

import sys, os, io, re, json, struct, zipfile, gzip
from pathlib import Path
import pandas as pd

# 선택 의존성(있는 경우만 사용)
try:
    import javaobj.v2 as javaobj
except Exception:
    javaobj = None

PNG_SIG = b"\x89PNG\r\n\x1a\n"
JPG_SIG = b"\xff\xd8\xff"
JAVA_SIG = b"\xac\xed\x00\x05"
ZIP_SIG = b"PK\x03\x04"
GZIP_SIG = b"\x1f\x8b"

OUTDIR = Path("gap_out")

def read_bytes(p):
    with open(p, "rb") as f:
        return f.read()

def ensure_out():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR/"plots").mkdir(exist_ok=True)
    (OUTDIR/"raw").mkdir(exist_ok=True)

def is_zip(b):  return b.startswith(ZIP_SIG)
def is_gzip(b): return b.startswith(GZIP_SIG)
def is_java(b): return b.startswith(JAVA_SIG)

def dump_text_probe(b, name):
    # 바이너리에서 텍스트 추출(휴리스틱)
    txt = b.decode("utf-8", errors="ignore")
    with open(OUTDIR/f"{name}.txt", "w", encoding="utf-8") as f:
        f.write(txt)
    return txt

def extract_embedded_images(b):
    """PNG/JPEG 시그니처 스캔으로 이미지 바이트 추출"""
    found = []
    # PNG
    start = 0
    while True:
        i = b.find(PNG_SIG, start)
        if i < 0: break
        j = b.find(b"IEND", i)
        if j < 0: break
        # IEND(4+4+IEND+CRC=12 bytes 근처) 보정
        end = b.find(b"\x82", j)  # 마지막 바이트까지 잡히게 느슨히
        if end < 0: end = j + 12
        blob = b[i:end+1]
        fname = OUTDIR/f"embedded_{len(found)+1}.png"
        with open(fname, "wb") as f: f.write(blob)
        found.append(fname)
        start = end+1
    # JPEG
    start = 0
    while True:
        i = b.find(JPG_SIG, start)
        if i < 0: break
        j = b.find(b"\xff\xd9", i+3)
        if j < 0: break
        blob = b[i:j+2]
        fname = OUTDIR/f"embedded_{len(found)+1}.jpg"
        with open(fname, "wb") as f: f.write(blob)
        found.append(fname)
        start = j+2
    return found

def try_parse_java(b):
    if not javaobj:
        return None, "javaobj-py3 미설치 또는 로딩 실패"
    try:
        obj = javaobj.loads(b)
        return obj, None
    except Exception as e:
        return None, f"javaobj 역직렬화 실패: {e}"

def walk_java(o, limit=20000):
    """Java 객체를 dict/리스트/원시형으로 best-effort 평탄화"""
    seen = set()
    def _conv(x):
        if id(x) in seen: return "<recursion>"
        seen.add(id(x))
        if isinstance(x, (str, bytes, int, float, bool)) or x is None:
            return x
        # javaobj 라이브러리의 POJO는 __dict__/fields 제공
        for attr in ("__dict__", "fields"):
            d = getattr(x, attr, None)
            if d and isinstance(d, dict):
                return {k:_conv(v) for k,v in list(d.items())[:1000]}
        # list-like
        if hasattr(x, "__iter__") and not isinstance(x, dict):
            try:
                return [_conv(v) for v in list(x)[:1000]]
            except Exception:
                return str(x)
        return str(x)
    return _conv(o)

def guess_tables_from_text(text):
    """
    텍스트 덤프에서 lane/band/intensity 등 표 형태를 찾는 간단 휴리스틱.
    실제 프로젝트에선 샘플 파일로 규칙을 고도화할 것.
    """
    # CSV 후보 라인만 추출
    lines = [ln for ln in text.splitlines() if ("," in ln and len(ln.split(","))>=3)]
    if not lines: return []
    # 헤더 추정
    candidates = []
    for i, ln in enumerate(lines[:100]):
        header = [c.strip().lower() for c in ln.split(",")]
        if any(k in ",".join(header) for k in ["lane","band","volume","intensity","mw"]):
            candidates.append(i)
    tables = []
    for idx in candidates:
        block = lines[idx: idx+200]
        csv_text = "\n".join(block)
        try:
            df = pd.read_csv(io.StringIO(csv_text))
            # 키워드 열 재명명(가능한 경우)
            rename = {}
            for col in df.columns:
                low = col.lower()
                if "lane" in low: rename[col] = "lane"
                elif "band" in low or "peak" in low: rename[col] = "band"
                elif "vol" in low or "intden" in low or "intensity" in low: rename[col] = "intensity"
                elif "mw" in low: rename[col] = "mw_kda"
                elif "bkg" in low or "back" in low: rename[col] = "background"
            if rename:
                df = df.rename(columns=rename)
            tables.append(df)
        except Exception:
            continue
    return tables

def main(path):
    ensure_out()
    b = read_bytes(path)
    info = {"path": str(path), "size": len(b)}
    info["magic"] = b[:8].hex()

    # 1) ZIP 컨테이너?
    if is_zip(b):
        info["container"] = "zip"
        with zipfile.ZipFile(io.BytesIO(b)) as zf:
            info["zip_entries"] = zf.namelist()
            # JSON/XML/CSV 우선 탐색
            extracted = []
            for name in zf.namelist():
                low = name.lower()
                if low.endswith((".json",".xml",".csv",".txt",".ini",".properties",".yaml",".yml",".png",".jpg",".jpeg")):
                    out = OUTDIR/Path(name).name
                    with zf.open(name) as f, open(out, "wb") as g:
                        g.write(f.read())
                    extracted.append(str(out))
            info["extracted"] = extracted

            # JSON/XML이 있으면 그걸 먼저 본다
            tables = []
            for out in extracted:
                if out.lower().endswith(".csv"):
                    try:
                        df = pd.read_csv(out)
                        tables.append(df)
                    except Exception:
                        pass
                elif out.lower().endswith(".json"):
                    try:
                        obj = json.load(open(out, "r", encoding="utf-8"))
                        # 흔한 키를 찾아 테이블로 변환 시도
                        for key in ["bands","lanes","results","table","data"]:
                            if isinstance(obj, dict) and key in obj and isinstance(obj[key], list):
                                df = pd.json_normalize(obj[key])
                                tables.append(df)
                    except Exception:
                        pass
            # 아무것도 못 찾으면 바이너리 전체에서 이미지/텍스트 추출
            if not tables:
                allbytes = b"".join(zf.read(n) for n in zf.namelist())
                imgs = extract_embedded_images(allbytes)
                txt = dump_text_probe(allbytes, "zip_text_probe")
                tables = guess_tables_from_text(txt)
    # 2) GZIP?
    elif is_gzip(b):
        info["container"] = "gzip"
        decomp = gzip.decompress(b)
        open(OUTDIR/"gap_gzip_payload.bin","wb").write(decomp)
        imgs = extract_embedded_images(decomp)
        txt = dump_text_probe(decomp, "gzip_text_probe")
        tables = guess_tables_from_text(txt)
    # 3) Java 직렬화?
    elif is_java(b):
        info["container"] = "java_serialization"
        obj, err = try_parse_java(b)
        if obj is not None:
            flat = walk_java(obj)
            with open(OUTDIR/"java_flat.json","w",encoding="utf-8") as f:
                json.dump(flat, f, ensure_ascii=False, indent=2)
            # dict/list에서 bands/lanes 후보를 테이블화
            tables = []
            def collect(o):
                if isinstance(o, dict):
                    for k,v in o.items():
                        if k.lower() in ("bands","lanes","results","table","data") and isinstance(v, list):
                            try:
                                tables.append(pd.json_normalize(v))
                            except Exception:
                                pass
                        collect(v)
                elif isinstance(o, list):
                    for v in o: collect(v)
            collect(flat)
        else:
            dump_text_probe(b, "java_bytes_probe")
            tables = []
    else:
        # 정체불명: 이미지/텍스트를 통으로 탐색
        info["container"] = "unknown"
        imgs = extract_embedded_images(b)
        txt = dump_text_probe(b, "raw_text_probe")
        tables = guess_tables_from_text(txt)

    # 테이블 후처리: 대표 컬럼 표준화 및 저장
    saved = []
    for i, df in enumerate(tables):
        # 열 이름 표준화(가능 시)
        rename = {}
        for col in df.columns:
            low = col.lower()
            if "lane" in low and "lane" not in rename.values(): rename[col] = "lane"
            if ("band" in low or "peak" in low) and "band" not in rename.values(): rename[col] = "band"
            if any(k in low for k in ["volume","intden","intensity","area"]) and "intensity" not in rename.values():
                rename[col] = "intensity"
            if "mw" in low and "mw_kda" not in rename.values(): rename[col] = "mw_kda"
            if any(k in low for k in ["bkg","back"]) and "background" not in rename.values():
                rename[col] = "background"
        df = df.rename(columns=rename)
        out = OUTDIR/f"table_{i+1}.csv"
        df.to_csv(out, index=False)
        saved.append(str(out))

    with open(OUTDIR/"meta.json","w",encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"[OK] 컨테이너: {info['container']}, 바이트={info['size']}")
    print(f" - 메타: {OUTDIR/'meta.json'}")
    if saved: print(" - 추정 테이블:", *saved, sep="\n   ")
    print(" - 추출 이미지는 gap_out/embedded_*.png|jpg (존재 시)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python read_gap.py <file.gap>")
        sys.exit(1)
    main(sys.argv[1])
#py read_gap.py "1 kb Plus DNA Ladder.gap"
