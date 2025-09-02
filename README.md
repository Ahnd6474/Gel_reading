# Gel Reading

Desktop UI for gel band detection with optional standard curve fitting.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python detect_gel.py
```

Load a gel image through the interface, adjust detection parameters, fit a
standard curve if desired, and export band measurements.

## Modules

- `gel_detection.py` – core image processing functions.
- `standard_curve.py` – standard curve parsing and fitting utilities.
