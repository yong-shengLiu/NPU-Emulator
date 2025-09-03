
# MaskU Emulator (Python)

Behavioral Python emulator for `masku.sv` and `masku_operands.sv`.

## Structure
```
masku_emulator/
  emulator/
    __init__.py
    types.py
    masku_operands.py
    masku.py
  run_demo.py
  README.md
```

## Quick Start
```bash
python run_demo.py
```

## Features
- Compare (lt, le, eq, ne, gt, ge)
- Logic (and, or, xor, andnot, ornot)
- Invert
- Select
- Apply (masked-off -> 0)
- Mask register file for named masks
