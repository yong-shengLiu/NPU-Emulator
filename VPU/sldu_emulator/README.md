
# SLDU Emulator (Python)

This package emulates a simple **Segment/2D Strided Load Unit** based on your RTL blocks:
- `sldu.sv` (top) → `emulator/sldu.py` (SLDU)
- `sldu_op_dp.sv` → `emulator/sldu_op_dp.py` (OpDP datapath)
- `p2_stride_gen.sv` → `emulator/p2_stride_gen.py` (2D stride address generator)

It is a **behavioral** model (not cycle-accurate). The goal is to match addressing semantics and data movement.

## Layout
```
sldu_emulator/
  emulator/
    __init__.py
    types.py
    memory.py
    p2_stride_gen.py
    sldu_op_dp.py
    sldu.py
  run_demo.py
  README.md
```

## Quick start
```bash
python /mnt/data/sldu_emulator/run_demo.py
```

You should see three cases:
1. **Byte-based strides** (row-major, 3x4 tile of 32-bit words)
2. **Element-based strides** (same geometry; strides multiplied by element size internally)
3. **Masked load** (some elements skipped)

## API
- `SlduReq`: fields include `base`, `sew`, `inner_count`, `outer_count`, `inner_stride`, `outer_stride`, `stride_unit {'bytes'|'elems'}`, `vd`, `mask`.
- `SLDU.execute(req)` runs the load, writes values into VRF under `vd`.
- Read results with `SLDU.read_vreg('vX')`.

## Notes
- Endianness defaults to little-endian; you can change via `SlduReq.little_endian`.
- Stride units: if `bytes`, strides are used directly; if `elems`, strides are scaled by `sew/8`.
- Errors like out-of-bounds reads will be returned in `SlduResp.exception`.
