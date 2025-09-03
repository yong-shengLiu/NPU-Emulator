
# Vector Lane Emulator (Python)

This package is a **behavioral** emulator derived from your RTL modules:
- `lane.sv` (top)                 → `emulator/lane.py`
- `vector_regfile.sv`             → `emulator/vector_regfile.py`
- `simd_alu.sv`                   → `emulator/simd_alu.py`
- `fixed_p_rounding.sv`           → `emulator/fixed_p_rounding.py`
- `operand_queue.sv`              → `emulator/operand_queue.py`
- `operand_queues_stage.sv`       → `emulator/operand_queues_stage.py`
- `operand_requester.sv`          → `emulator/operand_requester.py`
- `vector_fus_stage.sv`           → `emulator/vector_fus_stage.py`
- `lane_sequencer.sv`             → `emulator/lane_sequencer.py`
- `power_gating_generic.sv`       → `emulator/power_gating.py`

It focuses on the **functional semantics** (no cycle timing). You can load a list of `MicroOp`s and run them through a simplified pipeline.

## Layout
```
lane_emulator/
  emulator/
    __init__.py
    types.py
    vector_regfile.py
    simd_alu.py
    fixed_p_rounding.py
    operand_queue.py
    operand_queues_stage.py
    operand_requester.py
    vector_fus_stage.py
    lane_sequencer.py
    power_gating.py
    lane.py
  run_demo.py
  README.md
```

## Quick Start
```bash
python /mnt/data/lane_emulator/run_demo.py
```

## Notes
- Supported ops: `add, sub, and, or, xor, shl, shr, sar` (+ rounding for right-shifts).
- `MicroOp.mask` is accepted by the ALU; you can extend `run_demo.py` to exercise it.
- Power gating: when OFF, ops are skipped (destination keeps last written value).
- Extend easily to add more FUs, exception behavior, or your ISA mapping.
