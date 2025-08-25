from dataclasses import dataclass, field
from collections import deque, defaultdict
from typing import List, Dict, Optional, Tuple, Any


@dataclass
class Instr:
    """TODO: this will become Macro op in the future"""
    op: str                      # 'VADD','VMUL','VLOAD','VSTORE','GELU','SOFTMAX','LN','PERM','VRED'
    vd: Optional[int]            # 目的向量暫存器 (None 表示 store 或控制類)
    vs1: Optional[int]
    vs2: Optional[int]
    mask: Optional[int] = None   # 簡化：用 bitmask id 或 bool
    imm: Optional[Any] = None    # 立即數/形狀/stride
    len: int = 256               # vector 長度 (元素數)
    dtype: str = "int16"         # 'int8','int16','fp16','fp32'
    latency_hint: int = 1        # 預估 latency（可由 FU 覆寫）
    mem_addr: Optional[int] = None
    name: str = ""               # 除錯顯示


@dataclass
class MicroOp:
    instr:  Instr
    
    # hardware wire related
    op: str                       # 'ALU','LOAD','STORE','NAF','PERM','RED'
    srcs1: List[int]              # Source 1 Vreg id
    srcs2: Optional[List[int]]    # Source 2 Vreg id
    dst: Optional[int]            # destination VReg id
    scalar: Optional[int] = None  # scalar register id
    vstart: int = 0               # vstart (for strip-mining)
    vl: int = 256                 # vector length (elements)
    lmul: int = 1                 # vector register group (1,2,4,8)

    # VRF-aware scheduling
    lane_affinity: Optional[int] = None  # to a micro operation task for lane scheduling
    tokens: int = 1            # total number for strip-mining, increasing order
    tag: str = ""              # token to record the macro-op instance

    # performance model
    cycles: int                # predicted for each micro-op


def _size_bytes(instr: Instr) -> int:
    bpe = {'int8':1,'int16':2,'fp16':2,'fp32':4}.get(instr.dtype,1)
    print(instr.dtype)
    return instr.len * bpe


# 假設有一條 VLOAD，vector 長度=256，每個元素=int16 (2 bytes)
instr = Instr(op="VLOAD", vd=0, vs1=None, vs2=None, len=256, dtype="fp32")

# 呼叫 _size_bytes
bytes_needed = _size_bytes(instr)
print(f'{bytes_needed} byte')   # 輸出 512



# class VectorRegFile:
#     def __init__(self, num: int, vlen: int):
#         self.num = num
#         self.vlen = vlen
#         # (ready_cycle, data_stub) for each VReg
#         self.state: List[Tuple[int, Any]] = [(0, None) for _ in range(num)]

#     def ready(self, ridx: int, cycle: int) -> bool:
#         return self.state[ridx][0] <= cycle

#     def read(self, ridx: int, cycle: int) -> Any:
#         assert self.ready(ridx, cycle), f"VReg {ridx} not ready @ {cycle}"
#         return self.state[ridx][1]

#     def write(self, ridx: int, data: Any, ready_cycle: int):
#         self.state[ridx] = (ready_cycle, data)


# # 建立一個有 8 個暫存器、每個長度 256 的 VRegFile
# vregs = VectorRegFile(num=8, vlen=256)

# # 一開始，所有暫存器的 ready_cycle = 0，data = None
# print(vregs.state)
# # [(0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None)]

# # 假設 cycle=5，我要寫入暫存器 2，結果在 cycle=10 才 ready
# vregs.write(ridx=2, data="VADD_RESULT", ready_cycle=10)

# # 檢查 cycle=5 時，VReg 2 是否 ready？
# print(vregs.ready(2, cycle=5))   # False

# # 檢查 cycle=12 時，VReg 2 是否 ready？
# print(vregs.ready(2, cycle=12))  # True

# # 嘗試在 cycle=5 讀暫存器 2 → 會出錯，為了避免程式中斷才使用此語法
# try:
#     vregs.read(2, cycle=5)
# except AssertionError as e:
#     print("錯誤:", e)

# # 在 cycle=12 時讀暫存器 2 → OK
# print(vregs.read(2, cycle=12))   # "VADD_RESULT"