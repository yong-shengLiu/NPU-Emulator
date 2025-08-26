from dataclasses import dataclass, field
from collections import deque, defaultdict
from typing import List, Dict, Optional, Tuple, Any

#############################################
#  Out of Order buffer for temporary value  #
#############################################
tmp_exp = 1000
tmp_sum = 1001
@dataclass
class Instr:
    """
    TODO: this will become Macro op in the future
    """
    op: str                      # 'VLOAD','VSTORE','SOFTMAX','GELU','LN',...
    vd: Optional[int]            # dest vector register
    vs1: Optional[int] = None
    vs2: Optional[int] = None
    mask: Optional[int] = None   # mask reg id OR 1-bit enable
    imm: Optional[int] = None    # small immediate, mode flag, etc.

    # setup control instruction (TODO: Setup by CPU of instruction in the future?)
    len: int = 256               # vector length (elements)
    dtype: str = "int8"         # 'int8','int16','fp16','fp32'

    # performance model
    latency_hint: int = 1        # 預估 latency（可由 FU 覆寫）
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
    cycles: int = 0            # predicted for each micro-op


class Decoder:
    """
    Convert the macro-op (Instr) to micro-ops (MicroOp)
    (1) strip-mining
    """
    
    def __init__(self, hw_vlen=256):
        self.hw_vlen = hw_vlen  # the maximum vector length support by hardware

    def decode(self, instr: Instr) -> List[MicroOp]:
        micro_ops = []

        # 1. 判斷需要拆幾份 (strip-mining)
        total = instr.len
        num_chunks = (total + self.hw_vlen - 1) // self.hw_vlen

        for t in range(num_chunks):
            vstart = t * self.hw_vlen
            vl = min(self.hw_vlen, total - vstart)

            # 2. 轉換 op 類型
            if instr.op == "VLOAD":
                micro_ops.append(MicroOp(
                    instr=instr,
                    op="LOAD",
                    srcs1=[],   # LOAD 沒有來源 VReg
                    srcs2=None,
                    dst=instr.vd,
                    vstart=vstart,
                    vl=vl,
                    lmul=1,
                    tokens=num_chunks,
                    tag=f"{instr.op}_{t}",
                    cycles=5  # 假設 memory latency 5 cycles
                ))

            elif instr.op == "VSTORE":
                micro_ops.append(MicroOp(
                    instr=instr,
                    op="STORE",
                    srcs1=[instr.vs1] if instr.vs1 is not None else [],
                    srcs2=None,
                    dst=None,
                    vstart=vstart,
                    vl=vl,
                    lmul=1,
                    tokens=num_chunks,
                    tag=f"{instr.op}_{t}",
                    cycles=5
                ))

            elif instr.op == "SOFTMAX":
                # find maximum

                # subtract maximum

                # exp
                micro_ops.append(MicroOp(
                    instr=instr,
                    op="EXP",
                    srcs1=[instr.vs1],
                    srcs2=[instr.vs2] if instr.vs2 is not None else None,
                    dst=tmp_exp,
                    vstart=vstart,
                    vl=vl,
                    lmul=1,
                    tokens=num_chunks,
                    tag=f"{instr.op}_{t}",
                    cycles=4
                ))

                # reduce sum
                micro_ops.append(MicroOp(
                    instr=instr,
                    op="SLDU",
                    srcs1=tmp_exp,
                    srcs2=[instr.vs2] if instr.vs2 is not None else None,
                    dst=tmp_sum,
                    vstart=vstart,
                    vl=vl,
                    lmul=1,
                    tokens=num_chunks,
                    tag=f"{instr.op}_{t}",
                    cycles=4
                ))

                # divide
                micro_ops.append(MicroOp(
                    instr=instr,
                    op="DIV",
                    srcs1=tmp_sum,
                    srcs2=[instr.vs2] if instr.vs2 is not None else None,
                    dst=[instr.vd],
                    vstart=vstart,
                    vl=vl,
                    lmul=1,
                    tokens=num_chunks,
                    tag=f"{instr.op}_{t}",
                    cycles=4
                ))

            elif instr.op in ("GELU", "LN"):
                # fusion op → 可能拆成多個基本 micro ops
                # 這裡先簡化成一個 ALU micro-op
                micro_ops.append(MicroOp(
                    instr=instr,
                    op="ALU",
                    srcs1=[instr.vs1] if instr.vs1 is not None else [],
                    srcs2=[instr.vs2] if instr.vs2 is not None else None,
                    dst=instr.vd,
                    vstart=vstart,
                    vl=vl,
                    lmul=1,
                    tokens=num_chunks,
                    tag=f"{instr.op}_{t}",
                    cycles=10  # 比 ALU 更重，給個大一點 latency
                ))

            else:
                # 預設：一般 ALU op
                micro_ops.append(MicroOp(
                    instr=instr,
                    op="ALU",
                    srcs1=[instr.vs1] if instr.vs1 is not None else [],
                    srcs2=[instr.vs2] if instr.vs2 is not None else None,
                    dst=instr.vd,
                    vstart=vstart,
                    vl=vl,
                    lmul=1,
                    tokens=num_chunks,
                    tag=f"{instr.op}_{t}",
                    cycles=1
                ))

        return micro_ops


# 建立一個 Macro op (softmax)
instr = Instr(
    op="SOFTMAX",
    vd=1,
    vs1=2,
    len=256,
    dtype="int8"
)

decoder = Decoder(hw_vlen=256)
micro_ops = decoder.decode(instr)

for m in micro_ops:
    print(m)

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