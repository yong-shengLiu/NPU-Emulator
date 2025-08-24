# vpu_emul.py
from dataclasses import dataclass, field
from collections import deque, defaultdict
from typing import List, Dict, Optional, Tuple, Any
import math
import random

#############################################
#  Out of Order buffer for temporary value  #
#############################################
tmp_exp = 1000
tmp_sum = 1001

############################
# ISA / Instruction model  #
############################
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


#########################################
# Micro-ops (disassemble from Macro-op) #
#########################################
@dataclass
class MicroOp:
    instr: Instr
    kind: str                  # 'ALU','LOAD','STORE','NAF','PERM','RED'
    srcs: List[int]            # 來源 VReg id
    dst: Optional[int]         # 目的 VReg id
    cycles: int                # 預估 FU 週期
    lane_affinity: Optional[int] = None  # 指定 lane（可選）
    tokens: int = 1            # 要處理的 chunk 數 (for strip-mining)
    tag: str = ""              # debug

############################
# Register files / Preds   #
############################

class VectorRegFile:
    def __init__(self, num: int, vlen: int):
        self.num = num
        self.vlen = vlen
        # 每個暫存器存 (ready_cycle, data_stub)
        self.state: List[Tuple[int, Any]] = [(0, None) for _ in range(num)]

    def ready(self, rid: int, cycle: int) -> bool:
        return self.state[rid][0] <= cycle

    def read(self, rid: int, cycle: int) -> Any:
        assert self.ready(rid, cycle), f"VReg {rid} not ready @ {cycle}"
        return self.state[rid][1]

    def write(self, rid: int, data: Any, ready_cycle: int):
        self.state[rid] = (ready_cycle, data)

class PredRegFile:
    def __init__(self, num: int):
        self.mask: Dict[int, Any] = {i: None for i in range(num)}

############################
# Memory + Energy proxy    #
############################

class MemoryModel:
    def __init__(self, bandwidth_bytes_per_cycle=64, latency=50):
        self.bw = bandwidth_bytes_per_cycle
        self.lat = latency
        self.inflight = deque()  # (done_cycle, size_bytes)
        self.bytes_this_cycle = 0

    def begin_cycle(self, cycle: int):
        self.bytes_this_cycle = 0
        # retire finished
        while self.inflight and self.inflight[0][0] <= cycle:
            self.inflight.popleft()

    def can_issue(self, size_bytes: int) -> bool:
        return (self.bytes_this_cycle + size_bytes) <= self.bw

    def issue(self, size_bytes: int, cycle: int):
        self.bytes_this_cycle += size_bytes
        self.inflight.append((cycle + self.lat, size_bytes))
        return cycle + self.lat  # ready cycle

class EnergyModel:
    """非常粗略的能耗 proxy，供 Token/J 粗估"""
    COST = {
        'ALU': 1.0,       # per element
        'NAF': 2.5,       # GELU/Softmax/LN per element
        'PERM': 0.8,
        'RED': 1.2,
        'LOAD_PER_BYTE': 0.02,
        'STORE_PER_BYTE': 0.02,
        'XBAR': 0.3,      # lane 間搬移/互連
    }
    def __init__(self):
        self.energy = 0.0

    def add(self, kind: str, amount: float):
        self.energy += self.COST.get(kind, 0.0) * amount

############################
# Functional Units (FUs)   #
############################

class FUBase:
    def __init__(self, name: str, latency: int, throughput: int = 1):
        self.name = name
        self.latency = latency
        self.throughput = throughput
        self.busy_until = 0  # 可用的下一個 cycle

    def can_issue(self, cycle: int) -> bool:
        return cycle >= self.busy_until

    def issue(self, cycle: int) -> int:
        self.busy_until = cycle + self.latency
        return self.busy_until

class ALU(FUBase):
    pass

class NAF(FUBase):
    """Nonlinear Activation FU: GELU/Softmax/LN 等"""
    pass

class PERM(FUBase):
    pass

class RED(FUBase):
    pass

############################
# Lane & Interconnect      #
############################

class Lane:
    """一個 lane：包含 ALU/NAF/PERM/RED，可做 chaining（bypass write-back）"""
    def __init__(self, lane_id: int, vlen: int):
        self.id = lane_id
        self.vlen = vlen
        self.alu = ALU(f"ALU{lane_id}", latency=2)
        self.naf = NAF(f"NAF{lane_id}", latency=6)
        self.perm = PERM(f"PERM{lane_id}", latency=3)
        self.red = RED(f"RED{lane_id}", latency=4)
        self.bypass_buffer: Dict[str, Tuple[Any, int]] = {}  # tag -> (data, ready_cycle)

    def available_cycle(self, kind: str, now: int) -> int:
        fu = {'ALU': self.alu, 'NAF': self.naf, 'PERM': self.perm, 'RED': self.red}[kind]
        return max(now, fu.busy_until)

    def issue(self, uop: MicroOp, now: int) -> int:
        fu = {'ALU': self.alu, 'NAF': self.naf, 'PERM': self.perm, 'RED': self.red}[uop.kind]
        start = max(now, fu.busy_until)
        done = fu.issue(start)
        # chaining: 產生 bypass 可用（不必先寫回 VReg）
        self.bypass_buffer[uop.tag or uop.kind] = ("DATA", done)
        return done

class RingInterconnect:
    """簡單 deterministic ring，估計 lane 間傳輸延遲（for cross-lane permute/reduction）"""
    def __init__(self, num_lanes: int, hop_latency: int = 1):
        self.n = num_lanes
        self.hop = hop_latency

    def delay(self, src_lane: int, dst_lane: int) -> int:
        dist = (dst_lane - src_lane) % self.n
        return dist * self.hop

############################
# Decoder & Scheduler      #
############################

class Decoder:
    """
    Convert the macro-op (Instr) to micro-ops (MicroOp)
    (1) strip-mining
    """

    def decode(self, instr: Instr) -> List[MicroOp]:
        uops = []
        # mapping the micro operation to functional unit
        kind_map = {
            'VLE': 'LSU',   'VSE': 'LSU',
            'VADD': 'VALU', 'VSUB': 'VALU',
            'VMUL': 'VMFPU',
            'PERM': 'MASKU',
            'VREDMAX': 'SLDU', 'VREDSUM': 'SLDU',
            'SM': 'NAF', 'GELU': 'NAF', 'LN': 'NAF',
        }
        if instr.op in ('VLOAD', 'VSTORE'):
            # 在 scheduler 處理；這裡也可生成微指令
            pass
        elif instr.name == 'SoftMax':
            # find maximum

            # subtract maximum

            # exp
            uops.append(MicroOp(instr=instr, kind="NAF", srcs=[instr.vs1], dst=tmp_exp, cycles=4, tag="EXP"))

            # reduce sum
            uops.append(MicroOp(instr=instr, kind="SLDU", srcs=[tmp_exp], dst=tmp_sum, cycles=4, tag="RED_SUM"))

            # divide
            uops.append(MicroOp(instr=instr, kind="VALU", srcs=[tmp_exp, tmp_sum], dst=instr.vd, cycles=4, tag="DIV"))
        else:
            kind = kind_map[instr.op]
            print(f"Decoding {instr.name} to {kind}")
            uops.append(MicroOp(instr=instr, kind=kind, srcs=[instr.vs1, instr.vs2] if instr.vs2 is not None else [instr.vs1],
                                dst=instr.vd, cycles=instr.latency_hint, tokens=1, tag=instr.op))
        return uops

class Scheduler:
    """重點：ISSUE 規則、資源與鏈結限制、跨 lane 搬移、避免 hazard"""
    def __init__(self, lanes: List[Lane], vregs: VectorRegFile, mem: MemoryModel,
                 ic: RingInterconnect, energy: EnergyModel):
        self.lanes = lanes
        self.vregs = vregs
        self.mem = mem
        self.ic = ic
        self.energy = energy
        self.iq = deque()  # 發射佇列：uops 和 memory req

    def submit_instrs(self, decoded_uops: List[MicroOp], raw_instr: Instr):
        if raw_instr.op in ('VLOAD', 'VSTORE'):
            self.iq.append(raw_instr)  # 讓 memory 流程特別處理
        for u in decoded_uops:
            self.iq.append(u)

    def _size_bytes(self, instr: Instr) -> int:
        bpe = {'int8':1,'int16':2,'fp16':2,'fp32':4}.get(instr.dtype,2)
        return instr.len * bpe

    def step(self, cycle: int):
        """單 cycle 嘗試發射盡可能多的 uops（可限制每 cycle 最大發射數）"""
        issued = 0
        max_issue = 4  # 例如 4-issue
        next_q = deque()
        while self.iq and issued < max_issue:
            item = self.iq.popleft()
            if isinstance(item, MicroOp):
                # 檢查 operands 何時 ready（支援 chaining: 若上一個在同 lane 可 bypass）
                # 這裡做簡化：只檢查 VReg ready
                if any(r is not None and not self.vregs.ready(r, cycle) for r in item.srcs if r is not None):
                    next_q.append(item); continue

                # 選一條 lane（簡單 round-robin 或基於 kind 的 affinity）
                best_lane = min(self.lanes, key=lambda L: L.available_cycle(item.kind, cycle))
                # 若需要跨 lane 資料，加入 ring 延遲（此例簡化不做來源追蹤）
                start = best_lane.available_cycle(item.kind, cycle)
                done = best_lane.issue(item, start)

                # 能耗估：每元素成本
                if item.kind in ('ALU','NAF','PERM','RED'):
                    per_elem = {'ALU':'ALU','NAF':'NAF','PERM':'PERM','RED':'RED'}[item.kind]
                    self.energy.add(per_elem, amount=item.instr.len)

                # 寫回（若啟用 chaining，可延後或 bypass；這裡簡化：固定寫回）
                if item.dst is not None:
                    self.vregs.write(item.dst, data=f"{item.kind}_DATA", ready_cycle=done)

                issued += 1

            elif isinstance(item, Instr) and item.op in ('VLOAD','VSTORE'):
                size = self._size_bytes(item)
                if self.mem.can_issue(size):
                    ready = self.mem.issue(size, cycle)
                    if item.op == 'VLOAD' and item.vd is not None:
                        self.vregs.write(item.vd, data="MEM_DATA", ready_cycle=ready)
                    # 能耗 proxy
                    self.energy.add('LOAD_PER_BYTE' if item.op=='VLOAD' else 'STORE_PER_BYTE', amount=size)
                    issued += 1
                else:
                    next_q.append(item)  # 帶寬不夠，留待下個 cycle
            else:
                next_q.append(item)
        # 還沒發射的回補
        while self.iq:
            next_q.append(self.iq.popleft())
        self.iq = next_q

############################
# Top-level Simulator      #
############################

class VPUSim:
    def __init__(self, num_lanes=4, vlen=256, num_vregs=32):
        self.cycle = 0
        self.vregs = VectorRegFile(num_vregs, vlen)
        self.preds = PredRegFile(8)
        self.mem = MemoryModel(bandwidth_bytes_per_cycle=128, latency=60)
        self.energy = EnergyModel()
        self.lanes = [Lane(i, vlen) for i in range(num_lanes)]
        self.ic = RingInterconnect(num_lanes, hop_latency=1)
        self.decoder = Decoder()
        self.sched = Scheduler(self.lanes, self.vregs, self.mem, self.ic, self.energy)
        self.stats = defaultdict(int)

    def load_program(self, instrs: List[Instr]):
        for ins in instrs:
            uops = self.decoder.decode(ins)
            # print(f"Decoded {ins.name or ins.op}: {uops}")
            self.sched.submit_instrs(uops, ins)

    def step(self):
        self.mem.begin_cycle(self.cycle)
        self.sched.step(self.cycle)
        self.cycle += 1

    def run(self, max_cycles=100000):
        while self.sched.iq and self.cycle < max_cycles:
            self.step()
        return {
            "cycles": self.cycle,
            "energy_proxy": self.energy.energy,
            "token_per_j": None  # 你可依任務把 token/s 與 energy_proxy 換算
        }

############################
# Example usage            #
############################

def demo_program():
    """一個小小的 Transformer 子圖：LOAD -> VADD -> GELU -> LN -> STORE"""
    VLEN = 256
    return [
        Instr(op='SM',  vd=1, vs1=0, vs2=None, len=VLEN, dtype='int8', latency_hint=8, name='SoftMax'),
        # Instr(op='VLOAD',  vd=0, vs1=None, vs2=None, len=VLEN, dtype='fp16', mem_addr=0x1000, name='LdX'),
        # Instr(op='VLOAD',  vd=1, vs1=None, vs2=None, len=VLEN, dtype='fp16', mem_addr=0x2000, name='LdW'),
        # Instr(op='VADD',   vd=2, vs1=0,   vs2=1,     len=VLEN, dtype='fp16', name='Add'),
        # Instr(op='GELU',   vd=3, vs1=2,   vs2=None,  len=VLEN, dtype='fp16', latency_hint=6, name='GELU'),
        # Instr(op='LN',     vd=4, vs1=3,   vs2=None,  len=VLEN, dtype='fp16', latency_hint=8, name='LayerNorm'),
        # Instr(op='VSTORE', vd=None, vs1=4, vs2=None, len=VLEN, dtype='fp16', mem_addr=0x3000, name='StY'),
    ]

if __name__ == "__main__":
    sim = VPUSim(num_lanes=4, vlen=256, num_vregs=32)
    sim.load_program(demo_program())
    result = sim.run()
    print(result)
