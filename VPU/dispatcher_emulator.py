
from dataclasses import dataclass, field
from typing import Optional, List, Deque, Dict, Any, Tuple
from collections import deque
import enum


############################
# ISA / Instruction model  #
############################
@dataclass
class Instr:
    """
    TODO: this will add custom op, like exp, div, etc.
    """
    op:   str
    mask: Optional[int] = None   # mask reg id OR 1-bit enable

    # vector
    vs1: Optional[int] = None
    vs2: Optional[int] = None
    vs3: Optional[int] = None    # the source VREG to store main memory 
    vd:  Optional[int] = None    # dest vector register

    # scalar
    rs1: Optional[int] = None    # base addr / scalar source
    rs2: Optional[int] = None    # stride
    rd:  Optional[int] = None    # reduction dest

    # setup control instruction
    imm: Optional[int] = None    # small immediate, mode flag, etc.

    # performance model
    latency_hint: int = 1        # 預估 latency（可由 FU 覆寫）
    name: str = ""               # 除錯顯示


# this is the micro op. constructed by RISC-V RVV
@dataclass
class ACCRequest:
    '''
    typedef struct packed {
        logic                             req_valid;
        logic                             resp_ready;
        logic [31:0]                      insn;         // check
        logic [CVA6Cfg.XLEN-1:0]          rs1;          // check
        logic [CVA6Cfg.XLEN-1:0]          rs2;          // check
        fpnew_pkg::roundmode_e            frm;
        logic [CVA6Cfg.TRANS_ID_BITS-1:0] trans_id;     // check
        logic                             store_pending;
        logic                             acc_cons_en;
        logic                             inval_ready;
    } accelerator_req_t;
    '''
    instr: Instr
    rs1_value: int = 0
    rs2_value: int = 0
    trans_ID:  int = 0   # issue id for top scoreboard


# this is ara backend request
@dataclass
class ARARequest:
    '''
    typedef struct packed {
        ara_op_e op; // Operation                       // check

        // Stores and slides do not re-shuffle the
        // source registers. In these two cases, vl refers
        // to the target EEW and vtype.vsew, respectively.
        // Since operand requesters work with the old
        // eew of the source registers, we should rescale
        // vl to the old eew to fetch the correct number of Bytes.
        //
        // Another solution would be to pass directly the target
        // eew (vstores) or the vtype.vsew (vslides), but this would
        // create confusion with the current naming convention
        logic scale_vl;                                // check

        // Mask vector register operand
        logic vm;                                      // check
        rvv_pkg::vew_e eew_vmask;

        // 1st vector register operand
        logic [4:0] vs1;                               // check
        logic use_vs1;
        opqueue_conversion_e conversion_vs1;
        rvv_pkg::vew_e eew_vs1;
        rvv_pkg::vew_e old_eew_vs1;

        // 2nd vector register operand
        logic [4:0] vs2;                               // check
        logic use_vs2;
        opqueue_conversion_e conversion_vs2;
        rvv_pkg::vew_e eew_vs2;

        // Use vd as an operand as well (e.g., vmacc)
        logic use_vd_op;
        rvv_pkg::vew_e eew_vd_op;

        // Scalar operand
        elen_t scalar_op;                            // check
        logic use_scalar_op;

        // 2nd scalar operand: stride for constant-strided vector load/stores, slide offset for vector
        // slides
        elen_t stride;
        logic is_stride_np2;

        // Destination vector register
        logic [4:0] vd;                             // check
        logic use_vd;

        // If asserted: vs2 is kept in MulFPU opqueue C, and vd_op in MulFPU A
        logic swap_vs2_vd_op;

        // Effective length multiplier
        rvv_pkg::vlmul_e emul;

        // Number of segments in segment mem op
        logic [2:0] nf;

        // Is this a fault-only-first load?
        logic fault_only_first;

        // Rounding-Mode for FP operations
        fpnew_pkg::roundmode_e fp_rm;
        // Widen FP immediate (re-encoding)
        logic wide_fp_imm;
        // Resizing of FP conversions
        resize_e cvt_resize;

        // Vector machine metadata
        vlen_t vl;                                 // check
        vlen_t vstart;                             // check
        rvv_pkg::vtype_t vtype;                    // check

        // Request token, for registration in the sequencer
        logic token;
    } ara_req_t;
    '''

    op: str               # VADD, VLE, VSE, etc.
    vl: int               # vector length (elements)
    mask: bool = False

    use_vs1: bool = False
    vs1: Optional[int] = None

    use_vs2: bool = False
    vs2: Optional[int] = None

    use_vd: bool = False
    vd: Optional[int] = None

    use_scalar_op: bool = False
    scalar_op: Optional[int] = None


    vl: int
    vstart: int
    vtype: int
    # issue modules
    # 1. Vector Arithmetic: OPCFG, OPIVV, OPIVX, OPIVI, OPMVV, OPMVX, OPFVV, OPFVF
    # 2. Vector Loads:      VLE, VLSE, VLXE
    # 3. Vector Stores:     VSE, VSSE, VSXE
    # 4. CSR Read Write:    VSTART, VXRM, VXSAT, VCSR, VROFFSET, VRENAME, CONV_OP


class Dispatcher:
    def __init__(self):
        pass
    
    def decode(self, acc_req: ACCRequest) -> ARARequest:
        instr = acc_req.instr
        print(instr)
        
        
        
        # Decoder the ara backend
        ara_req = ARARequest(
            op=instr.op,
            vl=instr.imm if instr.imm is not None else 0,
            mask=(instr.mask is not None),
            use_vs1=(instr.vs1 is not None),
            vs1=instr.vs1,
            use_vs2=(instr.vs2 is not None),
            vs2=instr.vs2,
            use_vd=(instr.vd is not None),
            vd=instr.vd,
            use_scalar_op=(instr.rs1 is not None),
            scalar_op=acc_req.rs1_value if instr.rs1 is not None else None,
        )
        return ara_req


if __name__ == "__main__":
    disp = Dispatcher()
    acc_req = ACCRequest(
        instr=Instr(op="VADD", vs1=1, vs2=2, vd=3, imm=16),
        rs1_value=100,
        rs2_value=4
    )

    ara_req = disp.decode(acc_req)
    print(ara_req)

# @dataclass
# class Request:
#     id: int
#     payload: Dict[str, Any] = field(default_factory=dict)
#     preferred_lane: int = -1
#     exec_cycles: int = 1

# @dataclass
# class Response:
#     id: int
#     lane: int
#     cycles_spent: int
#     payload: Dict[str, Any] = field(default_factory=dict)

# class ArbPolicy(enum.Enum):
#     ROUND_ROBIN = "round_robin"
#     LEAST_LOAD = "least_load"

# class RoundRobinArbiter:
#     def __init__(self, n: int):
#         self.n = n
#         self.ptr = 0
#     def pick(self, ready: List[bool]) -> Optional[int]:
#         for off in range(self.n):
#             i = (self.ptr + off) % self.n
#             if ready[i]:
#                 self.ptr = (i + 1) % self.n
#                 return i
#         return None

# from collections import deque
# @dataclass
# class Lane:
#     id: int
#     queue_depth: int
#     credits: int
#     executing: Optional[Tuple[Request, int]] = None
#     q: Deque[Request] = field(default_factory=deque)
#     def can_accept(self) -> bool:
#         return len(self.q) < self.queue_depth
#     def enqueue(self, req: Request) -> bool:
#         if not self.can_accept():
#             return False
#         self.q.append(req)
#         return True
#     def tick(self) -> Optional[Response]:
#         finished: Optional[Response] = None
#         if self.executing is None and self.q and self.credits > 0:
#             req = self.q.popleft()
#             self.executing = (req, req.exec_cycles)
#             self.credits -= 1
#         if self.executing is not None:
#             req, rem = self.executing
#             rem -= 1
#             if rem <= 0:
#                 finished = Response(id=req.id, lane=self.id, cycles_spent=req.exec_cycles, payload=req.payload)
#                 self.executing = None
#                 self.credits += 1
#             else:
#                 self.executing = (req, rem)
#         return finished

# class Dispatcher:
#     def __init__(self, num_lanes: int, queue_depth: int = 4, init_credits: int = 1, arb_policy: ArbPolicy = ArbPolicy.ROUND_ROBIN, trace: bool = True):
#         self.num_lanes = num_lanes
#         self.lanes = [Lane(i, queue_depth=queue_depth, credits=init_credits) for i in range(num_lanes)]
#         self.incoming = deque()
#         self.time = 0
#         self.trace_enabled = trace
#         self.rr = RoundRobinArbiter(num_lanes)
#         self.arb_policy = arb_policy
#         self.completed = []
#         self.enq_drops = 0
#     def push_request(self, req: Request) -> None:
#         self.incoming.append(req)
#         if self.trace_enabled:
#             print(f"[t={self.time}] IN  req#{req.id} preferred={req.preferred_lane} cycles={req.exec_cycles}")
#     def lane_set_credits(self, lane_id: int, credits: int) -> None:
#         self.lanes[lane_id].credits = credits
#     def lane_load(self, lane_id: int) -> int:
#         l = self.lanes[lane_id]
#         return len(l.q) + (1 if l.executing else 0)
#     def pick_lane(self, req: Request) -> Optional[int]:
#         if 0 <= req.preferred_lane < self.num_lanes and self.lanes[req.preferred_lane].can_accept():
#             return req.preferred_lane
#         if self.arb_policy == ArbPolicy.ROUND_ROBIN:
#             ready = [ln.can_accept() for ln in self.lanes]
#             return self.rr.pick(ready)
#         else:
#             candidates = [(self.lane_load(i), i) for i in range(self.num_lanes) if self.lanes[i].can_accept()]
#             return min(candidates)[1] if candidates else None
#     def tick(self):
#         if self.incoming:
#             req = self.incoming[0]
#             lane_id = self.pick_lane(req)
#             if lane_id is not None:
#                 self.incoming.popleft()
#                 ok = self.lanes[lane_id].enqueue(req)
#                 if self.trace_enabled:
#                     if ok:
#                         print(f"[t={self.time}] DIS -> lane{lane_id} req#{req.id}")
#                     else:
#                         print(f"[t={self.time}] DROP lane{lane_id} full req#{req.id}")
#                         self.enq_drops += 1
#         finished = []
#         for ln in self.lanes:
#             res = ln.tick()
#             if res:
#                 finished.append(res)
#                 self.completed.append(res)
#                 if self.trace_enabled:
#                     print(f"[t={self.time}] OUT <- lane{res.lane} req#{res.id} done")
#         self.time += 1
#         return finished
#     def snapshot(self):
#         return {
#             "time": self.time,
#             "incoming": [r.id for r in list(self.incoming)],
#             "lanes": [
#                 {
#                     "id": ln.id,
#                     "q": [r.id for r in list(ln.q)],
#                     "executing": (ln.executing[0].id, ln.executing[1]) if ln.executing else None,
#                     "credits": ln.credits,
#                 }
#                 for ln in self.lanes
#             ],
#             "completed": [r.id for r in self.completed],
#             "drops": self.enq_drops,
#         }

# def demo_run():
#     disp = Dispatcher(num_lanes=4, queue_depth=2, init_credits=1, arb_policy=ArbPolicy.ROUND_ROBIN, trace=True)
#     rid = 0
#     for cyc in range(20):
#         if cyc < 8:
#             disp.push_request(Request(id=rid, exec_cycles=1 + (rid % 3), preferred_lane=-1))
#             rid += 1
#         disp.tick()
#     return disp.snapshot()

# if __name__ == "__main__":
#     snap = demo_run()
#     print(snap)
