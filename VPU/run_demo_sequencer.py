
from sequencer_emulator import Sequencer, Instruction
try:
    from dispatcher_emulator import Dispatcher, ArbPolicy
    use_dispatcher = True
except Exception:
    use_dispatcher = False

def main():
    if use_dispatcher:
        disp = Dispatcher(num_lanes=2, queue_depth=2, init_credits=1, arb_policy=ArbPolicy.ROUND_ROBIN, trace=True)
    else:
        disp = None
    seq = Sequencer(issue_width=1, fifo_depth=8, trace=True)
    # enqueue some instructions
    for i in range(8):
        seq.push_instruction(Instruction(id=i, opname='VOP', exec_cycles=(i%3)+1))
    # run cycles
    for cyc in range(24):
        print('\n==== Cycle', cyc, '====')
        snap = seq.tick(disp)
        if disp:
            disp.tick()
        print("Snapshot:", snap)
        if disp:
            print("Disp:", disp.snapshot())

if __name__ == "__main__":
    main()
