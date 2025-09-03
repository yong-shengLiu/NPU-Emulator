
from dispatcher_emulator import Dispatcher, ArbPolicy, Request
def main():
    disp = Dispatcher(num_lanes=3, queue_depth=2, init_credits=1, arb_policy=ArbPolicy.ROUND_ROBIN, trace=True)
    for i in range(6):
        disp.push_request(Request(id=i, exec_cycles=(i % 3) + 1))
    for _ in range(12):
        disp.tick()
    print("SNAPSHOT:", disp.snapshot())
if __name__ == "__main__":
    main()
