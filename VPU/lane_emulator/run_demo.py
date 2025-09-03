
#!/usr/bin/env python3
"""Lane emulator demo.

Demo flow:
  - Create a Lane with vlen=8
  - Initialize v0, v1 with patterns
  - Program:
      1) v2 = v0 + v1       (add, saturating off)
      2) v3 = v2 >> 2 (SAR) (arith shift-right with nearest_even rounding)
      3) power gate OFF
      4) v4 = v3 ^ v1       (this op is skipped due to power gate)
      5) power gate ON
      6) v4 = v3 ^ v1       (now executed)
"""
from emulator import Lane, LaneConfig, MicroOp

def show(label, vec):
    print(label, '[' + ', '.join(str(x) for x in vec) + ']')

def main():
    cfg = LaneConfig(vlen=8, nregs=32, dtype='i16')
    lane = Lane(cfg)

    # Initialize registers
    lane.vrf.write(0, [100, 200, -300, 400, 500, -600, 700, -800], 'i16')
    lane.vrf.write(1, [1, 2, 3, 4, 5, 6, 7, 8], 'i16')

    prog = [
        MicroOp(op='add', rd=2, rs1=0, rs2=1, dtype='i16', info='v2=v0+v1'),
        MicroOp(op='sar', rd=3, rs1=2, rs2=None, imm=2, dtype='i16', round_mode='nearest_even', info='v3=v2>>2'),
    ]

    lane.load_program(prog)
    lane.run()
    show('v0 =', lane.vrf.read(0))
    show('v1 =', lane.vrf.read(1))
    show('v2 =', lane.vrf.read(2))
    show('v3 =', lane.vrf.read(3))

    # Power gate OFF and attempt an operation (should be skipped)
    lane.pg.off()
    lane.load_program([MicroOp(op='xor', rd=4, rs1=3, rs2=1, dtype='i16', info='v4=v3^v1 (PG off)')])
    lane.run()
    show('v4 (PG off) =', lane.vrf.read(4))  # expect default zeros

    # Power gate ON and do it again
    lane.pg.on()
    lane.load_program([MicroOp(op='xor', rd=4, rs1=3, rs2=1, dtype='i16', info='v4=v3^v1')])
    lane.run()
    show('v4 (PG on) =', lane.vrf.read(4))

if __name__ == '__main__':
    main()
