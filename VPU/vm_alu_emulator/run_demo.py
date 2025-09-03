
#!/usr/bin/env python3
"""VMFPU/VALU/SIMD demo.
Demonstrates integer ALU ops, SIMD mul/div, and simple FP ops with masks.
"""
from emulator import VMFPU, OpReq

def show(label, vec):
    print(label, '[' + ', '.join(str(x) for x in vec) + ']')

def main():
    vm = VMFPU()

    # Integer vectors (i16)
    a = [100, 200, -300, 400, 500, -600, 700, -800]
    b = [1, 2, 3, 4, 5, 6, 7, 8]
    mask = [True, True, False, True, True, False, True, True]

    # 1) Integer add (i16), masked
    r1 = vm.execute(OpReq(op='iadd', dtype='i16', src0=a, src1=b, mask=mask, dst='v0'))
    print('1)', r1.ok, r1.info); show('v0 =', vm.read_vreg('v0'))

    # 2) Logic XOR (i16)
    r2 = vm.execute(OpReq(op='ixor', dtype='i16', src0=a, src1=b, dst='v1'))
    print('2)', r2.ok, r2.info); show('v1 =', vm.read_vreg('v1'))

    # 3) Shift left by 1 (using scalar b=[1])
    r3 = vm.execute(OpReq(op='shl', dtype='i16', src0=a, src1=[1], dst='v2'))
    print('3)', r3.ok, r3.info); show('v2 =', vm.read_vreg('v2'))

    # 4) Integer multiply (i16), saturating
    r4 = vm.execute(OpReq(op='imul', dtype='i16', src0=a, src1=b, dst='v3', saturating=True))
    print('4)', r4.ok, r4.info); show('v3 =', vm.read_vreg('v3'))

    # Float vectors (f32) — use Python floats
    fa = [1.5, -2.0, 3.25, 4.0]
    fb = [0.5,  2.0, -1.0, 0.25]

    # 5) FP add
    r5 = vm.execute(OpReq(op='fadd', dtype='f32', src0=fa, src1=fb, dst='vf0'))
    print('5)', r5.ok, r5.info); show('vf0 =', vm.read_vreg('vf0'))

    # 6) FP mul (via SIMD_MUL)
    r6 = vm.execute(OpReq(op='fmul', dtype='f32', src0=fa, src1=fb, dst='vf1'))
    print('6)', r6.ok, r6.info); show('vf1 =', vm.read_vreg('vf1'))

    # 7) FP div with mask (mask off lane 2)
    fmask = [True, True, False, True]
    r7 = vm.execute(OpReq(op='fdiv', dtype='f32', src0=fa, src1=fb, mask=fmask, dst='vf2'))
    print('7)', r7.ok, r7.info); show('vf2 =', vm.read_vreg('vf2'))

if __name__ == '__main__':
    main()
