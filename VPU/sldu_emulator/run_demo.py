
#!/usr/bin/env python3
"""SLDU emulator demo.

We create a 2D tile and fetch it with two different stride interpretations:
  1) Strides in BYTES (default): inner_stride/outer_stride are byte distances
  2) Strides in ELEMS: inner_stride/outer_stride are in elements, scaled by SEW bytes
We also show masking some elements.

Expected to run with plain Python 3.x (no extra deps).
"""
from emulator import SLDU, SlduReq

def hexdump(vec):
    return "[" + ", ".join(hex(x) for x in vec) + "]"

def main():
    sldu = SLDU(mem_size=4096)
    mem = sldu.mem

    # Prepare a 32-bit ascending array at 0x100 (64 words)
    base_arr = 0x100
    mem.fill_u32_inc(base_arr, 64)

    # Case 1: BYTES stride (row-major 3 x 4 block of 32-bit words)
    # inner_stride = 4 (bytes) moves to next 32-bit element
    # outer_stride = 4*8 (bytes) jumps 8 elements per row
    req1 = SlduReq(
        base=base_arr,
        sew=32,
        inner_count=4,
        outer_count=3,
        inner_stride=4,          # bytes
        outer_stride=32,         # bytes (8 elements * 4 bytes)
        stride_unit='bytes',
        vd='v0'
    )
    r1 = sldu.execute(req1)
    print('1) bytes-stride:', r1.ok, r1.info)
    print('   v0 =', hexdump(sldu.read_vreg('v0')))

    # Case 2: ELEMS stride (same geometry, but strides counted in elements)
    # inner_stride=1 elem, outer_stride=8 elems -> identical to case 1
    req2 = SlduReq(
        base=base_arr,
        sew=32,
        inner_count=4,
        outer_count=3,
        inner_stride=1,          # elems
        outer_stride=8,          # elems
        stride_unit='elems',
        vd='v1'
    )
    r2 = sldu.execute(req2)
    print('2) elems-stride:', r2.ok, r2.info)
    print('   v1 =', hexdump(sldu.read_vreg('v1')))

    # Case 3: Apply a mask to skip some elements
    mask = [True, False, True, True,
            False, True, False, True,
            True, True, False, False]
    req3 = SlduReq(
        base=base_arr,
        sew=32,
        inner_count=4,
        outer_count=3,
        inner_stride=1,
        outer_stride=8,
        stride_unit='elems',
        vd='v2',
        mask=mask
    )
    r3 = sldu.execute(req3)
    print('3) masked elems:', r3.ok, r3.info)
    print('   v2 =', hexdump(sldu.read_vreg('v2')))

if __name__ == '__main__':
    main()
