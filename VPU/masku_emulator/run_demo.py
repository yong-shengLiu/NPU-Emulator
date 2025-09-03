
#!/usr/bin/env python3
from emulator import MaskU, MaskReq

def show_mask(label, m):
    bits = ''.join('1' if x else '0' for x in m)
    print(f"{label}: {bits}  (popcnt={sum(1 for v in m if v)})")

def main():
    mu = MaskU()
    vec = [3, 7, 2, 9, 5, 1, 8, 4]
    vec2 = [10]*8
    thresh = 5

    r1 = mu.execute(MaskReq(mode="cmp", a=vec, b=thresh, cmp_op="lt", md="m0"))
    print("1) cmp:", r1.ok, r1.info); show_mask("m0", mu.read_mask("m0"))

    r2 = mu.execute(MaskReq(mode="cmp", a=vec2, b=10, cmp_op="eq", md="m1"))
    print("   ", r2.ok, r2.info); show_mask("m1", mu.read_mask("m1"))

    r3 = mu.execute(MaskReq(mode="logic", m0=mu.read_mask("m0"), m1=mu.read_mask("m1"), logic_op="and", md="m2"))
    print("2) logic:", r3.ok, r3.info); show_mask("m2", mu.read_mask("m2"))

    rinv = mu.execute(MaskReq(mode="invert", m0=mu.read_mask("m1"), md="m1n"))
    r4 = mu.execute(MaskReq(mode="logic", m0=mu.read_mask("m0"), m1=mu.read_mask("m1n"), logic_op="xor", md="m3"))
    print("   invert:", rinv.ok, rinv.info)
    print("   xor   :", r4.ok, r4.info); show_mask("m3", mu.read_mask("m3"))

    r5 = mu.execute(MaskReq(mode="apply", mask=mu.read_mask("m2"), data=vec))
    print("3) apply:", r5.ok, r5.info, "->", r5.data)

    r6 = mu.execute(MaskReq(mode="select", mask=mu.read_mask("m3"), x=vec, y=vec2))
    print("4) select:", r6.ok, r6.info, "->", r6.data)

if __name__ == "__main__":
    main()
