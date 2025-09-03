
from emulator import VLSU, Memory, PeReq
from pprint import pprint

def show_region(mem: Memory, base: int, nbytes: int):
    data = mem.read_bytes(base, nbytes)
    print(f"[0x{base:08x}..0x{base+nbytes-1:08x}] =", data.hex(" "))

def main():
    vlsu = VLSU()
    mem = vlsu.mem

    # Fill two regions with patterns
    src_base = 0x1000
    dst_base = 0x2000
    mem.fill_pattern(src_base, 256, pattern=0x10)
    mem.fill_pattern(dst_base, 256, pattern=0x80)

    print("Before:")
    show_region(mem, src_base, 32)
    show_region(mem, dst_base, 32)

    # 1) Unit-stride LOAD of 8 elements, 32-bit, into v0
    req_load_unit = PeReq(
        op="load", addr_mode="unit", base=src_base, vl=8, sew=32, vd="v0"
    )
    r = vlsu.execute(req_load_unit)
    print("Unit-stride load:", r.ok, r.info)
    print("v0 =", vlsu.read_v("v0"))

    # 2) STRIDED STORE of 8 elements, 32-bit, from v0 to dst with stride 8 bytes
    req_store_strided = PeReq(
        op="store", addr_mode="strided", base=dst_base, vl=8, sew=32, vs="v0", stride=8
    )
    r = vlsu.execute(req_store_strided)
    print("Strided store:", r.ok, r.info)
    show_region(mem, dst_base, 64)

    # 3) INDEXED LOAD into v1 using inline indices (in elements, not bytes)
    indices = [0, 2, 4, 6, 8, 10, 12, 14]  # gather even words from src_base
    req_load_indexed = PeReq(
        op="load", addr_mode="indexed", base=src_base, vl=8, sew=32, vd="v1",
        index=indices, index_is_bytes=False
    )
    r = vlsu.execute(req_load_indexed)
    print("Indexed load:", r.ok, r.info)
    print("v1 =", vlsu.read_v("v1"))

    # 4) MASKED STORE from v1 back to dst_base+0x100 (unit-stride), mask off odd lanes
    mask = [True if i % 2 == 0 else False for i in range(8)]
    req_store_masked = PeReq(
        op="store", addr_mode="unit", base=dst_base + 0x100, vl=8, sew=32, vs="v1",
        mask=mask
    )
    r = vlsu.execute(req_store_masked)
    print("Masked store:", r.ok, r.info)
    show_region(mem, dst_base + 0x100, 64)

    print("Done.")

if __name__ == "__main__":
    main()
