
from .types import RoundMode

class FixedPRounding:
    @staticmethod
    def round_right_shift(val: int, sh: int, mode: RoundMode, signed: bool, bits: int) -> int:
        if sh <= 0:
            return val
        mask = (1<<bits)-1
        if signed:
            v = val & mask
            if v & (1<<(bits-1)):  # sign extend
                v = v - (1<<bits)
        else:
            v = val & mask
        if mode == 'trunc':
            return v >> sh if v >= 0 else (-((-v) >> sh))
        # compute remainder for rounding
        base = v >> sh
        rem = abs(v) & ((1<<sh)-1)
        half = 1<<(sh-1)
        if mode == 'floor':
            return base if v >= 0 else (base - (1 if rem != 0 else 0))
        if mode == 'ceil':
            return base + (1 if (v >= 0 and rem != 0) else 0)
        if mode == 'nearest_even':
            if rem > half: return base + (1 if v >= 0 else -1)
            if rem < half: return base
            # tie
            return base + (1 if (base & 1)==1 and v>=0 else 0)
        return base
