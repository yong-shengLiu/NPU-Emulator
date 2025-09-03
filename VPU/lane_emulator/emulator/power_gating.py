
class PowerGate:
    """Simplified power gating: when off, computations are skipped and outputs hold previous values."""
    def __init__(self):
        self.enabled = True

    def on(self):
        self.enabled = True

    def off(self):
        self.enabled = False

    def is_on(self) -> bool:
        return self.enabled
