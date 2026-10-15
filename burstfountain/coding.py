import numpy as np

EXP = np.zeros(510, dtype=np.uint8)
LOG = np.zeros(256, dtype=np.int16)
value = 1
for exponent in range(255):
    EXP[exponent] = value
    LOG[value] = exponent
    value <<= 1
    if value & 256:
        value ^= 0x11D
EXP[255:] = EXP[:255]


def multiply(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, dtype=np.uint8), np.asarray(b, dtype=np.uint8))
    return np.where((a == 0) | (b == 0), 0, EXP[LOG[a] + LOG[b]]).astype(np.uint8)


class Basis:
    def __init__(self, symbols, payload=0):
        self.symbols = symbols
        self.payload = payload
        self.rows = {}
        self.operations = 0

    @property
    def rank(self):
        return len(self.rows)

    def add(self, vector):
        row = np.asarray(vector, dtype=np.uint8).copy()
        for pivot in sorted(self.rows):
            if row[pivot]:
                row ^= multiply(row[pivot], self.rows[pivot])
                self.operations += 2 * len(row)
        nonzero = np.flatnonzero(row[:self.symbols])
        if not len(nonzero):
            return False
        pivot = int(nonzero[0])
        inverse = EXP[255 - LOG[row[pivot]]]
        row = multiply(row, inverse)
        self.operations += len(row)
        for old, basis in self.rows.items():
            if basis[pivot]:
                self.rows[old] = basis ^ multiply(basis[pivot], row)
                self.operations += 2 * len(row)
        self.rows[pivot] = row
        return True

    def recode(self, rng):
        if not self.rows:
            return None
        rows = np.stack(list(self.rows.values()))
        coefficients = rng.integers(0, 256, len(rows), dtype=np.uint8)
        if not coefficients.any():
            coefficients[0] = 1
        self.operations += rows.size * 2
        return np.bitwise_xor.reduce(multiply(coefficients[:, None], rows), axis=0)

    def decode(self):
        if self.rank != self.symbols:
            raise ValueError("Insufficient innovative symbols")
        return np.stack([self.rows[pivot][self.symbols:] for pivot in range(self.symbols)])
