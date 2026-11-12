import random


def _smin(bits):
    return -(1 << (bits - 1))


def _smax(bits):
    return (1 << (bits - 1)) - 1


def constant_rows(n_vars, bits):
    """Les 5 vecteurs constants du papier Xyntia : 0, 1, -1, min, max. Communs à toutes les cibles."""
    return [[c] * n_vars for c in (0, 1, -1, _smin(bits), _smax(bits))]


def train_inputs(n_vars, bits=32, total=100, seed=0):
    """Protocole Xyntia : 95 vecteurs aléatoires dans [-50, 49], plus 5 vecteurs constants."""
    rnd = random.Random(seed)
    rows = [[rnd.randint(-50, 49) for _ in range(n_vars)] for _ in range(total - 5)]
    return rows + constant_rows(n_vars, bits)


def test_inputs(n_vars, bits=32, total=1000, seed=1):
    """Entrées de test sur tout le domaine signé, plus les vecteurs constants."""
    rnd = random.Random(seed)
    lo, hi = _smin(bits), _smax(bits)
    rows = [[rnd.randint(lo, hi) for _ in range(n_vars)] for _ in range(total)]
    return rows + constant_rows(n_vars, bits)


def to_env(row, bits):
    """Convertit une ligne signée en valeurs non signées sur `bits` bits."""
    mask = (1 << bits) - 1
    return {f"v{i}": x & mask for i, x in enumerate(row)}