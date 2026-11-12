import random


def _smin(bits):
    return -(1 << (bits - 1))


def _smax(bits):
    return (1 << (bits - 1)) - 1


def constant_rows(n_vars, bits):
    """Les 5 vecteurs constants du papier Xyntia : 0, 1, -1, min, max."""
    return [[c] * n_vars for c in (0, 1, -1, _smin(bits), _smax(bits))]


def bit_boundary_rows(n_vars, bits):
    """Valeurs qui exposent les bits de poids fort et les débordements."""
    shifts = sorted({0, 1, max(0, bits // 2), max(0, bits - 2)})
    values = [1 << shift for shift in shifts]
    values += [-(1 << shift) for shift in shifts]
    values += [_smin(bits) + 1, _smax(bits) - 1]
    return [[value] * n_vars for value in values]


def train_inputs(n_vars, bits=32, total=100, seed=0):
    """Paires d'observation : valeurs aléatoires et cas sensibles aux bits."""
    rnd = random.Random(seed)
    special = constant_rows(n_vars, bits) + bit_boundary_rows(n_vars, bits)
    special = special[:total]
    random_count = max(0, total - len(special))
    rows = [[rnd.randint(-50, 49) for _ in range(n_vars)] for _ in range(random_count)]
    return rows + special


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