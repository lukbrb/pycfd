import numpy as np
import pytest

from src import states
import src.params as params
from src.varindexes import IR, IBX, IBY, IBZ


def make_zero_grid():
    # Grid shape: (Ntx, Nty, Nfields)
    shape = (params.Ntx, params.Nty, params.Nfields)
    return np.zeros(shape, dtype=float)


def test_state_arithmetic_basic():
    s = states.State(np.arange(params.Nfields, dtype=float))
    t = states.State(np.ones(params.Nfields, dtype=float))

    # Addition
    a = s + t
    assert isinstance(a, states.State)
    assert np.allclose(a, np.arange(params.Nfields) + 1)

    # Subtraction
    b = s - t
    assert isinstance(b, states.State)
    assert np.allclose(b, np.arange(params.Nfields) - 1)

    # Multiplication
    c = s * 2
    assert isinstance(c, states.State)
    assert np.allclose(c, np.arange(params.Nfields) * 2)

    # True division
    d = c / 2
    assert isinstance(d, states.State)
    assert np.allclose(d, np.arange(params.Nfields))

    # In-place
    e = states.State(np.ones(params.Nfields))
    e += 3
    assert isinstance(e, states.State)
    assert np.allclose(e, 4)


def test_unary_and_comparisons():
    s = states.State(np.array([0.0, -1.0, 2.0, 0.0, 1.0, 0, 0, 0, 0]))
    neg = -s
    assert isinstance(neg, states.State)
    assert np.allclose(neg, -np.array(s))

    assert (s == s).all()
    assert not (s != s).any()


def test_cell_prim_cons_roundtrip():
    # build a primitive state q (rho, u, v, w, p, bx, by, bz, psi)
    q = states.State(np.array([1.2, 0.3, -0.1, 0.0, 2.5, 0.01, -0.02, 0.0, 0.0]))

    u = states.cell_primToCons(q)
    assert isinstance(u, states.State)
    # mass conserved
    assert pytest.approx(u[IR]) == q[IR]

    # convert back
    q2 = states.cell_consToPrim(u)
    assert isinstance(q2, states.State)
    # density and magnetic field components should match within tolerance
    assert pytest.approx(q2[IR], rel=1e-12) == q[IR]
    assert pytest.approx(q2[IBX], rel=1e-12) == q[IBX]
    assert pytest.approx(q2[IBY], rel=1e-12) == q[IBY]
    assert pytest.approx(q2[IBZ], rel=1e-12) == q[IBZ]


def test_get_set_state_into_array():
    Q = make_zero_grid()
    i, j = params.ibeg, params.jbeg

    q = states.State(np.array([2.0, 1.0, 0.0, 0.0, 3.0, 0.1, 0.2, 0.3, 0.0]))
    # write into Q
    states.set_state_into_array(Q, i, j, q)

    # read back
    q_read = states.get_state_from_array(Q, i, j)
    assert isinstance(q_read, states.State)
    assert np.allclose(q_read, q)


def test_grid_conversion_helpers():
    # Create grid arrays
    Q = make_zero_grid()
    U = make_zero_grid()

    # Fill Q with a simple primitive state at each domain cell
    for (ii, jj) in params.range_dom:
        states.set_state_into_array(Q, ii, jj, states.State(np.array([1.0, 0.5, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0])))

    # Convert prim->cons into U
    states.primToCons(Q, U)
    # Convert cons->prim back into Q2
    Q2 = make_zero_grid()
    states.consToPrim(U, Q2)

    # check a sample domain cell
    sample_i, sample_j = params.ibeg, params.jbeg
    q_orig = states.get_state_from_array(Q, sample_i, sample_j)
    q_round = states.get_state_from_array(Q2, sample_i, sample_j)
    assert np.allclose(q_orig, q_round)
