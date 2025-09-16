import numpy as np
import pytest

import src.params as params
from src.pycfd_types import IDir
from src import boundaries
from src.states import State, set_state_into_array, get_state_from_array


def make_zero_grid():
	shape = (params.Ntx, params.Nty, params.Nfields)
	return np.zeros(shape, dtype=float)


def test_fill_absorbing_returns_same_state():
	Q = make_zero_grid()
	# set a reference cell
	ref = State(np.array([1.0, 0.2, 0.0, 0.0, 2.0, 0.01, 0.02, 0.03, 0.0]))
	set_state_into_array(Q, params.ibeg, params.jbeg, ref)

	# call fillAbsorbing at (ibeg, jbeg) referencing itself -> should return same
	q = boundaries.fillAbsorbing(Q, params.ibeg, params.jbeg, None)
	assert isinstance(q, State)
	assert np.allclose(q, ref)


def test_fill_reflecting_x():
	Q = make_zero_grid()
	# Put a state near left domain interior and test reflecting on x
	interior = State(np.array([1.0, 0.5, 0.0, 0.0, 2.0, 0.1, 0.0, 0.0, 0.0]))
	set_state_into_array(Q, params.ibeg, params.jbeg, interior)

	# create a ghost cell left of ibeg
	i_ghost = params.ibeg - 1
	j = params.jbeg
	q = boundaries.fillReflecting(Q, i_ghost, j, params.ibeg, params.jbeg, IDir.IX)
	# velocity in x and bx should flip sign
	assert q[1] == pytest.approx(-interior[1])
	assert q[5] == pytest.approx(-interior[5])


def test_fill_periodic_x():
	Q = make_zero_grid()
	# Put a state at the leftmost domain cell (ibeg)
	left_cell_index = params.ibeg
	state_left = State(np.array([2.0, 0.7, 0.0, 0.0, 3.0, 0.2, 0.0, 0.0, 0.0]))
	set_state_into_array(Q, left_cell_index, params.jbeg, state_left)

	# ghost cell to the right (i == params.iend) should map to leftmost domain cell when periodic
	i_ghost = params.iend
	q = boundaries.fillPeriodic(Q, i_ghost, params.jbeg, IDir.IX)
	assert np.allclose(q, state_left)


def test_fill_reflecting_y():
	Q = make_zero_grid()
	# Put an interior state and test reflecting on y
	interior = State(np.array([1.0, 0.0, 0.6, 0.0, 2.0, 0.0, 0.1, 0.0, 0.0]))
	set_state_into_array(Q, params.ibeg, params.jbeg, interior)

	# ghost cell below jbeg
	i = params.ibeg
	j_ghost = params.jbeg - 1
	q = boundaries.fillReflecting(Q, i, j_ghost, i, params.jbeg, IDir.IY)
	# velocity in y and by should flip sign
	assert q[2] == pytest.approx(-interior[2])
	assert q[6] == pytest.approx(-interior[6])


def test_fill_periodic_y():
	Q = make_zero_grid()
	# Put a state at the bottom-most domain cell (jbeg)
	bottom_cell_index = params.jbeg
	state_bottom = State(np.array([2.0, 0.0, 0.7, 0.0, 3.0, 0.0, 0.2, 0.0, 0.0]))
	set_state_into_array(Q, params.ibeg, bottom_cell_index, state_bottom)

	# ghost cell below (j == params.jbeg - 1) should map to top when periodic
	j_ghost = params.jbeg - 1
	q = boundaries.fillPeriodic(Q, params.ibeg, j_ghost, IDir.IY)
	assert np.allclose(q, state_bottom)


def test_fill_boundaries_periodic_applies_to_ghosts():
	# Test fillBoundaries uses BC_PERIODIC by default in params
	# create grid and set one domain cell value, then call fillBoundaries
	Q = make_zero_grid()
	sample = State(np.array([1.0, 0.3, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]))
	set_state_into_array(Q, params.ibeg, params.jbeg, sample)

	# ensure boundary types are periodic for the test
	old_x = params.boundary_x
	old_y = params.boundary_y
	params.boundary_x = 'BC_PERIODIC'
	params.boundary_y = 'BC_PERIODIC'
	try:
		boundaries.fillBoundaries(Q)
		# check left ghost at ibeg-1 equals interior at ibeg+Nx-1 (periodic wrap)
		left_ghost = get_state_from_array(Q, params.ibeg-1, params.jbeg)
		wrapped = get_state_from_array(Q, params.iend-1, params.jbeg)
		assert np.allclose(left_ghost, wrapped)
		# check bottom ghost at jbeg-1 equals interior at jend-1 (periodic wrap in y)
		bottom_ghost = get_state_from_array(Q, params.ibeg, params.jbeg-1)
		wrapped_y = get_state_from_array(Q, params.ibeg, params.jend-1)
		assert np.allclose(bottom_ghost, wrapped_y)
	finally:
		params.boundary_x = old_x
		params.boundary_y = old_y

