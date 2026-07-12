import numpy as np

from IBLGF import sol


def _solver_shell(n=16):
    solver = sol.__new__(sol)
    solver.nx = n
    solver.ny = n
    solver.nx_ll = n // 2
    solver.ny_ll = n // 2
    return solver


def _cell_even(n):
    field = np.zeros((n, n))
    for j in range(n // 2):
        field[j, :] = j + 1
        field[n - 1 - j, :] = j + 1
    return field


def _node_odd(n):
    field = np.zeros((n, n))
    axis = n // 2
    for j in range(1, axis):
        j_ref = 2*axis - j
        if j_ref < n:
            field[j, :] = j
            field[j_ref, :] = -j
    return field


def test_clean_boundary_preserves_cell_centered_y_symmetry():
    solver = _solver_shell()
    field = _cell_even(solver.ny)[None, :, :]

    solver.cleanBdry(field, 2, grid_location="cell")

    error = solver.reflection_error_y(field[0], 1, y_offset=0.5)
    assert error["max_abs"] == 0


def test_clean_boundary_preserves_node_centered_y_symmetry():
    solver = _solver_shell()
    field = _node_odd(solver.ny)[None, :, :]

    solver.cleanBdry(field, 2, grid_location="node")

    error = solver.reflection_error_y(field[0], -1, y_offset=0.0)
    assert error["max_abs"] == 0


def test_clean_boundary_preserves_face_centered_y_symmetry():
    solver = _solver_shell()
    field = np.stack((_cell_even(solver.ny), _node_odd(solver.ny)))

    solver.cleanBdry(field, 2, grid_location="face")

    x_error = solver.reflection_error_y(field[0], 1, y_offset=0.5)
    y_error = solver.reflection_error_y(field[1], -1, y_offset=0.0)
    assert x_error["max_abs"] == 0
    assert y_error["max_abs"] == 0
