import numpy as np
import scipy

from IBLGF import sol


def test_assembled_projection_operators_match_marker_loops():
    solver = sol.__new__(sol)
    solver.ny = 3
    solver.nx = 4
    solver.nIBP = 2
    solver.P = [
        [scipy.sparse.lil_matrix((3, 4)), scipy.sparse.lil_matrix((3, 4))]
        for _ in range(solver.nIBP)
    ]
    solver.P[0][0][0, 1] = 0.25
    solver.P[0][1][1, 2] = 0.75
    solver.P[1][0][2, 3] = 0.5
    solver.P[1][1][0, 0] = 1.25
    solver.P = [[component.tocsr() for component in marker] for marker in solver.P]
    solver.assemble_projection_operators()

    grid = np.arange(24, dtype=float).reshape(2, 3, 4)
    projected = np.zeros((solver.nIBP, 2))
    solver.projection(grid, projected)
    expected_projection = np.array([
        [np.sum(solver.P[i][0].multiply(grid[0])),
         np.sum(solver.P[i][1].multiply(grid[1]))]
        for i in range(solver.nIBP)
    ])
    np.testing.assert_allclose(projected, expected_projection)

    marker_values = np.array([[2.0, 3.0], [5.0, 7.0]])
    spread = np.zeros_like(grid)
    solver.smearing(marker_values, spread)
    expected_spread = np.zeros_like(grid)
    for i in range(solver.nIBP):
        expected_spread[0] += solver.P[i][0].toarray()*marker_values[i, 0]
        expected_spread[1] += solver.P[i][1].toarray()*marker_values[i, 1]
    np.testing.assert_allclose(spread, expected_spread)


def test_batched_fft_operators_match_componentwise_evaluation():
    solver = sol.__new__(sol)
    solver.ny = 5
    solver.nx = 7
    solver.dx = 0.2
    rng = np.random.default_rng(3)
    solver.LGF = rng.normal(size=(2*solver.ny + 1, 2*solver.nx + 1))
    solver.IF = rng.normal(size=(3, 5, 5))
    solver.prepare_fast_lgf()
    source = rng.normal(size=(2, solver.ny, solver.nx))

    expected_lgf = np.stack([solver.Apply_lgf(component) for component in source])
    actual_lgf = np.zeros_like(source)
    solver.Apply_lgf_vec(source, actual_lgf)
    np.testing.assert_allclose(actual_lgf, expected_lgf, rtol=1e-13, atol=1e-13)

    expected_if = np.stack([solver.Apply_IF(component, 1) for component in source])
    actual_if = np.zeros_like(source)
    solver.Apply_IF_vec(source, actual_if, 1)
    np.testing.assert_allclose(actual_if, expected_if, rtol=1e-13, atol=1e-13)


def test_cached_lu_solve_matches_direct_solve():
    solver = sol.__new__(sol)
    solver.nIBP = 2
    matrix = np.array([
        [4.0, 1.0, 0.0, 0.0],
        [1.0, 3.0, 1.0, 0.0],
        [0.0, 1.0, 3.0, 1.0],
        [0.0, 0.0, 1.0, 2.0],
    ])
    solver.IBMat = np.stack((matrix, matrix, matrix))
    solver.IBMat_lu = [scipy.linalg.lu_factor(stage) for stage in solver.IBMat]
    source = np.array([[1.0, 2.0], [3.0, 4.0]])

    actual = solver.Direct_solve(source, 1)
    expected = scipy.linalg.solve(matrix, source.ravel()).reshape(source.shape)
    np.testing.assert_allclose(actual, expected)


def test_slice_stencils_match_signal_convolutions():
    solver = sol.__new__(sol)
    solver.dx = 0.2
    solver.dy = 0.125
    rng = np.random.default_rng(4)
    field = rng.normal(size=(6, 8))

    dx = scipy.signal.convolve(
        field, np.array([[0.0, 1/solver.dx, -1/solver.dx]]), mode="same"
    )
    dx_t = scipy.signal.convolve(
        field, np.array([[1/solver.dx, -1/solver.dx, 0.0]]), mode="same"
    )
    dy = scipy.signal.convolve(
        field, np.array([[0.0], [1/solver.dy], [-1/solver.dy]]), mode="same"
    )
    dy_t = scipy.signal.convolve(
        field, np.array([[1/solver.dy], [-1/solver.dy], [0.0]]), mode="same"
    )

    np.testing.assert_allclose(solver.Dx(field), dx)
    np.testing.assert_allclose(solver.Dx_t(field), dx_t)
    np.testing.assert_allclose(solver.Dy(field), dy)
    np.testing.assert_allclose(solver.Dy_t(field), dy_t)


def test_slice_nonlinear_averages_match_signal_convolutions():
    solver = sol.__new__(sol)
    solver.t = 0.3
    solver.U_infty = -1.0
    rng = np.random.default_rng(5)
    vort = rng.normal(size=(1, 6, 8))
    vel_raw = rng.normal(size=(2, 6, 8))
    vel = np.zeros_like(vel_raw)
    target = np.zeros_like(vel_raw)

    solver.nonlinear(vort, vel_raw, vel, target)

    expected_vel = vel_raw.copy()
    expected_vel[0] -= solver.U_inf()
    v_avg = scipy.signal.convolve(
        expected_vel[1], np.array([[0.0, 0.5, 0.5]]), mode="same"
    )
    u_avg = scipy.signal.convolve(
        expected_vel[0], np.array([[0.0], [0.5], [0.5]]), mode="same"
    )
    tmp_0 = -vort[0]*v_avg
    tmp_1 = vort[0]*u_avg
    expected = np.stack((
        scipy.signal.convolve(tmp_0, np.array([[0.5], [0.5], [0.0]]), mode="same"),
        scipy.signal.convolve(tmp_1, np.array([[0.5, 0.5, 0.0]]), mode="same"),
    ))

    np.testing.assert_allclose(vel, expected_vel)
    np.testing.assert_allclose(target, expected)
