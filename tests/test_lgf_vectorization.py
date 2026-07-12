import numpy as np

from IBLGF import sol


def _solver_shell(nx=8, ny=6):
    solver = sol.__new__(sol)
    solver.nx = nx
    solver.ny = ny
    solver.xyratio = 2.0
    solver.xyratio2 = 4.0
    solver.lgf_quad_n = 512
    solver.lgf_asym_cutoff = 10
    solver._lgf_asym_const = None
    return solver


def test_vectorized_asymptotic_evaluation_matches_scalars():
    solver = _solver_shell()
    n = np.array([[0, 1, 3], [8, 13, 21]])
    m = np.array([[0, 7, 4], [2, 11, 5]])

    vectorized = solver.LGF_asym_rect(n, m)
    scalar = np.array([
        [solver.LGF_asym_rect(n_ij, m_ij) for n_ij, m_ij in zip(n_row, m_row)]
        for n_row, m_row in zip(n, m)
    ])

    np.testing.assert_allclose(vectorized, scalar, rtol=0, atol=0)


def test_vectorized_lgf_construction_is_reflection_symmetric():
    solver = _solver_shell()

    solver.compute_LGF_int()

    np.testing.assert_allclose(solver.LGF, solver.LGF[::-1, :], rtol=0, atol=0)
    np.testing.assert_allclose(solver.LGF, solver.LGF[:, ::-1], rtol=0, atol=0)
    assert solver.LGF.shape == (2*solver.ny + 1, 2*solver.nx + 1)
