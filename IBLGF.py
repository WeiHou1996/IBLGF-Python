from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import multiprocessing as mp
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from numpy import linalg as LA
import scipy
import scipy.signal
import scipy.fft
from multiprocessing.pool import ThreadPool

import sys
sys.path.append('../Fast-Screened-Poisson-LGF/src')
import LGF_funcs as LGF

# define solution enviroment
class sol:
    def __init__(self, dx, dy, cfl, cg_th, nx, ny, nx_ll = 0, ny_ll = 0, Re = 100, dxdibdx = 1.5, lgf_quad_n = 4096, lgf_asym_cutoff = 500):
        # dx is the spatial resilution
        # cfl is the time steps size relative to dx
        # nx, ny are number of grid points in each direction
        # nx_ll and ny_ll are the location of the lower left corner coordinate
        self.dx = dx
        self.dy = dy
        self.xyratio = dx/dy
        self.xyratio2 = self.xyratio * self.xyratio
        self.dt = self.dx*cfl
        self.cg_th = cg_th
        self.nx = nx
        self.ny = ny
        self.Re = Re
        self.nx_ll = nx_ll
        self.ny_ll = ny_ll
        self.dxibdx = dxdibdx
        
        self.nIBP = 0
        
        self.t = 0.0



        self.dx_v = np.zeros((1,3))
        self.dx_v[0, 0] = 0
        self.dx_v[0, 1] = 1/self.dx
        self.dx_v[0, 2] = -1/self.dx

        self.dx_v_t = np.zeros((1,3))
        self.dx_v_t[0, 1] = -1/self.dx
        self.dx_v_t[0, 2] = 0
        self.dx_v_t[0, 0] = 1/self.dx

        self.dy_v = np.zeros((3,1))
        self.dy_v[0] = 0
        self.dy_v[1] = 1/self.dy
        self.dy_v[2] = -1/self.dy

        self.dy_v_t = np.zeros((3,1))
        self.dy_v_t[1] = -1/self.dy
        self.dy_v_t[2] = 0
        self.dy_v_t[0] = 1/self.dy

        self.use_direct_solve=True
        self.lgf_quad_n = lgf_quad_n
        self.lgf_asym_cutoff = lgf_asym_cutoff
        self._lgf_asym_const = None
        self.force_history = []
        self.symmetry_history = []
        
        
        
        #initialize some IFHERK data
        self.c_ = np.array([0.0, 1.0/3.0, 1.0, 1.0])
        self.RK = np.array([self.c_[1] - self.c_[0], self.c_[2] - self.c_[1], self.c_[3] - self.c_[2]])
        self.a_ = np.array([1.0/3.0, -1.0, 2.0, 0.0, 0.75, 0.25])
        self.alpha = self.RK*self.dt/self.dx/self.dx/Re
        self.U_infty = -1
        
        self.u   = np.zeros((2, self.ny, self.nx))
        self._u_refresh   = np.zeros((2, self.ny, self.nx))
        self.u_i = np.zeros((2, self.ny, self.nx))
        self.p   = np.zeros((1, self.ny, self.nx))
        self.p_i = np.zeros((1, self.ny, self.nx))
        self.stream = np.zeros((1, self.ny, self.nx))
        self.cell_aux2 = np.zeros((1, self.ny, self.nx))
        
        self.g_i = np.zeros((2, self.ny, self.nx))
        self.q_i = np.zeros((2, self.ny, self.nx))
        self.d_i = np.zeros((1, self.ny, self.nx))
        self.r_i = np.zeros((2, self.ny, self.nx))
        self.cell_aux = np.zeros((1, self.ny, self.nx))
        self.omega = np.zeros((1, self.ny, self.nx))
        self.face_aux = np.zeros((2, self.ny, self.nx))
        self.face_aux2 = np.zeros((2, self.ny, self.nx))
        self.w_1 = np.zeros((2, self.ny, self.nx))
        self.w_2 = np.zeros((2, self.ny, self.nx))
        
        self.coeff_a = np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                idx = int((i*(i-1))/2 + j -1)
                self.coeff_a[i, j] = self.a_[idx]
        
        #for i in range(self.nx):
        #    for j in range(self.ny):
        #        x = i*self.dx - self.nx*self.dx/2
        #        y = j*self.dx - self.ny*self.dx/2
        #        r = np.sqrt(x**2 + y**2)
                #self.u[0,j,i] = self.u_taylor_vort(x,y+self.dx/2,0,0)
                #self.u[1,j,i] = self.u_taylor_vort(x+self.dx/2,y,0,1)
                #self.u[0,j,i] = self.u_oseen_vort(x, y+self.dx/2, 0, 0)
                #self.u[1,j,i] = self.u_oseen_vort(x+self.dx/2, y, 0, 1)
                #self.u[0,j,i] = x
                #self.u[1,j,i] = y
                
        
        self.LGF = np.zeros((self.nx*2+1, self.ny*2+1))
        self.IF = np.zeros((3, 41, 41))
                
        f = pd.read_csv('lgf_more.txt', header=None, delimiter=',')
        self.lgf_dat = f.iloc[:,0].to_numpy()
        
        #self.generateLGF_read()
        self.compute_LGF_int()
        self.integratingFactor_init()
        self.prepare_fast_lgf()
        
        self.init_shape()
        self.construct_Projection_sparse()
        self.assemble_projection_operators()
        self.IBMat = np.zeros((3, 2*self.nIBP, 2*self.nIBP))
        self.ET_H_S_E_Mat()
        self.IBMat_lu = [scipy.linalg.lu_factor(mat) for mat in self.IBMat]
        
        
    def smoothstep(self, x):
        if x < 0:
            return 0
        elif x >= 1:
            return 1
        else:
            return x
        
    def U_inf(self):
        return self.U_infty * self.smoothstep(self.t)
        
    def generateLGF(self):
        self.LGF = np.zeros((self.ny*2+1, self.nx*2+1))
        for i in range(self.nx*2+1):
            for j in range(self.ny*2+1):
                i_abs = abs(i - self.nx)
                j_abs = abs(j - self.ny)
                res = self.eval_lgf(0, i_abs, j_abs)
                self.LGF[j,i] = res
                
    def generateLGF_read(self):
        self.LGF = np.zeros((self.ny*2+1, self.nx*2+1))
        for i in range(self.nx*2+1):
            for j in range(self.ny*2+1):
                i_abs = abs(i - self.nx)
                j_abs = abs(j - self.ny)
                if (i_abs <= 400 and j_abs <= 400):
                    idx = i_abs*402+j_abs
                    self.LGF[j,i] = -self.lgf_dat[idx]
                else:
                    self.LGF[j,i] = -self.LGF_asym(i_abs, j_abs)

    def compute_LGF_int(self):
        cutoff = self.effective_lgf_asym_cutoff()
        if cutoff is not None:
            self.LGF_asym_const()

        j_abs, i_abs = np.meshgrid(
            np.arange(self.ny + 1),
            np.arange(self.nx + 1),
            indexing="ij",
        )
        if cutoff is None:
            direct_mask = np.ones(j_abs.shape, dtype=bool)
        else:
            direct_mask = j_abs + i_abs <= cutoff

        quadrant = np.empty(j_abs.shape, dtype=float)
        asym_mask = ~direct_mask
        if np.any(asym_mask):
            quadrant[asym_mask] = self.LGF_asym_rect(
                j_abs[asym_mask], i_abs[asym_mask]
            )

        direct_points = np.argwhere(direct_mask)
        if len(direct_points) > 0:
            args = [(int(j), int(i)) for j, i in direct_points]
            with ThreadPool(processes=min(mp.cpu_count(), len(args))) as pool:
                results = pool.starmap(self.eval_lgf, args)
            quadrant[direct_points[:, 0], direct_points[:, 1]] = results

        j_reflect = np.abs(np.arange(self.ny*2 + 1) - self.ny)
        i_reflect = np.abs(np.arange(self.nx*2 + 1) - self.nx)
        self.LGF = quadrant[j_reflect[:, None], i_reflect[None, :]]
                
    def LGF_asym(self,n,m):
        Cfund = -0.18124942796
        Integral_val = -0.076093998454228
        r = np.sqrt(n*n + m*m)
        theta = np.arctan2(n,m)
        second_term = 1.0/24.0/np.pi*np.cos(4.0*theta)/r/r
        res = -0.5/np.pi*np.log(r) + Cfund + Integral_val + second_term
        return res

    def LGF_asym_const(self):
        if self._lgf_asym_const is not None:
            return self._lgf_asym_const

        rho = self.xyratio
        def integrand(t):
            if t == 0:
                return 0
            a = 2 + 2 * rho * rho - 2 * rho * rho * np.cos(t)
            K = (a + np.sqrt(a * a - 4)) / 2
            q = 2 * rho * t / (K - 1 / K)
            return (q - 1) / t

        val = scipy.integrate.quad(integrand, 0, np.pi, points=[0], epsabs=1e-10, epsrel=1e-10, limit=200)[0]
        self._lgf_asym_const = np.euler_gamma + np.log(np.pi) + val
        return self._lgf_asym_const

    def effective_lgf_asym_cutoff(self):
        if self.lgf_asym_cutoff is None:
            return None
        if self.lgf_asym_cutoff <= 0:
            return self.lgf_asym_cutoff

        safe_direct_cutoff = max(10, self.lgf_quad_n // 200)
        return min(self.lgf_asym_cutoff, safe_direct_cutoff)

    def LGF_asym_rect(self, n, m):
        scalar_input = np.ndim(n) == 0 and np.ndim(m) == 0
        n = np.abs(np.asarray(n, dtype=float))
        m = np.abs(np.asarray(m, dtype=float))

        rho = self.xyratio
        x = m
        y = n / rho
        R = np.sqrt(x * x + y * y)
        at_origin = R == 0
        safe_R = np.where(at_origin, 1.0, R)
        theta = np.arctan2(y, x)

        leading = (np.log(safe_R) + self.LGF_asym_const() + np.log(rho)) / (2 * np.pi * rho)
        correction = -(
            (1 + 1 / (rho * rho)) * np.cos(4 * theta) / (48 * np.pi * rho)
            + (1 / (rho * rho) - 1) * np.cos(2 * theta) / (24 * np.pi * rho)
        ) / (safe_R * safe_R)
        result = np.where(at_origin, 0.0, leading + correction)
        if scalar_input:
            return float(result)
        return result
                
    def eval_lgf(self, n, m):
        cutoff = self.effective_lgf_asym_cutoff()
        if cutoff is not None and abs(n) + abs(m) > cutoff:
            return self.LGF_asym_rect(n, m)

        t = -np.pi + (np.arange(self.lgf_quad_n) + 0.5) * 2 * np.pi / self.lgf_quad_n
        val = np.mean(self.integrand_g(t, n, m)) * 2 * np.pi
        return val.real
    
    def integrand_g(self, t, n, m):
        a = 2 + 2 * self.xyratio2 - np.cos(t)*2*self.xyratio2
        K = (a + np.sqrt(np.square(a) - 4))/2

        I = 1/2/np.pi*(1 - np.exp(1j*t*n) * (1/K)**m ) / (K - 1/K)

        return I
    
    def integratingFactor_init(self):
        self.IF = np.zeros((3, 41, 41))
        for n in range(3):
            alpha_t = self.alpha[n] 
            for i in range(41):
                for j in range(41):
                    i_abs = abs(i - 20)
                    j_abs = abs(j - 20)
                    #res = self.eval_lgf(0, i_abs, j_abs)
                    self.IF[n, i,j] = np.exp(-2*alpha_t-2*alpha_t*self.xyratio*self.xyratio)*scipy.special.iv(i_abs, 2*alpha_t*self.xyratio*self.xyratio) \
                    *scipy.special.iv(j_abs, 2*alpha_t)
                    
    def init_shape(self):
        r = 0.5
        self.nIBP = int(np.ceil(np.pi*2*r / min(self.dx, self.dy) / self.dxibdx))
        self.IBP = np.zeros((self.nIBP, 2))
        for i in range(self.nIBP):
            th = 2*np.pi*i/self.nIBP
            x = np.cos(th)*r
            y = np.sin(th)*r
            self.IBP[i, 0] = x
            self.IBP[i, 1] = y
            
    def construct_Projection(self):
        #use edge as benchmarking location
        self.P = np.zeros((2, self.ny, self.nx, self.nIBP))
        for i in range(self.nIBP):
            x = self.IBP[i,0]
            y = self.IBP[i,1]
            
            x_ctr = int(np.ceil(x/self.dx)) + self.nx_ll
            y_ctr = int(np.ceil(y/self.dy)) + self.ny_ll
            
            x_loc = x/self.dx + self.nx_ll
            y_loc = y/self.dy + self.ny_ll
            
            for j in range(-3, 4):
                for k in range(-3, 4):
                    x_pts = x_ctr + j
                    y_pts = y_ctr + k
                    self.P[0, y_pts, x_pts, i] = self.delta_func(x_pts - x_loc) * self.delta_func(y_pts + 0.5 - y_loc)
                    self.P[1, y_pts, x_pts, i] = self.delta_func(x_pts + 0.5 - x_loc) * self.delta_func(y_pts - y_loc)
    
    def construct_Projection_sparse(self):
        #use edge as benchmarking location
        self.P = []
        for i in range(self.nIBP):
            self.P.append([scipy.sparse.lil_matrix((self.ny, self.nx)), scipy.sparse.lil_matrix((self.ny, self.nx))])
            x = self.IBP[i,0]
            y = self.IBP[i,1]
            
            x_ctr = int(np.ceil(x/self.dx)) + self.nx_ll
            y_ctr = int(np.ceil(y/self.dy)) + self.ny_ll
            
            x_loc = x/self.dx + self.nx_ll
            y_loc = y/self.dy + self.ny_ll
            
            for j in range(-3, 4):
                for k in range(-3, 4):
                    x_pts = x_ctr + j
                    y_pts = y_ctr + k
                    v0 = self.delta_func(x_pts - x_loc) * self.delta_func(y_pts + 0.5 - y_loc)
                    v1 = self.delta_func(x_pts + 0.5 - x_loc) * self.delta_func(y_pts - y_loc)
                    if v0 != 0:
                        self.P[i][0][y_pts, x_pts] = v0
                    if v1 != 0:
                        self.P[i][1][y_pts, x_pts] = v1
                    #self.P[i][0][y_pts, x_pts] = self.delta_func(x_pts - x_loc) * self.delta_func(y_pts + 0.5 - y_loc)
                    #self.P[i][1][y_pts, x_pts] = self.delta_func(x_pts + 0.5 - x_loc) * self.delta_func(y_pts - y_loc)
            self.P[i][0] = self.P[i][0].tocsr()
            self.P[i][1] = self.P[i][1].tocsr()

    def assemble_projection_operators(self):
        n_grid = self.ny*self.nx
        self.P_matrix = [
            scipy.sparse.vstack(
                [self.P[i][component].reshape((1, n_grid)) for i in range(self.nIBP)],
                format="csr",
            )
            for component in range(2)
        ]
            
    def Schur(self, source, target):
        tmp = np.zeros(self.p.shape)
        self.Div(source, tmp)
        self.Apply_lgf_vec(tmp, tmp)
        self.Grad(tmp, target)
        
    def smearing(self, source, target):
        target[0] += np.asarray(self.P_matrix[0].T @ source[:, 0]).reshape(self.ny, self.nx)
        target[1] += np.asarray(self.P_matrix[1].T @ source[:, 1]).reshape(self.ny, self.nx)
            
    def projection(self, source, target):
        target[:, 0] = self.P_matrix[0] @ source[0].ravel()
        target[:, 1] = self.P_matrix[1] @ source[1].ravel()
        
    def ET_H_S_E(self, source, target, stage):
        tmpNp = np.zeros((2, self.ny, self.nx))
        self.smearing(source, tmpNp)
            
        #Apply IF
        self.Apply_IF_vec(tmpNp, tmpNp, stage)
        
        #Apply Schur
        self.face_aux2[:,:,:] = 0
        self.Schur(tmpNp, self.face_aux2)
        
        #Add
        tmpNp -= self.face_aux2
        
        self.projection(tmpNp, target)
            
    def ET_H_S_E_Mat(self):
        self.IBMat = np.zeros((3, 2*self.nIBP, 2*self.nIBP))
        for stage in range(3):
            self.IBMat[stage, :,:] = 0
            #smearing
            for i in range(self.nIBP):
                tmpNp = np.zeros((2, self.ny, self.nx))
                tmpNp[0] = self.P[i][0].toarray()
                self.Apply_IF_vec(tmpNp, tmpNp, stage)

                self.face_aux2[:,:,:] = 0
                self.Schur(tmpNp, self.face_aux2)

                tmpNp -= self.face_aux2

                self.IBMat[stage, 0::2, 2*i] = self.P_matrix[0] @ tmpNp[0].ravel()
                self.IBMat[stage, 1::2, 2*i] = self.P_matrix[1] @ tmpNp[1].ravel()

                tmpNp = np.zeros((2, self.ny, self.nx))
                tmpNp[1] = self.P[i][1].toarray()
                self.Apply_IF_vec(tmpNp, tmpNp, stage)

                self.face_aux2[:,:,:] = 0
                self.Schur(tmpNp, self.face_aux2)

                tmpNp -= self.face_aux2

                self.IBMat[stage, 0::2, 2*i + 1] = self.P_matrix[0] @ tmpNp[0].ravel()
                self.IBMat[stage, 1::2, 2*i + 1] = self.P_matrix[1] @ tmpNp[1].ravel()
            
    
    def LinearOperatorForCG(self, source, stage):
        res = np.zeros((self.nIBP, 2))
        source.shape = (self.nIBP, 2)
        self.ET_H_S_E(source, res, stage)
        res.shape = (self.nIBP * 2,)
        source.shape = (self.nIBP * 2,)
        return res
        
    def ib_solve(self, source, stage):
        uc = np.zeros((self.nIBP, 2))
        self.projection(source, uc)
        uc[:, 0] -= self.U_inf()
        if self.use_direct_solve:
            return self.Direct_solve(uc, stage)
        else:
            return self.CG_solve(uc, stage)
    
    def CG_solve(self, source, stage):
        source_tmp = np.zeros(source.shape)
        source_tmp[:,:] = source[:,:]
        source_tmp.shape = (self.nIBP * 2,)
        cur_LO = lambda v : self.LinearOperatorForCG(v, stage)
        solverFunc = scipy.sparse.linalg.LinearOperator((self.nIBP*2, self.nIBP*2), matvec=cur_LO)
        target_tmp, exit_code = scipy.sparse.linalg.cg(solverFunc, source_tmp, tol=1e-7)
        print('the exit code is', exit_code)
        target_tmp.shape = source.shape
        return target_tmp
    
    def Direct_solve(self, source, stage):
        source_tmp = np.asarray(source).reshape(self.nIBP*2)
        target_tmp = scipy.linalg.lu_solve(self.IBMat_lu[stage], source_tmp)
        return target_tmp.reshape(source.shape)

    def total_ib_force(self, stage=2):
        return np.sum(self.forcing, axis=0) * self.dx * self.dy / (self.dt*self.coeff_a[stage + 1, stage + 1])

    def ib_matrix_diagnostics(self):
        diag = []
        for stage in range(3):
            mat = self.IBMat[stage]
            diag.append({
                "stage": stage,
                "relative_symmetry_error": np.linalg.norm(mat - mat.T) / np.linalg.norm(mat),
                "condition_number": np.linalg.cond(mat),
            })
        return diag

    def _reflection_pairs_y(self, y_offset):
        rows = []
        for j in range(self.ny):
            j_ref = int(round(2*self.ny_ll - 2*y_offset - j))
            if 0 <= j_ref < self.ny and j <= j_ref:
                rows.append((j, j_ref))
        return rows

    def reflection_error_y(self, field, parity, y_offset=0.0):
        if parity not in (-1, 1):
            raise ValueError("parity must be 1 for even symmetry or -1 for odd symmetry")

        diff_norm2 = 0.0
        ref_norm2 = 0.0
        max_abs = 0.0
        for j, j_ref in self._reflection_pairs_y(y_offset):
            diff = field[j, :] - parity*field[j_ref, :]
            diff_norm2 += np.sum(diff*diff)
            ref_norm2 += np.sum(field[j, :]*field[j, :])
            if j != j_ref:
                ref_norm2 += np.sum(field[j_ref, :]*field[j_ref, :])
            max_abs = max(max_abs, np.max(np.abs(diff)))

        return {
            "relative_l2": np.sqrt(diff_norm2) / (np.sqrt(ref_norm2) + 1e-30),
            "max_abs": max_abs,
        }

    def flow_symmetry_diagnostics(self):
        return {
            "u_x_even": self.reflection_error_y(self.u[0], 1, y_offset=0.5),
            "u_y_odd": self.reflection_error_y(self.u[1], -1, y_offset=0.0),
            "omega_odd": self.reflection_error_y(self.omega[0], -1, y_offset=0.0),
        }

    def pressure_correction(self, source, target):
        tmp = np.zeros((2, self.ny, self.nx))
        self.smearing(source, tmp)
        self.face_aux2[:,:,:] = tmp
        self.Div(self.face_aux2, self.cell_aux2)
        self.Apply_lgf_vec(self.cell_aux2, self.cell_aux2)
        target -= self.cell_aux2
    
    def delta_func(self, x):
        r = np.abs(x)
        ddf = 0
        if r > 2:
            return 0
        
        r2 = r * r
        if r <= 1.0:
            ddf = 17.0 / 48.0 + np.sqrt(3) * np.pi / 108.0 + r / 4.0 - r2 / 4.0 + \
            (1.0 - 2.0 * r) / 16 * np.sqrt(-12.0 * r2 + 12.0 * r + 1.0) - \
            np.sqrt(3) / 12.0 * np.arcsin(np.sqrt(3) / 2.0 * (2.0 * r - 1.0))
        else:
            ddf = 55.0 / 48.0 - np.sqrt(3) * np.pi / 108.0 - 13.0 * r / 12.0 + r2 / 4.0 + \
            (2.0 * r - 3.0) / 48.0 * np.sqrt(-12.0 * r2 + 36.0 * r - 23.0) + \
            np.sqrt(3) / 36.0 * np.arcsin(np.sqrt(3) / 2.0 * (2 * r - 3.0))
        return ddf
            
    def prepare_fast_lgf(self):
        # Determine optimal FFT shape for linear convolution
        s_field = (self.ny, self.nx)
        s_lgf = self.LGF.shape
        shape = (s_field[0] + s_lgf[0] - 1, s_field[1] + s_lgf[1] - 1)
        self.fft_shape = (scipy.fft.next_fast_len(shape[0]), scipy.fft.next_fast_len(shape[1]))
        
        # Precompute FFT of LGF
        self.LGF_fft = scipy.fft.fft2(self.LGF, self.fft_shape)
        
        # Calculate slicing indices to emulate mode='same' (centered)
        start_y = (shape[0] - self.ny) // 2
        start_x = (shape[1] - self.nx) // 2
        self.lgf_slice = (slice(start_y, start_y + self.ny), slice(start_x, start_x + self.nx))
        
        # Precompute FFT for IF
        s_if = self.IF.shape[1:]
        shape_if = (s_field[0] + s_if[0] - 1, s_field[1] + s_if[1] - 1)
        self.if_fft_shape = (scipy.fft.next_fast_len(shape_if[0]), scipy.fft.next_fast_len(shape_if[1]))
        
        self.IF_fft = scipy.fft.fft2(self.IF, self.if_fft_shape, axes=(1, 2))
        
        start_y_if = (shape_if[0] - self.ny) // 2
        start_x_if = (shape_if[1] - self.nx) // 2
        self.if_slice = (slice(start_y_if, start_y_if + self.ny), slice(start_x_if, start_x_if + self.nx))

    def Apply_lgf(self, field, workers=-1):
        field_fft = scipy.fft.fft2(field, self.fft_shape, workers=workers)
        res_fft = field_fft * self.LGF_fft
        res = scipy.fft.ifft2(res_fft, workers=workers)
        res = res[self.lgf_slice].real
        res = res*self.dx*self.dx
        return res
    
    def Apply_lgf_vec(self, source, target):
        field_fft = scipy.fft.fft2(
            source, self.fft_shape, axes=(-2, -1), workers=-1
        )
        res = scipy.fft.ifft2(
            field_fft*self.LGF_fft, axes=(-2, -1), workers=-1
        )
        target[:] = res[:, self.lgf_slice[0], self.lgf_slice[1]].real*self.dx*self.dx
    
    def Apply_IF(self, field, stage, workers=-1):
        field_fft = scipy.fft.fft2(field, self.if_fft_shape, workers=workers)
        res_fft = field_fft * self.IF_fft[stage]
        res = scipy.fft.ifft2(res_fft, workers=workers)
        res = res[self.if_slice].real
        return res
    
    def Apply_IF_vec(self, source, target, stage):
        field_fft = scipy.fft.fft2(
            source, self.if_fft_shape, axes=(-2, -1), workers=-1
        )
        res = scipy.fft.ifft2(
            field_fft*self.IF_fft[stage], axes=(-2, -1), workers=-1
        )
        target[:] = res[:, self.if_slice[0], self.if_slice[1]].real
    
    def Dx(self, field):
        res = np.empty_like(field)
        res[:, 0] = field[:, 0]/self.dx
        res[:, 1:] = (field[:, 1:] - field[:, :-1])/self.dx
        return res
    
    def Dx_t(self, field):
        #Dx_t is f(k) = g(k) - g(k - 1)
        #so it is sum_{i+j = k} K(i)g(j) = sum_{i+j = 0} K(i)g(k+j)
        #K(0) = -1, K(-1) = 1

        res = np.empty_like(field)
        res[:, :-1] = (field[:, 1:] - field[:, :-1])/self.dx
        res[:, -1] = -field[:, -1]/self.dx
        return res
    
    def Dy(self, field):
        res = np.empty_like(field)
        res[0, :] = field[0, :]/self.dy
        res[1:, :] = (field[1:, :] - field[:-1, :])/self.dy
        return res
    
    def Dy_t(self, field):
        res = np.empty_like(field)
        res[:-1, :] = (field[1:, :] - field[:-1, :])/self.dy
        res[-1, :] = -field[-1, :]/self.dy
        return res
    
    def cleanBdry(self, field, n_grid=1, grid_location="cell"):
        """Zero boundary layers without breaking staggered-grid symmetry.

        Cell-centered fields use half-integer coordinates in both directions.
        Node-centered fields use integer coordinates, while face fields have
        integer x coordinates for the x component and integer y coordinates
        for the y component.  An integer-centered grid has one more point on
        its lower side, so that side needs one additional zeroed layer.
        """
        if n_grid <= 0:
            return
        if grid_location not in ("cell", "node", "face"):
            raise ValueError("grid_location must be 'cell', 'node', or 'face'")
        if grid_location == "face" and len(field) != 2:
            raise ValueError("face-centered fields must have two components")

        for i in range(len(field)):
            lower_y = n_grid
            lower_x = n_grid
            if grid_location == "node":
                lower_y += 1
                lower_x += 1
            elif grid_location == "face":
                if i == 0:  # x velocity: integer x, half-integer y
                    lower_x += 1
                else:       # y velocity: half-integer x, integer y
                    lower_y += 1

            field[i, 0:lower_y, :] = 0
            field[i, -n_grid:, :] = 0
            field[i, :, 0:lower_x] = 0
            field[i, :, -n_grid:] = 0
    
    def Div(self, source, target):
        if (len(source) == 1):
            print("wrong field for divergence")
        target[0] = self.Dx_t(source[0])
        target[0] += self.Dy_t(source[1])
        
    def Grad(self, source, target):
        if (len(source) != 1):
            print("wrong field for gradient")
        target[0] = self.Dx(source[0])
        target[1] = self.Dy(source[0])
        
    def Curl(self, source, target):
        if (len(source) == 1):
            print("wrong field for curl")
        target[0] = self.Dx(source[1]) - self.Dy(source[0])

    def Curl_t(self, source, target):
        if (len(source) != 1):
            print("wrong field for curl transpose")
        target[0] = self.Dy_t(source[0])
        target[1] = -self.Dx_t(source[0])

    def velocity_refresh(self, vel, vort):
        self.Curl(vel, vort)
        self.cleanBdry(vort, 6, grid_location="node")
        self.Apply_lgf_vec(vort, self.stream)
        self.Curl_t(self.stream, self._u_refresh)
        self.cleanBdry(self._u_refresh, 1, grid_location="face")
        self._u_refresh *= -1
        self.assign_bdry(self._u_refresh, vel, 6)
        
        
    def assign_bdry(self, field_from, field_to, depth):
        for i in range(len(field_from)):
            field_to[i, 0:depth, :] = field_from[i, 0:depth, :]
            field_to[i, -depth:, :] = field_from[i, -depth:, :]
            field_to[i, :, 0:depth] = field_from[i, :, 0:depth]
            field_to[i, :, -depth:] = field_from[i, :, -depth:]
    

    def nonlinear(self, vort, vel_raw, vel, target):
        #vel[:,:,:] = vel_raw[:,:,:] - self.U_inf()
        vel[0,:,:] = vel_raw[0,:,:] - self.U_inf()
        vel[1,:,:] = vel_raw[1,:,:]

        v_avg = np.empty_like(vel[1])
        v_avg[:, 0] = 0.5*vel[1, :, 0]
        v_avg[:, 1:] = 0.5*(vel[1, :, 1:] + vel[1, :, :-1])
        u_avg = np.empty_like(vel[0])
        u_avg[0, :] = 0.5*vel[0, 0, :]
        u_avg[1:, :] = 0.5*(vel[0, 1:, :] + vel[0, :-1, :])
        
        tmp_0 = -np.multiply(vort[0], v_avg)
        tmp_1 =  np.multiply(vort[0], u_avg)
        
        target[0, :-1, :] = 0.5*(tmp_0[:-1, :] + tmp_0[1:, :])
        target[0, -1, :] = 0.5*tmp_0[-1, :]
        target[1, :, :-1] = 0.5*(tmp_1[:, :-1] + tmp_1[:, 1:])
        target[1, :, -1] = 0.5*tmp_1[:, -1]
        
    
    def lin_sys_with_ib_solve(self, stage):
        self.Div(self.r_i, self.cell_aux)
        self.cleanBdry(self.cell_aux, 6)
        self.Apply_lgf_vec(self.cell_aux, self.d_i)
        
        self.face_aux2[:,:,:] = self.r_i[:,:,:]
        
        self.Grad(self.d_i, self.face_aux)
        self.cleanBdry(self.face_aux, 6, grid_location="face")
        
        self.face_aux2 -= self.face_aux
        
        #IB
        self.Apply_IF_vec(self.face_aux2, self.face_aux2, stage)
        self.forcing = self.ib_solve(self.face_aux2, stage)

        if stage == 2:
            F = self.total_ib_force(stage)
            self.force_history.append([self.t, F[0], F[1]])
            print('At ', self.t, ' Total IB force: ', F)
        
        self.pressure_correction(self.forcing, self.d_i)
        self.Grad(self.d_i, self.face_aux)
        self.cleanBdry(self.face_aux, 6, grid_location="face")
        
        tmp = np.zeros((2, self.ny, self.nx))
        self.smearing(self.forcing, tmp)
        
        self.face_aux += tmp
        
        
        self.r_i -= self.face_aux
        self.Apply_IF_vec(self.r_i, self.u_i, stage)
        
    def lin_sys_solve(self, stage):
        self.Div(self.r_i, self.cell_aux)
        self.cleanBdry(self.cell_aux, 6)
        self.Apply_lgf_vec(self.cell_aux, self.d_i)
        self.Grad(self.d_i, self.face_aux)
        self.cleanBdry(self.face_aux, 6, grid_location="face")
        self.r_i -= self.face_aux
        self.Apply_IF_vec(self.r_i, self.u_i, stage)
    
    def IFHERK_step(self, dt):
        self.q_i[:,:,:] = self.u[:,:,:]
        # stage 1
        self.g_i[:,:,:] = 0
        self.cell_aux[:,:,:] = 0
        self.face_aux[:,:,:] = 0
        self.w_1[:,:,:] = 0
        self.w_2[:,:,:] = 0
        self.omega[:,:,:] = 0
        
        
        self.Curl(self.u, self.omega)
        self.cleanBdry(self.omega, 6, grid_location="node")
        self.nonlinear(self.omega, self.u, self.face_aux, self.g_i)
        self.g_i *= (-dt)*self.coeff_a[1,1]
        self.r_i[:,:,:] = self.q_i[:,:,:]
        self.r_i += self.g_i
        
        if self.nIBP == 0:
            self.lin_sys_solve(0)
        else:
            self.lin_sys_with_ib_solve(0)
            
        self.t += dt*self.RK[0]
        
        #stage 2
        self.r_i[:,:,:] = 0
        self.d_i[:,:,:] = 0
        self.cell_aux[:,:,:] = 0
        
        self.face_aux -= self.g_i
        self.w_1[:,:,:] = self.face_aux[:,:,:]
        self.w_1 *= (-1/dt/self.coeff_a[1,1])
        self.Apply_IF_vec(self.q_i, self.q_i,0)
        self.Apply_IF_vec(self.w_1, self.w_1,0)
        
        self.r_i += self.q_i
        self.r_i += self.w_1 * self.coeff_a[2,1] * dt
        
        self.Curl(self.u_i, self.omega)
        self.cleanBdry(self.omega, 6, grid_location="node")
        self.nonlinear(self.omega, self.u_i, self.face_aux, self.g_i)
        self.g_i *= (-dt)*self.coeff_a[2,2]
        
        self.r_i += self.g_i
        
        if self.nIBP == 0:
            self.lin_sys_solve(1)
        else:
            self.lin_sys_with_ib_solve(1)
            
        self.t += dt*self.RK[1]
        
        #stage 3
        self.d_i[:,:,:] = 0
        self.cell_aux[:,:,:] = 0
        self.w_2[:,:,:] = 0
        self.face_aux -= self.g_i
        self.w_2[:,:,:] = self.face_aux[:,:,:]
        self.w_2 *= (-1/dt/self.coeff_a[2,2])
        self.r_i[:,:,:] = self.q_i[:,:,:]
        self.r_i += self.w_1 * self.coeff_a[3,1]*dt
        self.r_i += self.w_2 * self.coeff_a[3,2]*dt
        
        self.Apply_IF_vec(self.r_i, self.r_i, 1)
        
        self.Curl(self.u_i, self.omega)
        self.cleanBdry(self.omega, 6, grid_location="node")
        self.nonlinear(self.omega, self.u_i, self.face_aux, self.g_i)
        self.g_i *= (-dt)*self.coeff_a[3,3]
        self.r_i += self.g_i
        
        if self.nIBP == 0:
            self.lin_sys_solve(2)
        else:
            self.lin_sys_with_ib_solve(2)
            
        self.t += dt*self.RK[2]
        
        #finalize
        self.u[:,:,:] = self.u_i[:,:,:]
        self.p[:,:,:] = self.d_i[:,:,:]
        self.p /= (self.coeff_a[3,3] * dt)
        
    def time_march(self, n_steps, refresh_interval=None, record_symmetry=False, verbose=True):
        for i in range(n_steps):
            self.IFHERK_step(self.dt)
            if record_symmetry:
                self.symmetry_history.append({
                    "step": i,
                    "time": self.t,
                    **self.flow_symmetry_diagnostics(),
                })
            if verbose:
                print('step ',i)

            if refresh_interval is not None and refresh_interval > 0 and ((i + 1) % refresh_interval == 0):
                if verbose:
                    print('Refreshing velocity field at step ', i+1)
                self.velocity_refresh(self.u, self.omega)
                
        
    def u_taylor_vort(self, x, y, td, idx):
        R_ = 1
        #td = 0
        t_0 = self.Re / 2.0 / R_ / R_
        t_1 = td / R_ / R_ / R_ / R_
        t = t_0 + t_1
        r = np.sqrt(x * x + y * y)
        r2 = r * r
        exponent = 0.5 * (1.0 - r * r * t_0 / t / R_ / R_)
        expval = np.exp(exponent)
        u_theta = (t_0 / t) * (t_0 / t) * r / R_ * expval
        theta = np.arctan2(y, x)
        multiplier = 0
        if idx == 0: 
            multiplier = -np.sin(theta)
        else:
            multiplier = np.cos(theta)
        
        u_val = u_theta * multiplier
        return u_val
    
    def u_oseen_vort(self, x, y, td, idx):
        #td = 0
        mean_c = 2.24181; #if using non-dim in Panton, max vel happens at eta = 2.24181
        fac = 2.0 * mean_c * mean_c / (mean_c * mean_c + 2) #factor to make maxvelocity to be 1
        t0 = self.Re / mean_c / mean_c
        tc = t0 + td
        rd = np.sqrt(x * x + y * y)
        nu = 1.0 / self.Re
        eta = rd / np.sqrt(tc * nu)
        expVal = np.exp(-eta * eta / 4.0)
        
        denom = np.sqrt(tc * nu)
        
        u_theta = 2.0 / denom / eta * (1.0 - expVal) / fac
        theta = np.arctan2(y, x)
        multiplier = 0
        if (idx == 0):
            multiplier = -np.sin(theta)
        else:
            multiplier = np.cos(theta)
                    
        u_val = u_theta * multiplier
        return u_val
