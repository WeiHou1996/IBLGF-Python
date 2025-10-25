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

import sys
sys.path.append('../Fast-Screened-Poisson-LGF/src')
import LGF_funcs as LGF

# define solution enviroment
class sol:
    def __init__(self, dx, dy, cfl, cg_th, nx, ny, nx_ll = 0, ny_ll = 0, Re = 100):
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
        self.dxibdx = 1.5
        
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
        
        
        
        #initialize some IFHERK data
        self.c_ = np.array([0.0, 1.0/3.0, 1.0, 1.0])
        self.RK = np.array([self.c_[1] - self.c_[0], self.c_[2] - self.c_[1], self.c_[3] - self.c_[2]])
        self.a_ = np.array([1.0/3.0, -1.0, 2.0, 0.0, 0.75, 0.25])
        self.alpha = self.RK*self.dt/self.dx/self.dx/Re
        self.U_infty = -1
        
        self.u   = np.zeros((2, self.ny, self.nx))
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
        
        self.generateLGF_read()
        #self.compute_LGF_int()
        self.integratingFactor_init()
        
        self.init_shape()
        self.construct_Projection_sparse()
        self.IBMat = np.zeros((3, 2*self.nIBP, 2*self.nIBP))
        self.ET_H_S_E_Mat()
        
        
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
        self.LGF = np.zeros((self.ny*2+1, self.nx*2+1))
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.starmap(self.eval_lgf, [(j_abs, i_abs) for i in range(self.nx, self.nx*2+1) for j in range(self.ny, self.ny*2+1) for i_abs in [abs(i - self.nx)] for j_abs in [abs(j - self.ny)]])
            for (j,i), res in zip([(j,i) for i in range(self.nx, self.nx*2+1) for j in range(self.ny,self.ny*2+1)], results):
                self.LGF[j,i] = res

        for i in range(self.nx*2+1):
            for j in range(self.ny*2+1):
                i_abs = abs(i - self.nx)
                j_abs = abs(j - self.ny)
                self.LGF[j,i] = self.LGF[j_abs+self.ny,i_abs+self.nx]
                
    def LGF_asym(self,n,m):
        Cfund = -0.18124942796
        Integral_val = -0.076093998454228
        r = np.sqrt(n*n + m*m)
        theta = np.arctan2(n,m)
        second_term = 1.0/24.0/np.pi*np.cos(4.0*theta)/r/r
        res = -0.5/np.pi*np.log(r) + Cfund + Integral_val + second_term
        return res
                
    def eval_lgf(self, n, m):
        integrand = lambda t: self.integrand_g(t, n, m)
        val1 = scipy.integrate.quad(integrand, -np.pi, -1e-15)
        val2 = scipy.integrate.quad(integrand, 1e-15, np.pi)
        #val2 = scipy.integrate.quad(integrand, -np.pi, np.pi)
        val = val1[0] + val2[0]
        return val
    
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
        self.nIBP = int(np.ceil(np.pi*2*r / self.dx / self.dxibdx))
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
            self.P.append([scipy.sparse.csr_matrix((self.ny, self.nx)), scipy.sparse.csr_matrix((self.ny, self.nx))])
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
            
    def Schur(self, source, target):
        tmp = np.zeros(self.p.shape)
        self.Div(source, tmp)
        self.Apply_lgf_vec(tmp, tmp)
        self.Grad(tmp, target)
        
    def smearing(self, source, target):
        for i in range(self.nIBP):
            target[0] += self.P[i][0]*source[i][0]
            target[1] += self.P[i][1]*source[i][1]
            
    def projection(self, source, target):
        for i in range(self.nIBP):
            target[i][0] = np.sum(self.P[i][0].multiply(source[0]))
            target[i][1] = np.sum(self.P[i][1].multiply(source[1]))
        
    def ET_H_S_E(self, source, target, stage):
        tmp = [scipy.sparse.csr_matrix((self.ny, self.nx)), scipy.sparse.csr_matrix((self.ny, self.nx))]
        #smearing
        tmp[0][:,:] = 0
        tmp[1][:,:] = 0
        for i in range(self.nIBP):
            tmp[0] += self.P[i][0]*source[i][0]
            tmp[1] += self.P[i][1]*source[i][1]
            
        #Apply IF
        tmpNp = np.zeros((2, self.ny, self.nx))
        tmpNp[0] = tmp[0].todense()
        tmpNp[1] = tmp[1].todense()
        self.Apply_IF_vec(tmpNp, tmpNp, stage)
        
        #Apply Schur
        self.face_aux2[:,:,:] = 0
        self.Schur(tmpNp, self.face_aux2)
        
        #Add
        tmpNp -= self.face_aux2
        
        #Project
        for i in range(self.nIBP):
            target[i][0] = np.sum(self.P[i][0].multiply(tmpNp[0]))
            target[i][1] = np.sum(self.P[i][1].multiply(tmpNp[1]))
            
    def ET_H_S_E_Mat(self):
        self.IBMat = np.zeros((3, 2*self.nIBP, 2*self.nIBP))
        for stage in range(3):
            self.IBMat[stage, :,:] = 0
            tmp = [scipy.sparse.csr_matrix((self.ny, self.nx)), scipy.sparse.csr_matrix((self.ny, self.nx))]
            #smearing
            for i in range(self.nIBP):
                tmp[0][:,:] = self.P[i][0][:,:]
                tmp[1][:,:] = 0

                tmpNp = np.zeros((2, self.ny, self.nx))
                tmpNp[0] = tmp[0].todense()
                tmpNp[1] = tmp[1].todense()
                self.Apply_IF_vec(tmpNp, tmpNp, stage)

                self.face_aux2[:,:,:] = 0
                self.Schur(tmpNp, self.face_aux2)

                tmpNp -= self.face_aux2

                for j in range(self.nIBP):
                    self.IBMat[stage, 2 * i, j*2] = np.sum(self.P[j][0].multiply(tmpNp[0]))
                    self.IBMat[stage, 2 * i, j*2 + 1] = np.sum(self.P[j][1].multiply(tmpNp[1]))

                tmp[0][:,:] = 0
                tmp[1][:,:] = self.P[i][1][:,:]

                tmpNp = np.zeros((2, self.ny, self.nx))
                tmpNp[0] = tmp[0].todense()
                tmpNp[1] = tmp[1].todense()
                self.Apply_IF_vec(tmpNp, tmpNp, stage)

                self.face_aux2[:,:,:] = 0
                self.Schur(tmpNp, self.face_aux2)

                tmpNp -= self.face_aux2

                for j in range(self.nIBP):
                    self.IBMat[stage, 2 * i + 1, j*2] = np.sum(self.P[j][0].multiply(tmpNp[0]))
                    self.IBMat[stage, 2 * i + 1, j*2 + 1] = np.sum(self.P[j][1].multiply(tmpNp[1]))
            
    
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
        source_tmp = np.zeros(source.shape)
        source_tmp[:,:] = source[:,:]
        source_tmp.shape = (self.nIBP * 2,)
        
        target_tmp = scipy.linalg.solve(self.IBMat[stage], source_tmp)
        #print('the exit code is', exit_code)
        target_tmp.shape = source.shape
        return target_tmp
    
    def pressure_correction(self, source, target):
        tmp = [scipy.sparse.csr_matrix((self.ny, self.nx)), scipy.sparse.csr_matrix((self.ny, self.nx))]
        self.smearing(source, tmp)
        self.face_aux2[0,:,:] = tmp[0].todense()[:,:]
        self.face_aux2[1,:,:] = tmp[1].todense()[:,:]
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
            
    def Apply_lgf(self, field):
        res = scipy.signal.convolve(field, self.LGF, mode='same')
        res = res*self.dx*self.dx
        return res
    
    def Apply_lgf_vec(self, source, target):
        for i in range(len(source)):
            target[i] = self.Apply_lgf(source[i])
    
    def Apply_IF(self, field, stage):
        res = scipy.signal.convolve(field, self.IF[stage], mode='same')
        return res
    
    def Apply_IF_vec(self, source, target, stage):
        for i in range(len(source)):
            target[i] = self.Apply_IF(source[i], stage)
    
    def Dx(self, field):
        
        res = scipy.signal.convolve(field, self.dx_v, mode='same')
        return res
    
    def Dx_t(self, field):
        #Dx_t is f(k) = g(k) - g(k - 1)
        #so it is sum_{i+j = k} K(i)g(j) = sum_{i+j = 0} K(i)g(k+j)
        #K(0) = -1, K(-1) = 1

        res = scipy.signal.convolve(field, self.dx_v_t, mode='same')
        return res
    
    def Dy(self, field):
        res = scipy.signal.convolve(field, self.dy_v, mode='same')
        return res
    
    def Dy_t(self, field):
        res = scipy.signal.convolve(field, self.dy_v_t, mode='same')
        return res
    
    def cleanBdry(self, field, n_grid = 1):
        for i in range(len(field)):
            field[i, 0:n_grid,:] = 0
            field[i, -n_grid:, :] = 0
            field[i, :, 0:n_grid] = 0
            field[i, :, -n_grid:] = 0
    
    def Div(self, source, target):
        if (len(source) == 1):
            print("wrong field for divergence")
        target[0] = self.Dx_t(source[0])
        target[0] += self.Dy_t(source[1])
        self.cleanBdry(target, 2)
        
    def Grad(self, source, target):
        if (len(source) != 1):
            print("wrong field for gradient")
        target[0] = self.Dx(source[0])
        target[1] = self.Dy(source[0])
        self.cleanBdry(target, 2)
        
    def Curl(self, source, target):
        if (len(source) == 1):
            print("wrong field for curl")
        target[0] = self.Dx(source[1]) - self.Dy(source[0])
        self.cleanBdry(target, 2)

    def Curl_t(self, source, target):
        if (len(source) != 1):
            print("wrong field for curl transpose")
        target[0] = self.Dy_t(source[0])
        target[1] -= self.Dx_t(source[0])
        self.cleanBdry(target, 1)

    def velocity_refresh(self, vel, vort):
        self.Curl(vel, vort)
        self.cleanBdry(vort, 5)
        self.Apply_lgf_vec(vort, self.stream)
        self.Curl_t(self.stream, vel)
        vel *= -1
        
        
    def nonlinear(self, vort, vel_raw, vel, target):
        #vel[:,:,:] = vel_raw[:,:,:] - self.U_inf()
        vel[0,:,:] = vel_raw[0,:,:] - self.U_inf()
        vel[1,:,:] = vel_raw[1,:,:]
        avg_x_f = np.zeros((1,3))
        avg_x_f[0,0] = 0.5
        avg_x_f[0,1] = 0.5
        avg_x_f[0,2] = 0
        
        avg_x_b = np.zeros((1,3))
        avg_x_b[0,0] = 0
        avg_x_b[0,1] = 0.5
        avg_x_b[0,2] = 0.5
        
        avg_y_f = np.zeros((3,1))
        avg_y_f[0] = 0.5
        avg_y_f[1] = 0.5
        avg_y_f[2] = 0
        
        avg_y_b = np.zeros((3,1))
        avg_y_b[0] = 0
        avg_y_b[1] = 0.5
        avg_y_b[2] = 0.5
        
        v_avg = scipy.signal.convolve(vel[1], avg_x_b, mode='same')
        u_avg = scipy.signal.convolve(vel[0], avg_y_b, mode='same')
        
        tmp_0 = -np.multiply(vort[0], v_avg)
        tmp_1 =  np.multiply(vort[0], u_avg)
        
        target[0] = scipy.signal.convolve(tmp_0, avg_y_f, mode='same')
        target[1] = scipy.signal.convolve(tmp_1, avg_x_f, mode='same')
        
    
    def lin_sys_with_ib_solve(self, stage):
        self.Div(self.r_i, self.cell_aux)
        self.Apply_lgf_vec(self.cell_aux, self.d_i)
        
        self.face_aux2[:,:,:] = self.r_i[:,:,:]
        
        self.Grad(self.d_i, self.face_aux)
        
        self.face_aux2 -= self.face_aux
        
        #IB
        self.Apply_IF_vec(self.face_aux2, self.face_aux2, stage)
        self.forcing = self.ib_solve(self.face_aux2, stage)

        if stage == 2:
            F = np.sum(self.forcing, axis=0) * self.dx * self.dy / (self.dt*self.coeff_a[3,3])
            print('At ', self.t, ' Total IB force: ', F)
        
        self.pressure_correction(self.forcing, self.d_i)
        self.Grad(self.d_i, self.face_aux)
        
        tmp = [scipy.sparse.csr_matrix((self.ny, self.nx)), scipy.sparse.csr_matrix((self.ny, self.nx))]
        self.smearing(self.forcing, tmp)
        
        self.face_aux[0,:,:] += tmp[0].todense()[:,:]
        self.face_aux[1,:,:] += tmp[1].todense()[:,:]
        
        
        self.r_i -= self.face_aux
        self.Apply_IF_vec(self.r_i, self.u_i, stage)
        
    def lin_sys_solve(self, stage):
        self.Div(self.r_i, self.cell_aux)
        self.Apply_lgf_vec(self.cell_aux, self.d_i)
        self.Grad(self.d_i, self.face_aux)
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
        
    def time_march(self, n_steps):
        for i in range(n_steps):
            self.IFHERK_step(self.dt)
            print('step ',i)

            if ((i + 1) % 100 == 0):
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