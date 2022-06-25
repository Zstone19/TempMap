import numpy as np
from tqdm import tqdm
import astropy.constants as const
from scipy.sparse import csc_matrix

from numba import njit
from numba_progress import ProgressBar

import sys

from .utils import get_temperature_profile, get_Rin, get_F0






#############################################################################################
#                              Smearing Functions
#############################################################################################

@njit
def get_t1(tp, td, t0, dt):
    return max( tp-td-dt, -t0 )

@njit
def get_t2(tp, td, t0):
    return min( tp-td, t0)

@njit
def get_t3(tp, td, t0):
    return max(tp-td, -t0)

@njit
def get_t4(tp, td, t0, dt):
    return min(tp-td+dt, t0)
    
# @njit
# def G1(t0, ta, tb, dt):    
#     if tb < ta:
#         return 0
#     else:
#         return np.sqrt( abs(  (t0**2 - tb**2)**2 - (t0**2 - ta**2)**2  ) )/np.pi/dt
    

@njit
def G1(t0, ta, tb, dt):    
    if tb < ta:
        return 0
    else:
        return (  np.sqrt(t0**2 - tb**2) - np.sqrt(t0**2 - ta**2)  )/np.pi/dt

@njit
def G2(t0, ta, tb):
    if tb < ta:
        return 0
    else:
        return ( np.arcsin(tb/t0) - np.arcsin(ta/t0) )/np.pi


@njit
def divsum(arr):
    nan_mask = (np.isfinite(arr) == False)

    if len(arr) > 0:
        if arr[~nan_mask].sum() == 0.:
            return arr
        else:
            return arr / arr[~nan_mask].sum()
    else:
        return arr


#############################################################################################
#                    For Generic Perturbations
#############################################################################################

#The first index of W is : [nu0t0, nu1t0, nu2t0, ...]
#The second index of W is: [u0t0, u1t0, u2t0, ....]


@njit(nogil=True)
def make_W_all_in_one(row_dat, col_dat, input_dat, tp_vals, td_vals, lambda_vals, yvals, MBH, lambda_edd, dist, inc, progress_hook, 
                      errs=None, alpha=6,  h=const.h.cgs.value, c=const.c.cgs.value, kB=const.k_B.cgs.value, include_F0=True, 
                      dat_type='dToT', method='Yue', max_float=sys.float_info.max, min_float=sys.float_info.min, fluff_num=1e120):
    
    
    #row_dat, col_dat, input_dat are Awkward arrays
    #progress_hook is a numba_progress ProgressBar
    
    Nu = len(yvals)
    N_tp = len(tp_vals)
    N_td = len(td_vals)
    N_nu = len(lambda_vals)
    
    dtp = tp_vals[1] - tp_vals[0]
    dy = yvals[1] - yvals[0]
    
    Rin = get_Rin(MBH, alpha)
    T0_init = get_temperature_profile(yvals, MBH, lambda_edd, alpha)
    
    if errs is None:
        errs = np.ones(N_nu * N_td)
        
    for i in range(N_td):
        for j in range(Nu):

            t0 = (10**yvals[j]) * Rin * np.sin(inc) / c  /60/60/24
            Flux_vals = np.zeros(N_tp)                
                
            if method == 'Yue':
                good_ind = np.argwhere( np.abs(td_vals[i] - tp_vals) < t0  ).T[0]

                if len(good_ind) > 0:
                    theta = np.arccos( (tp_vals[good_ind] - td_vals[i])/t0 )

                    Flux_vals[good_ind] = dtp/t0/np.sin(theta)/np.pi
                    Flux_vals[good_ind] = divsum(Flux_vals[good_ind])
                   

            if method == 'NK22':
                td_full = np.full_like(tp_vals, td_vals[i])
                t0_full = np.full_like(tp_vals, t0)
                dt_full = np.full_like(tp_vals, dtp)
                
                t1 = np.array( list(map(get_t1, tp_vals, td_full, t0_full, dt_full)) )
                t2 = np.array( list(map(get_t2, tp_vals, td_full, t0_full)) )                
                t3 = np.array( list(map(get_t3, tp_vals, td_full, t0_full)) )    
                t4 = np.array( list(map(get_t4, tp_vals, td_full, t0_full, dt_full)) )

                term1 = np.array( list(map(G1, t0_full, t1, t2, dt_full)) )
                term2 = (td_vals[i] - tp_vals + dtp)*np.array( list(map(G2, t0_full, t1, t2)) )/dtp
                term3 = np.array( list(map(G1, t0_full, t3, t4, dt_full)) )
                term4 = (-td_vals[i] + tp_vals + dtp)*np.array( list(map(G2, t0_full, t3, t4)) )/dtp 

                Flux_vals = term1 + term2 + term3 + term4
                Flux_vals = divsum(Flux_vals)
                good_ind = np.nonzero(Flux_vals)[0] 
                    
                    
                    
            for k in good_ind:
                
                for l in range(N_nu):
                    F0 = get_F0(alpha, MBH, lambda_vals[l], dist, inc)
                    xval = h*c/lambda_vals[l] / T0_init[j] / kB

                    if xval > 100:
                        xterm = np.exp( np.log(fluff_num) + np.log(xval) - xval)
                    else:
                        xterm = fluff_num * xval / ( np.exp(xval) + np.exp(-xval) - 2 )
                        
                    log_integrand = np.log(xterm) + 2*yvals[j]*np.log(10) + np.log( np.log(10)*dy )
                    if include_F0:
                        log_integrand += np.log(F0)
                    if dat_type == 'dT':
                        log_integrand -= np.log(T0_init[j])
                        

                    #If integrand is too large/small, get integrand from max/min float allowed
                    if log_integrand > np.log(max_float) - 10:
                        integrand = np.exp( np.log(max_float) - 10 + 100 )
                    elif log_integrand < np.log(min_float) + 10:
                        integrand = np.exp( np.log(min_float) + 10 + 100 )
                    else:
                        integrand = np.exp(log_integrand)

                    if np.isinf(integrand):
                        integrand = np.sign(integrand) * ( max_float*1e100 / 1e10 )
                    elif np.isfinite(integrand) == False:
                        integrand = 0
                        
                    row_dat.integer( int(i*N_nu + l) )
                    col_dat.integer( int(k*Nu + j) )
                    input_dat.real( integrand * Flux_vals[k] / errs[i*N_nu + l] )
                
                    progress_hook.update(1)
    
    return row_dat, col_dat, input_dat



@njit(nogil=True)
def make_F_dF(input_dat, tp_vals, td_vals, lambda_vals, yvals, MBH, lambda_edd, dist, inc, progress_hook, alpha=6,
              h=const.h.cgs.value, c=const.c.cgs.value, kB=const.k_B.cgs.value, include_F0=True, dat_type='dToT',
              max_float=sys.float_info.max, min_float=sys.float_info.min, fluff_num=1e125):
    
    N_tp = len(tp_vals)
    N_td = len(td_vals)
    Nu = len(yvals)
    N_nu = len(lambda_vals)
    
    dy = yvals[1] - yvals[0]
    uvals = 10**yvals
    
    dF = np.zeros(N_nu*N_td)
    F = np.zeros(N_nu*N_td)
    
    T0_init = get_temperature_profile(yvals, MBH, lambda_edd, alpha)

    for i in range(N_td):
        
        td_ind = np.argmin( np.abs(td_vals[i] - tp_vals) )
        
        for j in range(N_nu):
            
            F0 = get_F0(alpha, MBH, lambda_vals[j], dist, inc)
            
            F_vals = np.zeros(Nu)
            dF_vals = np.zeros(Nu)
            for k in range(Nu):
                xval = h*c/lambda_vals[j] / T0_init[k] / kB                                         

                if xval > 100:
                    xterm = np.exp( np.log(fluff_num) + np.log(xval) - xval)
                else:
                    xterm = fluff_num * xval / ( np.exp(xval) + np.exp(-xval) - 2 )
                        
                log_integrand = np.log(xterm) + 2*yvals[k]*np.log(10) + np.log( np.log(10) ) + np.log(dy)
                if include_F0:
                    log_integrand += np.log(F0) 
                if dat_type == 'dT':
                    log_integrand -= np.log(T0_init[k])
            
                #If integrand is too large/small, get integrand from max/min float allowed
                if log_integrand > np.log(max_float) - 10:
                    integrand = np.exp( np.log(max_float) - 10 + 100 )
                elif log_integrand < np.log(min_float) + 10:
                    integrand = np.exp( np.log(min_float) + 10 + 100 )
                else:
                    integrand = np.exp(log_integrand)
        
                if np.isinf(integrand):
                    integrand = np.sign(integrand) * ( max_float*1e100 / 1e10 )
                elif np.isfinite(integrand) == False:
                    integrand = 0

                if xval > 250:
                    xterm = np.exp(  np.log(fluff_num) - xval )
                else:
                    xterm = fluff_num/( np.exp(xval) - 1 )
                    
                F_vals[k] = uvals[k]**2 * np.log(10) * xterm *dy
                if include_F0:
                    F_vals[k] *= F0
                
                dF_vals[k] = integrand * input_dat[k,td_ind]
                
            F[i*N_nu + j] = np.sum( F_vals )   
            dF[i*N_nu + j] = np.sum(dF_vals )
            progress_hook.update(1)
    
    
    return F, dF



@njit(nogil=True)
def make_F_dF_nonlinear(input_dat, tp_vals, td_vals, lambda_vals, yvals, MBH, lambda_edd, dist, inc, progress_hook, alpha=6,
              h=const.h.cgs.value, c=const.c.cgs.value, kB=const.k_B.cgs.value, include_F0=True, dat_type='dToT',
              max_float=sys.float_info.max, min_float=sys.float_info.min, fluff_num=1e125):
    
    N_tp = len(tp_vals)
    N_td = len(td_vals)
    Nu = len(yvals)
    N_nu = len(lambda_vals)
    
    dy = yvals[1] - yvals[0]
    uvals = 10**yvals
    
    dF = np.zeros(N_nu*N_td)
    F = np.zeros(N_nu*N_td)
    
    T0_init = get_temperature_profile(yvals, MBH, lambda_edd, alpha)
    
    for i in range(N_td):
        
        td_ind = np.argmin( np.abs(td_vals[i] - tp_vals) )

        if dat_type == 'dToT':
            Tvals = T0_init + input_dat[:,td_ind]*T0_init
            dT = input_dat[:,td_ind]*T0_init
        if dat_type == 'dT':
            Tvals = T0_init + input_dat[:,td_ind]
            dT = input_dat[:,td_ind]

        for j in range(N_nu):
            
            F0 = get_F0(alpha, MBH, lambda_vals[j], dist, inc)
            
            F_vals = np.zeros(Nu)
            dF_vals = np.zeros(Nu)
            for k in range(Nu):
                xval = h*c/lambda_vals[j] / Tvals[k] / kB                                         

                if xval > 100:
                    xterm = np.exp( np.log(fluff_num) + np.log(xval) - xval)
                else:
                    xterm = fluff_num * xval / ( np.exp(xval) + np.exp(-xval) - 2 )
                        
                log_integrand = np.log(xterm) + 2*yvals[k]*np.log(10) + np.log( np.log(10) ) + np.log(dy) - np.log(T0_init[k])
                if include_F0:
                    log_integrand += np.log(F0) 
            
                #If integrand is too large/small, get integrand from max/min float allowed
                if log_integrand > np.log(max_float) - 10:
                    integrand = np.exp( np.log(max_float) - 10 + 100 )
                elif log_integrand < np.log(min_float) + 10:
                    integrand = np.exp( np.log(min_float) + 10 + 100 )
                else:
                    integrand = np.exp(log_integrand)
        
                if np.isinf(integrand):
                    integrand = np.sign(integrand) * ( max_float*1e100 / 1e10 )
                elif np.isfinite(integrand) == False:
                    integrand = 0

                if xval > 250:
                    xterm = np.exp(  np.log(fluff_num) - xval )
                else:
                    xterm = fluff_num/( np.exp(xval) - 1 )
                    
                F_vals[k] = uvals[k]**2 * np.log(10) * xterm * dy
                if include_F0:
                    F_vals[k] *= F0
                
                dF_vals[k] = integrand * dT[k]
                
            F[i*N_nu + j] = np.sum( F_vals )   
            dF[i*N_nu + j] = np.sum(dF_vals )
            progress_hook.update(1)
    
    
    return F, dF


#############################################################################################
#                    For Light Curves of Differing Length
#############################################################################################

@njit(nogil=True)    
def make_W_lc(row_dat, col_dat, input_dat, yvals, tp_vals, td_vals, lambda_vals, 
              lambda_edd, MBH, dist, inc, alpha, progress_hook, errs=None, include_F0=False, dat_type='dToT', method='Yue', 
              c=const.c.cgs.value, h=const.h.cgs.value, kB=const.k_B.cgs.value, 
              max_float=sys.float_info.max, min_float=sys.float_info.min, fluff_num=1e100):

    #td_vals is an array of the observed times for each data point
    #lambda_vals is an array of the form [ lambda0, lambda0, lambda0, lambda0, ...., lambda0, lambda1, lambda1, lambda1, ... ]
    #The wavelengths are linked to each light curve data point
    
    
    Nu = len(yvals)
    N_tp = len(tp_vals)
    
    uvals = 10**yvals
    dy = yvals[1] - yvals[0]
    dtp = tp_vals[1] - tp_vals[0]
    
    Rin = get_Rin(MBH, alpha)
    T0_init = get_temperature_profile(yvals, MBH, lambda_edd, alpha)
    
    if errs is None:
        errs = np.ones( len(td_vals) )
        
    for i, (td, wl) in enumerate(  zip(td_vals, lambda_vals)  ):
        F0 = get_F0(alpha, MBH, wl, dist, inc)
        
        for j in range(Nu):
            t0 = (10**yvals[j]) * Rin * np.sin(inc) / c /60/60/24
            Flux_vals = np.zeros(N_tp)                
                
            if method == 'Yue':
                good_ind = np.argwhere( np.abs(td - tp_vals) < t0  ).T[0]
                scale = 1

                if len(good_ind) > 0:

                    theta = np.arccos( (tp_vals[good_ind] - td)/t0 )

                    Flux_vals[good_ind] = dtp/t0/np.sin(theta)/np.pi
                    scale = np.sum(Flux_vals[good_ind])

                    Flux_vals[good_ind] = Flux_vals[good_ind]/scale
                   

            if method == 'NK22':
                td_full = np.full_like(tp_vals, td_vals[i])
                t0_full = np.full_like(tp_vals, t0)
                dt_full = np.full_like(tp_vals, dtp)
                
                t1 = np.array( list(map(get_t1, tp_vals, td_full, t0_full, dt_full)) )
                t2 = np.array( list(map(get_t2, tp_vals, td_full, t0_full)) )                
                t3 = np.array( list(map(get_t3, tp_vals, td_full, t0_full)) )    
                t4 = np.array( list(map(get_t4, tp_vals, td_full, t0_full, dt_full)) )

                term1 = np.array( list(map(G1, t0_full, t1, t2, dt_full)) )
                term2 = (td_vals[i] - tp_vals + dtp)*np.array( list(map(G2, t0_full, t1, t2)) )/dtp
                term3 = np.array( list(map(G1, t0_full, t3, t4, dt_full)) )
                term4 = (-td_vals[i] + tp_vals + dtp)*np.array( list(map(G2, t0_full, t3, t4)) )/dtp 

                Flux_vals = term1 + term2 + term3 + term4
                good_ind = np.nonzero(Flux_vals)[0] 
                    
                    
                    
            for k in good_ind:
                
                xval = h*c/wl / T0_init[j] / kB
    
                if xval > 100:
                    xterm = np.exp( np.log(fluff_num) + np.log(xval) - xval)
                else:
                    xterm = fluff_num * xval / ( np.exp(xval) + np.exp(-xval) - 2 )

                log_integrand = np.log(xterm) + 2*yvals[j]*np.log(10) + np.log( np.log(10)*dy )  
                
                if include_F0:
                    log_integrand += np.log(F0) 
                if dat_type == 'dT':
                    log_integrand -= np.log(T0_init[j])

                #If integrand is too large/small, get integrand from max/min float allowed
                if log_integrand > np.log(max_float) - 10:
                    integrand = np.exp( np.log(max_float) - 10 + np.log(fluff_num) )
                elif log_integrand < np.log(min_float) + 10:
                    integrand = np.exp( np.log(min_float) + 10 + np.log(fluff_num) )
                else:
                    integrand = np.exp(log_integrand)

                if np.isinf(integrand):
                    integrand = np.sign(integrand) * ( max_float*fluff_num / 1e10 )
                elif np.isfinite(integrand) == False:
                    integrand = 0   

                row_dat.integer( int(i) )
                col_dat.integer( int(k*Nu + j) )
                input_dat.real( integrand * Flux_vals[k] / errs[i] )

                if progress_hook is not None:
                    progress_hook.update(1)
                
    return row_dat, col_dat, input_dat



@njit(nogil=True)
def make_F_dF_lc(input_dat, tp_vals, td_vals, lambda_vals, yvals, MBH, lambda_edd, dist, inc, progress_hook, alpha=6,
              h=const.h.cgs.value, c=const.c.cgs.value, kB=const.k_B.cgs.value, dat_type='dToT',
                 max_float=sys.float_info.max, min_float=sys.float_info.min, fluff_num=1e100, include_F0=False):
    
    N_tp = len(tp_vals)
    N_td = len(td_vals)
    Nu = len(yvals)
    
    dy = yvals[1] - yvals[0]
    uvals = 10**yvals
    
    dF = np.zeros(N_td)
    F = np.zeros(N_td)
    
    T0_init = get_temperature_profile(yvals, MBH, lambda_edd, alpha)

    for i in range(N_td):
        
        F0 = get_F0(alpha, MBH, lambda_vals[i], dist, inc)
        td_ind = np.argmin( np.abs(td_vals[i] - tp_vals) )
        
        F_vals = np.zeros(Nu)
        dF_vals = np.zeros(Nu)
        for k in range(Nu):
            xval = h*c/lambda_vals[i] / T0_init[k] / kB

            if xval > 100:
                xterm = np.exp( np.log(fluff_num) + np.log(xval) - xval)
            else:
                xterm = fluff_num * xval / ( np.exp(xval) + np.exp(-xval) - 2 )

            log_integrand = np.log(xterm) + 2*yvals[k]*np.log(10) + np.log( np.log(10) ) + np.log(dy) 
            
            if include_F0:
                log_integrand += np.log(F0)
            if dat_type == 'dT':
                log_integrand -= np.log(T0_init[k])
            
            #If integrand is too large/small, get integrand from max/min float allowed
            if log_integrand > np.log(max_float) - 10:
                integrand = np.exp( np.log(max_float) - 10 + 100 )
            elif log_integrand < np.log(min_float) + 10:
                integrand = np.exp( np.log(min_float) + 10 + 100 )
            else:
                integrand = np.exp(log_integrand)

            if np.isinf(integrand):
                integrand = np.sign(integrand) * ( max_float*1e100 / 1e10 )
            elif np.isfinite(integrand) == False:
                integrand = 0

            if xval > 250:
                xterm = np.exp(  np.log(fluff_num) - xval )
            else:
                xterm = fluff_num/( np.exp(xval) - 1 )

            F_vals[k] = uvals[k]**2 * np.log(10) * xterm *dy
            if include_F0:
                F_vals[k] *= F0
            
            dF_vals[k] = integrand * input_dat[k,td_ind]
            
        F[i] = np.sum(F_vals)   
        dF[i] = np.sum(dF_vals)

        if progress_hook is not None:
            progress_hook.update(1)
    
    
    return F, dF






@njit(nogil=True)
def make_F_lc(td_vals, lambda_vals, yvals, MBH, lambda_edd, dist, inc, progress_hook, alpha=6,
              h=const.h.cgs.value, c=const.c.cgs.value, kB=const.k_B.cgs.value, 
              fluff_num=1e100, include_F0=False):
    
    N_td = len(td_vals)
    Nu = len(yvals)
    
    dy = yvals[1] - yvals[0]
    uvals = 10**yvals
    
    F = np.zeros(N_td)
    
    T0_init = get_temperature_profile(yvals, MBH, lambda_edd, alpha)

    for i in range(N_td):
        F0 = get_F0(alpha, MBH, lambda_vals[i], dist, inc)
        
        F_vals = np.zeros(Nu)
        for k in range(Nu):
            xval = h*c/lambda_vals[i] / T0_init[k] / kB

            if xval > 250:
                xterm = np.exp(  np.log(fluff_num) - xval )
            else:
                xterm = fluff_num/( np.exp(xval) - 1 )

            F_vals[k] = uvals[k]**2 * np.log(10) * xterm *dy
            if include_F0:
                F_vals[k] *= F0
            
        F[i] = np.sum(F_vals)   

        if progress_hook is not None:
            progress_hook.update(1)
    
    
    return F