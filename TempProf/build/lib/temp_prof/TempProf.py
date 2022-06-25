import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np
from tqdm import tqdm
import astropy.constants as const
from scipy.sparse import csc_matrix


import multiprocessing as mp
from numba import njit
from numba_progress import ProgressBar

import sys



import matplotlib as mpl

mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.direction'] = 'in'

mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.direction'] = 'in'

mpl.rcParams["figure.autolayout"] = False

mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.format'] = 'pdf'


#############################################################################################
#                    Preliminary Functions
#############################################################################################

@njit
def get_temperature_profile(yvals, MBH, lambda_edd, alpha=6, eta=.1, 
                            sigma=const.sigma_sb.cgs.value, c=const.c.cgs.value,
                            G=const.G.cgs.value, method=0):
    
    uvals = 10**yvals
    
    if method == 0:
        Tin = (1.54e5) * (lambda_edd**(1/4)) * ( ( 1e9*1.99e33 / MBH )**(1/4) ) * ( (6/alpha)**(3/4) )
    else:
        Rin = get_Rin(MBH, alpha)
        Ledd = 1.26e38 * (MBH / 1.99e33)
        Mdot = Ledd*lambda_edd/eta/c/c
        
        Tin = ( 8*G*MBH*Mdot/8/np.pi/sigma/Rin/Rin/Rin )**(1/4)
        
    
    Tvals = Tin * ( uvals**(-3/4) ) * ( ( 1 - (uvals**(-1/2)) )**(1/4) )
    
    return Tvals

@njit
def get_Rin(MBH, alpha=6, G=const.G.cgs.value, c=const.c.cgs.value):
    return alpha*G*MBH/c/c


@njit
def get_F0(alpha, MBH, wl, dist, inc, c=const.c.cgs.value, h=const.h.cgs.value):    
    Rin = get_Rin(alpha, MBH)
    return 4*np.pi*h*c*c*np.cos(inc)*Rin*Rin/(wl**5)/(dist**2)


def rescale(vals, rescale_min=-1, rescale_max=1):
    min_val = np.nanmin(vals)
    max_val = np.nanmax(vals)
    
    return rescale_min + (rescale_max - rescale_min)*(vals - min_val)/(max_val - min_val)


def rescale_factor(vals, rescale_min=-1, rescale_max=1):
    min_val = np.nanmin(vals)
    max_val = np.nanmax(vals)
    
    return (rescale_max - rescale_min)/(max_val - min_val)


@njit
def get_xvals(xval, max_float, min_float):
    
    #If x is too large/small, get log(x) and exp(x)
    if np.isinf(xval):
        logx = np.log(max_float) - 10
        xval = np.exp(logx)
        exp_val = np.exp( np.log(max_float) - 10 )
    elif np.isfinite(xval) == False:
        logx = np.log(min_float) + 10
        xval = np.exp(logx)
        exp_val = np.exp( np.log(min_float) + 10 )
    elif np.isinf( np.exp(xval) ):
        logx = np.log(xval)
        exp_val = np.exp( np.log(max_float) - 10 )
    else:
        logx = np.log(xval)
        exp_val = np.exp(xval)

    return xval, logx, exp_val

@njit
def approx_xval(x, max_float):
    
    approx = [ x*np.exp(x) / ( np.exp(x) - 1 ),
             1/x - x/12 + x**3 / 240 - x**5 / 6048 + x**7 / 172800,
             1/x - x/12 + x**3 / 240 - x**5 / 6048,
             1/x - x/12 + x**3 / 240,
             1/x - x/12,
             1/x]
    
    val = np.inf
    i=-1
    while np.isinf(val):
        i += 1
        val = np.exp( approx[i] )
        
    return approx[i]


@njit
def approx_xval2(x, max_float):
    
    approx = [1/(np.exp(x) - 1),
             1/x - 1/2 + x/12 - x**3 / 720 + x**5 / 30240,
             1/x - 1/2 + x/12 - x**3 / 720,
             1/x - 1/2 + x/12,
             1/x]
    
    val = np.inf
    i=-1
    while np.isinf(val):
        i += 1
        val = np.exp( approx[i] )
        
    return approx[i]
    
    

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
    
@njit
def G1(t0, ta, tb, dt):    
    if np.abs(ta/t0) > 1 or np.abs(tb/t0) > 1:
        return 0
    else:
        return np.sqrt( abs(  (t0**2 - tb**2)**2 - (t0**2 - ta**2)**2  ) )/np.pi/dt
    
@njit
def G2(t0, ta, tb):
    if np.abs(ta/t0) > 1 or np.abs(tb/t0) > 1:
        return 0
    else:
        return ( np.arcsin(tb/t0) - np.arcsin(ta/t0) )/np.pi

    
#############################################################################################
#                    Temperature Profiles
#############################################################################################

def construct_temp_grid_const(yvals, tp_vals, MBH, lambda_edd,
                              pert_min, pert_max, alpha=6,
                              G=const.G.cgs.value, c=const.c.cgs.value):
        
    Temp_vals = np.zeros( ( len(yvals), len(tp_vals) ) )
    
    T0 = get_temperature_profile(yvals, MBH, lambda_edd, alpha)
    Rin = get_Rin(MBH, alpha, G=G, c=c)
    Rvals = Rin*(10**yvals)
    
    perturbation = np.zeros(len(yvals))
    pert_ind = np.argwhere( (np.log10(pert_min) <= yvals) & (yvals <= np.log10(pert_max) ) )
    perturbation[pert_ind] = 1
    
    for i in range(len(tp_vals)):        
        Temp_vals[:, i] = .1*perturbation
        
    return Temp_vals


def construct_temp_grid_const_out(yvals, tp_vals, MBH, lambda_edd,
                                  pert_min, pert_max, v,
                                  Nu=50, Nt=100, alpha=6,
                                  G=const.G.cgs.value, c=const.c.cgs.value):
    
            
    Temp_vals = np.zeros( ( len(yvals), len(tp_vals) ) )
    
    T0 = get_temperature_profile(yvals, MBH, lambda_edd, alpha)
    Rin = get_Rin(MBH, alpha, G=G, c=c)
    Rvals = Rin*(10**yvals)
    
    perturbation = np.zeros(len(yvals))
    pert_ind = np.argwhere( (np.log10(pert_min) <= yvals) & (yvals <= np.log10(pert_max) ) )
    perturbation[pert_ind] = 1
    
    for i in range(len(tp_vals)):
        min_u = pert_min * 10**(v* tp_vals[i]*24*3600 )
        max_u = pert_max * 10**(v* tp_vals[i]*24*3600 )
    
        if np.isnan(min_u) == True:
            min_u = 0
        
        perturbation = np.zeros(len(yvals))
        pert_ind = np.argwhere( (min_u <= 10**yvals) & (10**yvals <= max_u) ).T[0]
        perturbation[pert_ind] = 1       
        
        Temp_vals[:, i] = .1*perturbation
        
    return Temp_vals


@njit
def construct_temp_grid_out(yvals, tp_vals, MBH, lambda_edd, 
                           P=20, v=.8, alpha=6, 
                           G=const.G.cgs.value, c=const.c.cgs.value):
    
    Temp_vals = np.zeros( ( len(yvals), len(tp_vals) ) )
    
    T0 = get_temperature_profile(yvals, MBH, lambda_edd, alpha)
    Rin = get_Rin(MBH, alpha)
    per = P*24*60*60 #seconds

    velocities = np.full( len(yvals), v*c )

    Rvals = Rin*(10**yvals)
    omega = 2*np.pi/per

    for i in range(len(yvals)):
        Temp_vals[i, :] = .1*np.sin( omega*( tp_vals*24*60*60 - Rvals[i]/velocities[i] ) )
            
    return Temp_vals



#############################################################################################
#                    Creating a Light Curve
#############################################################################################

def make_lc(input_dat, yvals, tp_vals, td_vals, wl, MBH, inc, dist, lambda_edd, redshift=None, alpha=6, 
            c=const.c.cgs.value, kB=const.k_B.cgs.value, h=const.h.cgs.value, dat_type='dToT',
            max_float=sys.float_info.max, min_float=sys.float_info.min, fluff_num=1e100, include_F0=False):
    
    Nu = len(yvals)
    N_tp = input_dat.shape[1]
    uvals = 10**yvals
    dy = yvals[1] - yvals[0]
    
    Rin = get_Rin(MBH, alpha=alpha)
    T0 = get_temperature_profile(yvals, MBH, lambda_edd, alpha)
    F0 = get_F0(alpha, MBH, wl, dist, inc)
    
    dF_og = np.zeros( N_tp )
    
    for i in range(N_tp):
        
        integrand_vals = np.zeros(Nu)
        for j in range(Nu):
            
            xval = h*c/wl/T0[j]/kB
        
            if xval > 100:
                xterm = np.exp( np.log(fluff_num) + np.log(xval) - xval)
            else:
                xterm = fluff_num * xval / ( np.exp(xval) + np.exp(-xval) - 2 )    
    
            log_integrand = np.log(xterm) + 2*yvals[j]*np.log(10) + np.log( np.log(10) ) +np.log(dy)  
        
            if include_F0:
                log_integrand += np.log(F0)
            if dat_type == 'dT':
                log_integrand -= np.log(T0[j])

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
                
            integrand_vals[j] = integrand * input_dat[j,i]
            
        dF_og[i] = np.sum( integrand_vals )

    
    from scipy.interpolate import interp1d
    func = interp1d(tp_vals, dF_og, kind='linear')
    
    return func(td_vals)



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
    uvals = 10**yvals
    
    Rin = get_Rin(MBH, alpha)
    T0_init = get_temperature_profile(yvals, MBH, lambda_edd, alpha)
    
    if errs is None:
        errs = np.ones(N_nu * N_td)
        
    for i in range(N_td):
        
        td_ind = np.argmin( np.abs(td_vals[i] - tp_vals) )
        
        for j in range(Nu):

            t0 = (10**yvals[j]) * Rin * np.sin(inc) / c  /60/60/24
            Flux_vals = np.zeros(N_tp)                
                
            if method == 'Yue':
                good_ind = np.argwhere( np.abs(td_vals[i] - tp_vals) < t0  ).T[0]
                scale = 1

                if len(good_ind) > 0:

                    theta = np.arccos( (tp_vals[good_ind] - td_vals[i])/t0 )

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


#############################################################################################
#                    Filling the Matrix
#############################################################################################

@njit
def extract_indices(ak_arr, i1, i2):
    return list(ak_arr[i1:i2])


def chunk_fill(row_snap, col_snap, dat_snap, shape, Nchunk=1e5):
    
    #row_snap, col_snap, dat_snap are snapshots of the csr data Awkward arrays
    
    for n in tqdm( range( len(row_snap)//Nchunk ) ):

        row_dat = extract_indices( row_snap, n*Nchunk, (n+1)*Nchunk )
        col_dat = extract_indices( col_snap, n*Nchunk, (n+1)*Nchunk )
        input_dat = extract_indices( dat_snap, n*Nchunk, (n+1)*Nchunk )

        if n == 0:
            W_tot = csc_matrix( ( input_dat , (row_dat, col_dat) ), shape=shape )
            del row_dat, col_dat, input_dat

            continue

        W_tot = W_tot + csc_matrix( ( input_dat , (row_dat, col_dat) ), shape=shape )
        del row_dat, col_dat, input_dat

    row_dat = np.array(row_snap[(n+1)*Nchunk:])
    col_dat = np.array(col_snap[(n+1)*Nchunk:])
    input_dat = np.array(dat_snap[(n+1)*Nchunk:])

    W_input = W_tot + csc_matrix( ( input_dat , (row_dat, col_dat) ), shape=shape )
    del row_dat, col_dat, input_dat
    del W_tot
    
    return W_input



#############################################################################################
#                    Plotting the Temp Profile
#############################################################################################

from scipy.interpolate import interp1d

def get_chi2_nu_dF(new_spectra, resampled_dF_w_err, dF_err, 
                   td_vals, resampled_td_ind, nu_vals, resampled_nu_ind=None,
                   type='downsampled', resampled_nu_vals=None, func=np.mean):
    
    N_nu = len(nu_vals)
    
    chi2_vals = []
    
    
    if type == 'downsampled':
        if resampled_nu_ind is None:
            raise Exception
        
        resampled_N_nu = len(resampled_nu_ind)
        
        for i, ind in enumerate(resampled_td_ind):
            new_spectra_t = new_spectra[ind*N_nu:(ind+1)*N_nu][resampled_nu_ind]
            dF_t = resampled_dF_w_err[i*resampled_N_nu:(i+1)*resampled_N_nu]
            dF_err_t = dF_err[i*resampled_N_nu:(i+1)*resampled_N_nu]

            chi2_nu = np.sum( ( (new_spectra_t - dF_t)/dF_err_t )**2  )/resampled_N_nu
            chi2_vals.append( chi2_nu )
    
    if type == 'upsampled':

        if resampled_nu_vals is None:
            raise Exception
        
        resampled_N_nu = len(resampled_nu_vals)
        
        for i, ind in enumerate(resampled_td_ind):
            dF_t = resampled_dF_w_err[i*resampled_N_nu:(i+1)*resampled_N_nu]
            dF_err_t = dF_err[i*resampled_N_nu:(i+1)*resampled_N_nu]
            
            interp_func = interp1d( nu_vals, new_spectra[ind*N_nu:(ind+1)*N_nu] )
            upsampled_spectra = interp_func(resampled_nu_vals)
            
            chi2_nu = np.sum( ( (upsampled_spectra - dF_t)/dF_err_t )**2  )/resampled_N_nu
            chi2_vals.append( chi2_nu )
            
            
    return func( chi2_vals )    

            

  
def get_chi2_nu_lc(new_lc, input_lc, lc_err, lc_lengths, func=np.mean):
    
    N_lc = len(lc_lengths)
    chi2_vals = []
    
    for i in range(N_lc):
        
        if i !=0:
            ind1 = np.sum( lc_lengths[:i] )
        else:
            ind1 = 0
        
        ind2 = ind1 + lc_lengths[i]
        
        lc_in = np.array( rescale(input_lc[ind1:ind2])   )
        lc_out = np.array( rescale(new_lc[ind1:ind2])   )
        err = rescale_factor(input_lc[ind1:ind2])*np.array( lc_err[ind1:ind2] )
        
        chi2_nu = np.sum( ( ( lc_in - lc_out )/err  )**2  )/lc_lengths[i]
        chi2_vals.append(chi2_nu)
        
        
    return func( chi2_vals )



    
    
def plot_profs_inout(dToT_input, dToT_outputs_reshape, tp_vals, yvals, xi_vals, chi2_vals, fname=None, show=True, cmap_num=0):
    
    if cmap_num > 0:
        cmap = mpl.cm.get_cmap('RdBu_r', cmap_num)
    else:
        cmap = mpl.cm.get_cmap('RdBu_r')

    arrays = np.concatenate( [ [dToT_input], dToT_outputs_reshape ] )


    fig, ax = plt.subplots(1, len(arrays), figsize=(4*len(arrays), 3.5), sharey=True, sharex=True )

    for i, dToT_output_reshape in tqdm( enumerate(arrays) ):

        vals1 = dToT_output_reshape / np.percentile( np.abs(dToT_output_reshape), 99)

        im = ax[i].imshow(vals1, origin='lower', aspect='auto', 
                          extent=[ tp_vals[0], tp_vals[-1], yvals[0], yvals[-1] ],
                          cmap=cmap, vmin=-1, vmax=1)

        if i == 0:
            ax[i].set_ylabel(r'$\log_{10}(R / R_{in})$', fontsize=16)

        ax[i].set_xlabel(r'$t_p$ [days]', fontsize=16)

        if i == 0:
            ax[i].tick_params('both', labelsize=12)
        else:
            ax[i].tick_params('x', labelsize=12)

        ax[i].tick_params('both', which='major', length=7)
        ax[i].tick_params('both', which='minor', length=3)

        if i != 0:
            ax[i].set_title(r'$\xi$ = {}'.format(xi_vals[i-1]), fontsize=15)
        else:
            ax[i].set_title('Input', fontsize=15)

        vals2 = dToT_input / np.percentile( np.abs(dToT_input), 99)

        x1, x2 = ax[i].get_xlim()
        y1, y2 = ax[i].get_ylim()

        xtxt = x1 + (x2-x1)*.05 
        ytxt1 = y1 + (y2-y1)*.1
        ytxt2 = y1 + (y2-y1)*.2

        if i != 0:
            ax[i].text( xtxt, ytxt2, r'$\chi^2_\nu$ = {:.3f}'.format(chi2_vals[i-1]), fontsize=13, color='k' )

        ax[i].text( xtxt, ytxt1, 'scale = ' + '{:.4f}'.format( np.percentile( np.abs(dToT_output_reshape), 99) ), fontsize=13, color='k')

        interval = (tp_vals[-1] - tp_vals[0])//6

        
        if tp_vals[-1] > 10:
            ax[i].set_xticks([tp_vals[0] + interval*i for i in range(7)])
            ax[i].set_xticklabels(  np.concatenate([  [int(tp_vals[0] + interval*i) for i in range(6)] , [''] ]) )


    #Reduce spacing between subplots
    plt.subplots_adjust(wspace=.05, hspace=.5)

    #Make colorbar
    cbar = fig.colorbar(im, ax=[ax], location='right', shrink=1.0, pad=.01, aspect=15)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.tick_params('both', which='major', length=6)
    cbar.ax.tick_params('both', which='minor', length=3)
    cbar.ax.set_ylabel(r'$(\delta T / T)$ / scale', rotation=270, labelpad=25, fontsize=17)



    #Align colorbar ticks to the right
    r = plt.gcf().canvas.get_renderer()
    coord = cbar.ax.yaxis.get_tightbbox(r)

    ytickcoord = [yticks.get_window_extent() for yticks in cbar.ax.get_yticklabels()]


    inv = cbar.ax.transData.inverted()

    ytickdata = [inv.transform(a) for a in ytickcoord]
    ytickdatadisplay = [cbar.ax.transData.transform(a) for a in ytickdata]

    gap = [a[1][0]-a[0][0] for a in ytickdatadisplay]

    for tick in cbar.ax.yaxis.get_majorticklabels():
         tick.set_horizontalalignment("right")

    cbar.ax.yaxis.set_tick_params(pad=max(gap)+1)

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=200)
    
    if show:
        plt.show()
        return
    else:
        return fig, ax, cbar
    
    
    
    
    
    
    
    
    
    
    
    
def plot_profs_inout_dist(dToT_input, dToT_outputs_reshape, tp_vals, yvals, xi_vals, chi2_vals, fname=None, show=True, cmap_num=0):
        
    if cmap_num > 0:
        cmap = mpl.cm.get_cmap('RdBu_r', cmap_num)
    else:
        cmap = mpl.cm.get_cmap('RdBu_r')

    arrays = np.concatenate( [ [dToT_input], dToT_outputs_reshape ] )


    fig, ax = plt.subplots(2, len(arrays), figsize=(4*len(arrays), 7) )

    for i, dToT_output_reshape in tqdm( enumerate(arrays) ):

        vals1 = dToT_output_reshape / np.percentile( np.abs(dToT_output_reshape), 99)

        im = ax[0,i].imshow(vals1, origin='lower', aspect='auto', 
                          extent=[ tp_vals[0], tp_vals[-1], yvals[0], yvals[-1] ],
                          cmap=cmap, vmin=-1, vmax=1)

        if i == 0:
            ax[0,i].set_ylabel(r'$\log_{10}(R / R_{in})$', fontsize=17)
            ax[0,i].set_xlabel(r'$t_p$ [days]', fontsize=17)

        if i == 0:
            ax[0,i].tick_params('both', labelsize=14)
        else:
            ax[0,i].tick_params('x', labelsize=0)
            ax[0,i].tick_params('y', labelsize=0)

        ax[0,i].tick_params('both', which='major', length=7)
        ax[0,i].tick_params('both', which='minor', length=3)

        if i != 0:
            ax[0,i].set_title(r'$\xi$ = {}'.format(xi_vals[i-1]), fontsize=15)
        else:
            ax[0,i].set_title('Input', fontsize=15)

        vals2 = dToT_input / np.percentile( np.abs(dToT_input), 99)

        x1, x2 = ax[0,i].get_xlim()
        y1, y2 = ax[0,i].get_ylim()

        xtxt = x1 + (x2-x1)*.05 
        ytxt1 = y1 + (y2-y1)*.1
        ytxt2 = y1 + (y2-y1)*.2

        if i != 0:
            ax[0,i].text( xtxt, ytxt2, r'$\chi^2_\nu$ = {:.3f}'.format(chi2_vals[i-1]), fontsize=13, color='k' )

            
        if  i == 0:
            ax[0,i].text( xtxt, ytxt1, 'scale = $\mathbf{' + '{:.4f}'.format( np.percentile( np.abs(dToT_output_reshape), 99) ) + '}$', fontsize=13, color='k', fontweight='heavy', alpha=.75)
        else:
            ax[0,i].text( xtxt, ytxt1, 'scale = ' + '{:.4f}'.format( np.percentile( np.abs(dToT_output_reshape), 99) ), fontsize=13, color='k')

        interval = (tp_vals[-1] - tp_vals[0])//6

        
        if tp_vals[-1] > 10:
            ax[0,i].set_xticks([tp_vals[0] + interval*i for i in range(7)])
            
            if i == 0:
                ax[0,i].set_xticklabels(  np.concatenate([  [int(tp_vals[0] + interval*i) for i in range(6)] , [''] ]) )

    
        
    
    for i in range(len(dToT_outputs_reshape)):
   
        if len(np.argwhere(dToT_input == 0.)) != 0:
            dToT_input_alt = dToT_input.copy()
            zero_ind = np.argwhere(dToT_input == 0.)
            
            for inds in zero_ind:
                dToT_input_alt[ inds[0], inds[1] ] = 1e-100
                
            nz_mask = (dToT_input != 0.)
            dToT_input_nz = dToT_input[nz_mask]
            dToT_output_nz = dToT_outputs_reshape[i][nz_mask]
        else:
            dToT_input_alt = dToT_input
            dToT_input_nz = dToT_input
            dToT_output_nz = dToT_outputs_reshape[i]
        
        
        vals = (dToT_outputs_reshape[i] - dToT_input_alt)/dToT_input_alt
        vals_nz = (dToT_output_nz - dToT_input_nz)/dToT_input_nz
        
        im = ax[1, i+1].imshow(vals, origin='lower', aspect='auto', 
                          extent=[ tp_vals[0], tp_vals[-1], yvals[0], yvals[-1] ],
                          cmap=cmap, vmin=-1, vmax=1)
    
    
        if i == 0:
            ax[1,i+1].set_ylabel(r'$\log_{10}(R / R_{in})$', fontsize=17)

        ax[1,i+1].set_xlabel(r'$t_p$ [days]', fontsize=17)
        
        
        if i == 0:
            ax[1,i+1].tick_params('both', labelsize=14)
        else:
            ax[1,i+1].tick_params('x', labelsize=14)
            ax[1,i+1].tick_params('y', labelsize=0)
            
        ax[1,i+1].tick_params('both', which='both', color='k')
        ax[1,i+1].tick_params('both', which='major', length=7)
        ax[1,i+1].tick_params('both', which='minor', length=3)
        
    
        interval = (tp_vals[-1] - tp_vals[0])//6
        
        if tp_vals[-1] > 10:
            ax[1,i+1].set_xticks([tp_vals[0] + interval*i for i in range(7)])
            ax[1,i+1].set_xticklabels(  np.concatenate([  [int(tp_vals[0] + interval*i) for i in range(6)] , [''] ]) )
            
        if  i == 0:
            interval = .5
            tick_labels = [.5]
            j = .5
            while j < yvals[-1]:
                j += .5
                tick_labels.append(j)
            
            ax[1,i+1].set_yticks(tick_labels)
            ax[1,i+1].set_yticklabels(  np.concatenate([  tick_labels[:-1] , [''] ]) )
            
        ax[1,i+1].text( .05, .1, r'Med = $\mathbf{'+  '{:.3f}'.format(np.nanmedian(vals_nz)) + '}$', 
                       transform=ax[1,i+1].transAxes, c="xkcd:off white", fontsize=13, fontweight='heavy')
    
    
    ax[1,0].axis('off')
    
    
    
    
    ax[1,1].text( -1.25, .1, 
                 r'$\Delta t_p$ = {:.2f} days'.format( tp_vals[1] - tp_vals[0] ), fontsize=14, transform=ax[1,1].transAxes )
    
    ax[1,1].text( -1.25, 0., 
                 r'$\Delta \log_{10}(R / R_{in})$ =' + '{:.2f}'.format( yvals[1] - yvals[0] ), fontsize=14, transform=ax[1,1].transAxes )
    
    
    
    
    ax2 = ax[0, -1].twinx()
    ax2.imshow( np.zeros(( len(yvals), len(tp_vals) )) , origin='lower', aspect='auto', 
                      extent=[ tp_vals[0], tp_vals[-1], yvals[0], yvals[-1] ],
                      cmap=cmap, vmin=-1, vmax=1, alpha=0)
    ax2.set_ylabel(r'$(\delta T / T)$ / scale', fontsize=16, rotation=270, labelpad=20)
    ax2.tick_params('both', labelsize=0)
    ax2.tick_params('both', which='major', length=7)
    ax2.tick_params('both', which='minor', length=3)

    
    
    ax2 = ax[1, -1].twinx()
    ax2.imshow( np.zeros(( len(yvals), len(tp_vals) )) , origin='lower', aspect='auto', 
                      extent=[ tp_vals[0], tp_vals[-1], yvals[0], yvals[-1] ],
                      cmap=cmap, vmin=-1, vmax=1, alpha=0)
    ax2.set_ylabel('Fractional Difference', fontsize=15, rotation=270, labelpad=20)
    ax2.tick_params('both', labelsize=0)
    ax2.tick_params('both', which='major', length=7, color='k')
    ax2.tick_params('both', which='minor', length=3, color='k')
    
    
    #Reduce spacing between subplots
    plt.subplots_adjust(wspace=.05, hspace=.05)

    #Make colorbar
    cbar = fig.colorbar(im, ax=[ax], location='right', shrink=1.0, pad=.03, aspect=20)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.tick_params('both', which='major', length=8)
    cbar.ax.tick_params('both', which='minor', length=4)



    #Align colorbar ticks to the right
    r = plt.gcf().canvas.get_renderer()
    coord = cbar.ax.yaxis.get_tightbbox(r)

    ytickcoord = [yticks.get_window_extent() for yticks in cbar.ax.get_yticklabels()]


    inv = cbar.ax.transData.inverted()

    ytickdata = [inv.transform(a) for a in ytickcoord]
    ytickdatadisplay = [cbar.ax.transData.transform(a) for a in ytickdata]

    gap = [a[1][0]-a[0][0] for a in ytickdatadisplay]

    for tick in cbar.ax.yaxis.get_majorticklabels():
         tick.set_horizontalalignment("right")

    cbar.ax.yaxis.set_tick_params(pad=max(gap)+1)

    
    
    

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=200)
    
    if show:
        plt.show()
        return
    else:
        return fig, ax, cbar
    
    
    
    
def plot_profs_out(dToT_outputs_reshape, tp_vals, yvals, xi_vals, chi2_vals, fname=None, show=True, cmap_num=0, interval=50, Ncol=4):
    
    if cmap_num > 0:
        cmap = mpl.cm.get_cmap('RdBu_r', cmap_num)
    else:
        cmap = mpl.cm.get_cmap('RdBu_r')


    N = len(dToT_outputs_reshape)
    Nrow = np.ceil( N/Ncol ).astype(int)
    
        
    fig, ax = plt.subplots(Nrow, Ncol, figsize=(4.5*Ncol, 3.5*Nrow), sharey=True)

    for i, dToT_output_reshape in tqdm( enumerate(dToT_outputs_reshape) ):

        if N/Ncol <= 1:
            ax_ind = i
        else:
            ax_ind = (i//Ncol, i%Ncol)
        
        #Rescale data
        vals1 = dToT_output_reshape / np.percentile( np.abs(dToT_output_reshape), 99)
        
        #Plot
        im = ax[ax_ind].imshow(vals1, origin='lower', aspect='auto', 
                          extent=[ tp_vals[0], tp_vals[-1], yvals[0], yvals[-1] ],
                          cmap=cmap, vmin=-1, vmax=1)

        #Set tick length
        ax[ax_ind].tick_params('both', which='major', length=7)
        ax[ax_ind].tick_params('both', which='minor', length=3)

        
        #Set title
        if xi_vals[i] < 10000:
            xi_txt = '{}'.format(xi_vals[i])
        else:

            exp = int( np.log10(xi_vals[i]) )
            fact = xi_vals[i] / 10**exp

            if fact == 1:        
                xi_txt = r'$10^{' + '{:.0f}'.format( exp ) + '}$'
            elif int(fact) == fact:
                xi_txt = r'$' + '{:.0f}'.format(fact) + r' \times 10^{' + '{:.0f}'.format( exp ) + '}$'
            else:
                xi_txt = r'$' + '{:.2f}'.format(fact) + r' \times 10^{' + '{:.0f}'.format( exp ) + '}$'

        ax[ax_ind].set_title(r'$\xi$ = ' + xi_txt, fontsize=17)

        #Add text
        x1, x2 = ax[ax_ind].get_xlim()
        y1, y2 = ax[ax_ind].get_ylim()

        xtxt = x1 + (x2-x1)*.05 
        ytxt1 = y1 + (y2-y1)*.1
        ytxt2 = y1 + (y2-y1)*.2
        
        ax[ax_ind].text( xtxt, ytxt2, r'$\chi^2_\nu$ = {:.3f}'.format(chi2_vals[i]), fontsize=13, color='k' )
        ax[ax_ind].text( xtxt, ytxt1, 'scale = ' + '{:.3f}'.format( np.percentile( np.abs(dToT_output_reshape), 99) ), fontsize=13, color='k')
        
        #Add xticks
        Nticks = int( (tp_vals[-1] - tp_vals[0])//interval )
        
        if tp_vals[-1] > interval:
            val1 = np.ceil(tp_vals[0]/interval)*interval
           
            ax[ax_ind].set_xticks([val1 + interval*i for i in range(Nticks)])
            
            if tp_vals[-1] - interval/4 > val1 + interval*(Nticks-1):
                ax[ax_ind].set_xticklabels(  [int(val1 + interval*i) for i in range(Nticks)] )
            else:
                ax[ax_ind].set_xticklabels(  np.concatenate([  [int(val1 + interval*i) for i in range(Nticks-1)] , [''] ]) )

    #Add ylabel on left side
    for i in range(Nrow):
        if N/Ncol <= 1 :
            ax_ind = 0
        else:
            ax_ind = (i, 0)

        ax[ax_ind].set_ylabel(r'$\log_{10}(R / R_{in})$', fontsize=16)
        ax[ax_ind].tick_params('both', labelsize=12)
        
      
    #Add xlabel on bottom
    for i in range(Ncol):
        if N/Ncol <= 1:
            ax_ind = i
        else:
            ax_ind = (-1, i)
        
        ax[ax_ind].set_xlabel(r'MJD - 50000', fontsize=16)
        ax[ax_ind].tick_params('x', labelsize=12)

     
    #Set remove xtick labels on top if there is more than 1 row
    if N/Ncol > 1:
        for i in range(0, Nrow-1):
            for j in range(Ncol):
                ax[i,j].tick_params('x', labelsize=0)
        
    #Remove figures with no data
    if ( Nrow*Ncol > N ) & ( N/Ncol <= 1 ):
        for i in range( N, Nrow*Ncol ):
            ax[i].axis('off')

    elif Nrow*Ncol > N:
        for i in range( N, Nrow*Ncol ):
            ax[-1, i%Ncol].axis('off')
            
        for i in range(N%Ncol, Ncol):            
            ax[-2, i].tick_params('x', labelsize=12)
            ax[-2, i].set_xlabel(r'MJD - 50000', fontsize=16)
            
           
            

    #Reduce spacing between subplots
    plt.subplots_adjust(wspace=.05, hspace=.25)

    #Make colorbar
    cbar = fig.colorbar(im, ax=[ax], location='right', shrink=1.0, pad=.01, aspect=15 + 5*(Nrow-1))
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.tick_params('both', which='major', length=6)
    cbar.ax.tick_params('both', which='minor', length=3)
    cbar.ax.set_ylabel(r'$(\delta T / T)$ / scale', rotation=270, labelpad=25, fontsize=17)



    #Align colorbar ticks to the right
    r = plt.gcf().canvas.get_renderer()
    coord = cbar.ax.yaxis.get_tightbbox(r)

    ytickcoord = [yticks.get_window_extent() for yticks in cbar.ax.get_yticklabels()]


    inv = cbar.ax.transData.inverted()

    ytickdata = [inv.transform(a) for a in ytickcoord]
    ytickdatadisplay = [cbar.ax.transData.transform(a) for a in ytickdata]

    gap = [a[1][0]-a[0][0] for a in ytickdatadisplay]

    for tick in cbar.ax.yaxis.get_majorticklabels():
         tick.set_horizontalalignment("right")

    cbar.ax.yaxis.set_tick_params(pad=max(gap)+1)

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=200)
    
    if show:
        plt.show()
        return
    else:
        return fig, ax, cbar
    
    
    
#############################################################################################
#                    Animations
#############################################################################################


def animate_spectra(new_spectra, dF_pred, dF_input, F_input, resampled_dF_w_err, resampled_F, dF_err, 
                    td_vals, resampled_td_ind, nvals, resampled_nu_vals, xi_vals, fname, 
                    plot_err=False, fps=5):

    N_nu = len(nvals)
    resampled_N_nu = len(resampled_nu_vals)
    resampled_N_td = len(td_vals[resampled_td_ind])
    
    n1 = np.log10(resampled_nu_vals[0])
    n2 = np.log10(resampled_nu_vals[-1])
    
    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots(1, len(new_spectra), figsize=(5*len(new_spectra), 4), sharex=True, sharey=True)

    lines_arr_tot = []
    chi2_arr = []
    
    
    
    xmin = np.min(nvals)
    xmax = min(  n2+1.5, np.max(nvals) )
    
    ymin = -10
    ymax = 10
    
    for n, spectrum in enumerate(new_spectra):

        ax[n].tick_params('both', labelsize=12)
        ax[n].tick_params('both', which='major', length=8)
        ax[n].tick_params('both', which='minor', length=3)

        ax[n].set_xlabel('$\log_{10} ($Frequency [Hz]$)$', fontsize=15)        

        if n == 0:
            ax[n].set_ylabel(r'$\delta F / F_0$', fontsize=15)

        ax[n].set_ylim(ymin, ymax)
        ax[n].set_yscale('symlog')
        ax[n].set_xlim( xmin , xmax )

        y1, y2 = ax[n].get_ylim()
        x1, x2 = ax[n].get_xlim()

        if n == 0:
            time_text = ax[n].text(.1, .9, '', transform=ax[n].transAxes)

        chi2_text = ax[n].text(.1, .1, '', transform=ax[n].transAxes )
        chi2_arr.append(chi2_text)


        line3, = ax[n].plot([], [], lw=1, c='b', ls='--', alpha=.5, label='Input w/o Smearing')
        line1, = ax[n].plot([], [], lw=2, c='k', label='Output')
        
        
        
        if plot_err == True:
            line2a, = ax[n].plot([], [], lw=5, c='r', alpha=.5, label='Input')
            line2b, = ax[n].plot([], [], lw=5, c='r', alpha=.5)
            
            line4, = ax[n].plot([], [], lw=1, c='r', alpha=.5, zorder=1)
            lines_arr_tot.append([line1, line2a, line2b, line3, line4])    
        else:
            line2, = ax[n].plot([], [], lw=5, c='r', alpha=.5, label='Input')
            lines_arr_tot.append([line1, line2, line3])    
            
            
        ax[n].axvline(n1, ls='--', c='gray')
        ax[n].axvline(n2, ls='--', c='gray')

        if n == len(xi_vals)-1:
            ax[n].legend(loc='upper right', fontsize=8)
            
        ax[n].set_title(r'$\xi$ = {}'.format(xi_vals[n]), fontsize=13)

    #fig.subplots_adjust(bottom=0.2, left=0.2)

    # initialization function: plot the background of each frame
    def init():

        
        if plot_err == False:
            for n in range(len(new_spectra)):
                line1, line2, line3 = lines_arr_tot[n]

                line1.set_data([], [])
                line2.set_data([], [])
                line3.set_data([], [])

                chi2_text = chi2_arr[n]
                chi2_text.set_text('')

            time_text.set_text('')

            return lines_arr_tot[0][0], lines_arr_tot[0][1], lines_arr_tot[0][2], \
                   lines_arr_tot[1][0], lines_arr_tot[1][1], lines_arr_tot[1][2], \
                   lines_arr_tot[2][0], lines_arr_tot[2][1], lines_arr_tot[2][2], \
                   time_text, chi2_arr[0], chi2_arr[1], chi2_arr[2]

        else:
            for n in range(len(new_spectra)):
                line1, line2a, line2b, line3, line4 = lines_arr_tot[n]

                line1.set_data([], [])
                line2a.set_data([], [])
                line2b.set_data([],[])
                line3.set_data([], [])
                line4.set_data([], [])

                chi2_text = chi2_arr[n]
                chi2_text.set_text('')

            time_text.set_text('')

            return lines_arr_tot[0][0], lines_arr_tot[0][1], lines_arr_tot[0][2], lines_arr_tot[0][3], lines_arr_tot[0][4],\
                   lines_arr_tot[1][0], lines_arr_tot[1][1], lines_arr_tot[1][2], lines_arr_tot[1][3], lines_arr_tot[1][4],\
                   lines_arr_tot[2][0], lines_arr_tot[2][1], lines_arr_tot[2][2], lines_arr_tot[2][3], lines_arr_tot[2][4],\
                   lines_arr_tot[3][0], lines_arr_tot[3][1], lines_arr_tot[3][2], lines_arr_tot[3][3], lines_arr_tot[3][4],\
                   time_text, chi2_arr[0], chi2_arr[1], chi2_arr[2]

        
        

    # animation function.  This is called sequentially
    def animate(i):
        x = nvals

        td_ind = resampled_td_ind[i]

        if plot_err == False:
            for n, spectrum in enumerate(new_spectra):
                line1, line2, line3 = lines_arr_tot[n]
                chi2_text = chi2_arr[n]
                spectrum = new_spectra[n]

                y1 = spectrum[td_ind*N_nu:(td_ind+1)*N_nu]/F_input[td_ind*N_nu:(td_ind+1)*N_nu]    
                line1.set_data(x, y1)

                y2 = dF_pred[td_ind*N_nu : (td_ind+1)*N_nu]/F_input[td_ind*N_nu:(td_ind+1)*N_nu]                
                line2.set_data(x, y2)

                y3 = dF_input[td_ind*N_nu : (td_ind+1)*N_nu]/F_input[td_ind*N_nu:(td_ind+1)*N_nu]
                line3.set_data(x, y3)

                input_w_err = resampled_dF_w_err[i*resampled_N_nu:(i+1)*resampled_N_nu]
                dF_err_td = dF_err[i*resampled_N_nu:(i+1)*resampled_N_nu]
                
                
                interp_func = interp1d( 10**nvals, spectrum[td_ind*N_nu:(td_ind+1)*N_nu] )

                chi2_nu = np.sum( ( (  interp_func(resampled_nu_vals) - input_w_err)/dF_err_td )**2 ) / resampled_N_nu
                chi2_text.set_text(r'$\chi^2_{\nu}$ = %.3f' % chi2_nu)

            time_text.set_text(r'$t_d$ = %.2f' % td_vals[td_ind])

            return lines_arr_tot[0][0], lines_arr_tot[0][1], lines_arr_tot[0][2], \
                   lines_arr_tot[1][0], lines_arr_tot[1][1], lines_arr_tot[1][2], \
                   lines_arr_tot[2][0], lines_arr_tot[2][1], lines_arr_tot[2][2], \
                   time_text, chi2_arr[0], chi2_arr[1], chi2_arr[2]

        else:
            for n, spectrum in enumerate(new_spectra):
                line1, line2a, line2b, line3, line4 = lines_arr_tot[n]
                chi2_text = chi2_arr[n]
                spectrum = new_spectra[n]

                y1 = spectrum[td_ind*N_nu:(td_ind+1)*N_nu]/F_input[td_ind*N_nu:(td_ind+1)*N_nu]    
                line1.set_data(x, y1)

                y2 = dF_pred[td_ind*N_nu : (td_ind+1)*N_nu]/F_input[td_ind*N_nu:(td_ind+1)*N_nu]
                mask1 = (nvals < n1)
                mask2 = (nvals > n2)     
                    
                line2a.set_data(x[mask1], y2[mask1])
                line2b.set_data(x[mask2], y2[mask2])
                
                y3 = dF_input[td_ind*N_nu : (td_ind+1)*N_nu]/F_input[td_ind*N_nu:(td_ind+1)*N_nu]     
                line3.set_data(x, y3)

                input_w_err = resampled_dF_w_err[i*resampled_N_nu:(i+1)*resampled_N_nu]
                dF_err_td = dF_err[i*resampled_N_nu:(i+1)*resampled_N_nu]

                
                y4 = input_w_err/resampled_F[i*resampled_N_nu:(i+1)*resampled_N_nu]
                line4.set_data( np.log10(resampled_nu_vals), y4)


                interp_func = interp1d( 10**nvals, spectrum[td_ind*N_nu:(td_ind+1)*N_nu] )

                chi2_nu = np.sum( ( (  interp_func(resampled_nu_vals) - input_w_err)/dF_err_td )**2 ) / resampled_N_nu
                chi2_text.set_text(r'$\chi^2_{\nu}$ = %.3f' % chi2_nu)

            time_text.set_text(r'$t_d$ = %.2f' % td_vals[td_ind])

            return lines_arr_tot[0][0], lines_arr_tot[0][1], lines_arr_tot[0][2],  lines_arr_tot[0][3], lines_arr_tot[0][4],\
                   lines_arr_tot[1][0], lines_arr_tot[1][1], lines_arr_tot[1][2],  lines_arr_tot[1][3], lines_arr_tot[1][4],\
                   lines_arr_tot[2][0], lines_arr_tot[2][1], lines_arr_tot[2][2],  lines_arr_tot[2][3], lines_arr_tot[2][4],\
                   lines_arr_tot[3][0], lines_arr_tot[3][1], lines_arr_tot[3][2],  lines_arr_tot[3][3], lines_arr_tot[3][4],\
                   time_text, chi2_arr[0], chi2_arr[1], chi2_arr[2]
        
        
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=resampled_N_td, interval=20, blit=True, repeat_delay=10)

    anim.save(fname, fps=fps)
    
    return





def compare_smearing(dF_pred, dF_input, F_input, nvals, td_vals, dtp, 
                     fname, logy=False, fps=5):

    
    N_nu = len(nvals)
    N_td = len(td_vals)
    
    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    for i in range(2):
        ax[i].tick_params('both', labelsize=12)
        ax[i].tick_params('both', which='major', length=8)
        ax[i].tick_params('both', which='minor', length=3)

        ax[i].set_xlim(13, 18)
        ax[i].set_xlabel('$\log_{10} ($Frequency [Hz]$)$', fontsize=15)        

    ax[0].set_ylabel(r'$\delta F / F_0$', fontsize=15)
    
    
    if logy:
        ax[0].set_yscale('log')
        ax[0].set_ylim(1e-4, 1e2)
    else:
        ax[0].set_ylim(-.5, .5)

    ax[1].set_ylabel('|Difference|', fontsize=15)
    ax[1].set_yscale('log')
    ax[1].set_ylim(1e-6, 1e1)

    y1, y2 = ax[0].get_ylim()
    x1, x2 = ax[0].get_xlim()

    if logy:
        ytxt1 = 10**( np.log10(y1) + .9*(np.log10(y2) - np.log10(y1) )  )
        ytxt2 = 10**( np.log10(y1) + .8*(np.log10(y2) - np.log10(y1) )  )
    else:
        ytxt1 = y1 + .9*(y2-y1)
        ytxt2 = y1 + .8*(y2-y1)


    time_text = ax[0].text(x1 + .1*(x2-x1), ytxt1, '')
    ax[0].text( x1 + .1*(x2-x1), ytxt2, r'$\Delta t_p$ = ' + '{:.3f}'.format(dtp) )


    line3, = ax[0].plot([], [], lw=1, c='b', ls='--', alpha=.5, label='Input w/o Smearing')
    line2, = ax[0].plot([], [], lw=1, c='r', alpha=.5, label='Input')

    line4, = ax[1].plot([], [], lw=1, c='k')

    ax[0].legend(loc='upper right', fontsize=8)

    fig.subplots_adjust(bottom=0.2, left=0.2, wspace=.4)

    # initialization function: plot the background of each frame
    def init():

        line2.set_data([], [])
        line3.set_data([], [])
        line4.set_data([], [])

        time_text.set_text('')

        return line2, line3, line4, time_text


    # animation function.  This is called sequentially
    def animate(i):
        x = nvals

        y2 = dF_pred[i*N_nu : (i+1)*N_nu]/F_input[i*N_nu:(i+1)*N_nu]
        line2.set_data(x, y2)

        y3 = dF_input[i*N_nu : (i+1)*N_nu]/F_input[i*N_nu:(i+1)*N_nu]
        line3.set_data(x, y3)

        y4 = np.abs( np.abs( y2 - y3 ) )
        line4.set_data(x, y4)

        time_text.set_text(r'$t_d$ = %.2f' % td_vals[i])

        return line2, line3, line4, time_text

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=N_td, interval=20, blit=True, repeat_delay=10)

    anim.save(fname, fps=fps)
    
    return









def animate_LCs(new_lcs, dF_dat, err_dat, td_vals, dtp,
                lc_lengths, unique_lambda_vals, filters, xi_vals, z, min_td_obs, 
                fname, fps=.25, Ncol=4):

    N = len(new_lcs)
    Nrow = np.ceil(len(new_lcs)/Ncol).astype(int)


    fig, ax = plt.subplots( Nrow, Ncol, figsize=( 4.5*Ncol, 3.5*Nrow ), sharey=True, sharex=False )

    def animate_func(ind):
        sort_ind = np.argsort(unique_lambda_vals)
        a = sort_ind[ind]


        ind1 = np.sum(lc_lengths[:a]).astype(int)
        ind2 = ind1 + lc_lengths[a]

        if Nrow == 1:
            for i in range(N):
                ax[i].clear()
        else:
            for i in range(Nrow):
                for j in range(Ncol):
                    ax[i,j].clear()


        for n in range(len(new_lcs)):
            if n == 0:
                label1 = 'Input'
                label2 = 'Output'
            else:
                label1 = ''
                label2 = ''

            tvals = np.array( td_vals[ind1:ind2]*(1+z)+min_td_obs-50000)
            lc = np.array( rescale(dF_dat[ind1:ind2]) )
            lc_err = rescale_factor(dF_dat[ind1:ind2])*np.array( err_dat[ind1:ind2] )
            output = np.array( rescale(new_lcs[n][ind1:ind2]) )


            if N/Ncol <= 1:
                ax_ind = n
            else:
                ax_ind = (n//Ncol, n%Ncol)


            markers, caps, bars = ax[ax_ind].errorbar( tvals, lc, lc_err, fmt='.k', capsize=2, label=label1 )
            ax[ax_ind].plot( tvals, output, color='r', label=label2, zorder=100 )

            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]

            ax[ax_ind].set_ylim(-1.5, 1.5)

            if n//Ncol == (N-1)//Ncol:
                ax[ax_ind].set_xlabel(r'MJD - 50000', fontsize=16)

            ax[ax_ind].tick_params('both', which='both', labelsize=13)
            ax[ax_ind].tick_params('both', which='major', length=8)
            ax[ax_ind].tick_params('both', which='minor', length=3)


            if xi_vals[n] < 10000:
                xi_txt = '{}'.format(xi_vals[n])
            else:

                exp = int( np.log10(xi_vals[n]) )
                fact = xi_vals[n] / 10**exp

                if fact == 1:        
                    xi_txt = r'$10^{' + '{:.0f}'.format( exp ) + '}$'
                elif int(fact) == fact:
                    xi_txt = r'$' + '{:.0f}'.format(fact) + r' \times 10^{' + '{:.0f}'.format( exp ) + '}$'
                else:
                    xi_txt = r'$' + '{:.2f}'.format(fact) + r' \times 10^{' + '{:.0f}'.format( exp ) + '}$'

            ax[ax_ind].set_title(r'$\xi$ = ' + xi_txt, fontsize=17)


            chi2 = np.sum( ( (lc - output)/lc_err )**2  )/len(tvals)

            ax[ax_ind].text(.08, .85, r'$\chi^2_\nu$ = {:.3f}'.format(chi2), transform=ax[ax_ind].transAxes, fontsize=14, zorder=110)

        for i in range( Nrow ):

            if N/Ncol <= 1 :
                ax_ind = 0
            else:
                ax_ind = (i, 0)

            ax[ax_ind].set_ylabel('Normalized Flux', fontsize=16)

        if N/Ncol <= 1:
            ax_ind = -1
        else:
            ax_ind = (0, -1)


        filter_name = str(filters[a])[2:-1]
        if filter_name == 'W2':
            filter_name = 'Swift UVW2'
        if filter_name == 'W1':
            filter_name = 'Swift UVW1'
        if filter_name == 'M2':
            filter_name = 'Swift UVM2'

        ax[ax_ind].text(1.02, .55, filter_name, transform=ax[ax_ind].transAxes, fontsize=14)
        ax[ax_ind].text(1.02, .45, r'$\lambda$ = {:.1f} $\AA$'.format(unique_lambda_vals[a]/1e-8), transform=ax[ax_ind].transAxes, fontsize=14)
        ax[ax_ind].text(1.02, .2, r'$\Delta t_p$ = {:.2f} d'.format(dtp), transform=ax[ax_ind].transAxes, fontsize=14)


        if N/Ncol > 1:
            for i in range(0, Nrow-1):
                for j in range(Ncol):
                    ax[i,j].tick_params('x', labelsize=0)


        if ( Nrow*Ncol > N ) & ( N/Ncol <= 1 ):
            for i in range( N, Nrow*Ncol ):
                ax[i].axis('off')

        elif Nrow*Ncol > N:
            for i in range( N, Nrow*Ncol ):
                ax[-1, i%Ncol].axis('off')

            for i in range(N%Ncol, Ncol):            
                ax[-2, i].tick_params('x', labelsize=12)
                ax[-2, i].set_xlabel(r'MJD - 50000', fontsize=16)


        plt.subplots_adjust(wspace=.05, hspace=.25, bottom=.2)
        plt.figlegend(bbox_to_anchor=(.98,.89), fontsize=13)



    anim = animation.FuncAnimation(fig, animate_func, frames=len(unique_lambda_vals), interval=20, repeat_delay=10)
    anim.save(fname, fps=fps)
    
    return

    
    
#############################################################################################
#                    Legacy Functions
#############################################################################################
    
    
    
    
@njit(nogil=True)
def make_K_all_time(dToT, tp_vals, td_vals, lambda_vals, yvals, MBH, lambda_edd, progress_hook, alpha=6,
           h=const.h.cgs.value, c=const.c.cgs.value, kB=const.k_B.cgs.value):
    
    N_tp = len(tp_vals)
    N_td = len(td_vals)
    Nu = len(yvals)
    N_nu = len(lambda_vals)
    
    dy = yvals[1] - yvals[0]
    uvals = 10**yvals
    
    dF = np.zeros(N_nu*N_td)
    F = np.zeros(N_nu*N_td)
    K = np.zeros( (N_nu*N_td, Nu*N_tp) )
    
    T0_init = get_temperature_profile(yvals, MBH, lambda_edd, alpha)

    for i in range(N_td):
        
        td_ind = np.argmin( np.abs(td_vals[i] - tp_vals) )
        
        for j in range(N_nu):
            for k in range(N_tp):
                xvals = h*c/lambda_vals[j] / T0_init / kB

                xterms1 = 1/xvals - xvals/12 + (xvals**3)/240
                xterms2 = 1/xvals - 1/2 + xvals/12 - (xvals**3)/720

                K_term = xvals / ( np.exp(xvals) + np.exp(-xvals) - 2 ) * (uvals ** 2) * np.log(10) * dy
                nan_mask = np.argwhere( np.isfinite(K_term) == False ).T[0]

                if len(nan_mask) > 0:
                    for ind in nan_mask:
                        K_term[ind] = 0

                K[i*N_nu + j, k*Nu:(k+1)*Nu] = K_term
                F_vals = uvals**2 * np.log(10) * dy  / ( np.exp(xvals) - 1 )
                dF_vals = K[i*N_nu + j, k*Nu:(k+1)*Nu] * dToT[:,td_ind]

                F[i*N_nu + j] = np.sum( F_vals )
                dF[i*N_nu + j] = np.sum( dF_vals )

                progress_hook.update(1)
        
    return K, F, dF


@njit
def make_W_Yue_all_time(K, yvals, tp_vals, td_vals, MBH, inc, progress_hook, errs=None, alpha=6, c=const.c.cgs.value):

    Nu = len(yvals)
    N_tp = len(tp_vals)
    N_td = len(td_vals)
    N_nu = K.shape[0] // N_td
    
    row_dat = np.full(N_nu*N_td*Nu*N_tp//10, np.nan)    
    col_dat = np.full(N_nu*N_td*Nu*N_tp//10, np.nan)    
    input_dat = np.full(N_nu*N_td*Nu*N_tp//10, np.nan)
    
    n = 0
    
    Rin = get_Rin(MBH, alpha)
    dtp = tp_vals[1] - tp_vals[0]
    
    if errs is None:
        errs = np.ones(N_nu * N_td)
        
    for i in range(N_td):
        for j in range(Nu):

            t0 = (10**yvals[j]) * Rin * np.sin(inc) / c  /60/60/24
            Flux_vals = np.zeros(N_tp)

            good_ind = np.argwhere( np.abs(td_vals[i] - tp_vals) < t0  ).T[0]
            scale = 1
            
            if len(good_ind) > 0:
                theta = np.arccos( (tp_vals[good_ind] - td_vals[i])/t0 )

                Flux_vals[good_ind] = dtp/t0/np.sin(theta)/np.pi
                scale = np.sum(Flux_vals[good_ind])
                
                Flux_vals[good_ind] = Flux_vals[good_ind]/scale

                           
            for k in good_ind:
                row_dat[n:n+N_nu] = np.linspace(i*N_nu, (i+1)*N_nu-1, N_nu)
                col_dat[n:n+N_nu] = np.full( N_nu, k*Nu + j )            
                input_dat[n:n+N_nu] = K[i*N_nu : (i+1)*N_nu, k*Nu + j] * Flux_vals[k] / errs[i*N_nu : (i+1)*N_nu]
                
                n += N_nu
            
                progress_hook.update(1)
                 
    return row_dat, col_dat, input_dat




def get_chi2_nu_dFoF(new_spectra, input_F, resampled_dF_w_err, dF_err, resampled_F,
                   td_vals, resampled_td_ind, nu_vals, resampled_nu_ind=None,
                   type='downsampled', resampled_nu_vals=None, func=np.mean):
    
    N_nu = len(nu_vals)
    
    chi2_vals = []
    
    
    if type == 'downsampled':
        if resampled_nu_ind is None:
            raise Exception
        
        resampled_N_nu = len(resampled_nu_ind)
        
        for i, ind in enumerate(resampled_td_ind):
            new_spectra_t = (new_spectra/input_F)[ind*N_nu:(ind+1)*N_nu][resampled_nu_ind]
            dFoF_t = (resampled_dF_w_err/resampled_F)[i*resampled_N_nu:(i+1)*resampled_N_nu]
            dFoF_err_t = (dF_err/resampled_F)[i*resampled_N_nu:(i+1)*resampled_N_nu]

            chi2_nu = np.sum( ( (new_spectra_t - dFoF_t)/dFoF_err_t )**2  )/resampled_N_nu
            chi2_vals.append( chi2_nu )
    
    if type == 'upsampled':

        if resampled_nu_vals is None:
            raise Exception
        
        resampled_N_nu = len(resampled_nu_vals)
        
        for i, ind in enumerate(resampled_td_ind):
            dFoF_t = (resampled_dF_w_err/resampled_F)[i*resampled_N_nu:(i+1)*resampled_N_nu]
            dFoF_err_t = (dF_err/resampled_F)[i*resampled_N_nu:(i+1)*resampled_N_nu]
            
            interp_func = interp1d( nu_vals, (new_spectra/input_F)[ind*N_nu:(ind+1)*N_nu] )
            upsampled_spectra = interp_func(resampled_nu_vals)
            
            chi2_nu = np.sum( ( (upsampled_spectra - dFoF_t)/dFoF_err_t )**2  )/resampled_N_nu
            chi2_vals.append( chi2_nu )
            
            
    return func( chi2_vals )    



#For const perturbations
@njit(nogil=True)
def make_K(dToT, lambda_vals, yvals, MBH, lambda_edd, progress_hook, alpha=6,
           h=const.h.cgs.value, c=const.c.cgs.value, kB=const.k_B.cgs.value):
    
    Nu = len(yvals)
    N_nu = len(lambda_vals)
    dy = yvals[1] - yvals[0]
    
    dF = np.zeros(N_nu)
    F = np.zeros(N_nu)
    K = np.zeros( (N_nu, Nu) )
    
    T0_init = get_temperature_profile(yvals, MBH, lambda_edd, alpha)

    for i in range(N_nu):
        uvals = 10**yvals
        xvals = h*c/lambda_vals[i] / T0_init / kB

        K_term = xvals / ( np.exp(xvals) + np.exp(-xvals) - 2 ) * (uvals ** 2) * np.log(10) * dy
        nan_mask = np.argwhere( np.isfinite(K_term) == False ).T[0]
        
        if len(nan_mask) > 0:
            for ind in nan_mask:
                K_term[ind] = 0
        
        K[i,:] = K_term
        F_vals = uvals**2 * np.log(10) * dy  / ( np.exp(xvals) - 1 )
        dF_vals = K[i,:] * dToT[:,0]

        F[i] = np.sum( F_vals )
        dF[i] = np.sum( dF_vals )
        
        progress_hook.update(1)
        
    return K, F, dF

@njit(nogil=True)
def make_W_Yue(K, yvals, tp_vals, td_vals, MBH, inc, progress_hook, errs=None, alpha=6, c=const.c.cgs.value):

    N_nu = K.shape[0]
    Nu = len(yvals)
    N_tp = len(tp_vals)
    N_td = len(td_vals)
    
    W = np.zeros( (N_nu*N_td, Nu*N_tp) )
    Rin = get_Rin(MBH, alpha)
    dtp = tp_vals[1] - tp_vals[0]

    if errs is None:
        errs = np.ones(N_nu * N_td)
        
    for i in range(N_td):
        for j in range(Nu):

            t0 = (10**yvals[j]) * Rin * np.sin(inc) / c  /60/60/24
            Flux_vals = np.zeros(N_tp)

            good_ind = np.argwhere( np.abs(td_vals[i] - tp_vals) < t0  ).T[0]
            scale = 1
            
            if len(good_ind) > 0:
                theta = np.arccos( (tp_vals[good_ind] - td_vals[i])/t0 )

                Flux_vals[good_ind] = dtp/t0/np.sin(theta)/np.pi
                scale = np.sum(Flux_vals[good_ind])
                
                Flux_vals[good_ind] = Flux_vals[good_ind]/scale

                           
            for k in range(N_tp):
                W[ i*N_nu : (i+1)*N_nu, k*Nu + j ] = K[:, j] * Flux_vals[k] / errs[i*N_nu : (i+1)*N_nu]
                progress_hook.update(1)
                 
    return W

#The first index of W is : [nu0t0, nu1t0, nu2t0, ...]
#The second index of W is: [u0t0, u1t0, u2t0, ....]