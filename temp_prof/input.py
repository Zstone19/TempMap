import numpy as np
import astropy.constants as const
from scipy.sparse import csc_matrix

from numba import njit

import sys



from .utils import get_temperature_profile, get_Rin, get_F0
from .algorithm import get_t1, get_t2, get_t3, get_t4, G1, G2



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






def make_lc(input_dat, yvals, tp_vals, td_vals, wl, MBH, inc, dist, lambda_edd, alpha=6, 
            c=const.c.cgs.value, kB=const.k_B.cgs.value, h=const.h.cgs.value, dat_type='dToT', method='NK22',
            max_float=sys.float_info.max, min_float=sys.float_info.min, fluff_num=1e100, include_F0=False):
    
    Nu = len(yvals)
    dy = yvals[1] - yvals[0]
    N_td = len(td_vals)

    T0 = get_temperature_profile(yvals, MBH, lambda_edd, alpha)
    F0 = get_F0(alpha, MBH, wl, dist, inc)
    
    F_og = np.zeros( N_td )

    for i in range(N_td):
        td_ind = np.argmin( np.abs( td_vals[i] - tp_vals ) )

        if dat_type == 'dToT':
            Tvals = T0 + input_dat[:,td_ind]*T0
        elif dat_type == 'dT':
            Tvals = T0 + input_dat[:,td_ind]

        integrand_vals = np.zeros( Nu )
        for j in range(Nu):
            xval = h*c/wl/Tvals[j]/kB        
            if xval > 250:
                xterm = fluff_num*np.exp(-xval)
            else:
                xterm = fluff_num / ( np.exp(xval) - 1 )

            log_integrand = np.log(xterm) + 2*yvals[j]*np.log(10) + np.log( np.log(10) ) +np.log(dy)
        
            if include_F0:
                log_integrand += np.log(F0)

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
                
            integrand_vals[j] = integrand
            
        F_og[i] = np.sum( integrand_vals)

    return F_og

    # from scipy.interpolate import interp1d
    # func = interp1d(tp_vals, F_og, kind='linear')
    
    # return func(td_vals)