import numpy as np
import astropy.constants as const
from scipy.sparse import csc_matrix

from numba import njit

import sys



from .utils import get_temperature_profile, get_Rin, get_F0
from .algorithm import get_t1, get_t2, get_t3, get_t4, G1, G2



def const_perturbation(yvals, tp_vals, pert_min, pert_max):
        
    Temp_vals = np.zeros( ( len(yvals), len(tp_vals) ) )
        
    perturbation = np.zeros(len(yvals))
    pert_ind = np.argwhere( (np.log10(pert_min) <= yvals) & (yvals <= np.log10(pert_max) ) )
    perturbation[pert_ind] = 1
    
    for i in range(len(tp_vals)):        
        Temp_vals[:, i] = .1*perturbation
        
    return Temp_vals


def make_ring_out(yvals, tp_vals, pert_min, pert_max, v, tp_min=None, tp_max=None):
    
            
    Temp_vals = np.zeros( ( len(yvals), len(tp_vals) ) )

    if tp_min is None:
        tp_min = tp_vals[0]
    if tp_max is None:
        tp_max = tp_vals[-1]

    tp_ind = np.argwhere( (tp_vals >= tp_min) & (tp_vals <= tp_max) ).T[0]
    
    perturbation = np.zeros(len(yvals))
    pert_ind = np.argwhere( (np.log10(pert_min) <= yvals) & (yvals <= np.log10(pert_max) ) )
    perturbation[pert_ind] = 1
    
    for i in tp_ind:
        min_u = pert_min * 10**(v* (tp_vals[i] - tp_vals[tp_ind[0]])*24*3600 )
        max_u = pert_max * 10**(v* (tp_vals[i] - tp_vals[tp_ind[0]])*24*3600 )
    
        if np.isnan(min_u) == True:
            min_u = 0
        
        perturbation = np.zeros(len(yvals))
        pert_ind = np.argwhere( (min_u <= 10**yvals) & (10**yvals <= max_u) ).T[0]
        perturbation[pert_ind] = 1       
        
        Temp_vals[:, i] = .1*perturbation
        
    return Temp_vals


@njit
def make_outgo(yvals, tp_vals, MBH, P=20, v=.8, alpha=6, c=const.c.cgs.value, phase=0):
    
    Temp_vals = np.zeros( ( len(yvals), len(tp_vals) ) )
    
    Rin = get_Rin(MBH, alpha)
    per = P*24*60*60 #seconds

    velocities = np.full( len(yvals), v*c )

    Rvals = Rin*(10**yvals)
    omega = 2*np.pi/per

    for i in range(len(yvals)):
        Temp_vals[i, :] = .1*np.sin( omega*( tp_vals*24*60*60 - Rvals[i]/velocities[i] ) + phase )
            
    return Temp_vals


def make_ingo(tp_vals, yvals, vlog, P, offset=0):
    Nu = len(yvals)
    N_tp = len(tp_vals)
    dToT_input = np.zeros( ( Nu, N_tp ) )

    per = P * 24*60*60
    omega = 2*np.pi/per

    velocities = np.full( Nu, vlog )

    for i in range(Nu):
        dToT_input[i,:] = .1*np.sin( omega*( tp_vals*24*60*60 - yvals[i]/velocities[i] ) + offset )

    return dToT_input
