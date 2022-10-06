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


def make_in_and_out(tp_vals, yvals, MBH, vout, vin, Aout, Ain, 
                    Pout, Pin, offset):
    
    dToT_input = (Aout/(Aout+Ain))*make_outgo(yvals, tp_vals, MBH, P=Pout, v=vout)
    dToT_input += (Ain/(Aout+Ain))*make_ingo(tp_vals, yvals, vin, Pin, offset)
    
    return dToT_input



def make_bumps(tp_vals, yvals, v, P, offset):
    from skimage.draw import random_shapes
    
    Nu = len(yvals)
    N_tp = len(tp_vals)

    dToT_input = .5*make_ingo(tp_vals, yvals, v, P, offset)

    shape_arr_tot = np.zeros_like(dToT_input)
    shape_arr1 = random_shapes( dToT_input.shape, max_shapes=6,
                               min_shapes=3, allow_overlap=True, 
                               random_seed=1800, num_channels=1,
                               intensity_range=(1, 2) )[0][:,:,0]

    for i in range(Nu):
        for j in range(N_tp):
            if shape_arr1[i,j] == 255:
                shape_arr_tot[i,j] = 0

            if shape_arr1[i,j] == 1:
                shape_arr_tot[i,j] = -1
            
            if shape_arr1[i,j] == 2:
                shape_arr_tot[i,j] = 1


    shape_arr2 = random_shapes( dToT_input.shape, max_shapes=6,
                               min_shapes=3, allow_overlap=True, 
                               random_seed=18000, num_channels=1,
                               intensity_range=(1,2) )[0][:,:,0]

    for i in range(Nu):
        for j in range(N_tp):
            if shape_arr2[i,j] == 255:
                shape_arr_tot[i,j] += 0
            elif shape_arr2[i,j] == 1:
                shape_arr_tot[i,j] += -1
            elif shape_arr2[i,j] == 2:
                shape_arr_tot[i,j] += 1

    dToT_input += .05*shape_arr_tot
    
    return dToT_input






def make_multiple(yvals, tp_vals, params):
    tot_array = []
    
    for param in params:
        
        if param['type'] == 'ring_out':
            v = param['v']
            pert_min = param['pert_min']
            pert_max = param['pert_max']
            
            dToT_input = make_ring_out(yvals, tp_vals, pert_min, pert_max, v)
        
        if param['type'] == 'outgo':
            v = param['v']
            P = param['P']
            phase = param['offset']
            MBH = param['MBH']
            
            dToT_input = make_outgo(yvals, tp_vals, MBH, P=P, v=v, phase=phase)
            
            
        if param['type'] == 'ingo':
            v = param['v']
            P = param['P']
            offset = param['offset']
            
            dToT_input = make_ingo(tp_vals, yvals, v, P, offset)
            
        
        if param['type'] == 'in_and_out':
            Ain = param['Ain']
            Aout = param['Aout']
            
            vin = param['vin']
            vout = param['vout']
            
            Pin = param['Pin']
            Pout = param['Pout']
            
            MBH = param['MBH']
            offset = param['offset']
            
            dToT_input = make_in_and_out(tp_vals, yvals, MBH, vout, vin, Aout, Ain, Pout, Pin, offset)
            
            
        if param['type'] == 'bumps':
            v = param['v']
            P = param['P']
            offset = param['offset']
            
            dToT_input = make_bumps(tp_vals, yvals, v, P, offset)
            
        tot_array.append( dToT_input )
        
    return tot_array
    

