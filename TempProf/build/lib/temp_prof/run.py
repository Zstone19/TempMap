import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

import numpy as np
from pyrsistent import l
from tqdm import tqdm
from scipy.sparse import csc_matrix, diags, identity
import scipy.sparse.linalg as spla
import awkward as ak

from numba import njit
from numba_progress import ProgressBar
import astropy.constants as const
import astropy.units as u

import awkward as ak



from .input import make_lc
from .utils import get_temperature_profile, chunk_fill, add_error, get_chi2_nu_dF, rescale, rescale_factor, get_chi2_nu_lc, get_F0
from .algorithm import make_F_dF, make_W_all_in_one, make_F_dF_lc, make_W_lc
from .plotting import plot_profs_inout, plot_profs_inout_dist, animate_spectra, animate_LCs, plot_profs_out


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


########################################################################################################################
############################################### TEST: SPECTRA ##########################################################
########################################################################################################################


def analyze_temp_map_inout_spectra(temp_map, yvals, nvals, tp_vals, td_vals, AGN_params, map_fname, anim_fname, 
                                   alpha=6, R=500, n1=14.5, n2=15,  input_err=3, input_err_std=.5, 
                                   method='NK22', include_F0=False, dat_type='dToT', fluff_num=1,
                                   xi_vals=[1,10,100,1000], inversion='SPLA-INV', plot_type='dist', 
                                   fps=5, yscale='symlog'):


    """Function to analyze a given input temperature fluctuation map, given a set of 
    input parameters for the AGN it is attached to. Will create spectra from the input temperature map,
    given the frequencies and observed times given. These input spectra will be resampled at a given resolution (R)
    in a given frequency range (n1-n2), with a given uncertainty. The resampled spectra will then be used as input to the 
    inversion algorithm, which produces an output temperature fluctuation map. This map is then used to
    construct output spectra, which are then compared to the resampled input spectra.

    Arguments:
        temp_map {array}   -- 2D array of input temperature fluctuation map

        yvals {array}      -- array of radii within the map, more specifically y = log10(r/Rin) 

        nvals {array}      -- array of (log10) frequencies within the map, more specifically n = log10(freq)

        tp_vals {array}    -- array of times for the temperature map (in days)

        td_vals {array}    -- array of observed times (in days)

        AGN_params {dict}  -- dictionary of parameters for the AGN, must include the following keys:
                                  - 'MBH'        -- mass of the black hole (cgs)
                                  - 'lambda_edd' -- eddington ratio
                                  - 'inc'        -- inclination along the line of sight (in radians)
                                  - 'z'          -- redshift
                                  - 'dist'       -- distance to the AGN (cgs)

        map_fname {str}    -- name of the output file for the output temperature map

        anim_fname {str}   -- name of the output file for the animation of the output spectra (as a gif)

        alpha {float}      -- multiple of gravitational radii to use for Rin (Rin = alpha*Rg)
                              (default: 6)

        R {float}          -- resolution of the resampled spectra
                              (default: 500)

        n1 {float}         -- lower (log10) frequency bound for the resampled spectra. NOTE: n1 must be greater than min(nvals)
                              (default: 14.5)

        n2 {float}         -- upper (log10) frequency bound for the resampled spectra. NOTE: n2 must be less than max(nvals)
                              (default: 15)

        input_err {float}     -- mean percent error in the resampled spectra as a multiple of F_lambda
                              (default: 3)

        input_err_std {float} -- standard deviation of the percent error in the resampled spectra as a multiple of F_lambda
                              (default: .5)

        method {str}       -- method to use for the inversion, either 'NK22' or 'Yue'
                              (default: 'NK22')

        include_F0 {bool}  -- whether or not to multiply relevant variables by F0. If False, all spectra are divided by F0.
                              (default: False)

        dat_type {str}     -- type of temperature map input, either:
                               - 'dToT' -- input fluctuation maps are divided by the temperature profile T0
                               - 'dT'   -- unaltered input fluctuation maps (i.e. in units of K) 
                              (default: 'dToT')  

        fluff_num {float}  -- a 'fluff' number to multiply by to curb over/underflow errors
                              (default: 1)

        xi_vals {array}    -- array of xi values (smoothing factors) to use for the inversion
                              (default: [1,10,100,1000])

        inversion {str}    -- method to use for the inversion, can choose from:
                                - 'SPLA-INV'   -- scipy.sparse.lianalg.inv
                                - 'SPLA-LSMR'  -- scipy.sparse.linalg.lsmr
                                - 'SPLA-LSQR'  -- scipy.sparse.linalg.lsqr
                                - 'NPLA-INV'   -- numpy.linalg.inv
                                - 'NPLA-PINV'  -- numpy.linalg.pinv
                                - 'NPLA-LSTSQ' -- numpy.linalg.lstsq
                            (default: 'SPLA-INV')

        plot_type {str}    -- type of plot to make, either:
                                - 'dist'     -- include the distribution of fractional difference of the output fluctuation maps
                                - 'original' -- inlude only the fluctuation maps
                              (default: 'dist')

        fps {int}          -- frames per second for the output spectra animation
                              (default: 5)

    Returns:
       output_dict -- dictionary of output values, including:
                          - dToT_out                      -- output fluctuation divided by the temperature profile T0
                          - output_dF                     -- output spectra 
                          - resampled_input_spectra_w_err -- resampled input spectra after adding uncertainty
                          - resampled_input_spectra_err   -- resampled input spectra uncertainties
                          - predicted_dF                  -- input spectra
                          - resampled_F                   -- resampled input F
                          - resampled_freqs               -- resampled frequencies
    """
    

    N_tp = len(tp_vals)
    N_td = len(td_vals)
    N_nu = len(nvals)
    Nu = len(yvals)

    FLUFF = fluff_num

    nu_vals = 10**nvals
    lambda_vals = const.c.cgs.value / nu_vals

    #Extract AGN params
    MBH = AGN_params['MBH']
    lambda_edd = AGN_params['lambda_edd']
    inc = AGN_params['inc']
    z = AGN_params['z']
    dist = AGN_params['dist']

    T0 = get_temperature_profile(yvals, MBH, lambda_edd, alpha=alpha)


    if dat_type == 'dToT':

        dToT_input_flat = np.zeros(Nu*N_tp)
        for i in range(N_tp):
            for j in range(Nu):
                dToT_input_flat[i*Nu + j] = temp_map[j, i]

        dToT_input = temp_map

    elif dat_type == 'dT':

        dT_input_flat = np.zeros(Nu*N_tp)
        dToT_input_flat = np.zeros(Nu*N_tp)
        dToT_input = np.zeros( (Nu, N_tp) )
        for i in range(N_tp):
            for j in range(Nu):
                dT_input_flat[i*Nu + j] = temp_map[j, i]
                dToT_input_flat[i*Nu + j] = temp_map[j,i]/T0[j]
                dToT_input[j,i] = temp_map[j,i]/T0[j]

        dT_input = temp_map


    else:
        raise Exception('Invalid temperature map format')


    if np.any( np.array(['Yue', 'NK22']) == method ) == False:
        raise Exception('Invalid method')

    if np.any(  np.array(['SPLA-INV', 'SPLA-LSMR', 'SPLA-LSQR', 'NPLA-INV', 'NPLA-PINV', 'NPLA-LSTSQ']) == inversion ) == False:
        raise Exception('Invalid inversion method')

    if n1 < min(nvals) or n2 > max(nvals):
        raise Exception('Invalid frequency range')

    if plot_type not in ['dist', 'original']:
        raise Exception('Invalid plot type')
 

    #Plot temperature fluctuation map and a fluctuation profile at the first tp value
    fig, ax = plt.subplots(1,2, figsize=(17,5))

    ax[0].plot(yvals, dToT_input_flat[0:Nu])
    ax[0].set_xlabel(r'$\log_{10}(R / R_{in})$', fontsize=16)
    ax[0].set_ylabel(r'$\delta T \ / \ T$', fontsize=16)

    im = ax[1].imshow(dToT_input, origin='lower', aspect='auto', extent=[ tp_vals[0], tp_vals[-1], yvals[0], yvals[-1] ])
    ax[1].set_xlabel(r'$t_p$ [days]', fontsize=14)
    ax[1].set_ylabel(r'$\log_{10}(R / R_{in})$', fontsize=14)

    cbar = plt.colorbar(im, ax=ax[1])
    cbar.ax.set_ylabel(r'$\delta T \ / \ T$', rotation=270, fontsize=14, labelpad=25)

    plt.show()



    ################################################################################################################################################
    #                                                       Get Input F, dF, W
    ################################################################################################################################################


    #Get input F, dF
    print('Forming input F, dF ...')
    if dat_type == 'dToT':
        with ProgressBar(total=N_nu*N_td) as progress:
            F_input, dF_input = make_F_dF(dToT_input, tp_vals, td_vals, lambda_vals, yvals, MBH, lambda_edd, dist, inc, progress, 
                                            max_float=1e120, min_float=1e-120, fluff_num=FLUFF, include_F0=include_F0, dat_type=dat_type, alpha=alpha)

    elif dat_type == 'dT':
        with ProgressBar(total=N_nu*N_td) as progress:
            F_input, dF_input = make_F_dF(dT_input, tp_vals, td_vals, lambda_vals, yvals, MBH, lambda_edd, dist, inc, progress, 
                                            max_float=1e120, min_float=1e-120, fluff_num=FLUFF, include_F0=include_F0, dat_type=dat_type, alpha=alpha)



    #Get input W
    row_dat = ak.ArrayBuilder()
    col_dat = ak.ArrayBuilder()
    input_dat = ak.ArrayBuilder()

    print('Forming input W matrix...')
    with ProgressBar(total=N_td*Nu*N_tp*N_nu) as progress:
        rows, cols, input_dat = make_W_all_in_one(row_dat, col_dat, input_dat, 
                                                    tp_vals, td_vals, lambda_vals, yvals, 
                                                    MBH, lambda_edd, dist, inc, progress, method=method,
                                                    max_float=1e120, min_float=1e-120, fluff_num=FLUFF, 
                                                    include_F0=include_F0, dat_type=dat_type, alpha=alpha)


    row_snap = rows.snapshot()
    col_snap = cols.snapshot()
    dat_snap = input_dat.snapshot()

    print('Chunking in matrix...')
    W_input = chunk_fill(row_snap, col_snap, dat_snap, shape=(N_nu*N_td, Nu*N_tp), Nchunk=int(1e6) )


    ################################################################################################################################################
    #                                                          Get dF_pred
    ################################################################################################################################################


    #Get dF_pred
    if dat_type == 'dToT':
        dF_pred = W_input @ dToT_input_flat
    elif dat_type == 'dT':
        dF_pred = W_input @ dT_input_flat


    #Plot third timestep
    a = 3
    plt.errorbar(nvals, dF_pred[a*N_nu : (a+1)*N_nu]/F_input[a*N_nu : (a+1)*N_nu], fmt='.k', ms=2, label='Pred')
    plt.plot(nvals, dF_input[a*N_nu : (a+1)*N_nu]/F_input[a*N_nu : (a+1)*N_nu], label='Input')
    
    plt.ylim(-10, 10)
    plt.ylabel(r'$\delta F / F$')
    plt.xlabel(r'$\log_{10} ( $ Frequency [Hz] $)$')

    plt.legend()
    plt.show()


    ################################################################################################################################################
    #                                                          Resample dF_pred
    ################################################################################################################################################

    #RESAMPLE at v km/s resolution
    c = const.c.cgs.value/1e5
    v = c/R

    print('Resampled at v = {} km/s'.format(v))
    resampled_nvals = []

    nval = n1
    resampled_nvals.append( nval )

    while nval < n2:
        nval += v/c/np.log(10)
        resampled_nvals.append( nval )
        
        
    resampled_nvals = np.array(resampled_nvals)
    resampled_nu_vals = 10**resampled_nvals
    resampled_lambda_vals = const.c.cgs.value / resampled_nu_vals
        

        
    #Resample td
    resampled_td_ind = range(len(td_vals))
    resampled_td_ind = range( np.min(resampled_td_ind), np.max(resampled_td_ind), 1)

    resampled_td_vals = td_vals[resampled_td_ind]

    resampled_N_nu = len(resampled_nvals)
    resampled_N_td = len(resampled_td_vals)


    from scipy.interpolate import interp1d

    resampled_dF = np.zeros( resampled_N_td * resampled_N_nu )
    resampled_F = np.zeros( resampled_N_td * resampled_N_nu )
    for i, td_ind in enumerate(resampled_td_ind):
        
        #Interpolate dF_pred
        interp_func = interp1d( nvals, dF_pred[td_ind*N_nu:(td_ind+1)*N_nu] )
        resampled_dF[ i*resampled_N_nu:(i+1)*resampled_N_nu ] = interp_func(resampled_nvals)
        
        #Interpolate F_input
        interp_func = interp1d( nvals, F_input[td_ind*N_nu:(td_ind+1)*N_nu] )
        resampled_F[ i*resampled_N_nu:(i+1)*resampled_N_nu ] = interp_func(resampled_nvals)
    



    a = 3
    plt.errorbar(resampled_nvals, 
                np.nan_to_num( resampled_dF[a*resampled_N_nu : (a+1)*resampled_N_nu]/resampled_F[a*resampled_N_nu : (a+1)*resampled_N_nu] ),
                fmt='.k', ms=4, label='Resampled')
    plt.plot(nvals, 
            np.nan_to_num( dF_pred[resampled_td_ind[a]*N_nu : (resampled_td_ind[a]+1)*N_nu]/F_input[resampled_td_ind[a]*N_nu : (resampled_td_ind[a]+1)*N_nu] ), 
                        c='r', label='Full')
    
    plt.ylabel(r'$\delta F / F$')
    plt.xlabel(r'$\log_{10} ( $ Frequency [Hz] $)$')
    
    plt.legend()
    plt.show()


    ################################################################################################################################################
    #                                          Add error to dF_pred and make W for resampled dF_pred
    ################################################################################################################################################

    #Add error
    #Error is choseto be relative to the input F, chosen from a Gaussian with a mean=F0_err/100 and a stddev=F0_err_std/100
    resampled_dF_w_err, dF_err = add_error(resampled_dF, resampled_F, input_err/100, input_err_std/100)


    #Make resampled W matrix
    row_dat = ak.ArrayBuilder()
    col_dat = ak.ArrayBuilder()
    input_dat = ak.ArrayBuilder()

    print('Forming resampled W matrix...')
    with ProgressBar(total=N_td*Nu*N_tp*N_nu) as progress:
        rows, cols, input_dat = make_W_all_in_one(row_dat, col_dat, input_dat, 
                                                       tp_vals, resampled_td_vals, resampled_lambda_vals, yvals, 
                                                       MBH, lambda_edd, dist, inc, progress, errs=dF_err, method=method, 
                                                       fluff_num=FLUFF, include_F0=include_F0, dat_type=dat_type, alpha=alpha)


    row_snap = rows.snapshot()
    col_snap = cols.snapshot()
    dat_snap = input_dat.snapshot()

    print('Chunking in matrix...')
    W_output = chunk_fill(row_snap, col_snap, dat_snap, shape=(resampled_N_nu*resampled_N_td, Nu*N_tp), Nchunk=int(1e6) )


    ################################################################################################################################################
    #                                                   Solve system of equations
    ################################################################################################################################################


    WTW = W_output.transpose() @ W_output

    #Get smoothing matrices
    size = WTW.shape[0]

    if dat_type == 'dT':
        I = np.zeros( (size, size) )
        Dk = np.zeros( (size, size) )
        Dl = np.zeros( (size, size) )
        for i in tqdm( range(N_tp) ):
            for j in range(Nu):
                
                I[i*Nu +j, i*Nu + j] = 1/T0[j]/T0[j]
                
                if j+1 < Nu:
                    if i*Nu + j + 1 < Nu*N_tp:
                        Dk[ i*Nu + j, i*Nu + j ] = 1/T0[j]/T0[j+1]
                        Dk[ i*Nu + j, i*Nu + j + 1 ] = -1/T0[j]/T0[j+1]
                    else:
                        Dk[ i*Nu + j, i*Nu + j ] = 1/T0[j]/T0[j+1]
                
                if i*Nu + j + Nu < Nu*N_tp:
                    Dl[i*Nu + j, i*Nu + j] = 1/T0[j]/T0[j]
                    Dl[i*Nu + j, i*Nu + j + Nu] = -1/T0[j]/T0[j]
        
        I = csc_matrix(I)
        Dk = csc_matrix(Dk)
        Dl = csc_matrix(Dl)

    elif dat_type == 'dToT':
        I = identity(size)
        Dk = diags( [np.ones( size ), np.full( size-1,-1)], [0,1], shape=(size, size), format='csc' )
        Dl = diags( [np.ones( size ), np.full( size-Nu,-1)], [0,Nu], shape=(size, size), format='csc' )  




    WTb = W_output.transpose() @ (resampled_dF_w_err/dF_err)
    WTb = csc_matrix(WTb).transpose()



    print('Inverting...')
    inv_outputs = []
    for xi in tqdm(xi_vals):
        A = csc_matrix( WTW + xi*(I + Dk + Dl) )

        if inversion == 'SPLA-INV':
            res = spla.inv(A) @ WTb
            res = np.array( res.todense() ).T[0] 
        elif inversion == 'SPLA-LSMR':
            res = spla.lsmr(A, np.array(WTb.todense()).T[0])[0]
        elif inversion == 'SPLA-LSQR':
            res = spla.lsqr(A, np.array(WTb.todense()).T[0])[0]
        elif inversion == 'NPLA-INV':
            res = np.linalg.inv(A.todense()) @ WTb.todense()
            res = np.array(res).T[0]
        elif inversion == 'NPLA-PINV':
            res = np.linalg.pinv(A.todense()) @ WTb.todense()
            res = np.array(res).T[0]
        elif inversion == 'NPLA-LSTSQ':
            res = np.linalg.lstsq(A.todense(), WTb.todense())
            res = np.array(res[0]).T[0]

        inv_outputs.append( res )



    ################################################################################################################################################
    #                                                   Reshape output and get output spectra
    ################################################################################################################################################


    dToT_outputs_reshape = []

    if dat_type == 'dT':

        for n in range(len(inv_outputs)):
            dToT_output_reshape = np.zeros((Nu, N_tp))
            for i in range(inv_outputs[n].shape[0]):
                dToT_output_reshape[ i%Nu, i//Nu ] = inv_outputs[n][i]/T0[i%Nu]
                
            dToT_outputs_reshape.append(dToT_output_reshape)

    elif dat_type == 'dToT':

        for n in range(len(inv_outputs)):
            dToT_output_reshape = np.zeros((Nu, N_tp))
            for i in range(inv_outputs[n].shape[0]):
                dToT_output_reshape[ i%Nu, i//Nu ] = inv_outputs[n][i]
                
            dToT_outputs_reshape.append(dToT_output_reshape)



    new_spectra = []
    for i in range(len(inv_outputs)):
        new_spectra.append( np.array( (W_input @ inv_outputs[i]) ) )


    ################################################################################################################################################
    #                                                   Plot dToT and animate spectra
    ################################################################################################################################################


    chi2_mean = []
    chi2_med = []
    for i in range(len(xi_vals)):
        chi2 = get_chi2_nu_dF(new_spectra[i]/FLUFF, resampled_dF_w_err/FLUFF, dF_err/FLUFF, td_vals, resampled_td_ind, 
                                    nu_vals, type='upsampled', resampled_nu_vals=resampled_nu_vals)

            
        chi2_mean.append(chi2)
        
        chi2 = get_chi2_nu_dF(new_spectra[i]/FLUFF, resampled_dF_w_err/FLUFF, dF_err/FLUFF,
                            td_vals, resampled_td_ind, nu_vals, 
                            type='upsampled', resampled_nu_vals=resampled_nu_vals, func=np.median)

        chi2_med.append(chi2)







    if plot_type == 'dist':
        plot_profs_inout_dist(dToT_input, dToT_outputs_reshape, tp_vals, yvals, xi_vals, chi2_mean, 
                                   fname=map_fname, show=True, cmap_num=16)
    elif plot_type == 'original':
        plot_profs_inout(dToT_input, dToT_outputs_reshape, tp_vals, yvals, xi_vals, chi2_mean, 
                                   fname=map_fname, show=True, cmap_num=16)

    animate_spectra(new_spectra, dF_pred, dF_input, F_input, resampled_dF_w_err, resampled_F, dF_err, 
                    td_vals, resampled_td_ind, nvals, resampled_nu_vals, xi_vals, 
                     anim_fname, plot_err=True, fps=fps, yscale=yscale)



    out_dict = {'dToT_out': dToT_outputs_reshape,
                'output_dF': new_spectra,
                'resampled_input_spectra_w_err': resampled_dF_w_err,
                'resampled_input_spectra_err': dF_err,
                'predicted_dF': dF_pred,
                'resampled_F': resampled_F,
                'resampled_freq': resampled_nu_vals}

    return out_dict








########################################################################################################################
############################################### TEST: LIGHT CURVES #####################################################
########################################################################################################################

def analyze_temp_map_inout_LC(temp_map, yvals, nvals, tp_vals, td_vals, lc_lengths, AGN_params, map_fname, anim_fname, 
                              alpha=6,  input_err=3, input_err_std=.5, 
                              method='NK22', include_F0=False, dat_type='dToT', rescale_input=False, fluff_num=1,
                              xi_vals=[1,10,100,1000], inversion='SPLA-INV', plot_type='dist', 
                              fps=1, yscale='log'):

    """Function to analyze a given input temperature fluctuation map, given a set of 
    input parameters for the AGN it is attached to. Will create light curves from the input temperature map,
    given the frequencies, observed times, and light curve lengths given. Light curves will then be used as input to the 
    inversion algorithm, producing an output temperature fluctuation map. This map is then used to
    construct output light curves, which are compared to the input light curves.

    Arguments:
        temp_map {array}   -- 2D array of input temperature fluctuation map

        yvals {array}      -- array of radii within the map, more specifically y = log10(r/Rin) 

        nvals {array}      -- array of (log10) frequencies within the map, more specifically n = log10(freq)

        tp_vals {array}    -- array of times for the temperature map (in days)

        td_vals {array}    -- array of observed times (in days)

        lc_lengths {array} -- array of lengths of the light curves to be generated

        AGN_params {dict}  -- dictionary of parameters for the AGN, must include the following keys:
                                  - 'MBH'        -- mass of the black hole (cgs)
                                  - 'lambda_edd' -- eddington ratio
                                  - 'inc'        -- inclination along the line of sight (in radians)
                                  - 'z'          -- redshift
                                  - 'dist'       -- distance to the AGN (cgs)

        map_fname {str}    -- name of the output file for the output temperature map

        anim_fname {str}   -- name of the output file for the animation of the output spectra (as a gif)

        alpha {float}      -- multiple of gravitational radii to use for Rin (Rin = alpha*Rg)
                              (default: 6)

        input_err {float}     -- mean percent error in the light curves as a multiple of F_lambda
                              (default: 3)

        input_err_std {float} -- standard deviation of the percent error in the light curves as a multiple of F_lambda
                              (default: .5)

        method {str}       -- method to use for the inversion, either 'NK22' or 'Yue'
                              (default: 'NK22')

        include_F0 {bool}  -- whether or not to multiply relevant variables by F0. If False, all spectra are divided by F0.
                              (default: False)

        dat_type {str}     -- type of temperature map input, either:
                               - 'dToT' -- input fluctuation maps are divided by the temperature profile T0
                               - 'dT'   -- unaltered input fluctuation maps (i.e. in units of K) 
                              (default: 'dToT')  

        fluff_num {float}  -- a 'fluff' number to multiply by to curb over/underflow errors
                              (default: 1)

        xi_vals {array}    -- array of xi values (smoothing factors) to use for the inversion
                              (default: [1,10,100,1000])

        inversion {str}    -- method to use for the inversion, can choose from:
                                - 'SPLA-INV'   -- scipy.sparse.lianalg.inv
                                - 'SPLA-LSMR'  -- scipy.sparse.linalg.lsmr
                                - 'SPLA-LSQR'  -- scipy.sparse.linalg.lsqr
                                - 'NPLA-INV'   -- numpy.linalg.inv
                                - 'NPLA-PINV'  -- numpy.linalg.pinv
                                - 'NPLA-LSTSQ' -- numpy.linalg.lstsq
                            (default: 'SPLA-INV')

        plot_type {str}    -- type of plot to make, either:
                                - 'dist'     -- include the distribution of fractional difference of the output fluctuation maps
                                - 'original' -- inlude only the fluctuation maps
                              (default: 'dist')

        fps {int}          -- frames per second for the output light curve animation
                              (default: 1)

    Returns:
       output_dict -- dictionary of output values, including:
                          - dToT_out                      -- output fluctuation divided by the temperature profile T0
                          - input_LC                      -- input light curves constructed from the input temperature fluctuation map
                          - input_err                     -- error in the input light curves
                          - output_LC                     -- output light curves
                          - input_F                       -- input F
                          - input_dF                      -- input dF
    """
    

    N_tp = len(tp_vals)
    Nu = len(yvals)

    FLUFF = fluff_num

    nu_vals = 10**nvals
    lambda_vals = const.c.cgs.value / nu_vals
    dtp = tp_vals[1] - tp_vals[0]


    #Extract AGN params
    MBH = AGN_params['MBH']
    lambda_edd = AGN_params['lambda_edd']
    inc = AGN_params['inc']
    z = AGN_params['z']
    dist = AGN_params['dist']

    T0 = get_temperature_profile(yvals, MBH, lambda_edd, alpha=alpha)


    if dat_type == 'dToT':

        dToT_input_flat = np.zeros(Nu*N_tp)
        for i in range(N_tp):
            for j in range(Nu):
                dToT_input_flat[i*Nu + j] = temp_map[j, i]

        dToT_input = temp_map

    elif dat_type == 'dT':

        dT_input_flat = np.zeros(Nu*N_tp)
        dToT_input_flat = np.zeros(Nu*N_tp)
        dToT_input = np.zeros( (Nu, N_tp) )
        for i in range(N_tp):
            for j in range(Nu):
                dT_input_flat[i*Nu + j] = temp_map[j, i]
                dToT_input_flat[i*Nu + j] = temp_map[j,i]/T0[j]
                dToT_input[j,i] = temp_map[j,i]/T0[j]

        dT_input = temp_map


    else:
        raise Exception('Invalid temperature map format')


    if np.any( np.array(['Yue', 'NK22']) == method ) == False:
        raise Exception('Invalid method')

    if np.any(  np.array(['SPLA-INV', 'SPLA-LSMR', 'SPLA-LSQR', 'NPLA-INV', 'NPLA-PINV', 'NPLA-LSTSQ']) == inversion ) == False:
        raise Exception('Invalid inversion method')

    if plot_type not in ['dist', 'original']:
        raise Exception('Invalid plot type')
 


    tot_lc_array = []
    tot_err_array = []
    tot_F_input = []
    tot_dF_input = []

    tot_wl_array = []
    tot_td_array = []

    cmap = get_cmap('cool')
    norm = Normalize( vmin=np.log10(lambda_vals[-1]/1e-8), vmax=np.log10(lambda_vals[0]/1e-8) )

    smap = ScalarMappable(norm=norm, cmap=cmap)

    for i, lam in tqdm( enumerate(lambda_vals ) ):
        
        ind1 = np.sum(lc_lengths[:i])
        ind2 = ind1 + lc_lengths[i]
        td_lc = td_vals[ind1:ind2]

        wl_array = np.zeros_like(td_lc, lam)
        wl_array[:] = lam
        

        if dat_type == 'dToT':
            F, dF = make_F_dF_lc(dToT_input, tp_vals, td_lc, wl_array, yvals, MBH, lambda_edd, dist, inc, progress_hook=None, 
                                 fluff_num=FLUFF, include_F0=include_F0, alpha=alpha, dat_type=dat_type)
            lc_vals_no_err = make_lc(dToT_input, yvals, tp_vals, td_lc, lam, MBH, inc, dist, lambda_edd, 
                                    fluff_num=FLUFF, include_F0=include_F0, alpha=alpha, dat_type=dat_type)/FLUFF

        elif dat_type == 'dT':
            F, dF = make_F_dF_lc(dT_input, tp_vals, td_lc, wl_array, yvals, MBH, lambda_edd, dist, inc, progress_hook=None, 
                                 fluff_num=FLUFF, include_F0=include_F0, alpha=alpha, dat_type=dat_type)
            lc_vals_no_err = make_lc(dT_input, yvals, tp_vals, td_lc, lam, MBH, inc, dist, lambda_edd, 
                                    fluff_num=FLUFF, include_F0=include_F0, alpha=alpha, dat_type=dat_type)/FLUFF
        

        
        percent = np.random.normal(input_err, input_err_std, len(lc_vals_no_err))
        lc_errs = np.abs(F*percent)
        lc_vals = np.random.normal( lc_vals_no_err, lc_errs/FLUFF )*FLUFF
        
        dF_lc_err = np.sqrt( lc_errs**2 + np.std(lc_vals)**2  )
        dF_lc = lc_vals - np.mean(lc_vals)

        
        tot_wl_array.append(wl_array)
        tot_td_array.append(td_lc)
        tot_lc_array.append(dF_lc)
        tot_err_array.append(dF_lc_err)
        
        tot_F_input.append(F)
        tot_dF_input.append(dF)
        
        plt.errorbar(td_lc, rescale(dF_lc), rescale_factor(dF_lc)*dF_lc_err, c=smap.to_rgba(  np.log10(lam/1e-8) ))
        plt.fill_between(td_lc, rescale(dF_lc) - rescale_factor(dF_lc)*dF_lc_err, 
                                rescale(dF_lc) + rescale_factor(dF_lc)*dF_lc_err,
                        color=smap.to_rgba(  np.log10(lam/1e-8) ), alpha=.2)
        
        
    plt.colorbar(smap)
    plt.show()


    tot_wl_array = np.hstack(tot_wl_array)
    tot_td_array = td_vals
    tot_lc_array = np.hstack(tot_lc_array)
    tot_err_array = np.hstack(tot_err_array)

    tot_F_input = np.hstack(tot_F_input)
    tot_dF_input = np.hstack(tot_dF_input)


    ################################################################################################################################################
    #                                                       Get Input W
    ################################################################################################################################################


    #Get input W
    row_dat = ak.ArrayBuilder()
    col_dat = ak.ArrayBuilder()
    input_dat = ak.ArrayBuilder()

    print('Forming input W matrix...')
    with ProgressBar(total=Nu*N_tp*len(tot_td_array)) as progress:
        rows, cols, inputs = make_W_lc(row_dat, col_dat, input_dat, yvals, tp_vals, tot_td_array, tot_wl_array, 
                                       lambda_edd, MBH, dist, inc, alpha=alpha, progress_hook=progress, 
                                       errs=tot_err_array, method=method, fluff_num=FLUFF, include_F0=include_F0,
                                       dat_type=dat_type)


    row_snap = rows.snapshot()
    col_snap = cols.snapshot()
    dat_snap = input_dat.snapshot()

    print('Chunking in matrix...')
    W_input = chunk_fill(row_snap, col_snap, dat_snap, shape=( len(tot_td_array), Nu*N_tp ), Nchunk=int(1e4))

    ################################################################################################################################################
    #                                                   Solve system of equations
    ################################################################################################################################################


    WTW = W_input.transpose() @ W_input

    #Get smoothing matrices
    size = WTW.shape[0]

    if dat_type == 'dT':
        I = np.zeros( (size, size) )
        Dk = np.zeros( (size, size) )
        Dl = np.zeros( (size, size) )
        for i in tqdm( range(N_tp) ):
            for j in range(Nu):
                
                I[i*Nu +j, i*Nu + j] = 1/T0[j]/T0[j]
                
                if j+1 < Nu:
                    if i*Nu + j + 1 < Nu*N_tp:
                        Dk[ i*Nu + j, i*Nu + j ] = 1/T0[j]/T0[j+1]
                        Dk[ i*Nu + j, i*Nu + j + 1 ] = -1/T0[j]/T0[j+1]
                    else:
                        Dk[ i*Nu + j, i*Nu + j ] = 1/T0[j]/T0[j+1]
                
                if i*Nu + j + Nu < Nu*N_tp:
                    Dl[i*Nu + j, i*Nu + j] = 1/T0[j]/T0[j]
                    Dl[i*Nu + j, i*Nu + j + Nu] = -1/T0[j]/T0[j]
        
        I = csc_matrix(I)
        Dk = csc_matrix(Dk)
        Dl = csc_matrix(Dl)

    elif dat_type == 'dToT':
        I = identity(size)
        Dk = diags( [np.ones( size ), np.full( size-1,-1)], [0,1], shape=(size, size), format='csc' )
        Dl = diags( [np.ones( size ), np.full( size-Nu,-1)], [0,Nu], shape=(size, size), format='csc' )  



    if rescale_input == False:
        WTb = W_input.transpose() @ (tot_lc_array/tot_err_array)
        WTb = csc_matrix(WTb).transpose()

    else:
        tot_lc_array_rescaled = []
        tot_err_array_rescaled = []
        for i in range(len(lc_lengths)):
            ind1 = np.sum(lc_lengths[:i]).astype(int)
            ind2 = ind1 + lc_lengths[i]

            tot_lc_array_rescaled.append( rescale(tot_lc_array[ind1:ind2]) )
            tot_err_array_rescaled.append( rescale_factor(tot_err_array[ind1:ind2])*tot_err_array[ind1:ind2] )

        tot_lc_array_rescaled = np.hstack(tot_lc_array_rescaled)
        tot_err_array_rescaled = np.hstack(tot_err_array_rescaled)

        WTb = W_input.transpose() @ (tot_lc_array_rescaled/tot_err_array_rescaled)
        WTb = csc_matrix(WTb).transpose()


    print('Inverting...')
    inv_outputs = []
    for xi in tqdm(xi_vals):
        A = csc_matrix( WTW + xi*(I + Dk + Dl) )

        if inversion == 'SPLA-INV':
            res = spla.inv(A) @ WTb
            res = np.array( res.todense() ).T[0] 
        elif inversion == 'SPLA-LSMR':
            res = spla.lsmr(A, np.array(WTb.todense()).T[0])[0]
        elif inversion == 'SPLA-LSQR':
            res = spla.lsqr(A, np.array(WTb.todense()).T[0])[0]
        elif inversion == 'NPLA-INV':
            res = np.linalg.inv(A.todense()) @ WTb.todense()
            res = np.array(res).T[0]
        elif inversion == 'NPLA-PINV':
            res = np.linalg.pinv(A.todense()) @ WTb.todense()
            res = np.array(res).T[0]
        elif inversion == 'NPLA-LSTSQ':
            res = np.linalg.lstsq(A.todense(), WTb.todense())
            res = np.array(res[0]).T[0]

        inv_outputs.append( res )



    ################################################################################################################################################
    #                                                   Reshape output and get output LCs
    ################################################################################################################################################


    dToT_outputs_reshape = []

    if dat_type == 'dT':

        for n in range(len(inv_outputs)):
            dToT_output_reshape = np.zeros((Nu, N_tp))
            for i in range(inv_outputs[n].shape[0]):
                dToT_output_reshape[ i%Nu, i//Nu ] = inv_outputs[n][i]/T0[i%Nu]
                
            dToT_outputs_reshape.append(dToT_output_reshape)

    elif dat_type == 'dToT':

        for n in range(len(inv_outputs)):
            dToT_output_reshape = np.zeros((Nu, N_tp))
            for i in range(inv_outputs[n].shape[0]):
                dToT_output_reshape[ i%Nu, i//Nu ] = inv_outputs[n][i]
                
            dToT_outputs_reshape.append(dToT_output_reshape)



    new_lcs = []
    for i in range(len(inv_outputs)):
        new_lcs.append( np.array( (W_input @ inv_outputs[i]) ) )


    ################################################################################################################################################
    #                                                   Plot dToT and animate LCs
    ################################################################################################################################################

    chi2_mean = []
    chi2_med = []
    for i in range(len(xi_vals)):    
        chi2 = get_chi2_nu_lc(new_lcs[i], tot_lc_array/FLUFF, tot_err_array/FLUFF, lc_lengths)
        chi2_mean.append(chi2)
        
        
        chi2 = get_chi2_nu_lc(new_lcs[i], tot_lc_array/FLUFF, tot_err_array/FLUFF, lc_lengths, func=np.median)
        chi2_med.append(chi2)



    if plot_type == 'dist':
        plot_profs_inout_dist(dToT_input, dToT_outputs_reshape, tp_vals, yvals, xi_vals, chi2_mean, 
                                   fname=map_fname, show=True, cmap_num=16)
    elif plot_type == 'original':
        plot_profs_inout(dToT_input, dToT_outputs_reshape, tp_vals, yvals, xi_vals, chi2_mean, 
                                   fname=map_fname, show=True, cmap_num=16)


    filters =[]
    for _ in range(len(lambda_vals)):
        filters.append('')        

    animate_LCs( np.array(new_lcs), tot_lc_array/FLUFF, tot_err_array/FLUFF, tot_td_array, dtp,
                lc_lengths, lambda_vals, filters, xi_vals, 0, np.nanmin(td_vals),
                fname=anim_fname, fps=fps)



    out_dict = {'dToT_out': dToT_outputs_reshape,
                'input_dF_LC': tot_lc_array/FLUFF,
                'input_dF_err': tot_err_array/FLUFF,
                'output_LC': new_lcs,
                'input_F': tot_F_input,
                'input_dF': tot_dF_input
    }

    return out_dict





########################################################################################################################
##################################################### SPECTRA ##########################################################
########################################################################################################################





########################################################################################################################
################################################### LIGHT CURVES #######################################################
########################################################################################################################


@njit
def adjust_error(tvals, Fnu, Fnu_err, Niter=10000, deps=1e-4):
    
    assert len(tvals) == len(Fnu) == len(Fnu_err)

    N = len(Fnu)

    factor = 10**np.median(np.log10(Fnu))
    offset_errs = []

    for i in range(1,N-1):

        tvals_i = tvals[i-1:i+2] - tvals[0]
        lc_i = Fnu[i-1:i+2]/factor
        err_i = Fnu_err[i-1:i+2]/factor

        xavg = np.mean(tvals_i)
        yavg = np.mean(lc_i)

        m = np.sum( (tvals_i - xavg)*(lc_i - yavg) ) / np.sum( (tvals_i - xavg)**2 )
        b = yavg - m*xavg

        line_i = m*tvals_i + b

        chi2_vals = []
        eps = 0
        for _ in range(Niter):
            new_err = np.sqrt( err_i**2 + eps**2 )
            chi2 = np.sum( ( ( lc_i - line_i )/new_err  )**2  )

            chi2_vals.append(chi2)
            eps += deps

        ind = np.argmin( np.abs( np.array(chi2_vals)-1  )  )
        new_err = deps*ind

        offset_errs.append(new_err*factor)
        
    offset_errs = np.array(offset_errs)
    adjusted_errors = np.sqrt( Fnu_err**2 + np.mean(offset_errs)**2 )

    return adjusted_errors


def analyze_LCs(lightcurves, times, errors, lc_lengths, N_tp, yvals, AGN_params, filters, lambda_vals, map_fname, anim_fname,
                alpha=6, xi_vals=[1,10,100,1000], fluff_num=1, min_td=0, dat_type='dToT',
                include_F0=True, add_error=True, method='NK22', inversion='SPLA-INV', rescale_input=False,
                interval=50, fps=.25):


    if min_td == 0:
        min_td = np.nanmin(times)

    Nu = len(yvals)
    FLUFF = fluff_num

    #Extract AGN params
    MBH = AGN_params['MBH']
    lambda_edd = AGN_params['lambda_edd']
    inc = AGN_params['inc']
    z = AGN_params['z']
    dist = AGN_params['dist']


    min_wl = np.min(lambda_vals)
    max_wl = np.max(lambda_vals)
    norm = Normalize(vmax=np.log10( const.c.cgs.value/min_wl ), vmin=np.log10( const.c.cgs.value/max_wl ))
    cmap = get_cmap('cool')
    sm = ScalarMappable(norm=norm, cmap=cmap)



    print('Getting dF in rest frame...')
    flux_dat = []
    err_dat = []
    td_vals = []
    real_lc_lengths = []

    for i in tqdm(  range(len(filters))  ):
        nu = const.c.cgs.value / lambda_vals[i]
        F0 = get_F0(alpha, MBH, lambda_vals[i], dist, inc)

        ind1 = np.sum(lc_lengths[:i]).astype(int)
        ind2 = ind1 + lc_lengths[i]

        lc = np.array(lightcurves[ind1:ind2])
        lc_err = np.array(errors[ind1:ind2])
        tvals = np.array(times[ind1:ind2])

        #Remove NaNs
        nan_mask = np.isnan(lc)
        lc = lc[~nan_mask]
        lc_err = lc_err[~nan_mask]
        tvals = tvals[~nan_mask]
        
        #Remove earlier times if necessary
        time_mask = tvals > min_td
        lc = lc[time_mask]
        lc_err = lc_err[time_mask]
        tvals = (tvals[time_mask] - min_td)/(1+z)

        
        #Sort
        sort_ind = np.argsort(tvals)
        lc = lc[sort_ind]
        lc_err = lc_err[sort_ind]
        tvals = tvals[sort_ind]


        #Get spectral density from mag
        Fnu = 10**( (lc+48.6)/(-2.5) ) * nu
        Fnu_err = lc_err * (-.4*np.log(10))* 10**( -2*( lc + 243/5 )/5 ) * nu

        assert len(Fnu) == len(lc)

        if add_error:
            new_err = adjust_error(tvals, Fnu, Fnu_err, Niter=50000)
        
        #Get dF
        dF = Fnu - np.mean(Fnu)
        avg_Fnu_err = np.std(Fnu)/np.sqrt( len(Fnu) )
        dF_err = np.sqrt(new_err**2 + avg_Fnu_err**2)
        
        if include_F0 == False:
            dF /= F0
            dF_err /= F0
            
        plt.errorbar(tvals, rescale(dF), rescale_factor(dF)*dF_err, 
                     c=sm.to_rgba(np.log10(nu) ) )


        flux_dat.append(dF)
        err_dat.append(dF_err)
        td_vals.append(tvals)
        real_lc_lengths.append( len(dF) )
        
    flux_dat = np.hstack(flux_dat)
    err_dat = np.hstack(err_dat)
    td_vals = np.hstack(td_vals)

    output_lambda = []
    for i in range(len(lambda_vals)):
        output_lambda.append( np.full(lc_lengths[i], lambda_vals[i])  )

    output_lambda = np.hstack(output_lambda)
             
    plt.ylabel(r'Normalized $\delta F$', fontsize=15)
    plt.xlabel('$t_{rf}$ [days]', fontsize=15)
    
    cbar = plt.colorbar(sm)
    cbar.ax.set_ylabel(r'$\log_{10}$(Frequency [Hz])', rotation=270, labelpad=25, fontsize=15)
    plt.show()


    tp_vals = np.linspace(np.min(td_vals), np.max(td_vals), N_tp)


    print('Making W matrix...')
    row_dat = ak.ArrayBuilder()
    col_dat = ak.ArrayBuilder()
    input_dat = ak.ArrayBuilder()

    with ProgressBar(total=Nu*N_tp*np.sum(lc_lengths)) as progress:
        rows, cols, inputs = make_W_lc(row_dat, col_dat, input_dat, yvals, tp_vals, td_vals, output_lambda, 
                                            lambda_edd, MBH, dist, inc, alpha=alpha, progress_hook=progress, 
                                            errs=err_dat, method=method, fluff_num=FLUFF, 
                                            include_F0=include_F0, dat_type=dat_type)



    row_snap = rows.snapshot()
    col_snap = cols.snapshot()
    dat_snap = inputs.snapshot()

    print('Chunking in matrix...')
    W_input = chunk_fill(row_snap, col_snap, dat_snap, shape=( len(td_vals), Nu*N_tp ), Nchunk=int(1e4))


    WTW = W_input.transpose() @ W_input
    size = WTW.shape[0]
    T0 = get_temperature_profile(yvals, MBH, lambda_edd)

    if dat_type =='dT':
        I = np.zeros( (size, size) )
        Dk = np.zeros( (size, size) )
        Dl = np.zeros( (size, size) )
        for i in tqdm( range(N_tp) ):
            for j in range(Nu):
                
                I[i*Nu +j, i*Nu + j] = 1/T0[j]/T0[j]
                
                if j+1 < Nu:
                    if i*Nu + j + 1 < Nu*N_tp:
                        Dk[ i*Nu + j, i*Nu + j ] = 1/T0[j]/T0[j+1]
                        Dk[ i*Nu + j, i*Nu + j + 1 ] = -1/T0[j]/T0[j+1]
                    else:
                        Dk[ i*Nu + j, i*Nu + j ] = 1/T0[j]/T0[j+1]
                
                if i*Nu + j + Nu < Nu*N_tp:
                    Dl[i*Nu + j, i*Nu + j] = 1/T0[j]/T0[j]
                    Dl[i*Nu + j, i*Nu + j + Nu] = -1/T0[j]/T0[j]
        
        I = csc_matrix(I)
        Dk = csc_matrix(Dk)
        Dl = csc_matrix(Dl)

    elif dat_type == 'dToT':
        I = identity( size, format='csc' )        
        Dk = diags( [np.ones( size ), np.full( size-1,-1)], [0,1], shape=(size, size), format='csc' )
        Dl = diags( [np.ones( size ), np.full( size-Nu,-1)], [0,Nu], shape=(size, size), format='csc' )  



    if rescale_input == False:
        WTb = W_input.transpose() @ (flux_dat/err_dat)
        WTb = csc_matrix(WTb).transpose()

    else:
        flux_dat_rescaled = []
        err_dat_rescaled = []
        for i in range(len(real_lc_lengths)):
            ind1 = np.sum(real_lc_lengths[:i]).astype(int)
            ind2 = ind1 + real_lc_lengths[i]

            flux_dat_rescaled.append( rescale(flux_dat[ind1:ind2]) )
            err_dat_rescaled.append( rescale_factor(flux_dat[ind1:ind2])*err_dat[ind1:ind2] )

        flux_dat_rescaled = np.hstack(flux_dat_rescaled)
        err_dat_rescaled = np.hstack(err_dat_rescaled)

        WTb = W_input.transpose() @ (flux_dat_rescaled/err_dat_rescaled)
        WTb = csc_matrix(WTb).transpose()




    print('Inverting...')
    inv_outputs = []
    for xi in tqdm(xi_vals):
        A = csc_matrix( WTW + xi*(I + Dk + Dl) )

        if inversion == 'SPLA-INV':
            res = spla.inv(A) @ WTb
            res = np.array( res.todense() ).T[0] 
        elif inversion == 'SPLA-LSMR':
            res = spla.lsmr(A, np.array(WTb.todense()).T[0])[0]
        elif inversion == 'SPLA-LSQR':
            res = spla.lsqr(A, np.array(WTb.todense()).T[0])[0]
        elif inversion == 'NPLA-INV':
            res = np.linalg.inv(A.todense()) @ WTb.todense()
            res = np.array(res).T[0]
        elif inversion == 'NPLA-PINV':
            res = np.linalg.pinv(A.todense()) @ WTb.todense()
            res = np.array(res).T[0]
        elif inversion == 'NPLA-LSTSQ':
            res = np.linalg.lstsq(A.todense(), WTb.todense())
            res = np.array(res[0]).T[0]

        inv_outputs.append( res )






    dToT_outputs_reshape = []

    if dat_type == 'dT':

        for n in range(len(inv_outputs)):
            dToT_output_reshape = np.zeros((Nu, N_tp))
            for i in range(inv_outputs[n].shape[0]):
                dToT_output_reshape[ i%Nu, i//Nu ] = inv_outputs[n][i]/T0[i%Nu]
                
            dToT_outputs_reshape.append(dToT_output_reshape)

    elif dat_type == 'dToT':

        for n in range(len(inv_outputs)):
            dToT_output_reshape = np.zeros((Nu, N_tp))
            for i in range(inv_outputs[n].shape[0]):
                dToT_output_reshape[ i%Nu, i//Nu ] = inv_outputs[n][i]
                
            dToT_outputs_reshape.append(dToT_output_reshape)



    new_lcs = []
    for i in range(len(inv_outputs)):
        new_lcs.append( np.array( (W_input @ inv_outputs[i]) ) )




    chi2_mean = []
    chi2_med = []
    for i in range(len(xi_vals)):    
        chi2 = get_chi2_nu_lc(new_lcs[i], flux_dat/FLUFF, err_dat/FLUFF, real_lc_lengths)
        chi2_mean.append(chi2)
        
        
        chi2 = get_chi2_nu_lc(new_lcs[i], flux_dat/FLUFF, err_dat/FLUFF, real_lc_lengths, func=np.median)
        chi2_med.append(chi2)


    plot_profs_out(dToT_outputs_reshape, tp_vals*(1+z) + min_td-50000, yvals, xi_vals, chi2_mean, 
                   fname=map_fname, show=True, cmap_num=16, interval=interval)


    animate_LCs( np.array(new_lcs), flux_dat/FLUFF, err_dat/FLUFF, td_vals*(1+z) + min_td, tp_vals[1]-tp_vals[0],
                real_lc_lengths, lambda_vals, filters, xi_vals, z, min_td,
                fname=anim_fname, fps=fps)



    out_dict = {'dToT_out': dToT_outputs_reshape,
                'dF': flux_dat,
                'dF_err': err_dat,
                'LC_out': new_lcs,
                'Times': td_vals*(1+z) + min_td,
                'LC_lengths': real_lc_lengths
    }

    return out_dict