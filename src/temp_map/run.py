import numpy as np
from pyrsistent import l
from scipy.sparse import csc_matrix
import awkward as ak

from numba import njit
from numba_progress import ProgressBar
from sparse_dot_mkl import gram_matrix_mkl

from .utils import chunk_fill
from .algorithm import make_F_dF, make_W_spec_w_mean, make_smoothing_matrices, make_F_dF_nonlinear
from .plotting import plot_profs_inout_dist, plot_profs_out, animate_spectra_out


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

@njit(parallel=True)
def fast_res(A, b):
    return np.linalg.inv(A) @ b


def run_spectra(flux_dat, err_dat, mean_flux, tp_vals, yvals, td_vals, lambda_vals, 
                AGN_params, xi_vals, Nchunk=1e6, fps=10, verbose=True,
                show_tp=True, tp_fname=None, spec_fname=None, dat_fname=None):



    #AGN_params labels
    #-----------------
    # Object name:     'obj_name'
    # SMBH mass:       'MBH'
    # Distance:        'dist'
    # Eddington ratio: 'lambda_edd'
    # Redshft:         'z'
    # Alpha parameter: 'alpha'
    # Inclination:     'inc'
    
    
    #Units
    #-----
    # lambda:   cm
    # flux, err: erg/s/cm^3
    # tp, td:   days
    # y:        dimensionless
    # MBH:      g
    # dist:     cm
    # inc:      rad

    
    
    Nu = len(yvals)
    N_tp = len(tp_vals)
    N_td = len(td_vals)
    N_nu = len(lambda_vals)
    
    assert flux_dat.shape == (N_td*N_nu,)
    assert err_dat.shape == (N_td*N_nu,)
    Nchunk = int(Nchunk)
    
    
    obj_name = AGN_params['obj_name']
    MBH = AGN_params['MBH']
    lambda_edd = AGN_params['lambda_edd']
    dist = AGN_params['dist']
    z = AGN_params['z']
    inc = AGN_params['inc']
    alpha = AGN_params['alpha']
    
    
    if verbose:
        print('Making W matrix...')   
     
    row_dat = ak.ArrayBuilder()
    col_dat = ak.ArrayBuilder()
    input_dat = ak.ArrayBuilder()


    if verbose:
        with ProgressBar(total=Nu*N_tp*N_nu*N_td) as progress:
            rows, cols, inputs = make_W_spec_w_mean(row_dat, col_dat, input_dat, yvals, tp_vals, td_vals, lambda_vals,
                                                    lambda_edd, MBH, dist, inc, alpha=alpha, progress_hook=progress,
                                                    errs=err_dat, dat_type='dToT', include_F0=True)
    else:
        rows, cols, inputs = make_W_spec_w_mean(row_dat, col_dat, input_dat, yvals, tp_vals, td_vals, lambda_vals,
                                                lambda_edd, MBH, dist, inc, alpha=alpha, progress_hook=None,
                                                errs=err_dat, dat_type='dToT', include_F0=True)

    row_snap = rows.snapshot()
    col_snap = cols.snapshot()
    dat_snap = inputs.snapshot()

    W_input = chunk_fill(row_snap, col_snap, dat_snap, shape=( N_nu*N_td, Nu*N_tp + N_nu ), Nchunk=Nchunk)
    del row_dat, col_dat, input_dat
    del rows, cols, inputs
    del row_snap, col_snap, dat_snap, Nchunk
    del lambda_edd, MBH, dist, inc, alpha, z, obj_name
    
    
    
    WTW = gram_matrix_mkl(W_input, cast=True)
    size = WTW.shape[0]
    I, Dk, Dl = make_smoothing_matrices(Nu, N_tp, size)
    del size
    
    WTb = W_input.transpose() @ (flux_dat/err_dat)
    WTb = csc_matrix(WTb).transpose()
    
    
    if verbose:
        print('Inverting...')  
      
    inv_outputs = []
    for xi in xi_vals:
        A = csc_matrix( WTW + xi*(I + Dk + Dl) )

        res = fast_res(A.todense(), WTb.todense())
        res = np.array(res).T[0]

        inv_outputs.append( res )
        del A, res
        
    del WTW, WTb, I, Dk, Dl  
        
        
    #Get output 
    dToT_outputs_reshape = []
    for i in range(len(inv_outputs)):
        output_reshape_i = np.zeros( (Nu, N_tp) )
        
        for j in range(Nu*N_tp):
            output_reshape_i[ j%Nu, j//Nu ] = inv_outputs[i][j]
            
        dToT_outputs_reshape.append( output_reshape_i )

        
    new_spec = []
    for i in range(len(inv_outputs)):
        output = np.array( (W_input @ inv_outputs[i]) )
        new_spec.append( output * err_dat )


    offsets = []
    for i in range(len(inv_outputs)):
        offsets.append( inv_outputs[i][Nu*N_tp:] )
    
    del inv_outputs 
    
    
    
    #Reshape flux arrays
    flux_reshape = np.zeros( (N_nu, N_td) )
    err_reshape = np.zeros( (N_nu, N_td) )
    spec_out = np.zeros( ( len(xi_vals), N_nu, N_td) )

    for i in range(len(xi_vals)):
        for j in range(N_td):
            spec_out[i, :, j] = new_spec[i][j*N_nu:(j+1)*N_nu]
            
    for i in range(N_td):
        flux_reshape[:, i] = flux_dat[i*N_nu:(i+1)*N_nu]
        err_reshape[:, i] = err_dat[i*N_nu:(i+1)*N_nu]
        
        
        
        
    #Values are small, so use large fluff factor
    FLUFF = 1e9
        
    #Get Chi2-minimized coefficients
    m_coefs = np.zeros( ( len(xi_vals), N_nu ) )

    for i in range(len(xi_vals)):
        for j in range(N_nu):
            lc_in = flux_reshape[j,:]*FLUFF
            lc_err = err_reshape[j,:]*FLUFF
            lc_out = spec_out[i,j,:]
            
            sxy = np.sum(lc_out*lc_in / lc_err**2)
            sx2 = np.sum(lc_out**2 / lc_err**2)
            sx = np.sum(lc_out / lc_err**2)

            m_coefs[i,j] = (sxy - offsets[i][j]*sx)/sx2    
    
    
    
    
    #Fit output to input
    fitted_spec = np.zeros( ( len(xi_vals), N_nu, N_td) )
    chi2_tot = np.zeros( ( len(xi_vals), N_td ) )

    for i in range(N_td):
        spec_in = flux_reshape[:,i]
        spec_err = err_reshape[:,i]
        
        for j in range(len(xi_vals)):
            fitted_spec[j,:,i] = (m_coefs[j,:] * spec_out[j,:,i] + offsets[j])/FLUFF
            
            chi2_tot[j, i] = np.sum( (spec_in - fitted_spec[j,:,i])**2 / spec_err**2 )/N_nu
            
            
    chi2_tot = np.mean(chi2_tot, axis=1)
    
    
    
    #Save data
    if dat_fname is not None:
        np.savez_compressed(dat_fname,
                            temp_maps=dToT_outputs_reshape,
                            output_fitted_spectra=fitted_spec,
                            input_spectra=flux_reshape,
                            input_err=err_reshape,
                            mean_input_spectrum=mean_flux,
                            chi2_values=chi2_tot,
                            tp_vals=tp_vals,
                            yvals=yvals,
                            td_vals=td_vals,
                            lambda_vals=lambda_vals,
                            xi_vals=xi_vals,
                            AGN_params=AGN_params)
    
    
    #Plot output temp profile maps
    if (tp_fname is not None) or (show_tp == True):
        plot_profs_out(dToT_outputs_reshape, tp_vals, 
                                yvals, xi_vals, chi2_tot, 
                                fname=tp_fname,
                                show=show_tp, cmap_num=16,
                                percent=99, date_type='rest', Ncol= min( len(xi_vals), 5 ),
                                interpolation='gaussian')
    
    
    
    #Animate output spectra
    if spec_fname is not None:
        
        if verbose:
            print('Animating spectra...')
        
        animate_spectra_out(fitted_spec, flux_reshape, err_reshape, 
                                mean_flux, td_vals, lambda_vals, 
                                xi_vals, spec_fname, fps=fps)
        
        
    out_dict = dict( temp_maps=dToT_outputs_reshape,
                     output_fitted_spectra=fitted_spec,
                     input_spectra=flux_reshape,
                     input_err=err_reshape,
                     mean_input_spectrum=mean_flux,
                     chi2_values=chi2_tot,
                     tp_vals=tp_vals,
                     yvals=yvals,
                     td_vals=td_vals,
                     lambda_vals=lambda_vals,
                     xi_vals=xi_vals,
                     AGN_params=AGN_params)
    

    del dToT_outputs_reshape, fitted_spec, flux_reshape, err_reshape, mean_flux
    del chi2_tot, tp_vals, yvals, td_vals, lambda_vals, xi_vals, AGN_params
        
    return out_dict







def run_spectra_sim(dToT_input, tp_vals, yvals, td_vals, lambda_vals, AGN_params, xi_vals, 
                    err_mean=.03, err_std=.005, Nchunk=1e6, fps=10, verbose=True,
                    tp_fname=None, spec_fname=None, dat_fname=None, show_tp=True):
    
    Nu = len(yvals)
    N_tp = len(tp_vals)
    N_td = len(td_vals)
    N_nu = len(lambda_vals)
    
    assert dToT_input.shape == (Nu, N_tp)
    
    MBH = AGN_params['MBH']
    lambda_edd = AGN_params['lambda_edd']
    dist = AGN_params['dist']
    inc = AGN_params['inc']
    alpha = AGN_params['alpha']
    
    #Make spectra
    if verbose:
        print('Making spectra...')
        with ProgressBar(total=Nu*N_tp*N_nu*N_td) as progress:
            input_spec, _ = make_F_dF_nonlinear(dToT_input, tp_vals, td_vals, lambda_vals, yvals, 
                                            MBH, 
                                            lambda_edd, 
                                            dist, inc, progress, alpha=alpha,
                                            include_F0=True, dat_type='dToT')
    else:
        input_spec, _ = make_F_dF_nonlinear(dToT_input, tp_vals, td_vals, lambda_vals, yvals, 
                                        MBH, 
                                        lambda_edd, 
                                        dist, inc, None, alpha=alpha,
                                        include_F0=True, dat_type='dToT')
        
    
    
    #Get steady-state
    if verbose:
        print('Getting steady-state...')
        with ProgressBar(total=Nu*N_tp*N_nu*N_td) as progress:
            F_input, _ = make_F_dF(dToT_input, tp_vals, td_vals, 
                                            lambda_vals, yvals, 
                                            MBH, 
                                            lambda_edd, 
                                            dist, 
                                            inc, progress, alpha=alpha,
                                            dat_type='dToT', include_F0=True)
    else:
        F_input, _ = make_F_dF(dToT_input, tp_vals, td_vals, 
                                        lambda_vals, yvals, 
                                        MBH, 
                                        lambda_edd, 
                                        dist, 
                                        inc, None, alpha=alpha,
                                        dat_type='dToT', include_F0=True)
    

    #Add error      
    input_err = np.random.normal( F_input*err_mean, F_input*err_std )
    input_spec_w_err = np.random.normal(input_spec, input_err)
        
        
     
     
    #Subtract mean spectrum
    mean_flux = np.zeros(N_nu)
    for i in range(N_td):
        mean_flux += input_spec_w_err[i*N_nu:(i+1)*N_nu]

    mean_flux /= N_td

    processed_input_spec = np.zeros(N_td*N_nu)
    for i in range(N_td):
        processed_input_spec[i*N_nu:(i+1)*N_nu] = input_spec_w_err[i*N_nu:(i+1)*N_nu] - mean_flux
     
    
    
    
    output = run_spectra(processed_input_spec, input_err, mean_flux, 
                         tp_vals, yvals, td_vals, lambda_vals, 
                         AGN_params, xi_vals, verbose=verbose, 
                         Nchunk=Nchunk, show_tp=False)
    
    output['input'] = dToT_input
    
    
    
    dToT_outputs_reshape = output['temp_maps']
    fitted_spec = output['output_fitted_spectra']
    flux_reshape = output['input_spectra']
    err_reshape = output['input_err']
    chi2_tot = output['chi2_values']
    
    
    #Plot
    plot_profs_inout_dist(dToT_input, dToT_outputs_reshape, 
                                 tp_vals, yvals, xi_vals, chi2_tot,
                                 fname=tp_fname, show=show_tp, cmap_num=16,
                                 interpolation='gaussian')
    
    #Animate spectra
    if spec_fname is not None:
        
        if verbose:
            print('Animating spectra...')

        animate_spectra_out(fitted_spec, flux_reshape, err_reshape, 
                                mean_flux, td_vals, lambda_vals, 
                                xi_vals, spec_fname, fps=fps)
    
    
    #Save data
    if dat_fname is not None:
        np.savez_compressed( dat_fname,
                             input=dToT_input,
                             temp_maps=dToT_outputs_reshape,
                             output_fitted_spectra=fitted_spec,
                             input_spectra=flux_reshape,
                             input_err=err_reshape,
                             mean_input_spectrum=mean_flux,
                             chi2_values=chi2_tot,
                             tp_vals=tp_vals,
                             yvals=yvals,
                             td_vals=td_vals,
                             lambda_vals=lambda_vals,
                             xi_vals=xi_vals,
                             AGN_params=AGN_params)
    
    return output


