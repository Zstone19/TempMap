import numpy as np
import astropy.constants as const
from tqdm import tqdm
from scipy.sparse import csc_matrix
from scipy.interpolate import interp1d
from numba import njit


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





def add_error(dF, F, F0_err, sd):
    
    N = len(dF)
    
    F0_errs = np.abs( np.random.normal(F0_err, sd, N) )*F
    new_dF = np.random.normal(dF, F0_errs)
    
    return new_dF, F0_errs








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
            ind1 = np.sum( lc_lengths[:i] ).astype(int)
        else:
            ind1 = 0
        
        ind2 = ind1 + lc_lengths[i]
        
        lc_in = np.array( rescale(input_lc[ind1:ind2])   )
        lc_out = np.array( rescale(new_lc[ind1:ind2])   )
        err = rescale_factor(input_lc[ind1:ind2])*np.array( lc_err[ind1:ind2] )
        
        chi2_nu = np.sum( ( ( lc_in - lc_out )/err  )**2  )/lc_lengths[i]
        chi2_vals.append(chi2_nu)
        
        
    return func( chi2_vals )