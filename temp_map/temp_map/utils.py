import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
from tqdm import tqdm
from scipy.sparse import csc_matrix
from scipy.interpolate import interp1d
from numba import njit


@njit
def get_temperature_profile(yvals, MBH, lambda_edd, alpha=6):
        
    log_Tin = np.log10(1.54e5) + .25*np.log10(lambda_edd) + .25*( np.log10(1e9) + np.log10(1.99e33) - np.log10(MBH) ) + .75*( np.log10(6) - np.log10(alpha) )       
    log_Tvals = log_Tin - .75*yvals + .25*np.log10( 1 - 10**(-yvals/2) )


    bad_mask = ( 10**log_Tvals == 0. )
    log_Tvals[bad_mask] = log_Tin - .75*yvals[bad_mask] + .25*(  np.log(.5*yvals[bad_mask]*np.log(10))/np.log(10) - yvals[bad_mask]/4 + yvals[bad_mask]**2 * np.log(10)**3 / 96 )

    assert np.argwhere( 10**log_Tvals == 0. ).size == 0
    return 10**log_Tvals

@njit
def get_Rin(MBH, alpha=6, G=const.G.cgs.value, c=const.c.cgs.value):
    return alpha*G*MBH/c/c


@njit
def get_F0(alpha, MBH, wl, dist, inc, c=const.c.cgs.value, h=const.h.cgs.value):    
    Rin = get_Rin(alpha, MBH)
    return 4*np.pi*h*c*c*np.cos(inc)*Rin*Rin/(wl**5)/(dist**2)


#############################################################################################
#                             FILTER KERNELS
#############################################################################################

@njit
def filter_loop(wl, T0_init, uvals, dy, h=const.h.cgs.value, c=const.c.cgs.value, kB=const.k_B.cgs.value):

    out_vals = np.zeros( len(uvals) )
    for j, uval in enumerate(uvals):
        xval = h*c/wl/kB/T0_init[j]

        xterm = xval/( np.exp(xval) + np.exp(-xval) - 2 )
        out_vals[j] = uval**2 * dy * np.log(10) * xterm / T0_init[j]

    return out_vals



def get_filter_kernels(lambda_vals, obj_dict, frame='rest', plot=True, fname=None):

    import warnings
    warnings.filterwarnings("ignore")


    obj_name = obj_dict['obj_name']
    z = obj_dict['z']
    lambda_edd = obj_dict['lambda_edd']
    MBH = obj_dict['MBH']

    if frame == 'observed':
        lambda_vals = lambda_vals/(1+z)


    min_wl = np.min(lambda_vals)
    max_wl = np.max(lambda_vals)


    if plot:
        fig, ax = plt.subplots()

        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable, get_cmap

        cmap = get_cmap('coolwarm')
        norm = Normalize( vmin=min_wl/1e-8, vmax=max_wl/1e-8 )
        smap = ScalarMappable(norm=norm, cmap=cmap)

    min_yvals = []
    max_yvals = []

    #More restrict range of radii
    min_yvals2 = []
    max_yvals2 = []


    if len(lambda_vals) < 50:
        lam_loop = lambda_vals
    else:
        lam_loop = np.linspace( np.min(lambda_vals), np.max(lambda_vals), 20 )

    yvals, dy = np.linspace(1e-10, 5, 10000, retstep=True)
    uvals = 10**yvals

    T0_init = get_temperature_profile(yvals, MBH, lambda_edd)
    if ~np.all( np.isfinite(T0_init) ):
        print(np.log10(MBH/1.99e33), lambda_edd)
        print(T0_init)

    assert np.all( np.isfinite(T0_init) )



    for wl in lam_loop:
        out_vals = filter_loop(wl, T0_init, uvals, dy)
        peak_y = yvals[ np.argmax( out_vals ) ]

        mask1 = (yvals < peak_y) & ( (out_vals / np.nanmax(out_vals) ) > 1e-2) 
        mask2 = (yvals > peak_y) & ( (out_vals / np.nanmax(out_vals) ) > 1e-4) 

        mask3 = (yvals < peak_y) & ( (out_vals / np.nanmax(out_vals) ) > .5)
        mask4 = (yvals > peak_y) & ( (out_vals / np.nanmax(out_vals) ) > .5) 

        nan_mask = np.isnan( out_vals / np.nanmax(out_vals) )
        if len(yvals[nan_mask]) == len(yvals):
            print(obj_name, wl*(1+z)/1e-8, peak_y)
            continue

        min_y = np.nanmin(yvals[mask1])
        max_y = np.nanmax(yvals[mask2])
        min_yvals.append(min_y)
        max_yvals.append(max_y)

        min_y = np.nanmin(yvals[mask3])
        max_y = np.nanmax(yvals[mask4])
        min_yvals2.append(min_y)
        max_yvals2.append(max_y)

        if plot:
            ax.plot(yvals, out_vals/np.nanmax(out_vals), c=smap.to_rgba( wl/1e-8 ))

    ymin = np.min(min_yvals)
    ymax = np.max(max_yvals)

    ymin2 = np.min(min_yvals2)
    ymax2 = np.max(max_yvals2)

    if plot:
        x1, x2 = ax.get_xlim()

        ax.axvspan( x1, ymin, color='gray', alpha=.25 )
        ax.axvspan( ymax, x2, color='gray', alpha=.25 )
        ax.axvline( ymin, color='k', lw=.5 )
        ax.axvline( ymax, color='k', lw=.5 )
        ax.set_xlim(0, 4.5)
        ax.set_ylim(1e-4, 1.1)

        ax.set_ylabel(obj_name, fontsize=14)
        ax.text(ymin+.05, .25, 'u = {}'.format( int( 10**ymin) ), rotation=90, fontsize=12 )
        ax.text(ymax-.15, .25, 'u = {}'.format( int(10**ymax) ), rotation=90, fontsize=12 )

        ax.set_yticks([])
        ax.tick_params(labelsize=12)
        ax.tick_params('x', which='major', length=6, zorder=1000)
        ax.tick_params('x', which='minor', length=3, zorder=1000)



        ax.set_xlabel(r'$y = \log_{10}(u) =  \log_{10} \left( R / R_{\rm in} \right) $', fontsize=16)
        plt.subplots_adjust(hspace=0)

        cbar = plt.colorbar(smap, ax=[ax], location='top', pad=0.01)
        cbar.ax.set_xlabel(r'$\lambda_{\rm rest} \ / \ \AA$', rotation=0, labelpad=10, fontsize=14)
        cbar.ax.tick_params('both', labelsize=11)
        cbar.ax.tick_params('both', which='major', length=9)
        cbar.ax.tick_params('both', which='minor', length=5)


        if fname is not None:
            plt.savefig(fname, bbox_inches='tight', dpi=200)

        plt.show()
    

    return ymin, ymax, ymin2, ymax2


#############################################################################################
#                            NORMALIZATION FROM NK22
#############################################################################################

def rescale(vals, rescale_min=-1, rescale_max=1):
    min_val = np.nanmin(vals)
    max_val = np.nanmax(vals)
    
    return rescale_min + (rescale_max - rescale_min)*(vals - min_val)/(max_val - min_val)


def rescale_factor(vals, rescale_min=-1, rescale_max=1):
    min_val = np.nanmin(vals)
    max_val = np.nanmax(vals)
    
    return (rescale_max - rescale_min)/(max_val - min_val)


#############################################################################################
#                              FILLING THE W MATRIX
#############################################################################################

@njit
def extract_indices(ak_arr, i1, i2):
    return list(ak_arr[i1:i2])


def chunk_fill(row_snap, col_snap, dat_snap, shape, Nchunk=1e5):
    
    #row_snap, col_snap, dat_snap are snapshots of the csr data Awkward arrays
    
    for n in range( len(row_snap)//Nchunk ):

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
#                                FILTERING DATA
#############################################################################################



def hampel_filter(x, y, window_size, n_sigmas=3):
    """
    Perform outlier rejection using a Hampel filter
    
    x: time (list or np array)
    y: value (list or np array)
    window_size: window size to use for Hampel filter
    n_sigmas: number of sigmas to reject outliers past
    
    returns: x, y, mask [lists of cleaned data and outlier mask]
        
    Adapted from Eryk Lewinson
    https://towardsdatascience.com/outlier-detection-with-hampel-filter-85ddf523c73d
    """
    
    # Ensure data are sorted
    if np.all(np.diff(x) > 0):
        ValueError('Data are not sorted!')
        
    x0 = x[0]
    
    n = len(x)
    outlier_mask = np.zeros(n)
    k = 1.4826 # MAD scale factor for Gaussian distribution
    

    idx = []

    # Loop over data points
    for i in range(n):
        # Window mask
        mask = (x > x[i] - window_size) & (x < x[i] + window_size)
        if len(mask) == 0:
            idx.append(i)
            continue
        # Compute median and MAD in window
        y0 = np.median(y[mask])
        S0 = k*np.median(np.abs(y[mask] - y0))
        # MAD rejection
        if (np.abs(y[i] - y0) > n_sigmas*S0):
            outlier_mask[i] = 1
            
    outlier_mask = outlier_mask.astype(np.bool)
    
    return np.array(x)[~outlier_mask], np.array(y)[~outlier_mask], outlier_mask




def clean_filter_data(x, y, yerr, window_size=150):

    #Remove NaNs
    nan_mask = np.isnan(y)
    x = x[~nan_mask]
    y = y[~nan_mask]
    yerr = yerr[~nan_mask]

    #Sort data
    ind = np.argsort(x)
    x = x[ind]
    y = y[ind]
    yerr = yerr[ind]

    #Clean up data 
    cond = (y > 0)
    x = np.array(x)[cond]
    y = np.array(y)[cond]
    yerr = np.array(yerr)[cond]

    #Filter data
    x, y, mask = hampel_filter(x, y, window_size)
    yerr = yerr[~mask]

    return x, y, yerr


#############################################################################################
#                            ADDING ERROR TO LCs (from NK22)
#############################################################################################

@njit
def factor_loop(lc_i, line_i, err_i, Niter, deps):
    chi2_vals = []
    eps = 0
    for _ in range(Niter):
        new_err = np.sqrt( err_i**2 + eps**2 )
        chi2 = np.sum( ( ( lc_i - line_i )/new_err  )**2  )

        chi2_vals.append(chi2)
        eps += deps

    return chi2_vals




@njit
def adjust_error(tvals, Fnu, Fnu_err, Niter=10000, deps=1e-4):
    
    assert len(tvals) == len(Fnu) == len(Fnu_err)

    N = len(Fnu)

    factor = 10**np.nanmedian(np.log10(  np.abs(Fnu)  ))
    offset_errs = []

    for i in range(1,N-1):

        tvals_i = tvals[i-1:i+2] - tvals[0]
        lc_i = Fnu[i-1:i+2]
        err_i = Fnu_err[i-1:i+2]

        x_avg = np.mean(tvals_i)
        y_avg = np.mean(lc_i)
        xy_avg = np.mean(tvals_i*lc_i)
        x2_avg = np.mean(tvals_i**2)

        m = (xy_avg - x_avg*y_avg)/(x2_avg - x_avg**2)
        b = y_avg - m*x_avg

        line_i = m*tvals_i + b

        chi2_vals = factor_loop(lc_i, line_i, err_i, Niter, deps*factor)

        ind = np.argmin( np.abs( np.array(chi2_vals)-1  )  )
        new_err = deps*ind*factor

        offset_errs.append(new_err)
        
    offset_errs = np.array(offset_errs)

    adjusted_errors = np.sqrt( Fnu_err**2 + np.mean(offset_errs)**2 )

    return adjusted_errors
