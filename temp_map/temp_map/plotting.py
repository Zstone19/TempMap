import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np
from tqdm import tqdm
import astropy.constants as const

from .utils import rescale, rescale_factor



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


def get_ticks(tp_vals, interval):
    Nticks = int( (tp_vals[-1] - tp_vals[0])//interval ) + 1

    assert tp_vals[-1] > interval

    val1 = np.ceil(tp_vals[0]/interval)*interval
    ticks = [val1 + interval*i for i in range(Nticks)]
    
    if tp_vals[-1] - interval/4 > val1 + interval*(Nticks-1):
        tick_labels = [int(val1 + interval*i) for i in range(Nticks)]
        Nlabel = len(tick_labels)
    else:
        tick_labels = np.concatenate([  [int(val1 + interval*i) for i in range(Nticks-1)] , [''] ])
        Nlabel = len(tick_labels) - 1
        
    return ticks, tick_labels, Nlabel


def run_tick_loop(tp_vals):
    intervals = [100, 50, 25, 20, 15, 10, 5]

    #Add xticks
    for i in range(len(intervals)):
        interval = intervals[i] 
        
        if interval > tp_vals[-1]:
            continue
        
        ticks, tick_labels, Nlabel = get_ticks(tp_vals, interval)
        if Nlabel > 4:
            interval = intervals[i-1]
            ticks, tick_labels, Nlabel = get_ticks(tp_vals, interval)
            break
        
    if (i == len(intervals) - 1) & (ticks is None):
        interval = intervals[-1]
        ticks, tick_labels, Nlabel = get_ticks(tp_vals, interval)

    return ticks, tick_labels, Nlabel


#############################################################################################
#                    Plotting the Temp Profile
#############################################################################################

from scipy.interpolate import interp1d

def plot_profs_inout(dToT_input, dToT_outputs_reshape, tp_vals, yvals, xi_vals, chi2_vals, fname=None, show=True, 
                     Ncol=4, cmap_num=0, interpolation='antialiased'):
    
    if cmap_num > 0:
        cmap = mpl.cm.get_cmap('RdBu_r', cmap_num)
    else:
        cmap = mpl.cm.get_cmap('RdBu_r')

    arrays = np.concatenate( [ [dToT_input], dToT_outputs_reshape ] )



    ticks, tick_labels, Nlabel = run_tick_loop(tp_vals)



    N = len(dToT_outputs_reshape)
    Nrow = np.ceil( N/Ncol ).astype(int)

    fig, ax = plt.subplots(Nrow, Ncol+1, figsize=(4.5*(Ncol+1), 3.75*Nrow) )


    #Plot input
    vals1 = dToT_input / np.nanpercentile( np.abs(dToT_input), 99 )

    if Nrow > 1:
        ax_ind = (0,0)
    else:
        ax_ind = 0

    im = ax[ax_ind].imshow(vals1, origin='lower', aspect='auto', 
                    extent=[ tp_vals[0], tp_vals[-1], yvals[0], yvals[-1] ],
                    cmap=cmap, vmin=-1, vmax=1, interpolation=interpolation)

    ax[ax_ind].set_ylabel(r'$\log_{10}(R / R_{in})$', fontsize=22)
    ax[ax_ind].set_title('Input', fontsize=15)

    ax[ax_ind].tick_params('both', labelsize=13)
    ax[ax_ind].tick_params('both', which='major', length=7)
    ax[ax_ind].tick_params('both', which='minor', length=3)


    x1, x2 = ax[ax_ind].get_xlim()
    y1, y2 = ax[ax_ind].get_ylim()

    xtxt = x1 + (x2-x1)*.05 
    ytxt = y1 + (y2-y1)*.1
    ax[ax_ind].text( xtxt, ytxt, 'scale = ' + '{:.4f}'.format( np.percentile( np.abs(dToT_input), 99) ), fontsize=13, color='w', weight='bold')
    
    ax[ax_ind].set_xticks(ticks)
    ax[ax_ind].set_xticklabels(  tick_labels )


    for n, dToT_output_reshape in enumerate(dToT_outputs_reshape):

        if Nrow > 1:
            ax_ind = (n//Ncol, n%Ncol +1)
        else:
            ax_ind = n+1

        vals1 = dToT_output_reshape / np.percentile( np.abs(dToT_output_reshape), 99)

        im = ax[ax_ind].imshow(vals1, origin='lower', aspect='auto', 
                          extent=[ tp_vals[0], tp_vals[-1], yvals[0], yvals[-1] ],
                          cmap=cmap, vmin=-1, vmax=1, interpolation=interpolation)

        ax[ax_ind].tick_params('both', labelsize=0)
        ax[ax_ind].tick_params('both', which='major', length=7)
        ax[ax_ind].tick_params('both', which='minor', length=3)

        #Set title
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

        ax[ax_ind].set_title(r'$\xi$ = ' + xi_txt, fontsize=19)




        vals2 = dToT_input / np.percentile( np.abs(dToT_input), 99)

        x1, x2 = ax[ax_ind].get_xlim()
        y1, y2 = ax[ax_ind].get_ylim()

        xtxt = x1 + (x2-x1)*.05 
        ytxt1 = y1 + (y2-y1)*.1
        ytxt2 = y1 + (y2-y1)*.2

        ax[ax_ind].text( xtxt, ytxt2, r'$\chi^2 / N_d$ = {:.3f}'.format(chi2_vals[n]), fontsize=13, color='k' )
        ax[ax_ind].text( xtxt, ytxt1, 'scale = ' + '{:.3f}'.format( np.percentile( np.abs(dToT_output_reshape), 99) ), fontsize=13, color='k')
        
        ax[ax_ind].set_xticks(ticks)
        ax[ax_ind].set_xticklabels(  tick_labels )



    #Add ylabel on left side
    for i in range(1, Nrow):
        if Nrow <= 1 :
            ax_ind = 1
        else:
            ax_ind = (i, 1)

        ax[ax_ind].set_ylabel(r'$\log_{10}(R / R_{in})$', fontsize=16)
        ax[ax_ind].tick_params('y', labelsize=13)
        
      
    #Add xlabel on bottom
    for i in range(Ncol):
        if Nrow <= 1:
            ax_ind = i+1
        else:
            ax_ind = (-1, i+1)
    
        ax[ax_ind].tick_params('x', labelsize=13)




    #Set remove xtick labels on top if there is more than 1 row
    if Nrow > 1:
        for i in range(0, Nrow-1):
            for j in range(Ncol):
                ax[i,j+1].tick_params('x', labelsize=0)



    #Deal with plots with no data
    if ( Nrow*Ncol > N ) & ( Nrow <= 1 ):
        for i in range( N, Nrow*Ncol ):
            ax[i+1].axis('off')
    elif ( Nrow*Ncol > N ) & ( Nrow > 1 ) :
        for i in range( N, Nrow*Ncol ):
            ax[-1, i%Ncol + 1].axis('off')
            
        for i in range(N%Ncol, Ncol):            
            ax[-2, i+1].tick_params('x', labelsize=13)

    for i in range(1, Nrow):
        ax[i, 0].axis('off')

    #Reduce spacing between subplots
    plt.subplots_adjust(wspace=.05, hspace=.25)

    plt.figtext(.4, -.05, 'Rest-Frame Time [d]', fontsize=22)

    #Make colorbar
    cbar = fig.colorbar(im, ax=[ax], location='right', shrink=1.0, pad=.01, aspect=15 + 5*(Nrow-1) )
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.tick_params('both', which='major', length=6)
    cbar.ax.tick_params('both', which='minor', length=3)
    cbar.ax.set_ylabel(r'$(\delta T / T)$ / scale', rotation=270, labelpad=25, fontsize=20)



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
    
    
    
    
    
    
    
    
    
    
    
    
def plot_profs_inout_dist(dToT_input, dToT_outputs_reshape, tp_vals, yvals, xi_vals, chi2_vals, fname=None, show=True, 
                          cmap_num=0, interpolation='antialiased', percent=99):
        
    if cmap_num > 0:
        cmap = mpl.cm.get_cmap('RdBu_r', cmap_num)
    else:
        cmap = mpl.cm.get_cmap('RdBu_r')

    arrays = np.concatenate( [ [dToT_input], dToT_outputs_reshape ] )
    ticks, tick_labels, Nlabel = run_tick_loop(tp_vals)


    fig, ax = plt.subplots(2, len(arrays), figsize=(4*len(arrays), 7) )

    for i, dToT_output_reshape in enumerate(arrays):

        vals1 = dToT_output_reshape / np.percentile( np.abs(dToT_output_reshape), percent)

        im = ax[0,i].imshow(vals1, origin='lower', aspect='auto', 
                          extent=[ tp_vals[0], tp_vals[-1], yvals[0], yvals[-1] ],
                          cmap=cmap, vmin=-1, vmax=1, interpolation=interpolation)

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
            if xi_vals[i-1] < 10000:
                xi_txt = '{}'.format(xi_vals[i-1])
            else:

                exp = int( np.log10(xi_vals[i-1]) )
                fact = xi_vals[i-1] / 10**exp

                if fact == 1:        
                    xi_txt = r'$10^{' + '{:.0f}'.format( exp ) + '}$'
                elif int(fact) == fact:
                    xi_txt = r'$' + '{:.0f}'.format(fact) + r' \times 10^{' + '{:.0f}'.format( exp ) + '}$'
                else:
                    xi_txt = r'$' + '{:.2f}'.format(fact) + r' \times 10^{' + '{:.0f}'.format( exp ) + '}$'

            ax[0,i].set_title(r'$\xi$ = ' + xi_txt, fontsize=17)

        else:
            ax[0,i].set_title('Input', fontsize=15)

        vals2 = dToT_input / np.percentile( np.abs(dToT_input), percent)

        x1, x2 = ax[0,i].get_xlim()
        y1, y2 = ax[0,i].get_ylim()

        xtxt = x1 + (x2-x1)*.05 
        ytxt1 = y1 + (y2-y1)*.1
        ytxt2 = y1 + (y2-y1)*.2

        if i != 0:
            ax[0,i].text( xtxt, ytxt2, r'$\chi^2 / N_d$ = {:.3f}'.format(chi2_vals[i-1]), fontsize=13, color='k' )

            
        if  i == 0:
            ax[0,i].text( xtxt, ytxt1, 'scale = $\mathbf{' + '{:.4f}'.format( np.percentile( np.abs(dToT_output_reshape), percent) ) + '}$', fontsize=13, color='w', fontweight='bold')
        else:
            ax[0,i].text( xtxt, ytxt1, 'scale = ' + '{:.4f}'.format( np.percentile( np.abs(dToT_output_reshape), percent) ), fontsize=13, color='k')

        ax[0,i].set_xticks(ticks)
        ax[0,i].set_xticklabels(  tick_labels )

    
        
    
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
                          cmap=cmap, vmin=-1, vmax=1, interpolation=interpolation)
    
    
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


        #Add xticks
        ax[1,i+1].set_xticks(ticks)
        ax[1,i+1].set_xticklabels(  tick_labels )
            
        ax[1,i+1].text( .05, .1, r'Med = $\mathbf{'+  '{:.3f}'.format(np.nanmedian(vals)) + '}$', 
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




def plot_profs_out(dToT_outputs_reshape, tp_vals, yvals, xi_vals, chi2_vals, fname=None, show=True, cmap_num=0,
                   Ncol=4, interpolation='antialiased', percent=99, cmap_name='RdBu_r', date_type='obs'):
    
    if cmap_num > 0:
        cmap = mpl.cm.get_cmap(cmap_name, cmap_num)
    else:
        cmap = mpl.cm.get_cmap(cmap_name)

    ticks, tick_labels, Nlabel = run_tick_loop(tp_vals)

    N = len(dToT_outputs_reshape)
    Nrow = np.ceil( N/Ncol ).astype(int)
    
        
    fig, ax = plt.subplots(Nrow, Ncol, figsize=(4.5*Ncol, 3.5*Nrow), sharey=True)

    for i, dToT_output_reshape in enumerate(dToT_outputs_reshape):

        if N/Ncol <= 1:
            ax_ind = i
        else:
            ax_ind = (i//Ncol, i%Ncol)
        
        #Rescale data
        vals1 = dToT_output_reshape / np.percentile( np.abs(dToT_output_reshape), percent)
        
        #Plot
        im = ax[ax_ind].imshow(vals1, origin='lower', aspect='auto', 
                          extent=[ tp_vals[0], tp_vals[-1], yvals[0], yvals[-1] ],
                          cmap=cmap, vmin=-1, vmax=1, interpolation=interpolation)

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
        
        ax[ax_ind].text( xtxt, ytxt2, r'$\chi^2 / N_d$ = {:.3f}'.format(chi2_vals[i]), fontsize=13, color='k' )
        ax[ax_ind].text( xtxt, ytxt1, 'scale = ' + '{:.3f}'.format( np.percentile( np.abs(dToT_output_reshape), percent) ), fontsize=13, color='k')
        
        ax[ax_ind].set_xticks(ticks)
        ax[ax_ind].set_xticklabels(  tick_labels )

    #Add ylabel on left side
    for i in range(Nrow):
        if N/Ncol <= 1 :
            ax_ind = 0
        else:
            ax_ind = (i, 0)

        ax[ax_ind].set_ylabel(r'$\log_{10}(R / R_{in})$', fontsize=22)
        ax[ax_ind].tick_params('both', labelsize=12)
        
      
    #Add xlabel on bottom
    for i in range(Ncol):
        if N/Ncol <= 1:
            ax_ind = i
        else:
            ax_ind = (-1, i)
        
        ax[ax_ind].tick_params('x', labelsize=12)

     
    #Remove xtick labels on top if there is more than 1 row
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


    if date_type == 'obs':
        plt.figtext(.4, -.05, 'MJD - 50000', fontsize=22)
    elif date_type == 'rest':
        plt.figtext(.4, -.05, 'Rest-Frame Time [d]', fontsize=22)

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

import matplotlib.animation as animation

def animate_spectra_out(fitted_spec, flux_dat, err_dat, mean_spec, td_vals, lambda_vals, xi_vals, fname, fps=10):

    #Lambda must be in units of cm

    colors = ['r', 'green', 'orange', 'c', 'b', 'lime']

    N_td = len(td_vals)
    N_nu = len(lambda_vals)
    N = len(xi_vals)

    assert fitted_spec.shape == ( N, N_nu, N_td )
    assert flux_dat.shape == ( N_nu, N_td )

    flux_dat_rel = np.zeros_like(flux_dat)
    err_dat_rel = np.zeros_like(err_dat)
    for i in range(N_td):
        flux_dat_rel[:, i] = flux_dat[:,i]/mean_spec
        err_dat_rel[:, i] = err_dat[:,i]/mean_spec


    ymax = np.max(flux_dat_rel + err_dat_rel)
    ymin = np.min(flux_dat_rel - err_dat_rel)

    fig, ax = plt.subplots(constrained_layout=True)

    def animate(i):

        x = lambda_vals/1e-8
        ax.cla()

        for n in range(N):
            spec_in = flux_dat[:,i]/mean_spec
            spec_err = err_dat[:,i]/mean_spec
            spec_out = fitted_spec[n,:,i]/mean_spec

            _, _, bars = ax.errorbar(x, spec_in, spec_err, fmt='.k', markersize=1, elinewidth=.05)
            [bar.set_alpha(.2) for bar in bars]


            ax.plot(x, spec_out, c=colors[n], lw=.8, zorder=1000 - n, label=r'$\xi$ = {}'.format(xi_vals[n]))
            
            chi2_nu = np.sum( (spec_in - spec_out)**2 / spec_err**2 ) / N_nu
            ax.text(1.02, .93 - .07*n, r'$\chi^2 / N_d$ = %.4f' % chi2_nu, transform=ax.transAxes, fontsize=11, color=colors[n] )

            ax.tick_params('both', labelsize=12)
            ax.tick_params('both', which='major', length=8)
            ax.tick_params('both', which='minor', length=3)

            ax.set_xlabel(r'Rest-Frame Wavelength [$\AA$]', fontsize=15)
            ax.set_ylim(ymin, ymax)

            if n == 0:
                ax.set_ylabel(r'$\delta F_\lambda / \overline{F_\lambda}$', fontsize=11)

            if n == 0:
                ax.text(.1, .9, r'$t_d =$ ' + '%.1f' % td_vals[i], transform=ax.transAxes, fontsize=14)

            ax.legend(loc='upper right').set_zorder(1001)
        

    anim = animation.FuncAnimation(fig, animate,
                                   frames=N_td, interval=20, repeat_delay=10)

    anim.save(fname, writer='ffmpeg', fps=fps)
    
    return




def animate_arbitrary_data(lengths, fitted_dat, flux_dat, err_dat, mean_dat, 
                           td_vals, lambda_vals, xi_vals, fname, 
                           dat_type='spec', fps=10):

    #Lambda must be in units of cm
    colors = ['r', 'green', 'orange', 'c', 'b', 'lime']

    N_td = len(td_vals)
    N_nu = len(lambda_vals)
    N = len(xi_vals)
    
    assert np.sum(lengths).astype(int) == len(td_vals) == len(lambda_vals)

    flux_dat_rel = np.zeros_like(flux_dat)
    err_dat_rel = np.zeros_like(err_dat)
    
    if dat_type == 'spec':
        td_unique = np.unique(td_vals)
        sort_ind = np.argsort(td_unique)
        
        for i in range(len(td_unique)):
            ind1 = np.sum(lengths[:i]).astype(int)
            ind2 = ind1 + lengths[i]
            
            flux_dat_rel[ind1:ind2] = flux_dat[ind1:ind2]/mean_dat
            err_dat_rel[ind1:ind2] = err_dat[ind1:ind2]/mean_dat
        
        ymax = np.max(flux_dat_rel + err_dat_rel)
        ymin = np.min(flux_dat_rel - err_dat_rel)
        
        xmin = np.min(lambda_vals)/1e-8 - 20
        xmax = np.max(lambda_vals)/1e-8 + 20
        
        ms = 1
        ewidth = .05
        e_alpha = .2



    if dat_type == 'lc':
        lambda_unique = np.array( np.unique(lambda_vals) )
        sort_ind = np.argsort(lambda_unique)
        
        ymin = None
        ymax = None
        
        xmin = np.min(td_vals) - 10
        xmax = np.max(td_vals) + 10
        
        ms = 2
        ewidth = .1
        e_alpha = .5


    fig, ax = plt.subplots(constrained_layout=True)

    def animate(j):
        i = sort_ind[j]

        ind1 = np.sum(lengths[:i]).astype(int)
        ind2 = ind1 + lengths[i]

        if dat_type == 'spec':
            x = lambda_vals[ind1:ind2]/1e-8
            ybar = mean_dat
            
        if dat_type == 'lc':
            x = td_vals[ind1:ind2]
            ybar = mean_dat[i]

        y_in = flux_dat[ind1:ind2]/ybar
        y_err = err_dat[ind1:ind2]/ybar

        ax.cla()

        for n in range(N):
            y_out = fitted_dat[n, ind1:ind2]/ybar

            _, _, bars = ax.errorbar(x, y_in, y_err, fmt='.k', markersize=ms, elinewidth=ewidth)
            [bar.set_alpha(e_alpha) for bar in bars]


            ax.plot(x, y_out, c=colors[n], lw=.8, zorder=1000 - n, label=r'$\xi$ = {}'.format(xi_vals[n]))
            
            chi2_nu = np.sum( (y_in - y_out)**2 / y_err**2 ) / len(y_in)
            ax.text(1.02, .93 - .07*n, r'$\chi^2 / N_d$ = %.4f' % chi2_nu, transform=ax.transAxes, fontsize=11, color=colors[n] )

            ax.set_ylim(ymin, ymax)
            ax.set_xlim(xmin, xmax)
            
            
            if dat_type == 'lc':
                y1, y2 = ax.get_ylim()
                ax.set_ylim( y1*1.1, y2*1.1 )

            if n == 0:
                ax.set_ylabel(r'$\delta F_\lambda / \overline{F_\lambda}$', fontsize=11)

            if n == 0:
                if dat_type == 'spec':
                    ax.text(.1, .9, r'$t_d =$ ' + '%.1f' % td_unique[i], transform=ax.transAxes, fontsize=14)
                if dat_type == 'lc':
                    ax.text(.1, .9, r'$\lambda_{\rm rest} =$ ' + '{:.1f}'.format(lambda_unique[i]/1e-8) + r' $\AA$', transform=ax.transAxes, fontsize=14)


            ax.legend(loc='upper right').set_zorder(1001)

        ax.tick_params('both', labelsize=12)
        ax.tick_params('both', which='major', length=8)
        ax.tick_params('both', which='minor', length=3)
        
        if dat_type == 'spec':
            ax.set_xlabel(r'Rest-Frame Wavelength [$\AA$]', fontsize=15)
        if dat_type == 'lc':
            ax.set_xlabel(r'Rest-Frame Time [d]', fontsize=15)

        

    anim = animation.FuncAnimation(fig, animate,
                                   frames=len(lc_lengths), interval=20, repeat_delay=10)

    anim.save(fname, writer='ffmpeg', fps=fps)
    
    return