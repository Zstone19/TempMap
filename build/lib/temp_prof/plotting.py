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
#                    Plotting the Temp Profile
#############################################################################################

from scipy.interpolate import interp1d

def plot_profs_inout(dToT_input, dToT_outputs_reshape, tp_vals, yvals, xi_vals, chi2_vals, fname=None, show=True, Ncol=4, cmap_num=0, interval=50):
    
    if cmap_num > 0:
        cmap = mpl.cm.get_cmap('RdBu_r', cmap_num)
    else:
        cmap = mpl.cm.get_cmap('RdBu_r')

    arrays = np.concatenate( [ [dToT_input], dToT_outputs_reshape ] )


    N = len(dToT_outputs_reshape)
    Nrow = np.ceil( N/Ncol ).astype(int)

    fig, ax = plt.subplots(Nrow, Ncol+1, figsize=(4.5*(Ncol+1), 3.75*Nrow) )


    #Plot input
    vals1 = dToT_input

    if Nrow > 1:
        ax_ind = (0,0)
    else:
        ax_ind = 0

    im = ax[ax_ind].imshow(vals1, origin='lower', aspect='auto', 
                    extent=[ tp_vals[0], tp_vals[-1], yvals[0], yvals[-1] ],
                    cmap=cmap, vmin=-1, vmax=1)

    ax[ax_ind].set_ylabel(r'$\log_{10}(R / R_{in})$', fontsize=16)
    ax[ax_ind].set_xlabel(r'$t_p$ [days]', fontsize=16)
    ax[ax_ind].set_title('Input', fontsize=15)

    ax[ax_ind].tick_params('both', labelsize=12)
    ax[ax_ind].tick_params('both', which='major', length=7)
    ax[ax_ind].tick_params('both', which='minor', length=3)


    x1, x2 = ax[ax_ind].get_xlim()
    y1, y2 = ax[ax_ind].get_ylim()

    xtxt = x1 + (x2-x1)*.05 
    ytxt = y1 + (y2-y1)*.1
    ax[ax_ind].text( xtxt, ytxt, 'scale = ' + '{:.4f}'.format( np.percentile( np.abs(dToT_input), 99) ), fontsize=13, color='k')
    
    
    
    #Add xticks
    Nticks = int( (tp_vals[-1] - tp_vals[0])//interval )
    
    if tp_vals[-1] > interval:
        val1 = np.ceil(tp_vals[0]/interval)*interval
        
        ax[ax_ind].set_xticks([val1 + interval*i for i in range(Nticks)])
        
        if tp_vals[-1] - interval/4 > val1 + interval*(Nticks-1):
            ax[ax_ind].set_xticklabels(  [int(val1 + interval*i) for i in range(Nticks)] )
        else:
            ax[ax_ind].set_xticklabels(  np.concatenate([  [int(val1 + interval*i) for i in range(Nticks-1)] , [''] ]) )





    for n, dToT_output_reshape in tqdm( enumerate(dToT_outputs_reshape) ):

        if Nrow > 1:
            ax_ind = (n//Ncol, n%Ncol +1)
        else:
            ax_ind = n+1

        vals1 = dToT_output_reshape / np.percentile( np.abs(dToT_output_reshape), 99)

        im = ax[ax_ind].imshow(vals1, origin='lower', aspect='auto', 
                          extent=[ tp_vals[0], tp_vals[-1], yvals[0], yvals[-1] ],
                          cmap=cmap, vmin=-1, vmax=1)

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

        ax[ax_ind].set_title(r'$\xi$ = ' + xi_txt, fontsize=17)




        vals2 = dToT_input / np.percentile( np.abs(dToT_input), 99)

        x1, x2 = ax[ax_ind].get_xlim()
        y1, y2 = ax[ax_ind].get_ylim()

        xtxt = x1 + (x2-x1)*.05 
        ytxt1 = y1 + (y2-y1)*.1
        ytxt2 = y1 + (y2-y1)*.2

        ax[ax_ind].text( xtxt, ytxt2, r'$\chi^2_\nu$ = {:.3f}'.format(chi2_vals[n]), fontsize=13, color='k' )
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
    for i in range(1, Nrow):
        if Nrow <= 1 :
            ax_ind = 1
        else:
            ax_ind = (i, 1)

        ax[ax_ind].set_ylabel(r'$\log_{10}(R / R_{in})$', fontsize=16)
        ax[ax_ind].tick_params('y', labelsize=12)
        
      
    #Add xlabel on bottom
    for i in range(Ncol):
        if Nrow <= 1:
            ax_ind = i+1
        else:
            ax_ind = (-1, i+1)
        
        ax[ax_ind].set_xlabel(r'$t_p$ [days]', fontsize=16)
        ax[ax_ind].tick_params('x', labelsize=12)




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
            ax[-2, i+1].tick_params('x', labelsize=12)
            ax[-2, i+1].set_xlabel(r'$t_p$ [days]', fontsize=16)


    for i in range(1, Nrow):
        ax[i, 0].axis('off')

    #Reduce spacing between subplots
    plt.subplots_adjust(wspace=.05, hspace=.25)

    #Make colorbar
    cbar = fig.colorbar(im, ax=[ax], location='right', shrink=1.0, pad=.01, aspect=15 + 5*(Nrow-1) )
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
                    plot_err=False, fps=5, yscale='symlog'):

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
    
    if yscale == 'symlog':
        ymin = -10
        ymax = 10
    elif yscale == 'linear':
        ymin = -1
        ymax = 1
    
    for n, spectrum in enumerate(new_spectra):

        ax[n].tick_params('both', labelsize=12)
        ax[n].tick_params('both', which='major', length=8)
        ax[n].tick_params('both', which='minor', length=3)

        ax[n].set_xlabel('$\log_{10} ($Frequency [Hz]$)$', fontsize=15)        

        if n == 0:
            ax[n].set_ylabel(r'$\delta F / F_0$', fontsize=15)

        ax[n].set_ylim(ymin, ymax)
        ax[n].set_yscale(yscale)
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


        if len(filters[a]) > 0:
            filter_name = str(filters[a])[2:-1]
        else:
            filter_name = ''

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