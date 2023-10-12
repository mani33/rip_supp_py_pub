# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 15:23:53 2022

@author: Mani

Here are the functions related to plotting ripple suppression data

"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42 # to make fonts editable in Illustrator
#matplotlib.rcParams['svg.fonttype'] = 42

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle as rect
import warnings
import numpy as np
import plot_helpers as ph
import general_functions as gfun
import rip_data_processing as rdp
import rip_data_plotting as rplt
import util_py as utp


# import scipy.stats as stats 

# For separate window, use %matplotlib qt
# For inline plots, use %matplotlib inline
# To save use: plt.savefig('test.pdf',dpi='figure')
     
def plot_all_chan_mouse_rip_rate_matrix(grdata,wh):
    """ Plot trial-averaged ripple rate of each channel of each mouse
    Inputs: grdata = rip_data_processing.collect_mouse_group_rip_data(dd,args)
            wh = [w,h] width and height in inches of the axis plot box (excluding ticks and labels)
    """
    rr,bin_cen,n = rdp.get_all_chan_mouse_rip_rate_matrix(grdata)
    nChan = np.sum(n)   
    rplt.set_common_subplot_params(plt)
    
    #%%
    fig,ax = utp.make_axes(plt, wh)
    bw = grdata[0][0]['args'].bin_width/1000
    extents = [bin_cen[0]-bw/2,bin_cen[-1]+bw/2,nChan,0]
    im = ax.imshow(rr,cmap='coolwarm',norm=TwoSlopeNorm(1),extent=extents,origin='upper')
    # Mark beginning and end of channel group of each mouse
    yticks = np.cumsum(np.hstack((0, n)))
    for yt in yticks[1:-2]:
        ax.plot(ax.get_xlim(),[yt,yt],linestyle='--',color='k',linewidth=0.5)
    ax.set_yticks(yticks)
    ax.set_xlabel('Time (s) relative to photostimulation onset')
    ax.set_ylabel('Mouse #')
    fig.colorbar(im,location='top',orientation='horizontal',aspect=15,shrink=0.25) 
    
def plot_lightpulse_ripple_modulation (rdata, args, **kwargs):
    """ Plot ripple data when a pulse or train of pulses of light was given
    Inputs: 
        rdata, args = rip_data_processing.get_processed_rip_data(...)
    Outputs:
        None
    """
    for k, v in kwargs.items():
        setattr(args, k, v)
        
    if 'fig_num' not in kwargs:
        args.fig_num = None
    
    fig = plt.figure(num =args.fig_num, figsize=(11.15,6.15), dpi = 150, tight_layout = True)
       
    # All plot beautification params such as tick label size, spine linewidth etc
    set_common_subplot_params(plt) 
    
    # Subplots
    G = plt.GridSpec(3, 2)
    ax = [0]*4
    ax[0] = fig.add_subplot(G[0,0]) # Ripples as dots
    ax[1] = fig.add_subplot(G[1,0]) # Ripples as histogram
    ax[2] = fig.add_subplot(G[2,0]) # head movement - averaged
    ax[3] = fig.add_subplot(G[:,1]) # trial-by-trial head movement
    
    for aax in ax:
        ph.boxoff(aax)
     
    # Raster plot of ripples 
    plot_ripples_as_dots(rdata, args, ax[0])
    plot_light_pulses(args.pulse_width, args.pulse_per_train, args.pulse_freq, args.laser_color, ax[0])
    
    # Histogram of ripples
    plot_ripples_hist(rdata, args, ax[1])
    
    # Plot average motion
    t_list = [v['rel_mt'] for _ , v in rdata.items()]
    v_list = [v['head_disp'] for _ , v in rdata.items()]
    plot_average_motion(t_list, v_list, args.xmin, args.xmax, ax[2])    

    
    # Plot head dispacement trial by trial
    plot_head_disp_by_trial(rdata, args, ax[3])
    plot_light_pulses(args.pulse_width, args.pulse_per_train, args.pulse_freq, args.laser_color, ax[3])
    
    #Title
    add_sup_title(args, fig)
    
    plt.show()
    
def add_sup_title(args, fh):
    # Super-title for the given figure handle fh
    n_sess = len(args.sess_str)
    t_str = []
    for i_sess in range(n_sess):
        t_str.append(f"M{args.mouse_id},{args.chan_name}, {args.title}, {args.sess_str[i_sess]} " 
                     f"({args.pulse_per_train} x {args.pulse_width} ms pulse) "  
                f"{args.pulse_freq} Hz std = {args.std[i_sess]} "
                f"minwidth = {args.minwidth[i_sess]} {args.beh_state}")
    # Join strings
    title_str = '\n'.join(t_str)
    fh.suptitle(title_str)    
    
def plot_head_disp_by_trial(rdata, args, hdax):
    # Plot individual trial head displacement data    
    """
    Inputs: 
        rdata, args = rip_data_processing.get_processed_rip_data(...)
        hdax - axis handle on which to plot data
    Outputs:
        None
    """            
    c = 0
    for _ , v in rdata.items(): 
        hdax.plot(v['rel_mt'], v['head_disp'] + c, color='k', linewidth = 0.5)
        c += 1
    hdax.set_xlabel('Time (s)')
    hdax.set_ylabel('Head disp (pix/frame)')
    hdax.set_xlim([args.xmin, args.xmax])
    hdax.margins(0.0,0.05)
    
def plot_average_motion(t_list, v_list, xmin, xmax, dax, **kwargs):
    """
    Averaged head displacement plot
    Use interpolation and sample head displacement at the same time points for 
    all photostimulation trials (interpolation is needed because video camera sampled
    frames at slightly different times)  
    
    Inputs:
        t_list - list of numpy array of times (sec) relative to photostim. 
                 len(t_list) = num trials
        v_list - list of numpy array of head displacement. Same len as t_list
        xmin, xmax - int or float, time window boundary (sec) for all trials. 
                     e.g. xmin = -10, xmax = 15
        dax - axes handle in which to plot the data
    Outputs:
        None
    """
    t_vec, mi_list = gfun.interp_based_event_trig_data_average(t_list, v_list)
    
    # Average: first change into 2D array where rows are trials
    mi_array = np.stack(mi_list, axis=0)
    mi_avg = np.mean(mi_array, axis=0)
    
    if 'median' in kwargs.keys() and kwargs['median']:
        mi_mdn = np.median(mi_array, axis=0)
        dax.plot(t_vec, mi_mdn, color='b', linewidth = 0.5, label='median')
    
    dax.plot(t_vec, mi_avg, color='k', linewidth = 0.5, label='mean')
    dax.set_xlim([xmin, xmax])
    dax.set_ylim([0, 4])
    dax.set_xlabel('Time (s)')
    dax.set_ylabel('Head disp (pix/frame)')
    dax.legend(loc='upper right')
    
def plot_ripples_hist(rdata, args, hax):
    """
    Histogram of ripples
    Inputs:
        rdata, args = rip_data_processing.get_processed_rip_data(...)
        hax - axes on which to plot data
    Outputs:
        None
    """
    # Pool ripple event times
    rip_rate, bins, _ = rdp.get_ripple_rate(rdata, args.bin_width, args.xmin, args.xmax)
    hax.hist(bins[:-1], bins,  weights = rip_rate, color = 'k', rwidth = 1)
    # x-axis data is set tight.
    # hax.margins(0.0,0.075)
    hax.set_xlim([args.xmin, args.xmax])
    hax.set_xlabel('Time (s) relative to photostimulation onset')
    hax.set_ylabel('Ripples/s')
    
def plot_ripples_as_dots(rdata, args, rax, markersize=1,
                         xlabel='Time (s) relative to photostimulation onset',
                         ylabel='Photostimulation trial',
                         marker='.'):
    """
    Raster plot of ripples
    Inputs:
        rdata, args = rip_data_processing.get_processed_rip_data(...)
        rax - axes on which to plot data
    Outputs:
        None
    """
    for idx in rdata:
        rip_times = rdata[idx]['rip_evt']
        rax.plot(rip_times, np.ones(rip_times.shape)*idx,\
                 marker = marker, markersize=markersize, color = 'k', linestyle = 'none')
    
    # x-axis data is set tight.
    rax.margins(0.00,0.075)
    rax.set_xlim([args.xmin, args.xmax])
    # Leave space for light pulse on the top but keep bottom tight
    rax.set_ylim(-2, rax.get_ylim()[1]) 
    rax.set_xlabel(xlabel)
    rax.set_ylabel(ylabel)

def plot_light_pulses(pulse_width, pulse_per_train, pulse_freq, laser_color, rax, **kwargs):
    """
    plot the light pulses as boxes on the given axes (rax)
    Inputs:
        pulse_width - int or float, width of light pulse in msec
        pulse_per_train - int or float, num of light pulses per train
        pulse_freq - int or float, Hz
        laser_color - char, 'g' for green, 'b' for blue
        rax - axes for plotting data
    Outputs:
        None
    """
    # Where to plot - top or bottom
    if 'loc' not in kwargs.keys():
        loc = 'top'
    else:
        loc = kwargs['loc']
            
    if loc=='top':
        y = rax.get_ylim()[1]*0.95
    elif loc=='bottom':
        y = rax.get_ylim()[0]*(-0.01)
    else:
        raise ValueError('loc param should be either "top" or "bottom"')
        
    pw = pulse_width * 1e-3 # convert from ms to sec
    if pw < 0.01:
        pw = 0.01
        warnings.warn('Pulse width was too short for plotting so setting it to 10ms')
    
    if pulse_per_train==1:
        x = 0
        rax.add_patch(rect((x,y), pw, 5, edgecolor = 'none', facecolor = laser_color))
    else:
        ipi = 1/pulse_freq # Interpulse interval
        for i in range(pulse_per_train):
            x = i * ipi           
            rax.add_patch(rect((x,y), pw, 5, edgecolor = 'none', facecolor = laser_color))           
  
def set_common_subplot_params(plt):
    """
    # Set parameters that will be common for all subplots
    Inputs: 
        plt - matplotlib.pyplot object
    Ouputs:
        None
    """
    fontlabel_size = 9
    tick_len = 3
    line_width= 0.5
    fontname = 'Arial'
    params = {
        'font.family': 'sans-serif',
        'font.sans-serif': [fontname],
        'axes.titlesize': fontlabel_size,
        'axes.linewidth': line_width,
        'axes.labelsize': fontlabel_size,
        'xtick.labelsize': fontlabel_size,
        'ytick.labelsize': fontlabel_size,
        'xtick.major.size': tick_len,
        'xtick.major.width': line_width,
        'ytick.major.size': tick_len,
        'ytick.major.width': line_width,
        'text.usetex': False
        }
    plt.rcParams.update(params)
 
   