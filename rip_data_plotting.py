# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 15:23:53 2022

@author: Mani

Here are the functions related to plotting ripple suppression data

"""
import copy
import djutils as dju
import util_py as utp
import rip_data_plotting as rplt
import rip_data_processing as rdp
import general_functions as gfun
import plot_helpers as ph
import numpy as np
import warnings
from matplotlib.patches import Rectangle as rect
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import matplotlib
# to make fonts editable in Illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['svg.fonttype'] = 42


linewidth = 0.5


# For separate window, use %matplotlib qt
# For inline plots, use %matplotlib inline
# To save use: plt.savefig('test.pdf',dpi='figure')


def add_sup_title(args, fh):
    # Super-title for the given figure handle fh
    n_sess = len(list(args.sess_str))
    t_str = []
    for i_sess in range(n_sess):
        t_str.append(f"M{args.mouse_id},{args.chan_name}, {args.title}, {args.sess_str[i_sess]} "
                     f"({args.pulse_per_train[i_sess]} x {args.pulse_width[i_sess]} ms pulse) "
                     f"{args.pulse_freq[i_sess]} Hz std = {args.std[i_sess]} "
                     f"minwidth = {args.minwidth[i_sess]} {args.beh_state}")
    # Join strings
    title_str = '\n'.join(t_str)
    fh.suptitle(title_str)


def add_yticklab_for_last_trial(rax, n_trials):
    # Adjust counting of trials to starts with 1 not zero which is the default
    # Inputs:
    #    rax - raster plot axis handle
    yticks = np.array(rax.get_yticks())
    d = np.median(np.diff(yticks))

    yticks = np.append(yticks, n_trials)
    dt = np.diff(yticks[-2::])
    # if the last but one tick label is too close to the ntrials
    # tick label, we will delete it.
    if np.abs(dt) < 0.25*d:
        yticks = np.delete(yticks, -2)
    ytlabels = copy.copy(yticks)
    pos_ticks = yticks >= 0
    rax.set_yticks(yticks[pos_ticks])
    rax.set_yticklabels(ytlabels[pos_ticks].astype(int))


def change_raster_yticklab_0_to_1(rax, ymax):
    # Adjust counting of trials to starts with 1 not zero which is the default
    # Inputs:
    #    rax - raster plot axis handle
    yticks = np.array(rax.get_yticks())
    yticks = yticks[yticks <= ymax]
    ytlabels = copy.copy(yticks)
    pos_ticks = yticks >= 0
    yticks[yticks == 0] = 1
    rax.set_yticks(yticks[pos_ticks])
    ytlabels[ytlabels == 0] = 1
    rax.set_yticklabels(ytlabels[pos_ticks].astype(int))


def plot_head_mov_by_trial(rdata, args, hdax=None, mov_metric='inst_speed'):
    # Plot individual trial head displacement data
    """
    Inputs: 
        rdata, args = rip_data_processing.get_processed_rip_data(...)
        hdax - axis handle on which to plot data
        mov_metric - 'disp' or 'inst_speed'; We will plot instantaneous speed (mm/s) 
                    by default. If 'disp', instantaneous displacement in 
                    pixels will be plotted.
    Outputs:
        None
    """
    if hdax==None:
        plt.figure()
        hdax = plt.gca()
        
    c = 1
    sf = 1/20  # scaling factor to reduce clutter in the plot
    for v in rdata:
        # Get bin centers for rel_mt since head_disp was computed from two
        # adjacent frames
        rt = v['rel_mt']
        bin_cen = rt[0:-1]+np.median(np.diff(rt))/2
        match mov_metric:  # noqa
            case 'inst_speed':
                mov_data = v['inst_speed']
                ylabel = 'Instantaneous head speed (mm/s)'
            case 'disp':
                mov_data = v['disp']
                ylabel = 'Instantaneous head displacement (pix)'
            case _:
                raise ValueError('Undefined mov metric')
        hdax.plot(bin_cen, (mov_data*sf)+c, color='k', linewidth=0.5)
        c += 1
    hdax.set_xlabel('Time (s) relative to photostimulation onset')
    hdax.set_ylabel(ylabel)
    hdax.set_xlim([args.xmin, args.xmax])
    hdax.set_ylim([-1, c+2])
    hdax.margins(0.0, 0.05)
    change_raster_yticklab_0_to_1(hdax, c+2)
    # Add a ytick label for the last trial
    add_yticklab_for_last_trial(hdax, c-1)


def plot_mean_med_motion(t_list, v_list, xmin, xmax, dax, ylim=[]):
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
    t_vec, mi_list = gfun.sample_event_trig_data_evenly(t_list, v_list)
    # First change into 2D array where rows are trials
    mi_array = np.stack(mi_list, axis=0)
    mov_mean = np.nanmean(mi_array, axis=0)
    # d_high = mi_cen + np.std(mi_array,axis=0)
    # d_low = mi_cen - np.std(mi_array,axis=0)
    mov_med = np.nanmedian(mi_array, axis=0)
    d_high = np.nanquantile(mi_array, 0.75, axis=0)
    d_low = np.nanquantile(mi_array, 0.25, axis=0)

    error_col = [0.75, 0.75, 0.75]
    edge_col = [0.75, 0.75, 0.75]

    dax.fill_between(t_vec, d_low, d_high,
                     facecolor=error_col, edgecolor=edge_col)
    t = np.concatenate((t_vec[0::5], [t_vec[-1]]))
    m = np.concatenate((mov_med[0::5], [mov_med[-1]]))
    dax.plot(t, m, color='k', marker='o', markersize=0.5, linestyle='-',
             markerfacecolor='k', linewidth=linewidth, label='Median')
    dax.plot(t_vec, mov_mean, color='k', linestyle='-',
             linewidth=linewidth, label='Mean')
    ph.boxoff(dax)
    # plt.legend()
    dax.set_xlim([xmin, xmax])
    if len(ylim) == 2:
        dax.set_ylim(ylim)
    dax.set_xlabel('Time (s) relative to photostimulation onset')
    dax.set_ylabel('Inst speed (mm/s)')


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
    rip_rate, bins, _ = rdp.get_ripple_rate(
        rdata, args.bin_width, args.xmin, args.xmax)
    hax.hist(bins[:-1], bins,  weights=rip_rate, color='k', rwidth=1)
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
    for idx, rd in enumerate(rdata):
        rip_times = rd['rip_evt']
        rax.plot(rip_times, (np.ones(rip_times.shape)*idx)+1, marker=marker,
                 markersize=markersize, color='k', linestyle='none')

    # x-axis data is set tight.
    rax.margins(0.00, 0.075)
    rax.set_xlim([args.xmin, args.xmax])
    # Leave space for light pulse on the top but keep bottom tight
    ylim_max = rax.get_ylim()[1]
    rax.set_ylim(-2, ylim_max)
    rax.set_xlabel(xlabel)
    rax.set_ylabel(ylabel)
    change_raster_yticklab_0_to_1(rax, ylim_max)
    # Add a ytick label for the last trial
    add_yticklab_for_last_trial(rax, idx+1)


def plot_light_pulses(pulse_width, pulse_per_train, pulse_freq, laser_color,
                      rax, **kwargs):
    """
    plot the light pulses as boxes on the given axes (rax)
    Inputs:
        pulse_width - list, width of light pulse in msec
        pulse_per_train - list, num of light pulses per train
        pulse_freq - list, Hz
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
    pulse_h_frac = 0.05 # fraction of ylimit
     
    if loc == 'top':
        y = rax.get_ylim()[1]*(1 - pulse_h_frac * np.arange(len(pulse_per_train)+1))
    elif loc == 'bottom':
        y = rax.get_ylim()[0]*(0 + pulse_h_frac * np.arange(len(pulse_per_train)+1))
    else:
        raise ValueError('loc param should be either "top" or "bottom"')
    pulse_h = pulse_h_frac * np.max(y) * 0.95
    for i,(pw, ppt, pf) in enumerate(zip(pulse_width, pulse_per_train, pulse_freq)):
        pw = pw * 1e-3  # convert from ms to sec
        if pw < 0.01:
            pw = 0.01
            warnings.warn(
                'Pulse width was too short for plotting so setting it to 10ms')
        if ppt == 1:
            x = 0
            rax.add_patch(
                rect((x, y[i]), pw, pulse_h, edgecolor='none', facecolor=laser_color))
        else:
            ipi = 1/pf  # Interpulse interval
            for j in range(ppt):
                x = j * ipi
                rax.add_patch(
                    rect((x, y[i]), pw, pulse_h, edgecolor='none', facecolor=laser_color))
    rax.set_ylim([rax.get_ylim()[0],np.max(y)+pulse_h])

def plot_ripples_one_session(session_ts, chan_num, pulse_per_train, std, minwidth=30):
    args = rdp.Args()
    args.title = ''
    args.n_std_mov_art = 10
    args.pulse_per_train = 1
    if args.pulse_per_train > 2:
        args.xmin = -10
        args.xmax = 20
    else:
        args.xmin = -4
        args.xmax = 6
    key = dju.get_key_from_session_ts(session_ts)[0]
    key['chan_num'] = chan_num
    rdata, args = rdp.get_processed_rip_data(
        [key], args.pulse_per_train, [std], [minwidth], args)
    args.laser_color = 'b'
    args.sess_str = [session_ts]
    args.std = [std]
    args.chan_num = chan_num
    args.minwidth = [minwidth]
    plot_lightpulse_ripple_modulation(rdata, args)


def plot_lightpulse_ripple_modulation(rdata, args, **kwargs):
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

    fig = plt.figure(num=args.fig_num, figsize=(
        11.15, 6.15), dpi=150, tight_layout=True)

    # All plot beautification params such as tick label size, spine linewidth etc
    set_common_subplot_params(plt)

    # Subplots
    G = plt.GridSpec(3, 2)
    ax = [0]*4
    ax[0] = fig.add_subplot(G[0, 0])  # Ripples as dots
    ax[1] = fig.add_subplot(G[1, 0])  # Ripples as histogram
    ax[2] = fig.add_subplot(G[2, 0])  # head movement - averaged
    ax[3] = fig.add_subplot(G[:, 1])  # trial-by-trial head movement

    for aax in ax:
        ph.boxoff(aax)

    # Raster plot of ripples
    plot_ripples_as_dots(rdata, args, ax[0])
    
    if type(args.pulse_per_train)==list:
        # Find unique pulse params combination. If pulse_per_train is a list,
        # then pulse_width and pulse_freq will also be lists of the same size
        # as that of pulse_per_train.
        p_array = np.vstack((args.pulse_per_train, args.pulse_width,
                             args.pulse_freq))
        uarray = np.unique(p_array,axis=1)
        pulse_per_train = uarray[0,:].astype(int)
        pulse_width = uarray[1,:]
        pulse_freq = uarray[2,:]
    else:
        pulse_per_train = args.pulse_per_train
        pulse_width = args.pulse_width
        pulse_freq = args.pulse_freq
    
    #     if np.unique(pulse_per_train).size == 1:
    #         pulse_per_train = args.pulse_per_train[0]
    #     else:
    #         raise ValueError(f'Pulse per train has different values: {args.pulse_per_train}')        
        
    # if type(args.pulse_width)==list:
    #     # Pick one
    #     if np.unique(args.pulse_width).size == 1:
    #         pulse_width = args.pulse_width[0]
    #     else:
    #         raise ValueError(f'Pulse width has different values: {args.pulse_width}')
    
    # if type(args.pulse_freq)==list:
    #     # Pick one
    #     if np.unique(args.pulse_freq).size == 1:
    #         pulse_freq = args.pulse_freq[0]
    #     else:
    #         raise ValueError(f'Pulse width has different values: {args.pulse_width}')
    
    
    plot_light_pulses(pulse_width, pulse_per_train,
                      pulse_freq, args.laser_color, ax[0])

    # Histogram of ripples
    plot_ripples_hist(rdata, args, ax[1])

    # Plot average motion
    t_list = [v['rel_mt'] for v in rdata]
    # Since we computed head displacement using movement info from two adjacent
    # video frames, we will create time bin centers
    t_list = [x[0:-1]+np.median(np.diff(x))/2 for x in t_list]
    v_list = [v['inst_speed'] for v in rdata]
    # plot_average_motion(t_list, v_list, args.xmin, args.xmax, ax[2])
    plot_mean_med_motion(t_list, v_list, args.xmin, args.xmax, ax[2], ylim=[])

    # Plot head dispacement trial by trial
    plot_head_mov_by_trial(rdata, args, ax[3])
    plot_light_pulses(pulse_width, pulse_per_train,
                      pulse_freq, args.laser_color, ax[3])

    # Title
    add_sup_title(args, fig)

    plt.show()


def plot_all_chan_mouse_rip_rate_matrix(grdata, wh):
    """ Plot trial-averaged ripple rate of each channel of each mouse
    Inputs: grdata = rip_data_processing.collect_mouse_group_rip_data(dd,args)
            wh = [w,h] width and height in inches of the axis plot box (excluding ticks and labels)
    """
    rr, bin_cen, n = rdp.get_all_chan_mouse_rip_rate_matrix(grdata)
    nChan = np.sum(n)
    rplt.set_common_subplot_params(plt)

    fig, ax = utp.make_axes(plt, wh)
    bw = grdata[0][0]['args'].bin_width/1000
    extents = [bin_cen[0]-bw/2, bin_cen[-1]+bw/2, nChan, 0]
    im = ax.imshow(rr, cmap='coolwarm', norm=TwoSlopeNorm(1),
                   extent=extents, origin='upper',interpolation='none')
    # Mark beginning and end of channel group of each mouse
    yticks = np.cumsum(np.hstack((0, n)))
    for yt in yticks[1:-2]:
        ax.plot(ax.get_xlim(), [yt, yt],
                linestyle='--', color='k', linewidth=0.5)
    ax.set_yticks(yticks)
    ax.set_xlabel('Time (s) relative to photostimulation onset')
    ax.set_ylabel('Mouse #')
    fig.colorbar(im, location='top', orientation='horizontal',
                 aspect=15, shrink=0.25)


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
    line_width = 0.5
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
