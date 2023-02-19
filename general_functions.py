# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 17:49:38 2022

@author: Mani

"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress

import plot_helpers as ph
import rip_data_processing as rdp
import rip_data_plotting as rdpl
#------------------------------------------------------------------------------

def parse_var_args(params,**kwargs):
    """
    This function updates the keys in the params dictionary based on user supplied key-value pairs. If 
    the key is not already present in the dict, then it is added to the dict.

    Inputs:
        params - dictionary of parameters
        kwargs - user's key-value params
    Output:
        params - updated dictionary
        
    """    
    # We will pool all keys and then we will update those that are present in 
    # the kwargs
    
    # If user does not supply kwargs, then do nothing. The user should be wise-enough
    # not to call the function this way in the first place.
    if len(kwargs)==0:
        return params
    
    # Deepcopy so that params doesn't get altered outside this function
    import copy
    nparams = copy.deepcopy(params)
    
    param_keys = set(nparams.keys())
    user_keys = set(kwargs.keys())
    
    all_keys = param_keys.union(user_keys)
    
    # Go through each key and keep/update value
    for key in all_keys:
        # If this key is in the user supplied keys, pick the user's value
        if key in kwargs.keys():
            nparams.update({key:kwargs[key]})
    return nparams
#------------------------------------------------------------------------------        
        
# Function that collects x and y movement tracking coordinates around event times
def get_perievent_motion(key, event_ts, rt_pre, rt_post, same_len):
    """Fetches the x and y coordinate points of animal movement around the time of
    events:
        Inputs:
            key - database key
            event_ts - numpy array of event times of datatype double
            rt_pre - negative number, a double, seconds before event
            rt_post - positive number, a double, seconds after event
            same_len - Boolean, should all perievent data be of the same length?
        Outputs:
            mt - list of numpy array of Neuralynx time (us); len(mt) = num of trials
            mx - list of numpy array of x-coodinate of LED tracker, same size as mt
            my - list of numpy array of y-coodinate of LED tracker, same size as mt
            good_trials - numpy array of boolean, size(good_trials) = num of trials
    """
            
    # First get the video tracker file path
    # Get required database tables
    import acq
    import os.path
    import scipy.io
    import numpy as np
       
    sourceFolder = (acq.Ephys & key).fetch('ephys_path', as_dict = False)[0]
    _ , ephysFolder = os.path.split(sourceFolder);
    vtfile = os.path.join('D:\\ephys\\processed',ephysFolder,'vt_for_python.mat');
    data = scipy.io.loadmat(vtfile)
    vtd = data['vt']
    # In the vt dict, 0,1,2 lists are t and, x and y respectively
    # See save_neuralynx_nvt_for_python.m for how the .mat file was created.
      
    t = np.array(vtd[0])
    x = np.array(vtd[1])
    y = np.array(vtd[2])
        
    # Convert time to microsec. mt is already in microsec
    cf = 1.0e6 # for conversion from sec to usec
    rt_pre = rt_pre * cf
    rt_post = rt_post * cf
       
    # Go to each event time (event_ts) and pick the x and y values 
    mt = []
    mx = []
    my = []
    for evt in event_ts:
        t1 = evt + rt_pre
        t2 = evt + rt_post
        in_win = (t >= t1) & (t < t2)
       
        # Time is in microsec
        mt.append(np.array(t[in_win]))
        mx.append(np.array(x[in_win]))
        my.append(np.array(y[in_win]))
    
    """Set all trials to have the same length. We will use the median length as the common
    trial length. Due to sampling, some trials will be longer or shorter by 1 or 2 
    data points. The last trial may be shorter by large amount because the 
    recording may have been stopped before the rt_post time. Similarly the first stimulation
    trial may have started too soon after recording started."""
    
    # Sampling created time jitter - because video based position tracking was done at
    # 29Hz, we will allow for one datapoint jitter which is equal to about 34ms
        
    tsize = np.array([len(y) for y in mt])
    md = int(np.median(tsize))
    
    # Trials with more than one datapoint jitter - should be marked as bad
    good_trials = np.abs(tsize-md) <= 1
    if tsize.size < good_trials.sum():
        print(f'{tsize.size-good_trials.sum()} trials were marked as too short')
    
   
    # Fix lengths. If a trial is shorter, We will simply replicate the last data
    # point to fill the missing one at the end. If it is longer, we will cut it off
    # at the end.
     
    for idx, t in enumerate(mt):
        # Fix good trials only
        if good_trials[idx]:
            v = mt[idx] 
            dt = np.median(np.diff(v))           
            if t.size < md:
                mt[idx] = np.append(v,v[-1]+dt)
                mx[idx] = np.append(mx[idx],mx[idx][-1])
                my[idx] = np.append(my[idx],my[idx][-1])
            elif t.size > md:
                mt[idx] = v[0:md]
                mx[idx] = mx[idx][0:md]
                my[idx] = my[idx][0:md]
                
    return mt, mx, my, good_trials
        

def interp_based_event_trig_data_average(t_list, v_list):
    """ Average the data of multiple trials into a single vector. Using linear interpolation, 
    all values will be sampled at the same time point. time vector will be decided by the
    min and max of all time vectors pooled. This code assumes that data is event triggered
    and time is relative to the event. Therefore, you should have time zero in the
    t_list.
    Inputs:
        t_list: list of 1D np.arrays of time
        v_list: list of 1D np.arrays of values (motion for eg)
    Outputs:
        t_vec: 1d numpy array of time vector at which data was sampled
        vi_list: list of 1d numpy array of interpolated data
    """
    all_t = np.concatenate(t_list)
    tmin = np.min(all_t)
    tmax = np.max(all_t)
    
    # Determine median sampling interval for creating time vector
    mdv = np.concatenate([np.diff(tval) for tval in t_list])
    ifi = np.nanmedian(mdv)
    
    # To ensure that we have zero in the time vector:
    left = np.flip(np.arange(0, tmin, -ifi))
    right = np.arange(ifi, tmax, ifi)
    t_vec = np.concatenate([left, right]) 
    
    vi_list = [np.interp(t_vec, ti, vi) for ti, vi in zip(t_list, v_list)]
        
    return t_vec, vi_list
    

def plot_peristim_head_disp(sess_ts, pulse_per_train, xmin, xmax, laser_color, **kwargs):
    """
    Function that plots head displacement in a time window (xmin to xmax) around
    the onset of light pulse stimulation. Data will be plotted as a stack of
    trials.
    MS 2022-02-07
    
    Inputs:
    sess_ts : session timestamp string, eg. "2022-02-14_10-28-34"
    pulse_per_train : int, number of light pulses per stimulus train
    xmin : int or float, a negative number (sec) before pulse train onset
    xmax : int or float, a positive number (sec) after pulse train onset
    laser_color: char, 'g' (for green) or 'b' (for blue)
    
    Returns
    -------
    None.
    
    """
    import matplotlib.pyplot as plt
    import rip_data_processing as rdpr
    import rip_data_plotting as rdpl
    import djutils as dju
    
    # Get key
    sess_key = dju.get_key_from_session_ts(sess_ts)[0]
        
    # Create a new axes if user did not supply one
    if 'axes' not in kwargs:
        _, (hdax, avg_ax) = plt.subplots(2,1) 
    else:
        hdax, avg_ax = kwargs['axes']
        _ = plt.gcf()
    print(kwargs)
    if 'ylim' in kwargs:
        ylim = kwargs['ylim']
    else:
        ylim = [-0.1,4]
       
        
    # Get light pulse train info
    pon, _ , pulse_width, pulse_freq = rdpr.get_light_pulse_train_info(sess_key, pulse_per_train) 
    
    # Get motion data
    same_len = True
    mt, mx, my, good_len = get_perievent_motion(sess_key, pon, xmin, xmax, same_len)
    mt = list(compress(mt, good_len))
    mx = list(compress(mx, good_len))
    my = list(compress(my, good_len))
    pon = list(compress(pon, good_len))
    
    rel_mt_mid_lst = []
    head_disp_lst = []
    for idx in range(len(mt)): 
        head_disp, t_mid = compute_head_disp_info(mt[idx], mx[idx], my[idx])
        rel_mt_mid = (t_mid-pon[idx]) * 1e-6   
         
        rel_mt_mid_lst.append(rel_mt_mid)
        head_disp_lst.append(head_disp)
        
        hdax.plot(rel_mt_mid, head_disp + idx, color='k', linewidth = 0.5)
           
    hdax.set_xlabel('Time (s)')
    hdax.set_ylabel('Head disp (pix/frame)')
    hdax.set_xlim([xmin, xmax])   
    hdax.margins(0.0,0.05)        
        
    # Add light pulses to plot   
    rdpl.plot_light_pulses(pulse_width, pulse_per_train, pulse_freq, laser_color,\
                           hdax, loc='bottom')
    
    # Plot average
    rdpl.plot_average_motion(rel_mt_mid_lst, head_disp_lst, xmin, xmax, avg_ax, median=True)
    avg_ax.set_ylim(ylim)
    ph.boxoff(avg_ax)
    ph.boxoff(hdax)
    # Add title
    tt = "M%d %s" % (sess_key['animal_id'], sess_ts)
    plt.suptitle(tt)
    plt.show()
        
def compute_head_disp_info(mt, mx, my):
    """
    Calculate head displacement from x and y coordinate of motion tracker
    Inputs:
        mt: 1D numpy array of time stamps (us)
        mx: 1D numpy array of x coordinate of LED tracker
        my: 1D numpy array of y coordinate of LED tracker
    Outputs:
        t_mid: 1D numpy array of time stamps (us) - these are mid points of the
               mt input array. Therefore, size(t_mid) = size(mt)-1 
        head_disp: 1D numpy array of head displacement, same length as t_mid

    """   
    xc = np.diff(mx)**2
    yc = np.diff(my)**2
    # Because of diff operation, we lose a data point. Since displacement
    # needs two data points, it is fair to assign that value to the middle 
    # of the two corresponding time points. So, we will get mid points of
    # the time vector. This way, the lengths of time and displacement match.
    delta = np.median(np.diff(mt))/2
    t_mid = mt[:-1] + delta
    head_disp = np.sqrt(xc + yc)
    
    return head_disp, t_mid
    
def plot_mouse_group_motion(group_data, within_mouse_operator, **kwargs):
    """
    Plot motion (head displacement) pooled across mice. By default, individual 
    mouse data and pooled data will be plotted in two subplots. If you don't want
    the population data, set 'pop_data'=False in the optional arguments.
    
    In each mouse first mean or median of head displacement is computed across
    trials. This function calls the pool_head_disp_across_mice function 
    of rip_data_processing module.
    
    Inputs:
        group_data: list (mice) of list(channels) of dict (ripple data), this is an output from
                     collect_mouse_group_rip_data(...) function call.
        within_mouse_operator: str, should be 'mean' or 'median' - tells you if mean
                              or median is computed across trials within a mouse 
        kwargs: 'axes' - list of axis objects onto which to plot data, user given. 
                        Should have two axis objects if both pooled and population
                        motion need to be plotted.               
    Outputs:
        None
        
    MS 2022-03-16
    """
    t_vec, mean_rr, std_rr, all_rr = rdp.pool_head_disp_across_mice(group_data, within_mouse_operator)
    se = std_rr/all_rr.shape(0)
    
    # Prepare axes for plotting
    if 'axes' not in kwargs:
        ax1, ax2 = kwargs['axes']
    else:       
        fig = plt.figure(num=1, figsize=(8,6), dpi=150, frameon=False)
        G = plt.GridSpec(2, 1)         
        ax1 = fig.add_sub_plot(G[0,0])
        ax2 = fig.add_sub_plot(G[1,0])
    rdpl.set_common_subplot_params(plt)
    
    # Plot - averaged across mice
    ax1.plot(t_vec, mean_rr, color='k')
    ax1.fill_between(t_vec, mean_rr-se, mean_rr+se, color=(0.75, 0.75, 0.75))
    # Plot all mouse data
    i = 0
    for mdata in all_rr:
        ax2.plot(t_vec, mdata+i, color='k')
        i +=1
    
    
        
        
    
    
    
    
    
            
    
    