# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 17:34:39 2022
@author: Mani
This module gets trial by trial ripple time points for the light stimulation-mediated
ripple suppression.
"""

# Get database tables
import itertools
from itertools import compress
import logging
import numpy as np
import acq as acq
import cont as cont
import ripples as ripples
import general_functions as gfun
import djutils as dju
import util_py as utp
import copy
import scipy.signal as sig
from scipy import interpolate
import matplotlib.pyplot as plt

# Class for default values of parameters
class Args():
    def __init__(self):
        # Data length pre and post light pulse
        self.xmin = -5.0 # Add sign appropriately. If you want 1sec before light pulse, use -1
        self.xmax = 10.0
        self.bin_width = 200 # bin width in msec
        self.beh_state = 'nrem' #  beh state to collect ripples from: 'all','rem','nrem','awake'       
        self.motion_quantile_to_keep = 1.0; # below this quantile value, we will retain stimulation trials, list if >1 session
        
        # in sec: time window to detect motion. Can be 4 element [-5 0 2 3] where [-5 0] and [2 3] are two separate windows.
        self.motion_det_win = [self.xmin, self.xmax]
        
        # Remove trials without ripples?
        self.remove_norip_trials = True
        self.correct_high_amp_mov_art = True
        # Maximum cumulative duration of artifacts after which trial will be 
        # excluded
        self.max_art_duration = 1 
        # Height-to-width ratio threshold for calling a detected peak as 
        # artifact in the head displacement values
        self.art_peak_hw_ratio_th = 10
        self.pre_and_stim_only = True

# main function that calls other functions
def average_rip_rate_across_mice(group_data, elec_sel_meth, **kwargs):
    """
    Normalize ripple rate of each mouse by its own baseline. Then average the
    normalized ripple rate across mice. When multiple electrodes had ripples, 
    combine their data based on given selection method.
    Inputs:
        group_data : list (mice) of list(channels) of dict (ripple data), 
                    this is an output from collect_mouse_group_rip_data(...) 
                    function call.       
        elec_sel_meth: str, should be one of 'avg','random','max_effect', 
                    'max_baseline_rate'. When more than one electrode had 
                    ripples, tells you which electrode to pick.
        kwargs:
            'light_effect_win': 2-element list of bounds (in sec) of the 
            light mediated effect.

    Outputs: 
        bin_cen - 1D numpy array of bin-center time in sec.
        mean_rr - 1D numpy array, mean ripple rate(Hz)
        std_rr - 1D numpy array, standard deviation of each time bin
        all_mouse_rr - 2D numpy array, nMice-by-nTimeBins of ripple rates
    
    MS 2022-03-02        
    """
    all_rr_norm = []
    for md in group_data: # loop over mice       
        # For each mouse pick a channel or average across channels
        n_chan = len(md)
        if elec_sel_meth == 'random':
            chd = md[np.random.randint(n_chan)]
            args = chd['args']
            mouse_rr,_,bin_cen = get_ripple_rate(chd['rdata'], 
                                                   args.bin_width, 
                                                   args.xmin, args.xmax)               
        else: # Other methods that require computing ripple rate
            norm_rip_rate = []
            for chd in md: # loop over channels of each mouse
                args = chd['args']
                rd,_,bin_cen = get_ripple_rate(chd['rdata'],args.bin_width, 
                                                   args.xmin, args.xmax)
                # Normalize the ripple rate for each channel by the mean ripple
                # rate during baseline period (i.e., period until stim onset time)
                # This method will give equal weight to all channels
                norm_rd = rd/np.mean(rd[bin_cen < 0])
                norm_rip_rate.append(norm_rd)
            # Apply selection on the ripple rate
            norm_rip_rate = np.array(norm_rip_rate)
            match elec_sel_meth:
                case 'avg':
                    norm_mouse_rr = np.mean(norm_rip_rate, axis=0)
                case _:
                    raise ValueError("The provided method %s is not implemented\n" % elec_sel_meth)                
        all_rr_norm.append(norm_mouse_rr)
    
    # # Normalize ripple rate to baseline before averaging across mice
    # all_rr_norm = [rr/np.mean(rr[bin_cen < 0]) for rr in all_rr]
    
    # Average across mice
    all_mouse_rr = np.array(all_rr_norm)
    mean_rr = np.mean(all_rr_norm, axis=0)
    std_rr = np.std(all_rr_norm, axis=0)
    
    return bin_cen, mean_rr, std_rr, all_mouse_rr

def calc_displacement(x,y):
    # Calculate instantaneous displacement based on x and y coordinates    
    return np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    
def center_and_scale_time(mt, evt):
    """
    Center the time vector to event time and convert units from microsec to sec
    Inputs: 
        mt - list of numpy arrays of time in microsec
        evt - numpy array of event times in microsec corresponding to the list
                    of times in mt
    Outputs: 
        rel_mt - list of numpy arrays of time in sec relative to event times
    """
    rel_mt = [(t-et)*1e-6 for t,et in zip(mt, evt)]   
    return rel_mt

def collect_mouse_group_rip_data(data_sessions,args):
    """
    For a given list of sessions, collect ripple data.
    Inputs:
        data_sessions : Pandas data frame containing the following columns:
                        animal_id, session_ts, pulse_per_train, pulse_width, chan, std, minwidth, 
                        laser_color, laser_knob       
        args -  Args object with updated params-values
       
    Outputs:
        group_data : list (mice) of list(channels) of dict (ripple data)
    """
    """
    To combine repeated sessions from mouse subjects, we will group their row
    indices and iterate through this index groups
    """
    ids = np.array(data_sessions.animal_id)
    uids = np.unique(ids)
    group_data = [[] for _ in range(uids.size)]
    for iMouse,m_id in enumerate(uids):
        # Get data slice corresponding to the current mouse
        dd = data_sessions[ids==m_id]        
        sess_keys = [dju.get_key_from_session_ts(sess_ts)[0] for sess_ts in list(dd.session_ts)]        
        # It is assumed that all repeated sessions of any mouse have the same 
        # set of experimental parameters excluding recording channels. 
        # So parameters come from the first session. But in some rare circumstances
        # like the wild type mice, we need to pool two different conditions such
        # as 1 and 2 pulses per train and 3 and 5 ms pulse widths.
        dd_one = dd.iloc[0,:]        
        # When the same experimental condition (say pulse width, std, minwidth,
        # laser color etc) was repeated several days apart, often recording 
        # channels were different though some channels overlapped. So, we will 
        # pool channels across the repeated sessions in which they were recorded.
        unique_chans = np.unique(list(itertools.chain.from_iterable([str(x).split(',')  for x in dd.chan])))
                
        # Go through each unique channel
        for ch in unique_chans:
            ch = int(ch)
            # Find which sessions had this channel
            sel_keys,std,minwidth = [],[],[]
            # Some channels may not be in all sessions
            for iKey,sess_key in enumerate(sess_keys):
                chans_of_ikey = [int(x) for x in str(dd.iloc[iKey,:]['chan']).split(',')]
                if ch in chans_of_ikey:
                    sess_key['chan_num'] = ch
                    sel_keys.append(sess_key)
                    std.append(dd.iloc[iKey,:]['std'])
                    minwidth.append(dd.iloc[iKey,:]['minwidth'])            
            rdata, args = get_processed_rip_data(sel_keys,list(dd['pulse_per_train']),
                                std, minwidth,args)
            ch_dic = {'animal_id': sess_keys[0]['animal_id'], 
                      'chan_num': ch,'rdata': rdata, 'args': args}
            group_data[iMouse].append(ch_dic)
        print(f'Done with mouse {iMouse}')     
    return group_data

def collapse_rip_events_across_chan_for_each_trial(group_data,elec_sel_meth):
    """
    Within a given recording session, for each photostim trial, 
    we will average binned ripple counts across all available channels. Then, we
    will aggregate the averaged trial data from all available recording sessions 
    from a given mouse. We need to do this trial wise data for
    statistical tests on individual mice.
    
    Inputs:
        group_data: list (mice) of list(channels) of dict ('animal_id','chan_num'
                    rdata(list of dict),args). 
                        This is an output from collect_mouse_group_rip_data(...) 
                        function call. 
                        rdata is a list of dict:
                        len(rdata) = num of trials. Each dict contains the following keys:
                        'rel_mt' - numpy array of time (sec) relative to stim train onset
                        'mx','my' - numpy array of tracker LED x and y coodinate respectively
                        'head_disp' - numpy array of head displacement per video frame (pix/frame)
                        'rip_evt'- numpy array of ripple event times (sec) relative to stim train onset
                        'session_start_time' - scalar; session start time of the trial
        elec_sel_meth: str, should be one of 'avg' or'random'
    Outputs:
        cgroup_data: list(of length nMice) of dict('animal_id','args','trial_data') where
                    trial_data is a list (of length nTrials) of 
                    dict('rel_mt','mx','my','head_disp','bin_cen_t','rip_cnt')                 
    """ 
    cgroup_data = []
    args = group_data[0][0]['args']
    bw = args.bin_width/1000
    bin_edges,bin_cen = utp.create_psth_bins(args.xmin,args.xmax,bw) 
    # Loop through each mouse
    for iMouse,md in enumerate(group_data):
        # To find out how many recording sessions were there for this mouse:
        sess_starts = [md[i]['rdata'][0]['session_start_time'] for i in range(len(md))]
        nMouseChan = len(sess_starts)
        unique_starts = np.unique(sess_starts)        
        # Go through each unique recording session and average each trial data
        # across the channels of that session
        all_trial_data = []
        for sess_start in unique_starts:
            # Check each channel data and if that channel belongs to the current
            # recording session, retain it for later averaging
            sess_chan_data = []
            for iChan in range(nMouseChan):
                # Analyze the trials' session start time to check if any belong
                # to the current session
                trial_sess_starts = np.array([x['session_start_time'] for x in md[iChan]['rdata']])
                sel_ind = np.nonzero(trial_sess_starts==sess_start)[0]
                # Pick trials that belonged to the current session
                if sel_ind.size !=0:
                    sess_chan_data.append(copy.deepcopy([md[iChan]['rdata'][i] for i in sel_ind]))
            # Average each trial data across all the channels of the given session
            # All channels will have the same number of trials
            nTrials = len(sess_chan_data[0])
            sess_n_chan = len(sess_chan_data)
            for iTrial in range(nTrials):
                # Motion data are same for all channels. So pick those from
                # the first channel
                cdata = sess_chan_data[0][iTrial]
                hdata = np.zeros((sess_n_chan,bin_cen.size))                
                match elec_sel_meth:
                    case 'avg':
                        # Average current trial data from all channels of the current session
                       for iSessChan in range(sess_n_chan):
                           evt_t = sess_chan_data[iSessChan][iTrial]['rip_evt']
                           # Bin the event times
                           hdata[iSessChan,:],_ = np.histogram(evt_t,bin_edges)                           
                       # Collapse (average) across channels                       
                       rip_cnt = np.mean(hdata,axis=0)
                    case _:
                        raise ValueError('electrode selection method must be "avg"')
                # Recreate cgroup_data data structure so it looks like group_data
                # except that there will be only one collapsed channel now.
                # Build trial data structure as a dict
                one_trial_data = {'rel_mt': cdata['rel_mt'],'mx':cdata['mx'],
                                  'my':cdata['my'],'head_disp':cdata['head_disp'],
                                  'rip_cnt':rip_cnt,
                                  'session_start_time':cdata['session_start_time']}
                # This list will accumulate trials of all sessions of the
                # current mouse
                all_trial_data.append(one_trial_data)
            
        one_mouse_data = {'animal_id':md[0]['animal_id'],'args':md[0]['args'],
                          'bin_cen_t':bin_cen,'trial_data':all_trial_data}
        cgroup_data.append(one_mouse_data)
        
    return cgroup_data
        
def convert_motion_traj_to_inst_disp(px,py):    
    # Convert mouse motion in pixels to millimeter using video calibration
    #     Inputs:   #         
    #         px - 1d numpy array (size n) of x-coordinates of the LED on mouse head
    #         py - 1d numpy array (size n) of y-coordinates of the LED on mouse head
    #    Outputs:
    #         d - 1d numpy array (size n-1) of instantaneous displacement (mm)         
    #         fps - frames per second
    #     Mani Subramaniyan 2023-11-10
    #     
    # The constants used here were taken from the file:
    # r'C:\Users\maniv\Documents\ripples_manuscript\data\video_size_calibration.pkl'
    # To avoid loading everytime, I have hard-coded these numbers here. If you
    # change the video calibration info, these numbers should be updated
    # accordingly.
    fps = 29.97    
    mm_per_pix_x = 0.47047365470852015
    b = np.array([ 3.93429543e+02, -7.32063842e-02])
    # dt = np.median(np.diff(t)) # should be close to 1/29.97
    cage_width_mm = 180
    dx = np.diff(px)  
    dy = np.diff(py)
    # Get the mid point of consecutive px values to get the 
    # y scaling factor corresponding to x values
    x = (px[0:-1] + px[1:])/2
    # Scale dx and dy according to video calibration
    dx_mm = dx*mm_per_pix_x
    # Because the pixel to mm conversion factor changes as a function of
    # x (due to video camera angle), we need to compute this factor for
    # all x in our trial.
    mm_per_pix_y = cage_width_mm/(b[0]+b[1]*x)
    dy_mm = dy*mm_per_pix_y
    d = np.sqrt((dx_mm**2)+(dy_mm**2))
    
    return d,fps


def correct_abnorm_high_mov_artifacts(rdata,art_peak_hw_ratio_th=10):
    """ Remove data points with abnormally high amplitude of head displacement
        and replace them with interpolated datapoints.
    Inputs:
        rdata - list of dict, output of get_processed_rip_data(...) function call
        art_peak_hw_ratio_th - artifacts' height-to-width ratio threshold above
                which peaks will be interpolated with surrounding values.
    Output:
        tot_art_dur - 1d numpy array (nTrials,) of total duration of artifacts
                        within each trial
        
        rdata - same as the input but with artifact-corrected motion data and 
                with a new dict item art_idx, which is a list (len = nTrials) 
                of 1d numpy array of indices of artifact locations in the 
                displacement values within each trial 
        
    """
    debug=False
    # The idea: we will detect unusually big artifacts in the head displacement
    # but won't correct it first. Instead, we will go and fix the artifacts in 
    # the x and y coordinates and then recompute head displacement based on the
    # new values.
    
    dd = [d['head_disp'] for d in rdata]
    dd = np.concatenate(dd)
    s = np.nanstd(dd)
        
    trials_with_art = []
    fps = 29.97
    tot_art_dur = []
    
    for iTrial,rd in enumerate(rdata):
        d = rd['head_disp']
        mx = rd['mx']
        my = rd['my']
        
        # head_disp was calculated with two adjacent points leaving the length 
        # of y one less than that of t. So we will exclude the last time point.
        # We can do this because the exact value of t is unimportant        
        t = rd['rel_mt'][:-1]
        tf = rd['rel_mt'] # full time for x and y coordinates
        # Replace outliers with interpolated values        
       
        pdata = sig.find_peaks(d,height=s,width=0.5,rel_height=0.9)
        hw_ratio = pdata[1]['peak_heights']/pdata[1]['widths'] 
        
        art_dur = 0        
        outliers = np.zeros(t.size,dtype=bool)
        if hw_ratio.size>0:            
            real_bad = hw_ratio > art_peak_hw_ratio_th
            if np.any(real_bad):
                if debug:
                    print(f'Height_width ratios: {hw_ratio}')
                    print(f'max hw ratio: {np.round(np.max(hw_ratio),2)}')
                    plt.clf()
                    plt.subplot(1,2,1)
                    plt.plot(t,d)
                    plt.subplot(1,2,2)
                    plt.stem(hw_ratio)
                    plt.tight_layout()
                    hw_th = art_peak_hw_ratio_th
                    plt.plot(plt.xlim(),[hw_th,hw_th],color='gray')
                    plt.draw()                
                art_dur = (pdata[1]['widths'][real_bad])*(1/fps)
                left_ips = pdata[1]['left_ips'][real_bad].astype(int)
                right_ips = pdata[1]['right_ips'][real_bad].astype(int)
                for w1,w2 in zip(left_ips,right_ips):
                    outliers[w1:w2+1]=True
        
        tot_art_dur.append(np.sum(art_dur))
        
        # Check if trial needs to be removed due to too much artifact
        art = np.zeros(t.size,dtype=bool)
        if np.any(outliers):
            trials_with_art.append(iTrial)
            art = outliers | np.isnan(d)
            good_part_ind = np.nonzero(~art)[0]
            # Shift indices by 1 to account for differencing in computing 
            # displacement
            gmx = mx[good_part_ind+1]
            gmy = my[good_part_ind+1]
            gtf = tf[good_part_ind+1]            
            outlier_ind = np.nonzero(outliers)[0]+1
            # Fix artifact segments by interpolation
            gx_out = np.interp(tf[outlier_ind],gtf,gmx)
            gy_out = np.interp(tf[outlier_ind],gtf,gmy)            
            # The following line is equivalent to rdata[iTrial]['mx'][outliers] = gx_out
            # because in the line mx = rd['mx'], mx copies the reference to
            # data. So when mx gets updated, the original rdata gets updated. 
            
            mx[outlier_ind] = gx_out
            my[outlier_ind] = gy_out
            
            # Recalculate displacement based on artifact-corrected x and y 
            # coorindates
            d_fixed = calc_displacement(mx, my)
            rd['head_disp'] = d_fixed
        rdata[iTrial]['art_idx'] = art
        
    if len(trials_with_art)>0:
        print('Trials (Python indices) for which artifacts corrected: ',trials_with_art)
   
    return np.array(tot_art_dur),rdata


def filter_trials_by_beh_state(tr_on, tr_off, beh_state, key):
    """
    Filter photostim trials based on their presence within a given behavioral state
    Inputs:
        tr_on and tr_off  - numpy arrays of photostim train on and off times
        beh_state - string, should be 'awake','nrem','rem' or 'all'
        key - database key, a dict
    Outputs:
        good_trials_beh - numpy array of boolean, with True indicating if a given 
                          photostim trial was within the given behavioral state.
    """
    good_trials_beh = np.full((tr_on.size,),False)
    
    #First get the behavior state segment on and off times
    if beh_state == 'awake':        
        st1, st2 = (cont.AwakeSegManual & key).fetch('seg_begin','seg_end')
        st1 = st1[0][0]
        st2 = st2[0][0]
        ephys_start, ephys_end = (acq.Ephys & key).fetch('ephys_start_time','ephys_stop_time')
       
    elif beh_state == 'nrem':
        t1, t2 = (cont.SleepAuto & key).fetch('nrem_seg_begin','nrem_seg_end')
        #t1, t2 = (cont.NremSegManual & key).fetch('seg_begin','seg_end')
        t1 = t1[0][0]
        t2 = t2[0][0]
        
    elif beh_state == 'rem':
        t1, t2 = (cont.SleepAuto & key).fetch('rem_seg_begin','rem_seg_end')
        # t1, t2 = (cont.RemSegManual & key).fetch('seg_begin','seg_end')
        t1 = t1[0][0]
        t2 = t2[0][0]
        
    elif beh_state == 'all':
        good_trials_beh = np.full((tr_on.size,), True)
        return good_trials_beh
    else:
        raise ValueError(f'the behavior state {beh_state} is unaccepted:' 
                         f'should be one of "awake","rem","nrem" or "all"')
    
    # Mark as True if a given photostim train is within any of the behavioral state segments
    for i_train, cton in enumerate(tr_on):
        ctoff = tr_off[i_train]
        # Take on and off times of each train. If train is inside any one of the NREM
        # segment, it's a good trial.
        inseg = np.full((t1.size,), False)
        for idx, st1 in enumerate(t1):
            st2 = t2[idx]
            if (cton >= st1) & (ctoff < st2):
                inseg[idx] = True
        if any(inseg):
            good_trials_beh[i_train] = True
    
    return good_trials_beh
        
def filter_trials_by_head_disp(mt, mx, my, motion_det_win, motion_quantile_to_keep): 
    """       
    Determine head displacement level in the given window and remove trials above
    threshold obtained based user specified quantile value of head dispalcement
    Inputs:
        mt, mx, and my are outputs of get_perievent_motion(...)
        motion_det_win - list of 2 or 4 relative times (sec) marking boundaries of
                        one or two time windows respecively. e.g: [-5, 10] or [-5,0,2,10] 
        motion_quantile_to_keep - quantile value below which trials will be kept. e.g: 0.9
    Outputs:
        good_trials -  numpy array of boolean, with True indicating trials satisfying 
                       head displacement limit condition.
    """      
    # For the selected time windows, compute averaged head displacement.
    r = np.zeros((len(mx),))
    print('Filtering trials by head displacement ...')
    for idx, x in enumerate(mx):
        tt = mt[idx]
        # Selected time period for computing motion
        sel_t = (tt >= motion_det_win[0]) & (tt < motion_det_win[1])
        if len(motion_det_win)==4:
            sel_t = sel_t | (tt >= motion_det_win[2]) & (tt < motion_det_win[3])
            
        y = my[idx]
        # Get data for time windows
        dx = np.diff(x[sel_t])
        dy = np.diff(y[sel_t])
        r[idx] = np.mean(np.sqrt(dx**2 + dy**2))
        
    q = np.quantile(r, motion_quantile_to_keep)
    good_trials = r <= q
    
    return good_trials


def get_all_chan_mouse_rip_rate_matrix(mouse_group_data,normalize=True):
    """
    Inputs: mouse_group_data = collect_mouse_group_rip_data(data_sessions,args)
            normalize - Boolean, normalize rate by baseline mean rate (bins with bin_cen < 0)
    Outputs:
        rip_rate_matrix - 2d numpy array (nTotChanAllMice x nBins) of ripple rate
        bin_cen - 1d numpy array of bin centers (sec) relative to photostim onset
        n_chan - list (length = nMice) of number of channels in each mouse
    """
    all_rip_rate = []
    n_chan = [] # list of number of channels in each mouse
    for md in mouse_group_data:
        n_chan.append(len(md))
        for chd in md:            
            args = chd['args']
            rip_rate,_,bin_cen = get_ripple_rate(chd['rdata'],args.bin_width,args.xmin,args.xmax)
            if normalize:
                rip_rate = rip_rate/np.mean(rip_rate[bin_cen <0])
            all_rip_rate.append(rip_rate)
    rip_rate_matrix = np.vstack(all_rip_rate)
    return rip_rate_matrix,bin_cen,n_chan        

def get_light_pulse_train_info(key, pulse_per_train):
    
    """
    Inputs:
        key : dict, key of one session or one recording channel
        pulse_per_train : int, number of light pulses per stimulus train
    
    Outputs:
        pon: light pulse on times (us) - 1D numpy array of int64 time stamps
        poff: light pulse off times (us) - 1D numpy array of int64 time stamps
        pulse_width: float, pulse width in ms
        pulse_freq: Hz, float (if pulse train) or NaN (if single pulse)
        
    """
    # Get pulse onset and offset times (microsec)
    pon, poff = dju.get_light_pulse_times(key)
    
    
    # Get pulse width in ms
    all_pulse_widths = (poff-pon)*1e-3
    pulse_width = np.median(all_pulse_widths)
    # Inform user how many pulses had widths different from the median pulse width
    minority = all_pulse_widths != pulse_width
    minority_pulse_widths = all_pulse_widths[minority]   
    minority_pulse_count = np.where(minority)[0].size
    
    if minority_pulse_count > 0:
        # Compute relative difference in the pulse width between majoriy and minority
        rel_change = 100*(np.abs(np.mean(minority_pulse_widths)-pulse_width)/pulse_width)
        if rel_change > 1: # 1% tolerance
            print('%u pulses had widths different from the majority pulse width of %0.2f'
                  % (minority_pulse_count,pulse_width))
            print('Pulse widths are:')
            print(minority_pulse_widths)
    # Compute pulse freq
    if pulse_per_train == 1:
        pulse_freq = np.NaN
    else:
        # Get the first train as an example & get pulse freq in Hz
        train_on = pon[0:pulse_per_train]        
        pulse_freq = 1/(np.median(np.diff(train_on))*1e-6)
            
       
    return pon, poff, pulse_width, pulse_freq   


# def get_disp_artifacts(rdata,art_peak_hw_ratio_th=10):
#     """ Identify displacement data point locations with abnormally high 
#         amplitude of head displacement.
#     Inputs:
#         rdata: dict containing keys (rel_mt,head_disp etc)
#         art_peak_hw_ratio_th - artifacts' height-to-width ratio threshold above
#                 which peaks will be interpolated with surrounding values.
#     Output:        
#         art_idx - list (len = nTrials) of 1d numpy array of indices of artifact 
#                 locations in the displacement values within each trial
#         art_dur - 1d numpy array (nTrials,) of artifact duration in sec
        
#     """
#     debug=True
   
#     all_hd = np.concatenate(head_disp)
#     s = np.nanstd(all_hd)    
    
#     trials_with_art = []
#     fps = 29.97
#     art_dur = []
#     art_idx = []
#     for iTrial,(t,hd,cmx,cmy) in enumerate(zip(rel_mt,head_disp,mx,my)):       
#         # head_disp was calculated with two adjacent points leaving the length 
#         # of hd one less than that of t. So we will exclude the last time point.
#         # We can do this because the exact value of t is unimportant        
#         t = t[:-1] 
#         # Replace outliers with interpolated values       
       
#         pdata = sig.find_peaks(hd,height=s,width=0.5,rel_height=0.9)
#         hw_ratio = pdata[1]['peak_heights']/pdata[1]['widths'] 
        
#         trial_art_dur = 0        
#         outliers = np.zeros(t.size,dtype=bool)
#         if hw_ratio.size>0:            
#             real_bad = hw_ratio > art_peak_hw_ratio_th
#             if np.any(real_bad):
#                 if debug:
#                     print(f'Height_width ratios: {hw_ratio}')
#                     print(f'max hw ratio: {np.round(np.max(hw_ratio),2)}')
#                     plt.clf()
#                     plt.subplot(1,3,1)
#                     plt.plot(t,hd)
#                     plt.subplot(1,3,2)
#                     plt.plot(t,cmx[:-1],color='aqua')
#                     plt.plot(t,cmy[:-1],color='plum')
#                     plt.subplot(1,3,3)
#                     plt.stem(hw_ratio)
#                     plt.tight_layout()
#                     hw_th = art_peak_hw_ratio_th
#                     plt.plot(plt.xlim(),[hw_th,hw_th],color='gray')
#                     plt.draw()                
#                 trial_art_dur = (pdata[1]['widths'][real_bad])*(1/fps)
#                 left_ips = pdata[1]['left_ips'][real_bad].astype(int)
#                 right_ips = pdata[1]['right_ips'][real_bad].astype(int)
#                 for w1,w2 in zip(left_ips,right_ips):
#                     outliers[w1:w2+1]=True
        
#         art_dur.append(np.sum(trial_art_dur))
        
#         # Correct artifact segments
#         sel = np.zeros(t.size,dtype=bool)
#         if np.any(outliers):
#             trials_with_art.append(iTrial)
#             # sel = (~outliers) & ~np.isnan(hd)
#             sel = outliers | np.isnan(hd)
            
#             # # For directly correcting mx and my, we need to increment the location
#             # # of the artifact in the head displacemtn indices by 1.
#             # sel_xy_idx = np.nonzero(sel)[0]+1
#             # outlier_xy_idx = np.nonzero(outliers)[0]+1
            
#             sel_t = t[sel]
#             sel_hd = hd[sel]
#             # sel_x = cmx[sel_xy_idx]
#             # sel_y = cmy[sel_xy_idx]
#             outlier_replace_hd = np.interp(t[outliers],sel_t,sel_hd)
#             # outlier_replace_x = np.interp(t[outlier_xy_idx],sel_t,sel_x)
#             # outlier_replace_y = np.interp(t[outlier_xy_idx],sel_t,sel_y)
            
#             # # When hd gets updated here, the original head_disp gets 
#             # # updated.
#             hd[outliers] = outlier_replace_hd
#             # cmx[outlier_xy_idx] = outlier_replace_x
#             # cmy[outlier_xy_idx] = outlier_replace_y
#             if debug:
#                 plt.subplot(1,3,1)
#                 plt.plot(t,hd,color='m')
#                 plt.subplot(1,3,2)
#                 plt.plot(t,cmx[:-1])
#                 plt.plot(t,cmy[:-1])
#                 plt.draw()
            
#     if len(trials_with_art)>0:
#         print('Trials for which artifacts corrected: ',trials_with_art)
   
#     return art_idx,np.array(art_dur)


def get_processed_rip_data(keys,pulse_per_train,std,minwidth,args):
    """ Get ripple data for a single channel of a mouse recorded in one or more 
        session that used the same stimulation protocol.
        Inputs:
            keys - list of dict. Must be a list even if just one dict. All keys
                   must be from the same mouse and same channel and with the 
                   same light pulse stimulation protocol
            pulse_per_train - list; number of pulses given in each train; list
                    size must match that of keys
            std - list of std values for restricting ripples table
            minwidth - list of minwidth params for restricting ripples table
            args -  Args object with updated params-values          
        Outputs:
            rdata - list of dict. len(rdata) = num of trials. Each dict contains the following keys:
                    'rel_mt' - numpy array of time (sec) relative to stim train onset
                    'mx','my' - numpy array of tracker LED x and y coodinate respectively
                    'head_disp' - numpy array of head displacement per video frame (pix/frame)
                    'rip_evt'- numpy array of ripple event times (sec) relative to stim train onset
                    'session_start_time' - session start time of the trial.
                    
            args - Updated Args class object with newly added experiment-related info.
        """        
    # Data type check for keys, std and minwidth
    assert (type(keys)==list) and (type(std)==list) and (type(minwidth)==list),\
    'keys, std and minwidth must be lists'
    assert len(set([x['animal_id'] for x in keys]))==1, 'all keys must be from the same mouse'
    # Make sure chan_num exists in keys    
    assert all(['chan_num' in kk for kk in keys]), 'chan_num does not exist in given keys'
    assert len(set([x['chan_num'] for x in keys]))==1, 'all keys must have the same chan_num'
    # Ensure that each key has its own associated std and minwidth
    assert (len(keys)==len(std)==len(minwidth)), 'number of keys must match \
    number of std and minwidth'
   
    # Create binning params
    pre = args.xmin * 1e6 # to microsec
    post = args.xmax * 1e6 # to microsec
    
    # Go through each channel key and collect ripple and motion data
    cc = 0
    rdata = []
    all_pulse_width = set()
    all_pulse_freq = set()
    for ikey, key in enumerate(keys):
        ppt = pulse_per_train[ikey]
        # Get light pulse train info
        pon_times, poff_times, pulse_width, pulse_freq = get_light_pulse_train_info(key, ppt)
        all_pulse_width.add(pulse_width)
        all_pulse_freq.add(pulse_freq)
        # For plotting purpose, extract one pulse train on and off times in sec
        # We assume that all keys had the same stimulus train parameters, so we pick one key
        if ikey==0:
            one_train_on = (pon_times[0:ppt] - pon_times[0]) * 1e-6
            one_train_off = (poff_times[0:ppt] - poff_times[0]) * 1e-6
        # Pick the first pulse times in the pulse train for time referencing
        tr_on = pon_times[::ppt] # shape: (rows,), int64
        tr_off = poff_times[::ppt] # shape: (rows,), int64
        
        # Get motion info from tracker data. tr_on is returned because it may be trimmed
        # inside the get_perievent_motion function for shorter trial length
        # Note - output of get_perievent_motion, mt, will be in seconds.
        same_len = True
        mt, mx, my, good_trials_len = gfun.get_perievent_motion(key, tr_on, args.xmin, 
                                                                args.xmax, same_len)
        
        rel_mt = center_and_scale_time(mt, tr_on)
     
        # # Mark (True/False) good trials by amount of head movement 
        # good_trials_mov = filter_trials_by_head_disp(rel_mt, mx, my, 
        #                                              args.motion_det_win, 
        #                                              args.motion_quantile_to_keep)
        
        # Mark (True/False) trials by their presence inside given behavioral state (e.g NREM)
        good_trials_beh = filter_trials_by_beh_state(tr_on, tr_off, args.beh_state, key)
        
        # Combine all selections and re-filter event times and trial data
        good_trials = good_trials_len & good_trials_beh
        tr_on = tr_on[good_trials]
        rel_mt = list(compress(rel_mt, good_trials))
        mx = list(compress(mx, good_trials))
        my = list(compress(my, good_trials))
        
        # Get ripple events
        fstr = f'std = {std[ikey]} and minwidth = {minwidth[ikey]}'
        print(fstr)        
        rpt = np.array((ripples.RipEventsBs & key & 
                        (ripples.DetParamsBs & fstr)).fetch('peak_t'))
        # Check if the RipEvents table was populated for sessions for nrem behavioral state
        if args.beh_state=='nrem':
            assert rpt.size > 0, 'No ripples! Check if the RipEvents table is populated'
        
        # Go through each pulse train and collect ripples near by        
        for t_idx, ct_on in enumerate(tr_on):
            twin = np.array([pre, post]) + ct_on
            # Pick ripples in window
            sel_rt = rpt[(rpt >= twin[0]) & (rpt < twin[1])]
            cmx = mx[t_idx]
            cmy = my[t_idx]
            head_disp = calc_displacement(cmx,cmy)
            # # Add a filler value to compensate for one sample loss due to diff
            # We will replicate the last valid head disp value as the filler
            # head_disp = np.append(head_disp, head_disp[-1])
          
            # Center ripple event times also as above
            re_rel_t = (sel_rt - ct_on)*1e-6
            rdata.append({'rel_mt': rel_mt[t_idx], 'mx': mx[t_idx], 'my': my[t_idx],\
                        'head_disp': head_disp, 'rip_evt': re_rel_t,
                        'session_start_time':key['session_start_time']})            
                                 
    # Remove high amplitude head disp artifacts
    if args.correct_high_amp_mov_art:
        # Correct artifactual segments 
        art_dur,rdata = correct_abnorm_high_mov_artifacts(rdata,
                                art_peak_hw_ratio_th=args.art_peak_hw_ratio_th)
        rdata = [rdata[i] for i in np.nonzero(art_dur < args.max_art_duration)[0]]
            
    if not len(all_pulse_freq)==len(all_pulse_width)==1:
        print('**************************************************************')
        print('pulse frequencies: ',all_pulse_freq)
        print('pulse widths: ', all_pulse_width)
        logging.warning('all keys must have the same pulse frequency and pulse widths\
                      make sure that the difference in values are acceptable')
        print('**************************************************************')
        
    args.one_train_on = one_train_on
    args.one_train_off = one_train_off
    args.pulse_width = pulse_width
    args.pulse_freq = pulse_freq
    args.pulse_per_train = pulse_per_train
    args.mouse_id = key['animal_id']
    args.chan_name = (cont.Chan & key & f'chan_num = {key["chan_num"]}').fetch('chan_name')[0]
    
    return rdata, args
        

def get_ripple_rate(rdata, bin_width, xmin, xmax):
    """
    Computes ripple rates in rip/s at bins of width bin_width between xmin and xmax
    times points relative to event onset.
    
    Inputs: -----------------------------------
        rdata: list of dict, output of get_processed_rip_data(...) function call
        bin_width: in millisec
        xmin: a negative number in sec
        xmax: a positive number in sec
    
    Outputs:-----------------------------------
        rip_rate: 1D np array of ripple rate (rip/sec)
        bin_edges: 1D np array of bin edges
        bin_cen: 1D np array of bin centers in sec, same length as rip_rate.
    
    """
    evt = np.concatenate([v['rip_evt'] for v in rdata])
    # Create binedges
    bw =  bin_width/1000
    bin_edges,bin_cen = utp.create_psth_bins(xmin,xmax,bw)    
    # Get histogram counts
    counts,_ = np.histogram(evt,bin_edges)
    n_trials = len(rdata)
    rip_rate = (counts/n_trials)/bw # rip/sec
        
    return rip_rate,bin_edges,bin_cen


def pool_head_mov_across_mice(group_data,metric,within_mouse_operator):
    """
    Pool head displacement across mice. We will not normalize within each mouse
    Inputs:
        group_data - list (mice) of list(channels) of dict (ripple data), this 
                    is an output from collect_mouse_group_rip_data(...) function call.
        metric - string; must be either 'disp' (head displacement) or 'inst_speed'
        within_mouse_operator - str, should be 'mean' or 'median' - tells you if mean
                              or median is computed across trials within a mouse 
    Outputs: 
        t_bin_cen_vec - 1D numpy array of bin center times(s) relative to 
                        stimulus onset
        all_rr - 2D numpy array, nMice-by-nTimeBins of head movement metric 
                (mean or median)
    
    MS 2022-03-14/2024-08-08
        
    """
    all_rr = []
    for md in group_data: # loop over mice       
        # For each mouse pick head disp data     
        crdata = copy.deepcopy(md[0]['rdata'])        
        n = len(crdata)
        t_list = []
        vx_list = []
        vy_list = []
        art_idx = []
        for jj in range(n):
            t_list.append(crdata[jj]['rel_mt'])
            vx_list.append(crdata[jj]['mx'])
            vy_list.append(crdata[jj]['my'])
            art_idx.append(crdata[jj]['art_idx'])
        
        # Use interpolation and compute a common time vector for all stimulation 
        # trials, and the x and y coordinates corresponding to the common time points.
        t_vec, mxi_list = gfun.interp_based_event_trig_data_average(t_list, vx_list)
        _, myi_list = gfun.interp_based_event_trig_data_average(t_list, vy_list)
        
        # Compute head displacement  or instantaneuous speed 
        # using movement info from two adjacent video frames        
        # Create time bin centers
        ifi = np.median(np.diff(t_vec)) # inter-frame-interval
        t_bin_cen = t_vec[0:-1]+ifi/2
        
        # Compute head displacement in mm
        d_array = [convert_motion_traj_to_inst_disp(px,py)[0] \
             for px,py in zip(mxi_list,myi_list)]
        # Interpolate accepted-for-correction artifactual time periods
        for outliers,dis in zip(art_idx,d_array):
            good_idx = ~outliers
            gt = t_bin_cen[good_idx]
            gy = dis[good_idx]
            outliers_corr = np.interp(t_bin_cen[outliers],gt,gy)
            # In-place correction of d_array's each element
            dis[outliers] = outliers_corr  
        match metric:
            case 'inst_speed':
                mi_list = [di/ifi for di in d_array] # mm/sec
            case 'disp':
                mi_list = d_array # mm
            case _:
                raise ValueError('metric must be either inst_speed or disp')
                    
        # Pool data within mouse: 
        # first change into 2D array where rows are trials
        mi_array = np.stack(mi_list, axis=0)
        if within_mouse_operator == 'mean':
            mi_cen = np.mean(mi_array, axis=0)
        elif within_mouse_operator == 'median':
            mi_cen = np.median(mi_array, axis=0)
        else:
            raise ValueError('within_mouse_operator should be "mean" or "median"')
            
        all_rr.append(mi_cen)
        
    all_rr = np.array(all_rr)
   
    return t_bin_cen, all_rr
                
 


   

        