# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 17:34:39 2022
@author: Mani
This module gets trial by trial ripple time points for the light stimulation-mediated
ripple suppression.
"""

# Get database tables
from itertools import compress, chain
import logging
import numpy as np
import acq as acq
import cont as cont
import ripples as ripples
import general_functions as gfun
import djutils as dju
import util_py as utpy
import copy
import scipy.signal as sig
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import neuralynx_io as nio
import scipy

#%% Functions
# Class for default values of parameters
class Args():
    def __init__(self):
        # Data length pre and post light pulse
        self.xmin = -4.0  # Add sign appropriately. If you want 1sec before light pulse, use -1
        self.xmax = 6.0
        self.bin_width = 100  # bin width in msec
        self.beh_state = 'nrem'  # beh state to collect ripples from: 'all','rem','nrem','awake'

        # in sec: time window to detect motion. Can be 4 element [-5 0 2 3] where [-5 0] and [2 3] are two separate windows.
        self.motion_det_win = [self.xmin, self.xmax]

        # Remove trials without ripples?
        self.remove_norip_trials = True
        self.correct_high_amp_mov_art = True
        # Maximum cumulative duration of artifacts after which trial will be
        # excluded
        self.max_art_duration = 1
        # Threshold value of distance between 5th and 95th percentile value of
        # the differential of the head displacement for removing artifacts
        # arising from recording cable crossing LED back and forth
        self.art_trial_removal_th = 5
        # Height-to-width ratio threshold for calling a detected peak as
        # artifact in the head displacement values
        self.art_peak_hw_ratio_th = 7
        self.pre_and_stim_only = True
        # Threshold (ripple rate Hz in baseline) below which we will exclude 
        # a recording session
        self.baseline_rip_rate_th = 0.2

# main function that calls other functions
def get_raw_traces(data_filename, t_pre, t_post, dec_factors=None, 
                                                   plot=False, t_pad=0):
    """ Get raw traces optionally decimated for each channel of each mouse
    Inputs:
        data_filename - filename of rocessed data containing all ripple and 
                        motion related data       
        t_pre - time (must be a negative number: like -4) in sec before light
                pulse train onset
        t_post - time (must be a positive number: like 6) in sec after light
                pulse train onset
        dec_factors - list of decimation factors; e.g., [4,4,4,2]. If None, pure
                  raw data will be returned
        t_pad - buffer time at the beginning and end of the trial raw data to 
                deal with decimation and convolution artifacts
    Outputs:
        raw - dict with keys fs(final sampling rate), t_pre, t_post, t_pad,
              and data; data is list (len=n_mice) of lists(len=n_chan); 
              inner list containing a dict with keys animal_id, chan_num, t, 
              and trial_data; trial_data is a list (len=n_trials) of 
              raw (or decimated) data traces corresponding to different trials
    """
    ch_map = ['t1c1','t1c2','t1c3','t1c4','t2c1','t2c2','t2c3','t2c4',
              't3c1','t3c2','t3c3','t3c4','t4c1','t4c2','t4c3','t4c4']
    data = utpy.get_pickled_data(data_filename)
    ax_size = [2.7, 0.4]
    scale_fac = 1000
    fs = 32000 if dec_factors is None else 32000/np.prod(dec_factors)
    raw = dict(fs=fs, t_pre=t_pre, t_post=t_post, t_pad=t_pad, data=[])
    for md in data: 
        chdata_temp = []
        for chd in md:
            chan_num = chd['chan_num']
            ch_name = ch_map[chan_num]
            rdata = chd['rdata']    
            sess_times = [x['session_start_time'] for x in rdata]
            u_sess_times = np.unique(sess_times)
            sdata = dict()
            down_fac = np.prod(dec_factors)
            for ust in u_sess_times:
                dfold = dju.get_sess_str(ust)
                file_path = os.path.join(r'D:\ephys\raw', dfold, f'{ch_name}.ncs')
                hf = nio.load_ncs(file_path)
                sdata.update({ust: hf})            
            if plot:
                fig, ax = utpy.make_axes(plt, ax_size)
            trdata_temp = []
            for i_trial, td in enumerate(rdata):
                print(i_trial)
                hf = sdata[td['session_start_time']]
                cond_pre = td['train_onset'] - ((abs(t_pre) + t_pad) * 1e6)
                cond_post = td['train_onset'] + ((t_post + t_pad)*1e6)
                sel = (hf['time'] > cond_pre) & (hf['time'] <= cond_post)
                dd = hf['data'][sel]
                if dec_factors is not None:
                    for dec_fac in dec_factors:
                        dd = sig.decimate(dd, dec_fac, ftype='fir')
                tr0 = float(td['train_onset'])
                t = ((hf['time'][sel]).astype(float)-tr0)
                t = t[::down_fac]*1e-6               
                trdata_temp.append(dd)
                if plot:
                    ax.plot(t, dd + (i_trial*scale_fac),color='k',linewidth=0.5)   
                    
            chdata_temp.append(dict(animal_id=chd['animal_id'], 
                                    chan_num=chan_num, 
                                    trial_data=trdata_temp,
                                    t=t))
            print(f'Done with mouse {chd["animal_id"]}, chan {chan_num}')
        raw['data'].append(chdata_temp)
        
    return raw

def get_spectrogram(data, fs, t_resol=0.05, fmax=None):
    """ Compute spectrogram
    Inputs:
        data - list of time series of signals
        fs - sampling rate (Hz)
        t_resol - time resolution (s) of spectrogram
        fmax - maximum analysis frequency for spectrogram
    Outputs:
        ps - list of 2d numpy arrays of spectrograms
        f - analysis frequencies
        t - time vector corresponding to the spectrogram
    """
    t_win = 1.5 # sec sliding window   
    M = int(fs*t_win)
    g_std = int(M/5)
     
    hop = int(t_resol * fs)
    win = scipy.signal.windows.gaussian(M, g_std, sym=True)
    sft = scipy.signal.ShortTimeFFT(win, hop, fs)
    sel = slice(None) if fmax is None else sft.f <= fmax
    Sxx = []
    for d in data:        
        Sx = abs(sft.stft(d))    
        Sxx.append(Sx[sel,:])
    f = sft.f[sel]
    t = sft.t(len(d))
    return Sxx, f, t

def get_average_spectrogram(raw, fmax=None):
    """ Compute spectrogram for each trial, average it across trials, then
    average it across channels for each mouse. We will also remove any padding
    of data used to mitigate edge artifacts
    Inputs:
        raw - output data from get_raw_trial_traces() function; it is a 
                dict with keys fs(final sampling rate), t_pre, t_post, t_pad,
                and data; data is list (len=n_mice) of lists(len=n_chan); 
                inner list containing a dict with keys animal_id, chan_num, t, 
                and trial_data; trial_data is a list (len=n_trials) of 
                raw (or decimated) data traces corresponding to different trials
        fmax - maximum of analysis frequency for spectrogram
    Outputs:
        sxx_data - dict with keys f (analysis frequency vector),t and data; 
                data is a list (len=n_mice) of dicts with keys animal_id and sxx
    """
    sxx_data = dict(t_pre=raw['t_pre'], t_post=raw['t_post'], data = [])
    tdr_params = (cont.TDratioParams()).fetch1()
    for md in raw['data']: # mice
        sxx_ch = []
        for chd in md: # channels
            sxx, f, t = get_spectrogram(chd['trial_data'], raw['fs'], fmax=fmax)          
            # Adjust t output as spectrogram function assumed t=0 at the first 
            # time-domain sample input
            t = t - (abs(raw['t_pre']) + raw['t_pad'])
            # it is assumed to start
            sel = (t >= raw['t_pre']) & (t <= raw['t_post'])
            # average across trials & remove padding if any
            sxx = np.dstack(sxx).mean(axis=2)[:, sel]
            sxx_ch.append(sxx)
        # Average across channels 
        sxx = np.dstack(sxx_ch).mean(axis=2)       
        # Z-score across frequencies
        sxx_zs = (sxx - np.mean(sxx, axis=0))/np.std(sxx, axis=0)        
        # Theta/delta ratio
        f_theta = (f >= tdr_params['theta_begin']) & (f <= tdr_params['theta_end'])
        f_delta = (f >= tdr_params['delta_begin']) & (f <= tdr_params['delta_end'])
        tdr = (sxx[f_theta,:].mean(axis=0))/(sxx[f_delta,:].mean(axis=0))
        # Save
        sxx_data['data'].append(dict(animal_id = chd['animal_id'],
                                sxx=sxx, sxx_zs=sxx_zs, tdr=tdr))
    sxx_data['f'] = f
    sxx_data['t'] = t[sel]
    
    return sxx_data
          
def get_raw_traces_one_chan(data_filename, mouse_id, chan_num, t_pre, t_post, 
                                       dec_factors=None, plot=False, t_pad=1):
    """ Get raw traces optionally decimated
    Inputs:
        data_filename - filename of rocessed data containing all ripple and 
                        motion related data
        mouse_id - id of mouse
        chan_num - channel number
        t_pre - time (must be a negative number: like -4) in sec before light
                pulse train onset
        t_post - time (must be a positive number: like 6) in sec after light
                pulse train onset
        dec_factors - list of decimation factors; e.g., [4,4,4,2]. If None, pure
                  raw data will be returned
        t_pad - buffer time at the end of the trial raw data to deal with
                    decimation artifact
    """
    ch_map = ['t1c1','t1c2','t1c3','t1c4','t2c1','t2c2','t2c3','t2c4']
    ch_name = ch_map[chan_num]
    d = utpy.get_pickled_data(data_filename)
    mouse_data = [x for x in d if x[0]['animal_id']==mouse_id][0]
    ch_data = [x for x in mouse_data if x['chan_num']==chan_num][0]
    rdata = ch_data['rdata']   
    scale_fac = 1000
    sess_times = [x['session_start_time'] for x in rdata]
    u_sess_times = np.unique(sess_times)
    sdata = dict()
    down_fac = np.prod(dec_factors)
    for ust in u_sess_times:
        dfold = dju.get_sess_str(ust)
        file_path = os.path.join(r'D:\ephys\raw', dfold, f'{ch_name}.ncs')
        hf = nio.load_ncs(file_path)
        sdata.update({ust: hf})
    raw = []
    if plot:
        ax_size = [2.7,0.4]
        fig, ax = utpy.make_axes(plt, ax_size)
    for i_trial, td in enumerate(rdata):
        hf = sdata[td['session_start_time']]
        cond_pre = td['train_onset'] + (t_pre * 1e6)
        cond_post = td['train_onset'] + ((t_post + t_pad)*1e6)
        sel = (hf['time'] > cond_pre) & (hf['time'] <= cond_post)
        dd = hf['data'][sel]
        if dec_factors is not None:
            for dec_fac in dec_factors:
                dd = sig.decimate(dd, dec_fac, ftype='fir')
        tr0 = float(td['train_onset'])
        t = ((hf['time'][sel]).astype(float)-tr0)
        t = t[::down_fac]*1e-6
        dd = dd[t <= t_post]
        t = t[t <= t_post]
        raw.append(dd)
        if plot:
            ax.plot(t, dd + (i_trial*scale_fac),color='k',linewidth=0.5)       
    return raw, t

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
    for md in group_data:  # loop over mice
        # For each mouse pick a channel or average across channels
        n_chan = len(md)
        if elec_sel_meth == 'random':
            chd = md[np.random.randint(n_chan)]
            args = chd['args']
            mouse_rr, _, bin_cen = get_ripple_rate(chd['rdata'],
                                                   args.bin_width,
                                                   args.xmin, args.xmax)
        else:  # Other methods that require computing ripple rate
            norm_rip_rate = []
            for chd in md:  # loop over channels of each mouse
                args = chd['args']
                rd, _, bin_cen = get_ripple_rate(chd['rdata'], args.bin_width,
                                                 args.xmin, args.xmax)
                # Normalize the ripple rate for each channel by the mean ripple
                # rate during baseline period (i.e., period until stim onset time)
                # This method will give equal weight to all channels
                norm_rd = 100*rd/np.nanmean(rd[bin_cen < 0])
                norm_rip_rate.append(norm_rd)
            # Apply selection on the ripple rate
            norm_rip_rate = np.array(norm_rip_rate)
            match elec_sel_meth:  # noqa
                case 'avg':
                    norm_mouse_rr = np.nanmean(norm_rip_rate, axis=0)
                case _:
                    raise ValueError(
                        "The provided method %s is not implemented\n" % elec_sel_meth)
        all_rr_norm.append(norm_mouse_rr)

    # # Normalize ripple rate to baseline before averaging across mice
    # all_rr_norm = [rr/np.mean(rr[bin_cen < 0]) for rr in all_rr]

    # Average across mice
    all_mouse_rr = np.array(all_rr_norm)
    mean_rr = np.nanmean(all_rr_norm, axis=0)
    std_rr = np.nanstd(all_rr_norm, axis=0)

    return bin_cen, mean_rr, std_rr, all_mouse_rr


def calc_displacement(x, y):
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
    rel_mt = [(t-et)*1e-6 for t, et in zip(mt, evt)]
    return rel_mt


def collect_mouse_group_rip_data(data_sessions, args):
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
    for iMouse, m_id in enumerate(uids):
        # Get data slice corresponding to the current mouse
        mouse_dd = data_sessions[ids == m_id]
        mouse_keys = [dju.get_key_from_session_ts(
            sess_ts)[0] for sess_ts in list(mouse_dd.session_ts)]
        # It is assumed that all repeated sessions of any mouse have the same
        # set of experimental parameters excluding recording channels.
        # So parameters come from the first session. But in some rare circumstances
        # like the wild type mice, we need to pool two different conditions such
        # as 1 and 2 pulses per train and 3 and 5 ms pulse widths.

        # When the same experimental condition (say pulse width, std, minwidth,
        # laser color etc) was repeated several days apart, often recording
        # channels were different though some channels overlapped. So, we will
        # pool channels across the repeated sessions in which they were recorded.
        
        all_chans = [re.findall(r'(\d+)',str(x)) for x in mouse_dd.chan]
        unique_chans = np.unique([int(x) for x in chain(*all_chans)])
        # Go through each unique channel
        for ch in unique_chans:
            ch = int(ch)
            # Find which sessions had this channel
            sel_keys, std, minwidth, pp_train = [], [], [], []
            # Some channels may not be in all sessions
            sel_key_ind = np.full(len(mouse_keys), False)
            for iKey, sess_key in enumerate(mouse_keys):
                chan_str_list = re.findall(r'(\d+)',str(mouse_dd.iloc[iKey, :]['chan']))
                chans_of_ikey = [int(x) for x in chan_str_list]
                if ch in chans_of_ikey:
                    sess_key['chan_num'] = ch
                    sel_keys.append(sess_key)
                    pp_train.append(int(mouse_dd.iloc[iKey,:]['pulse_per_train']))
                    std.append(mouse_dd.iloc[iKey, :]['std'])
                    minwidth.append(mouse_dd.iloc[iKey, :]['minwidth'])
                    # Save index
                    sel_key_ind[iKey] = True
            rdata, args_out = get_processed_rip_data(sel_keys, pp_train, std,
                                                 minwidth, args)            
            if args.beh_state=='nrem':
                # ============ Exclude sessions having low ripple rate ============
                rdata, good_sess, good_sess_start_times = \
                                        remove_sess_with_low_rip_rate(rdata, args) 
            else:
                good_sess = np.full(len(sel_keys), True)
                
            animal_id = mouse_keys[0]['animal_id']
            if len(rdata) > 0:               
                # Add useful info to args_out
                args_out.pulse_width = list(mouse_dd['pulse_width'][sel_key_ind][good_sess])
                args_out.pulse_freq = list(np.array(args_out.pulse_freq)[good_sess])
                args_out.pulse_per_train = list(mouse_dd['pulse_per_train'][sel_key_ind][good_sess])            
                args_out.session_type = list(mouse_dd['session_type'][sel_key_ind][good_sess])
                args_out.laser_color = list(mouse_dd['laser_color'][sel_key_ind][good_sess])
                args_out.opsin = list(mouse_dd['opsin'][sel_key_ind][good_sess])
                args_out.session_ts = list(mouse_dd['session_ts'][sel_key_ind][good_sess])
                ch_dic = {'animal_id': animal_id,
                          'chan_num': ch, 'rdata': rdata, 'args': args_out}
                group_data[iMouse].append(ch_dic)
            else:
                print(f'Mouse {animal_id} was excluded due to low ripple rate')
        print(f'Done with mouse {iMouse}')
    # Some mice may not have data with sufficient baseline ripple rate. For 
    # them, data will be []. We will remove those here:   
    group_data = [gd for gd in group_data if len(gd) > 0]        
    return group_data

def remove_sess_with_low_rip_rate(rdata, args):
    """ Remove bad sessions
    Inputs:
        rdata - comes from: rdata, args_out = get_processed_rip_data(...)
        args -  Args object with updated params-values
    Outputs:
        rdata - rdata with trials from bad sessions removed
        good_sess_start_times - 1d np array of session start times of good sessions
    """
    # Get unique sessions
    sess_start_times = np.array([x['session_start_time'] for x in rdata])
    # Maintain order when finding unique sessions
    _,idx = np.unique(sess_start_times, return_index=True)
    u_sess_start_times = sess_start_times[np.sort(idx)]
    good_sess_data = []
    n_sess = u_sess_start_times.size    
    good_sess = np.zeros(idx.size,dtype=bool)
    for isess,ust in enumerate(u_sess_start_times):
        # Collect trials of given session
        sel = np.nonzero(sess_start_times==ust)[0]
        sel_rdata = [rdata[s] for s in sel]
        rip_rate, bins, _ = get_ripple_rate(sel_rdata, args.bin_width, 
                                                args.xmin, args.xmax)
        baseline_rip_rate = np.nanmean(rip_rate[bins[0:-1] < 0])
        if baseline_rip_rate >= args.baseline_rip_rate_th:
            good_sess_data.append(sel_rdata)            
            good_sess[isess] = True
    rdata = list(chain(*good_sess_data))
    n_sess_excluded = n_sess-np.count_nonzero(good_sess)
    good_sess_start_times = u_sess_start_times[good_sess]
    if n_sess_excluded > 0:
        print(f'{n_sess_excluded} sessions were excluded due to low baseline ripple rate')       
    return rdata, good_sess, good_sess_start_times

def collapse_rip_events_across_chan_for_each_trial(group_data, elec_sel_meth='avg'):
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
                        'inst_speed' - numpy array of instantaneous speed (mm/s)
                        'rip_evt'- numpy array of ripple event times (sec) relative to stim train onset
                        'session_start_time' - scalar; session start time of the trial
        elec_sel_meth: str, should be one of 'avg' or'random'
    Outputs:
        cgroup_data: list(of length nMice) of dict('animal_id','args','trial_data') where
                    trial_data is a list (of length nTrials) of
                    dict('rel_mt','mx','my','inst_speed','bin_cen_t','rip_cnt')
    """
    cgroup_data = []
    args = group_data[0][0]['args']
    bw = args.bin_width/1000
    bin_edges, bin_cen = utpy.create_psth_bins(args.xmin, args.xmax, bw)
    # Loop through each mouse
    for iMouse, md in enumerate(group_data):
        # Find out how many recording sessions were there for this mouse:       
        all_sess_starts = []
        for chd in md:
            all_sess_starts.append([rd['session_start_time'] for rd in chd['rdata']])
        unique_starts = np.unique(list(chain(*all_sess_starts)))
        nMouseChan = len(md)
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
                trial_sess_starts = np.array(
                    [x['session_start_time'] for x in md[iChan]['rdata']])
                sel_ind = np.nonzero(trial_sess_starts == sess_start)[0]
                # Pick trials that belonged to the current session
                if sel_ind.size != 0:
                    sess_chan_data.append(copy.deepcopy(
                        [md[iChan]['rdata'][i] for i in sel_ind]))
            # Average each trial data across all the channels of the given session
            # All channels will have the same number of trials
            nTrials = len(sess_chan_data[0])
            sess_n_chan = len(sess_chan_data)
            for iTrial in range(nTrials):
                # Motion data are same for all channels. So pick those from
                # the first channel
                cdata = sess_chan_data[0][iTrial]
                hdata = np.zeros((sess_n_chan, bin_cen.size))
                match elec_sel_meth:
                    case 'avg':
                        # Average current trial data from all channels of the current session
                        for iSessChan in range(sess_n_chan):
                            evt_t = sess_chan_data[iSessChan][iTrial]['rip_evt']
                            # Bin the event times
                            hdata[iSessChan, :], _ = np.histogram(evt_t, bin_edges)
                        # Collapse (average) across channels
                        rip_cnt = np.mean(hdata, axis=0)
                    case _:
                        raise ValueError(
                            'electrode selection method must be "avg"')
                # Recreate cgroup_data data structure so it looks like group_data
                # except that there will be only one collapsed channel now.
                # Build trial data structure as a dict
                one_trial_data = {'rel_mt': cdata['rel_mt'], 'mx': cdata['mx'],
                                  'my': cdata['my'], 'inst_speed': cdata['inst_speed'],
                                  'rip_cnt': rip_cnt,
                                  'session_start_time': cdata['session_start_time']}
                # This list will accumulate trials of all sessions of the
                # current mouse
                all_trial_data.append(one_trial_data)

        one_mouse_data = {'animal_id': md[0]['animal_id'], 'args': md[0]['args'],
                          'bin_cen_t': bin_cen, 'trial_data': all_trial_data}
        cgroup_data.append(one_mouse_data)

    return cgroup_data


def convert_motion_traj_to_inst_disp(px, py):
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
    dx = np.diff(px)
    dy = np.diff(py)
    # Get the mid point of consecutive px values to get the
    # y scaling factor corresponding to x values
    x_mid = (px[0:-1] + px[1:])/2   
    dx_mm, dy_mm = convert_pix_dist_to_mm(x_mid, dx, dy)
    d = np.sqrt((dx_mm**2)+(dy_mm**2))
    
    return d, fps

def convert_pix_dist_to_mm(avg_loc_x_pix, dx_pix, dy_pix):
    """Convert x and y distances in pixels to mm
     Inputs:
        avg_loc_x_pix -  mid point of consecutive x values from which dx_pix was
            computed. This is needed to get the y scaling factor which changes as
            a function x coordinate
        dx_pix - x distance in pixels. Can be a 1d numpy array or a number
        dy_pix - y distance in pixels.  Can be a 1d numpy array or a number
     Outputs:
         dx_mm - x distance in mm. Same shape as dx_pix
         dy_mm - y distance in mm. Same shape as dy_pix
         
     The constants used here were taken from the file:
     r'C:/Users/maniv/Documents/ripples_manuscript/data/video_size_calibration.pkl'
     To avoid loading everytime, I have hard-coded these numbers here. If you
     change the video calibration info, these numbers should be updated
     accordingly."""
     
    mm_per_pix_x = 0.47047365470852015
    b = np.array([3.93429543e+02, -7.32063842e-02])
    cage_width_mm = 180   
   
    # Scale dx and dy according to video calibration
    dx_mm = dx_pix * mm_per_pix_x
    # Because the pixel to mm conversion factor changes as a function of
    # x (due to video camera angle), we need to compute this factor for
    # all x in our trial.
    mm_per_pix_y = cage_width_mm/(b[0]+b[1] * avg_loc_x_pix)
    dy_mm = dy_pix * mm_per_pix_y
    
    return dx_mm, dy_mm

def correct_abnorm_high_mov_artifacts(rdata,args,art_peak_hw_ratio_th=7,
                                                                  debug=False):
    """ Remove data points with abnormally high amplitude of head displacement
        and replace them with interpolated datapoints.
    Inputs:
        rdata - list of dict, output of get_processed_rip_data(...) function call
        args - Args object containing preset parameters
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
    
    # The idea: we will detect unusually big artifacts in the head displacement
    # but won't correct them first. Instead, we will go and fix the artifacts in
    # the x and y coordinates and then recompute head displacement based on the
    # new values.
    
    # Add a buffer at edges to capture any artifacts that 
    # start or end high at the edges. Later we will remove them before
    # saving.
    nbp = 1 # buffer points
    dd = [d['head_disp'] for d in rdata]
    dd = np.concatenate(dd)
    s = np.nanstd(dd)
    
    trials_with_art = []
    fps = 29.97
    tot_art_dur = []    
    qd = []
    
    # if rdata[0]['session_start_time']==5315118162:
    #     rdpl.plot_head_mov_by_trial(rdata, args)
    for iTrial, rd in enumerate(rdata):
        d = rd['head_disp']
        inst_speed = rd['inst_speed']
        # Add a buffer at edges to capture any artifacts that 
        # start or end high at the edges. Later we will remove them before
        # saving.
        bvi = np.min(inst_speed)*np.ones(nbp)
        bvd = np.min(d)*np.ones(nbp)
        inst_speed = np.hstack((bvi,inst_speed,bvi))
        d = np.hstack((bvd,d,bvd))
        # n_samples = inst_speed.size
        # nb = 0  # extension samples on either side of an artifact
        # head_disp and speed were calculated with two adjacent points leaving
        # the length of y one less than that of t. So we will exclude the last
        # time point and shift all times forward by half bin width
        bw = np.nanmedian(np.diff(rd['rel_mt']))
        hbw = bw/2
        bin_cen_t = rd['rel_mt'][:-1] + hbw
        b_pre = np.arange(-nbp,0)*bw + bin_cen_t[0]
        b_post = np.arange(1,nbp+1)*bw + bin_cen_t[-1]
        # Add buffer
        bin_cen_t = np.hstack((b_pre,bin_cen_t,b_post))
        
        # Replace outliers with interpolated values    
        pdata = sig.find_peaks(d, height=s, width=0.5, rel_height=0.9)
        peak_ind = pdata[0]
        hw_ratio = pdata[1]['peak_heights']/pdata[1]['widths']
        
        d1 = np.diff(d);
        q = np.quantile(d1,[0.05,0.95])
        qd.append(q[1]-q[0])
        # if (iTrial >= 60) & (rd['session_start_time']==5375673492):
        #     plt.figure() 
        #     plt.plot(bin_cen_t,inst_speed)
        #     plt.title(str(iTrial))
        art_dur = 0
        outliers = np.zeros(bin_cen_t.size, dtype=bool)
        # if (rdata[0]['session_start_time']==5315118162) :
        #     debug = True
        # else:
        #     debug = False
        if hw_ratio.size > 0:
            real_bad = hw_ratio > art_peak_hw_ratio_th
            if np.any(real_bad):
                if debug:
                    print(f'Height_width ratios: {hw_ratio}')
                    print(f'max hw ratio: {np.round(np.max(hw_ratio),2)}')
                    plt.figure()
                    plt.subplot(1, 3, 1)
                    plt.plot(bin_cen_t, inst_speed)
                    plt.scatter(bin_cen_t[peak_ind],inst_speed[peak_ind],c='r',marker='*')
                    yl = plt.ylim()
                    plt.xlabel('Relative time (s)')
                    plt.ylabel('Displacement (pix)')
                    plt.title('Before artifact correction')
                    
                    plt.subplot(1, 3, 3)
                    plt.stem(hw_ratio)
                    plt.tight_layout()
                    plt.xlabel('peak index')
                    plt.ylabel('Height-width ratio')
                    hw_th = art_peak_hw_ratio_th
                    plt.plot(plt.xlim(), [hw_th, hw_th], color='gray')
                    plt.draw()
                    plt.gcf().suptitle(f'Trial index: {iTrial}')
                art_dur = (pdata[1]['widths'][real_bad])*(1/fps)
                left_ips = pdata[1]['left_ips'][real_bad].astype(int)
                right_ips = pdata[1]['right_ips'][real_bad].astype(int)
                for w1, w2 in zip(left_ips, right_ips):              
                    outliers[w1:(w2+1)] = True

        tot_art_dur.append(np.sum(art_dur))

        # Check if trial needs to be removed due to too much artifact
        art = np.zeros(bin_cen_t.size, dtype=bool)
        if np.any(outliers):
            trials_with_art.append(iTrial)
            art = outliers | np.isnan(d)
            good_part_ind = np.nonzero(~art)[0]
            # Shift indices by 1 to account for differencing in computing
            # displacement
            gis = inst_speed[good_part_ind]
            gdisp = d[good_part_ind]
            gt = bin_cen_t[good_part_ind]
            outlier_ind = np.nonzero(outliers)[0]
            # Fix artifact segments by interpolation
            gis_out = np.interp(bin_cen_t[outlier_ind], gt, gis)
            gdisp_out = np.interp(bin_cen_t[outlier_ind], gt, gdisp)
            inst_speed[outlier_ind] = gis_out
            d[outlier_ind] = gdisp_out
            if debug:
                plt.subplot(1, 3, 2)
                plt.plot(bin_cen_t, inst_speed, color='r')
                plt.scatter(bin_cen_t[peak_ind],inst_speed[peak_ind],c='b',marker='*')
                plt.tight_layout()
                plt.title('After artifact correction')
                plt.ylabel('inst speed (mm/s)')
                plt.xlabel('Rel time (s)')
                plt.ylim(yl)
            
            # Trim away buffer & store
            rdata[iTrial]['inst_speed'] = inst_speed[nbp:-nbp]
            rdata[iTrial]['head_disp'] = inst_speed[nbp:-nbp]
        rdata[iTrial]['art_idx'] = art[nbp:-nbp]
    
    n_trials_corrected = len(trials_with_art)
    corrected_trials = trials_with_art
    
    if n_trials_corrected > 0:
        print('Trials (Python indices) for which artifacts corrected: ',
              trials_with_art)
    if debug:
        plt.figure()
        plt.subplot(1,3,1)       
        plt.subplot(1,3,3)
        plt.stem(qd)
    tot_art_dur = np.array(tot_art_dur)
    removed_trials = np.nonzero(np.array(qd) > args.art_trial_removal_th)[0]
    trial_dur = np.nansum(np.diff(rd['rel_mt']))
    tot_art_dur[removed_trials] = trial_dur
    n_full_trials_removed = removed_trials.size  
    if n_full_trials_removed > 0:
        print(f'bad whole trials: {removed_trials}')
    
    # Some corrected trials are removed. So we will adjust for that.
    corrected_trials = list(set(corrected_trials)-set(removed_trials))
    
    # if rdata[0]['session_start_time']==5315118162:
    #     rdpl.plot_head_mov_by_trial(rdata, args)
    #     plt.figure()
    #     rdpl.plot_head_mov_by_trial([rr for ii,rr in enumerate(rdata) if ii not in removed_trials], args)
    #     1
    return tot_art_dur, rdata, corrected_trials, list(removed_trials)


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
    good_trials_beh = np.full((tr_on.size,), False)

    # First get the behavior state segment on and off times
    if beh_state == 'awake':
        # st1, st2 = (cont.AwakeSegManual & key).fetch('seg_begin', 'seg_end')
        # st1 = st1[0][0]
        # st2 = st2[0][0]
        st1, st2 = (cont.SleepAuto & key).fetch1('sleep_trans_begin', 
                                                'sleep_trans_end')
        ephys_start, ephys_end = (acq.Ephys & key).fetch1(
            'ephys_start_time', 'ephys_stop_time')
        if np.all(np.isnan(st1)):
            st1,st2 = [[]],[[]]
        t1 = np.array(list(chain([ephys_start],st2[0])))
        t2 = np.array(list(chain(st1[0],[ephys_end])))
    elif beh_state == 'nrem':
        t1, t2 = (cont.SleepAuto & key).fetch('nrem_seg_begin', 'nrem_seg_end')
        # t1, t2 = (cont.NremSegManual & key).fetch('seg_begin','seg_end')
        t1 = t1[0][0]
        t2 = t2[0][0]

    elif beh_state == 'rem':
        t1, t2 = (cont.SleepAuto & key).fetch('rem_seg_begin', 'rem_seg_end')
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
    for i_train, (cton, ctoff) in enumerate(zip(tr_on, tr_off)):        
        # Take on and off times of each train. If train is inside any one of the NREM
        # segment, it's a good trial.       
        inseg = np.full(t1.size, False)
        for idx, (seg_t1, seg_t2) in enumerate(zip(t1,t2)):
            if (cton >= seg_t1) & (ctoff < seg_t2):
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
        if len(motion_det_win) == 4:
            sel_t = sel_t | (tt >= motion_det_win[2]) & (
                tt < motion_det_win[3])

        y = my[idx]
        # Get data for time windows
        dx = np.diff(x[sel_t])
        dy = np.diff(y[sel_t])
        r[idx] = np.mean(np.sqrt(dx**2 + dy**2))

    q = np.quantile(r, motion_quantile_to_keep)
    good_trials = r <= q

    return good_trials


def get_all_chan_mouse_rip_rate_matrix(mouse_group_data, normalize=True):
    """
    Inputs: mouse_group_data = collect_mouse_group_rip_data(data_sessions,args)
            normalize - Boolean, normalize rate by baseline mean rate (bins with bin_cen < 0)
    Outputs:
        rip_rate_matrix - 2d numpy array (nTotChanAllMice x nBins) of ripple rate
        bin_cen - 1d numpy array of bin centers (sec) relative to photostim onset
        n_chan - list (length = nMice) of number of channels in each mouse
    """
    all_rip_rate = []
    n_chan = []  # list of number of channels in each mouse
    for md in mouse_group_data:
        n_chan.append(len(md))
        for chd in md:
            args = chd['args']
            rip_rate, _, bin_cen = get_ripple_rate(
                chd['rdata'], args.bin_width, args.xmin, args.xmax)
            if normalize:
                rip_rate = rip_rate/np.mean(rip_rate[bin_cen < 0])
            all_rip_rate.append(rip_rate)
    rip_rate_matrix = np.vstack(all_rip_rate)
    return rip_rate_matrix, bin_cen, n_chan


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
        rel_change = 100 *\
            (np.abs(np.mean(minority_pulse_widths)-pulse_width)/pulse_width)
        if rel_change > 1:  # 1% tolerance
            print('%u pulses had widths different from the majority pulse width of %0.2f'
                  % (minority_pulse_count, pulse_width))
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

def get_processed_rip_data(keys, pulse_per_train, std, minwidth, args_in,debug=False):
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
            args_in -  Args object with updated params-values
            debug - boolean, for debugging artifact correction
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
    assert (type(keys) == list) and (type(std) == list) and (type(minwidth) == list), \
        'keys, std and minwidth must be lists'
    assert len(set([x['animal_id'] for x in keys])
               ) == 1, 'all keys must be from the same mouse'
    # Make sure chan_num exists in keys
    assert all(['chan_num' in kk for kk in keys]
               ), 'chan_num does not exist in given keys'
    assert len(set([x['chan_num'] for x in keys])
               ) == 1, 'all keys must have the same chan_num'
    # Ensure that each key has its own associated std and minwidth
    assert (len(keys) == len(std) == len(minwidth)), 'number of keys must match \
    number of std and minwidth'
    
    args = copy.deepcopy(args_in)
    # Create binning params
    pre = args.xmin * 1e6  # to microsec
    post = args.xmax * 1e6  # to microsec

    # Go through each channel key and collect ripple and motion data
    rdata = []
    all_pulse_width = []
    all_pulse_freq = []
    
    for ikey, key in enumerate(keys):
        ppt = pulse_per_train[ikey]
        # Get light pulse train info
        pon_times, poff_times, pulse_width, pulse_freq = get_light_pulse_train_info(
            key, ppt)
        all_pulse_width.append(pulse_width)
        all_pulse_freq.append(pulse_freq)
        # For plotting purpose, extract one pulse train on and off times in sec
        # We assume that all keys had the same stimulus train parameters, so we pick one key
        if ikey == 0:
            one_train_on = (pon_times[0:ppt] - pon_times[0]) * 1e-6
            one_train_off = (poff_times[0:ppt] - pon_times[0]) * 1e-6
        
        # Pick the first pulse times in the pulse train for time referencing
        tr_on = pon_times[::ppt]  # shape: (rows,), int64
        tr_off = poff_times[::ppt]  # shape: (rows,), int64
        
        args.n_trials_total = len(tr_on)

        # Get motion info from tracker data. tr_on is returned because it may be trimmed
        # inside the get_perievent_motion function for shorter trial length
        # Note - output of get_perievent_motion, mt, will be in seconds.
        same_len = True
        mt, mx, my, good_trials_len,v_t0 = gfun.get_perievent_motion(key, tr_on, args.xmin,
                                                                args.xmax, same_len)
        rel_mt = center_and_scale_time(mt, tr_on)

        # Mark (True/False) trials by their presence inside given behavioral state (e.g NREM)
        good_trials_beh = filter_trials_by_beh_state(tr_on, tr_off, 
                                                     args.beh_state, key)

        # Combine all selections and re-filter event times and trial data
        good_trials = good_trials_len & good_trials_beh
        tr_on = tr_on[good_trials]
        rel_mt = list(compress(rel_mt, good_trials))
        mx = list(compress(mx, good_trials))
        my = list(compress(my, good_trials))
        
        args.n_trials_in_beh_state = tr_on.size
        # Get ripple events
        fstr = f'std = {std[ikey]} and minwidth = {minwidth[ikey]}'
        print(fstr)
        rpt = np.array((ripples.RipEventsBs & key &
                        (ripples.DetParamsBs & fstr)).fetch('peak_t'))
        # Check if the RipEvents table was populated for sessions for nrem behavioral state
        if args.beh_state == 'nrem':
            assert rpt.size > 0, 'No ripples! Check if the RipEvents table is populated'

        # Go through each pulse train and collect ripples near by
        for t_idx, ct_on in enumerate(tr_on):
            twin = np.array([pre, post]) + ct_on
            # Pick ripples in window
            sel_rt = rpt[(rpt >= twin[0]) & (rpt < twin[1])]
            cmx = mx[t_idx]
            cmy = my[t_idx]
            head_disp = calc_displacement(cmx, cmy)
           
            # Center ripple event times also as above
            re_rel_t = (sel_rt - ct_on)*1e-6
            # We will put all trials of sessions (keys) together. That is why
            # append operation for rdata is inside this loop!
            rdata.append({'v_t0':v_t0,'train_onset':ct_on,'rel_mt': rel_mt[t_idx], 
                          'mx': mx[t_idx], 'my': my[t_idx],
                          'head_disp': head_disp, 'rip_evt': re_rel_t,
                          'session_start_time': key['session_start_time']})

    # Compute instantaneous speed, after correcting for artifacts
    for rd in rdata:
        inst_disp, fps = convert_motion_traj_to_inst_disp(rd['mx'], rd['my'])
        rd['inst_speed'] = inst_disp*fps

    # Remove high amplitude head disp artifacts
    if args.correct_high_amp_mov_art:
        # Correct artifactual segments of inst_speed. Note: we do not
        # correct artifactual segments of mx, my, or head_disp.
        art_dur, rdata,args.art_trials_corrected,args.art_trials_removed = \
        correct_abnorm_high_mov_artifacts(rdata, args,
                                art_peak_hw_ratio_th=args.art_peak_hw_ratio_th,
                                debug=debug)
        rdata = [rdata[i]
                 for i in np.nonzero(art_dur < args.max_art_duration)[0]]
    else: # Add dummy art_idx key to rdata dict to be consistent with above
        for rd in rdata:
            rd['art_idx'] = np.full(rd['inst_speed'].size, False)

    if not np.unique(all_pulse_freq).size == np.unique(all_pulse_width).size == 1:
        print('**************************************************************')
        print('pulse frequencies: ', all_pulse_freq)
        print('pulse widths: ', all_pulse_width)
        logging.warning('all keys must have the same pulse frequency and pulse widths\
                      make sure that the difference in values are acceptable')
        print('**************************************************************')

    args.one_train_on = one_train_on
    args.one_train_off = one_train_off   
    args.mouse_id = key['animal_id']
    # pulse_width, pulse_freq, pulse per train and session_ts may be updated 
    # by collect_mouse_group_rip_data function if it decides that some sessions 
    # are not good
    args.pulse_width = all_pulse_width
    args.pulse_freq = all_pulse_freq
    args.pulse_per_train = pulse_per_train
    args.session_ts = dju.get_sess_str_from_keys(keys)
    
    args.fps = fps  # frames per sec (video frame rate)
    args.minwidth = minwidth
    args.std = std
    args.chan_name = (cont.Chan & key & f'chan_num = {key["chan_num"]}').fetch(
        'chan_name')[0]
    
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
    bw = bin_width/1000
    bin_edges, bin_cen = utpy.create_psth_bins(xmin, xmax, bw)
    # Get histogram counts
    counts, _ = np.histogram(evt, bin_edges)
    n_trials = len(rdata)
    rip_rate = (counts/n_trials)/bw  # rip/sec

    return rip_rate, bin_edges, bin_cen


def pool_head_mov_across_mice(group_data, within_mouse_operator):
    """
    Pool head displacement across mice. We will not normalize within each mouse
    Inputs:
        group_data - list (mice) of list(channels) of dict (ripple data), this
                    is an output from collect_mouse_group_rip_data(...) function call.
        within_mouse_operator - str, should be 'mean' or 'median' - tells you if mean
                              or median is computed across trials within a mouse
    Outputs:
        t_bin_cen_vec - 1D numpy array of bin center times(s) relative to
                        stimulus onset
        all_rr - 2D numpy array, nMice-by-nTimeBins of head inst speed
                (mean or median)

    MS 2022-03-14/2024-08-08

    """
    all_rr_list = []
    tm_bin_cen_list = []
    for md in group_data:  # loop over mice
        # For each mouse pick head disp data
        crdata = copy.deepcopy(md[0]['rdata'])
        
        # Compute head displacement  or instantaneuous speed
        # using movement info from two adjacent video frames
        # Create time bin centers
        
        t_bin_cen_list = []
        for cr in crdata:
            rt = cr['rel_mt']
            ifi = np.nanmedian(np.diff(rt))  # inter-frame-interval
            tbc = rt[0:-1]+ifi/2
            t_bin_cen_list.append(tbc)
        
        # vx_list = [cr['mx'] for cr in crdata]
        # vy_list = [cr['my'] for cr in crdata]
        ins_list = [cr['inst_speed'] for cr in crdata]        

        # # Use interpolation and compute a common time vector for all stimulation
        # # trials, and the x and y coordinates corresponding to the common time points.
        # t_bin_cen, mxi_list = gfun.interp_based_event_trig_data_average(
        #     t_bin_cen_list, vx_list)
        # _, myi_list = gfun.interp_based_event_trig_data_average(
        #     t_bin_cen_list, vy_list)
        t_bin_cen, ins_speedi_list = gfun.sample_event_trig_data_evenly(
            t_bin_cen_list, ins_list)
        
        tm_bin_cen_list.append(t_bin_cen)
        # # Compute head displacement in mm
        # d_array = [convert_motion_traj_to_inst_disp(px, py)[0]
        #            for px, py in zip(mxi_list, myi_list)]
        # match metric:
        #     case 'inst_speed':              
        mi_list = ins_speedi_list # mm/sec
            # case 'disp':
            #     mi_list = d_array  # mm
            # case _:
            #     raise ValueError('metric must be either inst_speed or disp')

        # Pool data within mouse:
        # first change into 2D array where rows are trials
        mi_array = np.stack(mi_list, axis=0)
        if within_mouse_operator == 'mean':
            mi_cen = np.nanmean(mi_array, axis=0)
        elif within_mouse_operator == 'median':
            mi_cen = np.nanmedian(mi_array, axis=0)
        else:
            raise ValueError(
                'within_mouse_operator should be "mean" or "median"')
        all_rr_list.append(mi_cen)

    # Make sure all mice have the same time base
    t_bin_cen, all_rri_list = gfun.sample_event_trig_data_evenly(
        tm_bin_cen_list, all_rr_list)  
    all_rr = np.array(all_rri_list)

    return t_bin_cen, all_rr

def get_motion_vs_rip_rate_corr_data(group_data, stats_data=None, 
                                                 rel_time_win=None, qval=1.00,
                                                 normalize_motion=False):
    """ For assessing the correlation between head movement speed and ripple rate
    at the trial by trial level, we will create a dataframe with relevant info
    Inputs:
        group_data - output from collapse_rip_events_across_chan_for_each_trial(...) 
        stats_data - output from create_data.pute_and_save_statistics(...). Not
                    needed if rel_time_win is specified (i.e., not set to None)
        rel_time_win - 2 element list; window start and end times (s) relative
                        to stimulus onset; if None, the time window during which
                        significant ripple modulation was found
        q_val - upper threshold for instantaneuous speed for outlier removal
        normalize_motion - if True, normalize (%) post-stim motion by t
                        rial-averaged baseline motion
    Outputs:
        df - dataframe; data in long format; each row is a trial; columns are
            mouse_id, trial_num, inst_speed, and rip_rate
    """
    
    if rel_time_win is None:
        smt = stats_data['sig_mod_times']
        rel_time_win = [min(smt), max(smt)];  
     
    animal_id, inst_speed, rip_rate, rip_cnt, baseline_rate = [],[],[],[],[]
    mouse_baseline_inst_speed = []
    trial_num = [] # For each mouse, give sequential numbers starting from 0 to
    # be able to use mouse-specific random offsets in mixed model analysis
    for mouse_num, md in enumerate(group_data): # mouse
        bw = md['args'].bin_width # msec
        rip_bin_cen_t = md['bin_cen_t'] # for ripple counts
        iTrial = 0
        baseline_inst_speed = []
        imouse_inst_speed = []
        for td in md['trial_data']:
            # Convert to ripples/sec
            rr = td['rip_cnt']*(1000/bw) # Used 1000 ms because bw is in ms
            # Skip trial if no ripples were found in the baseline period
            if ~np.all(rr[rip_bin_cen_t < 0]==0):
                trial_num.append(iTrial)
                iTrial += 1
                b_rate = np.mean(rr[rip_bin_cen_t < 0])
                # motion bin center times
                mbw = np.diff(td['rel_mt'][0:2]) # in sec
                mov_bin_cen = td['rel_mt'][0:-1]+mbw/2
                sel_mv_t = (mov_bin_cen >= rel_time_win[0]) & \
                                        (mov_bin_cen <= rel_time_win[1])
                sel_inst_speed = td['inst_speed'][sel_mv_t] # mm/s
                mean_inst_speed = np.mean(sel_inst_speed)
                # Accumulate baseline inst speed to compute an overall average
                # for normalizing post-stim motion later
                baseline_inst_speed.append(np.mean(td['inst_speed'][mov_bin_cen < 0]))
                # Accumulate animal id
                animal_id.append(md['animal_id'])
                imouse_inst_speed.append(mean_inst_speed)
                sel_rip_t = (rip_bin_cen_t >= rel_time_win[0]) & \
                                           (rip_bin_cen_t <= rel_time_win[1])
                sel_rip_rate = rr[sel_rip_t]
                norm_rip_rate = 100 * np.mean(sel_rip_rate)/b_rate
                rip_rate.append(norm_rip_rate)
                baseline_rate.append(b_rate)
                rip_cnt.append(np.sum(td['rip_cnt'][sel_rip_t]))
        # Average across all trials and and time period
        mean_baseline_inst_speed = np.mean(baseline_inst_speed)
        mouse_baseline_inst_speed.append(mean_baseline_inst_speed)
        if normalize_motion:
            imouse_inst_speed = [100*x/mean_baseline_inst_speed for x in imouse_inst_speed]
        inst_speed.append(imouse_inst_speed)
        
        print(f'Done with mouse {mouse_num}')
        # Convert to data frame
    baseline_mov_df = pd.DataFrame(dict(animal_id=[md['animal_id'] for md in group_data],
                                        baseline_inst_speed=mouse_baseline_inst_speed))
    # Remove outliers
    animal_id = np.array(animal_id)
    inst_speed = np.array(list(chain(*inst_speed)))
    rip_rate = np.array(rip_rate)
    baseline_rate = np.array(baseline_rate)
    rip_cnt = np.array(rip_cnt)
    trial_num = np.array(trial_num)
    
    qx = np.quantile(inst_speed, qval)
    qy = np.quantile(rip_rate, qval)
    
    sel_x = inst_speed < qx
    sel_y = rip_rate < qy
    sel = sel_x & sel_y    
    
    df = pd.DataFrame(dict(animal_id=animal_id[sel], 
                           trial_num = trial_num[sel],
                           inst_speed=inst_speed[sel],
                           rip_rate=rip_rate[sel], 
                           baseline_rate=baseline_rate[sel],
                           rip_count=rip_cnt[sel]))
    
    return df, baseline_mov_df