# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 18:00:49 2022
Statistics for the ripple suppression paper
Mani Subramaniyan
"""
import cluster_based_nonparam_test as cmt
#import pandas as pd
import rip_data_processing as rdp
import numpy as np
# For Figure 1: Is the effect significant in individual wild type mouse?

# Common parameters for statistics
class Args():
    def __init__(self):
        self.nBoot = 20
        self.alpha = 0.05/2 # two sided test        

def get_sig_mod_times_for_each_mouse(group_data,stat_test,nBoot,
                                     data_type,elec_sel_meth=None):
    """ For each mouse, compute significance of modulation at each time point 
    after light pulse onset.
    Inputs:        
        group_data = rdp.collect_mouse_group_rip_data(data_sessions,args)
        where rdp is ripple data processing module        
        stat_test - 't' or 'ranksum'
        nBoot - number of replicates for bootstrapping
        data_type - string; data type ('ripples' or 'inst_speed') for which
                    statistics need to be computed
        elec_sel_meth - 'avg' or 'random'; default is None; irrelevant for
                        data_type 'inst_speed'
    Outputs:
        sig_mod_times_imouse - dict(animal_id:1d numpy array)-significant modulation 
                                times for each mouse
    """
    #--------------------------------------------------------------------------
    # Find signicantly modulated times for each mouse
    match data_type:
        case 'ripples':
            grdata = rdp.collapse_rip_events_across_chan_for_each_trial(
                                                    group_data,elec_sel_meth)    
            # grdata: list(of length nMice) of dict('animal_id','args','bin_cen_t','trial_data') 
            # where trial_data is a list (of length nTrials) of 
            # dict('rel_mt','mx','my','head_disp','rip_cnt')
        case 'inst_speed':
            # All channels will have the same movement data. So we will pick the
            # motion data from the first channel
            grdata = [md[0] for md in group_data]
    
    # Go through each mouse and get the times of significant modulation
    sig_mod_times_imouse = {} # for individual mouse statistics  
    for iMouse,md in enumerate(grdata):
        mouse_id = md['animal_id']
        args = md['args']
        match data_type:
            case 'ripples':
                # catenate trial data into a matrix
                nTrials = len(md['trial_data'])
                nBins = md['bin_cen_t'].size
                trial_rip_cnt = np.zeros((nTrials,nBins))
                for iTrial,td in enumerate(md['trial_data']):
                    trial_rip_cnt[iTrial,:] = td['rip_cnt']
                data = np.array([td['rip_cnt'] for td in md['trial_data']])
                bin_cen_t = md['bin_cen_t']
            case 'inst_speed':                
                """ Each trial will likely have different rel_mt as the frame
                times will not be guaranteed to be identical for all trials.
                Therefore, to keep a common time vector for all trials, and to 
                have equal number of time bins before and after stimulus onset,
                we will do the following:
                """
                # 1. Determine the latest minimum and the earliest maximum time 
                # across all trials
                tmin = np.max([x['rel_mt'][0] for x in md['rdata']])
                tmax = np.min([x['rel_mt'][-1] for x in md['rdata']])
                assert (-args.xmin)==args.xmax, 'xmin and xmax must be the same length'
                # 2. Pick the smaller of the tmin and tmax abs values
                tlen = np.min(np.abs([tmin,tmax]))
                # 3. Create time vector
                step = 1/args.fps
                bin_cen_t = create_symmetric_bin_center_t(tlen,step)
                # 4. We will pick inst_speed data corresponding to these points
                # from each trial using interpolation
                inst_speed = []
                for d in md['rdata']:
                    # Create bin centers for each trial
                    bw = np.median(np.diff(d['rel_mt']))
                    # The actual times matching inst_speed in this trial
                    bct = d['rel_mt'][0:-1]+bw/2                    
                    # Interpolate to get inst_speed at our desired time points
                    inst_speed.append(np.interp(bin_cen_t,bct,d['inst_speed']))                    
                # Change list into 2d array (nTrials-by-nTime points)
                data = np.array(inst_speed)        
        # Get significant modulation times
        smi = cmt.cluster_mass_test(bin_cen_t,data,stat_test,
                                    nBoot=nBoot)    
        # smi: 1d numpy array of bin_cen_t-indices of clusters that are 
        # significantly modulated        
        smt = bin_cen_t[smi]
        sig_mod_times_imouse.update({mouse_id:smt})   
    
    return sig_mod_times_imouse

def get_sig_mod_times_for_mouse_population(group_data,stat_test,
                                           nBoot,data_type,elec_sel_meth=None):
    """ For the entire mouse population, compute significance of modulation at 
    each time point after light pulse onset.
    Inputs:        
        group_data = rdp.collect_mouse_group_rip_data(data_sessions,args)
        where rdp is ripple data processing module        
        stat_test - 't' or 'ranksum'
        nBoot - number of replicates for bootstrapping
        data_type - string; data type ('ripples' or 'inst_speed') for which
                    statistics need to be computed
        elec_sel_meth - 'avg' or 'random'; default is None; irrelevant for
                        data_type 'inst_speed'
    Outputs:        
        sig_mod_times - 1d numpy array of bin_cen_t of clusters that are 
                        significantly modulated when doing mouse-level statistics
    """

    # Perform statistical test on mouse-level, meaning, for each mouse, we first
    # average across trials, and then we do statistics by swapping baseline and
    # post-stim periods for each mouse during bootstrapping
    match data_type:
        case 'ripples':
            bin_cen_t,_,_,mdata = rdp.average_rip_rate_across_mice(group_data,
                                                                elec_sel_meth)
        case 'inst_speed':
            # 1. Create time vector: determine the latest minimum and the 
            # earliest maximum time across all trials and all mice
            t1 = [[td['rel_mt'][0] for td in gd[0]['rdata']] for gd in group_data]
            t1 = np.max(np.concatenate(t1))
            t2 = [[td['rel_mt'][-1] for td in gd[0]['rdata']] for gd in group_data]
            t2 = np.min(np.concatenate(t2))
            tlen = np.min(np.abs([t1,t2]))
            args = group_data[0][0]['args']
            assert (-args.xmin)==args.xmax, 'xmin and xmax must be the same length'
            step = 1/args.fps
            bin_cen_t = create_symmetric_bin_center_t(tlen,step)
            
            # 2. Averaging across trials of a mouse: In each trial of a mouse, 
            # we will pick inst_speed data corresponding to these points from 
            # each trial using interpolation. Then we will average across 
            # trials in each mouse.
            inst_speed = []
            for md in group_data: # each mouse
                inst_speed_imouse = []
                # Will use first channel data as motion is same for all channels
                for d in md[0]['rdata']: # each trial
                    # Create bin centers for each trial
                    bw = np.median(np.diff(d['rel_mt']))
                    # The actual times matching inst_speed in this trial
                    bct = d['rel_mt'][0:-1]+bw/2                    
                    # Interpolate to get inst_speed at our desired time points
                    inst_speed_imouse.append(np.interp(bin_cen_t,bct,d['inst_speed']))
                # Average across trials in each mouse
                inst_speed.append(np.mean(np.array(inst_speed_imouse),axis=0))
            mdata = np.array(inst_speed) # nMice-by-nTimeBins
            
    # mdata is 2D numpy array, nMice-by-nTimeBins of ripple rates or inst_speed
    # bin_cen_t - 1D numpy array of bin-center time in sec.
    smi = cmt.cluster_mass_test(bin_cen_t,mdata,stat_test,nBoot=nBoot)
    # smi: 1d numpy array of bin_cen_t-indices of clusters that are 
    # significantly modulated
    sig_mod_times = bin_cen_t[smi]
    
    return sig_mod_times

def create_symmetric_bin_center_t(t_len,bin_width):
    # Create a bin center time vector containing equal number of positive and 
    # negative time bins. Because of this requirement, the time vector will not 
    # have 0.
    # 
    # MS 2024-08-11
    
    a = np.arange(0,t_len+2*bin_width,bin_width)
    a = a[(a>0) & (a<=t_len)]-bin_width/2
    bin_cen_t = np.concatenate((-np.flip(a),a))
    
    return bin_cen_t