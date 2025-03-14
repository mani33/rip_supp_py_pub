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
        pval -  dict(animal_id:1d array): p-values of significantly modulated clusters. 
                Note that the size of this array will not generally match that of 
                sig_mod_times since each cluster could contain more than one
                data point.
    """
    #--------------------------------------------------------------------------
    # Find signicantly modulated times for each mouse
    grdata = rdp.collapse_rip_events_across_chan_for_each_trial(group_data,
                                                                elec_sel_meth)
    # Go through each mouse and get the times of significant modulation
    sig_mod_times_imouse = {} # for individual mouse statistics 
    pval = {} # p-values of significantly modulated clusters
    for iMouse,md in enumerate(grdata):
        mouse_id = md['animal_id']
        # Trim the right side end to match the length of pre-stim time
        sel_t_bins = md['bin_cen_t'] <= np.abs(np.min(md['bin_cen_t']))        
        match data_type:
            case 'ripples':
                # catenate trial data into a matrix               
                data = np.array([td['rip_cnt'][sel_t_bins] for td in md['trial_data']])
                bin_cen_t = md['bin_cen_t'][sel_t_bins]
            case 'inst_speed':                
                data, bin_cen_t = compute_across_chan_pooled_inst_speed(md)
        # Get significant modulation times
        smi, pval_i = cmt.cluster_mass_test(bin_cen_t,data,stat_test,
                                    nBoot=nBoot)    
        # smi: 1d numpy array of bin_cen_t-indices of clusters that are 
        # significantly modulated        
        smt = bin_cen_t[smi]
        sig_mod_times_imouse.update({mouse_id:smt})   
        pval.update({mouse_id:pval_i})
    return sig_mod_times_imouse, pval

def compute_across_chan_pooled_inst_speed(mdata):
    """ Pool all instantaneous speed data from all unique trials from all recorded
      sessions of a given mouse
      Inputs:
        mdata - one element (one mouse) of the output list from the function call - 
        rdp.collapse_rip_events_across_chan_for_each_trial(group_data,elec_sel_meth)
      Outputs:
          data - 2d numpy array (nTrials-by-nTime points) inst speed data
    """
          
    """ Each trial will likely have different rel_mt as the frame
    times will not be guaranteed to be identical for all trials.
    Therefore, to keep a common time vector for all trials, and to 
    have equal number of time bins before and after stimulus onset,
    we will do the following:
    """
    # 1. Determine the latest minimum and the earliest maximum time 
    # across all trials
    tmin = np.max([x['rel_mt'][0] for x in mdata['trial_data']])              
    # 2. Pick the smaller of the tmin abs values
    tlen = np.abs(tmin)
    # 3. Create time vector
    step = 1/mdata['args'].fps
    bin_cen_t = create_symmetric_bin_center_t(tlen,step)
    # 4. We will pick inst_speed data corresponding to these points
    # from each trial using interpolation
    inst_speed = []
    for d in mdata['trial_data']:
        # Create bin centers for each trial
        bw = np.median(np.diff(d['rel_mt']))
        # The actual times matching inst_speed in this trial
        bct = d['rel_mt'][0:-1]+bw/2                    
        # Interpolate to get inst_speed at our desired time points
        inst_speed.append(np.interp(bin_cen_t,bct,d['inst_speed']))                    
    # Change list into 2d array (nTrials-by-nTime points)
    data = np.array(inst_speed)
    
    return data, bin_cen_t
       
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
        pval -  1d array of p-values of significantly modulated clusters. Note
                that the size of this array will not generally match that of 
                sig_mod_times since each cluster could contain more than one
                data point.
    """

    # Perform statistical test on mouse-level, meaning, for each mouse, we first
    # average across trials, and then we do statistics by swapping baseline and
    # post-stim periods for each mouse during bootstrapping
    match data_type:
        case 'ripples':
            bin_cen_t,_,_,mdata = rdp.average_rip_rate_across_mice(group_data,
                                                                elec_sel_meth)
            # Trim the right side end to match the length of pre-stim time
            sel_t_bins = bin_cen_t <= np.abs(np.min(bin_cen_t))
            bin_cen_t = bin_cen_t[sel_t_bins]
            mdata = mdata[:, sel_t_bins]
        case 'inst_speed':
            grdata = rdp.collapse_rip_events_across_chan_for_each_trial(group_data,
                                                               elec_sel_meth)
            mdata = []
            bct = []
            for iMouse, md in enumerate(grdata):
                data, bin_cen_t = compute_across_chan_pooled_inst_speed(md) # 2d np array: (nTrials, nTime points)
                bct.append(bin_cen_t)
                mdata.append(np.mean(data, axis=0))            
            bin_cen_t = np.median(np.array(bct), axis=0)
            mdata = np.array(mdata) # nMice-by-nTimeBins
            
    # mdata is 2D numpy array, nMice-by-nTimeBins of ripple rates or inst_speed
    # bin_cen_t - 1D numpy array of bin-center time in sec.
    smi, pval = cmt.cluster_mass_test(bin_cen_t,mdata,stat_test,nBoot=nBoot)
    # smi: 1d numpy array of bin_cen_t-indices of clusters that are 
    # significantly modulated
    sig_mod_times = bin_cen_t[smi]
    
    return sig_mod_times, pval

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