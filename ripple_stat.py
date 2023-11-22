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

def get_sig_mod_times_for_each_mouse(group_data,elec_sel_meth,stat_test,nBoot):
    """ For each mouse, compute significance of modulation at each time point 
    after light pulse onset.
    Inputs:        
        group_data = rdp.collect_mouse_group_rip_data(data_sessions,args)
        where rdp is ripple data processing module
        elec_sel_meth - 'avg' or 'random'
        stat_test - 't' or 'ranksum'
        nBoot - number of replicates for bootstrapping
    Outputs:
        sig_mod_times_imouse - dict(animal_id:1d numpy array)-significant modulation 
                                times for each mouse
    """

    #--------------------------------------------------------------------------
    # Find signicantly modulated times for each mouse
    gdata = rdp.collapse_rip_events_across_chan_for_each_trial(group_data,elec_sel_meth)    
    # gdata: list(of length nMice) of dict('animal_id','args','bin_cen_t','trial_data') 
    # where trial_data is a list (of length nTrials) of 
    # dict('rel_mt','mx','my','head_disp','rip_cnt')
    
    # Go through each mouse and get the times of significant modulation
    sig_mod_times_imouse = {} # for individual mouse statistics  
    for iMouse,md in enumerate(gdata):
        mouse_id = md['animal_id']
        # catenate trial data into a matrix
        nTrials = len(md['trial_data'])
        nBins = md['bin_cen_t'].size
        trial_rip_cnt = np.zeros((nTrials,nBins))
        for iTrial,td in enumerate(md['trial_data']):
            trial_rip_cnt[iTrial,:] = td['rip_cnt']
        # Get significant modulation times
        smi = cmt.cluster_mass_test(md['bin_cen_t'],trial_rip_cnt,stat_test,
                                    nBoot=nBoot)        
        # smi: 1d numpy array of bin_cen_t-indices of clusters that are 
        # significantly modulated        
        smt = md['bin_cen_t'][smi]
        sig_mod_times_imouse.update({mouse_id:smt})   
    
    return sig_mod_times_imouse

def get_sig_mod_times_for_mouse_population(group_data,elec_sel_meth,stat_test,nBoot):
    """ For the entire mouse population, compute significance of modulation at 
    each time point after light pulse onset.
    Inputs:        
        group_data = rdp.collect_mouse_group_rip_data(data_sessions,args)
        where rdp is ripple data processing module
        elec_sel_meth - 'avg' or 'random'
        stat_test - 't' or 'ranksum'
        nBoot - number of replicates for bootstrapping
    Outputs:        
        sig_mod_times - 1d numpy array of bin_cen_t of clusters that are 
                        significantly modulated when doing mouse-level statistics
    """

    # Perform statistical test on mouse-level, meaning, for each mouse, we first
    # average across trials, and then we do statistics by swapping baseline and
    # post-stim periods for each mouse during bootstrapping
    bin_cen_t,_,_,mdata = rdp.average_rip_rate_across_mice(group_data,elec_sel_meth)
    # mdata is 2D numpy array, nMice-by-nTimeBins of ripple rates
    # bin_cen_t - 1D numpy array of bin-center time in sec.
    smi = cmt.cluster_mass_test(bin_cen_t,mdata,stat_test,nBoot=nBoot)
    # smi: 1d numpy array of bin_cen_t-indices of clusters that are 
    # significantly modulated
    sig_mod_times = bin_cen_t[smi]
    
    return sig_mod_times