# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 18:00:49 2022
Statistics for the ripple suppression paper
Mani Subramaniyan
"""
import cluster_based_nonparam_test as cmt
import pandas as pd
import rip_data_processing as rdp
import numpy as np
# For Figure 1: Is the effect significant in individual wild type mouse?

# Common parameters for statistics
class Args():
    def __init__(self):
        self.nBoot = 2000
        self.alpha = 0.05/2 # two sided test        

def get_figure_1_stat(group_data):
    """ Perform statistics on indivisual mouse ripple suppression data.
    Inputs:        
        group_data = rdp.collect_mouse_group_rip_data(dd, beh_state, xmin, xmax,
                                args.bin_width)
        where rdp is ripple data processing module
    Outputs:
        sig_mod_times - list (length of nMice) of 1d numpy array of bin_cen_t 
                        of clusters that are significantly modulated
        nBoot - number of bootstrap simulations used
    """
    # Read the first three rows of wild-type sheet in the excel sheet
    fn = r"C:\Users\maniv\Documents\upenn_projects\ripple_suppression\docs\ripp_supp_rec_sessions.xlsx"
    dd = pd.read_excel(fn,'wild_type')
    # Slice the dataframe to include only the first three sessions
    dd = dd.iloc[[0,1,2],:]    
    args = Args()  
    elec_sel_meth = 'avg'
    stat_test = 't'
    gdata = rdp.collapse_rip_events_across_chan_for_each_trial(group_data,elec_sel_meth)
    # gdata: list(of length nMice) of dict('animal_id','args','bin_cen_t','trial_data') 
    # where trial_data is a list (of length nTrials) of 
    # dict('rel_mt','mx','my','head_disp','rip_cnt')
    
    # Go through each mouse and get the times of significant modulation
    sig_mod_times = []
    for iMouse,md in enumerate(gdata):
        # catenate trial data into a matrix
        nTrials = len(md['trial_data'])
        nBins = md['bin_cen_t'].size
        trial_rip_cnt = np.zeros((nTrials,nBins))
        for iTrial,td in enumerate(md['trial_data']):
            trial_rip_cnt[iTrial,:] = td['rip_cnt']
        # Get significant modulation times
        smt = cmt.cluster_mass_test(md['bin_cen_t'],trial_rip_cnt,stat_test,
                                    nBoot=args.nBoot)
        for ci in smt:           
            sig_mod_times.append(md['bin_cen_t'][ci])
    return sig_mod_times,args.nBoot
