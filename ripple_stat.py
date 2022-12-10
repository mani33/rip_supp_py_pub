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
        self.nBoot = 10
        self.alpha = 0.05/2 # two sided test
        self.test = 't' # 't' or 'ranksum'
#def get_sig_mod_times(bin_cen_t,rdata,)

def figure_1_stat():
    # Read the first three rows of wild-type sheet in the excel sheet
    fn = r"C:\Users\maniv\Documents\upenn_projects\ripple_suppression\docs\ripp_supp_rec_sessions.xlsx"
    dd = pd.read_excel(fn,'wild_type')
    # Slice the dataframe to include only the first three sessions
    dd = dd.iloc[[0,1,2],:]
    beh_state = 'nrem'
    xmin = -5
    xmax = 5
    bin_width = 200
    elec_sel_meth = 'avg'
    stat_test = 't'
    nBoot = 10
    group_data = rdp.collect_mouse_group_rip_data(dd, beh_state, xmin, xmax, 
                            bin_width)
    gdata = rdp.collapse_rip_events_across_chan_for_each_trial(group_data,elec_sel_meth)
    # cgroup_data: list(of length nMice) of dict('animal_id','args','bin_cen_t','trial_data') 
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
        smt = cmt.cluster_mass_test(md['bin_cen_t'],trial_rip_cnt,stat_test,nBoot=nBoot)
        for ci in smt:
            print(md['bin_cen_t'][ci])
        sig_mod_times.append(smt)
    return sig_mod_times
