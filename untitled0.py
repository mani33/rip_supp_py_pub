# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 11:10:21 2023

@author: maniv
"""

import pandas as pd
import rip_data_processing as rdp
import util_py as utpy
import ripple_stat as rstat
import os

parent_dir = r"C:\Users\maniv\Documents\ripples_manuscript\data\vglut2"
if not os.path.exists(parent_dir):
    os.mkdir(parent_dir)
    
#%% -----
group_data = utpy.get_pickled_data(os.path.join(parent_dir,'vglut2_5ms_grdata_stat.pkl'))
elec_sel_meth = 'avg'
stat_test = 't'
nBoot = 10
#%% -----
smt = rstat.get_sig_mod_times_for_each_mouse(group_data,elec_sel_meth,stat_test,nBoot)