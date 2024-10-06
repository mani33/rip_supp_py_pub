# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 09:08:40 2024

@author: maniv
"""
import rip_data_processing as rdp
import rip_data_plotting as rdpl
import djutils as dju
import matplotlib.pyplot as plt
# %%
plt.close('all')
sess_str = '2021-06-22_10-37-51'
chan = 3
key = dju.get_key_from_session_ts(sess_str)[0]
key['chan_num'] = chan
pulse_per_train = [50]

xmin = -15
xmax = 20
args = rdp.Args()
args.xmin = xmin
args.xmax = xmax
args.max_art_duration = 1
std, minwidth = [8], [30]

rdata, args = rdp.get_processed_rip_data(
    [key], pulse_per_train, std, minwidth, args, debug=False)
args.laser_color = 'b'
args.sess_str = [sess_str]
args.std = std
args.chan_num = chan
args.minwidth = minwidth
args.title = ''

args.pulse_per_train = args.pulse_per_train[0]
rdpl.plot_lightpulse_ripple_modulation(rdata, args)
