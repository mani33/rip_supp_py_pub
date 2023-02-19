# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 08:04:12 2022

@author: Mani
"""

#------------------------------------------------------------------------------ 
def get_fontsize_for_string(str, str_len_in_inches):
    """This function will calculate the fontsize suitable for fitting the input 
    string 'str' within the length given by 'str_len_in_inches'. Useful for deciding
    fontsize when you know the figure size and want to know the appropriate fontsize
    for say the xlabel so that it nicely fits in
    
    We use the fact that a fontsize unit (point) of 1 = 1/72 inches"""
    
    fsize = str_len_in_inches * 72.0/len(str)
    return fsize

#------------------------------------------------------------------------------
def boxoff(ax):
    # Remove the top and right spines of the given single axes (ax)
    # MS 2022-01-29
    
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    
#------------------------------------------------------------------------------    
def boxon(ax):
    # Add the top and right spines of the given single axes (ax)
    # MS 2022-01-29
    # Get color from bottom axis
    c = ax.spines['bottom'].get_edgecolor()
    ax.spines['top'].set_color(c)
    ax.spines['right'].set_color(c)
    
#------------------------------------------------------------------------------    
def set_spine_linewidth(ax, linewidth):
    # Sets the spine linewidth for the given single axes (ax)
    # MS 2022-01-29
    # change all spines
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(linewidth)
        
#------------------------------------------------------------------------------          
def set_axis_linewidth(ax, linewidth):
    # Set linewidth for the spine and tick for the given single axes (ax)
    # MS 2022-01-29
    # First set spine linewidth
    set_spine_linewidth(ax, linewidth)     
    ax.tick_params(width = linewidth)
    