# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 16:32:21 2022

@author: Mani Subramaniyan

Utilities of general nature.


"""

def munique(A):
    """
    Finds unique values and returns indices of first occurrences and all other 
    occurrences similar to MATLAB such that 
    "[C,IA,IC] = UNIQUE(A) also returns index vectors IA and IC such that
    C = A(IA) and A = C(IC)""
    
    Inputs:
        A: list or 1D numpy array
    Returns:
        C: list or 1D numpy array of unique values
        IA: list or 1D np array of of Indices of values in C such that C = A(IA)
        IC: list or 1D np array of of Indices of values in A such that A = C(IC)
        
        
    """
    
