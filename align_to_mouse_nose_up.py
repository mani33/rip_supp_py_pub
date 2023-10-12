# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 12:06:25 2023

@author: maniv
"""
import numpy as np
# import time
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
import cv2
#%%
v = VideoFileClip(r'D:\ephys\raw\2022-02-21_12-28-33\VT1.mpg')
# v = VideoFileClip(r'D:\ephys\raw\2022-04-19_10-25-43\VT1.mpg')
T = v.reader.nframes/v.reader.fps
frame_size = v.reader.size
max_x,max_y = frame_size[0],frame_size[1]
cx,cy = frame_size[0]/2,frame_size[1]/2
inter_blob_dist_th = 15
print('Video minutes: ',T/60)
plt.close('all')
es = 10
plt.rcParams['image.origin']='upper'
for f in np.arange(0.3,1,0.3):
    # frame = cv2.flip(cv2.cvtColor(v.get_frame(T*f),cv2.COLOR_BGR2GRAY),0)
    frame = cv2.cvtColor(v.get_frame(T*f),cv2.COLOR_BGR2GRAY)
    plt.figure(dpi=300)
    gs = cv2.medianBlur(frame, 5)
    plt.subplot(1,3,1)
    plt.imshow(frame,cmap='gray')
    plt.title(np.round(f,3))  
    
    gs = cv2.medianBlur(frame, 25)
    # threshold image
    th = np.quantile(np.ravel(gs),0.025)
    mouse_pix_rc = np.nonzero(gs[10:-10,10:-10]<th)
  
    # Centroid
    mouse_cen_rc = np.array([[np.median(x)] for x in mouse_pix_rc]).astype(int)
    mouse_cen_xy = np.flipud(mouse_cen_rc)
    plt.plot(mouse_cen_xy[0],mouse_cen_xy[1],color='r',marker='o',markersize=2)
    
    # Now get the LED  
    gs = cv2.medianBlur(frame,5)
    th = np.min([np.quantile(np.ravel(gs),0.9995),253])
    gs = cv2.threshold(gs,th,255,cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    gs = cv2.erode(gs,kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    gs = cv2.dilate(gs,kernel)
    
    
    gs[:es,:] = 0
    gs[-es:,:] = 0
    gs[:,:es] = 0
    gs[:,-es:] = 0
    plt.subplot(1,3,2)
    plt.imshow(gs,cmap='gray')
    
    led_pix_rc = np.nonzero(gs)
    led_pix_x,led_pix_y = led_pix_rc[1],led_pix_rc[0]
    for _ in range(2):
        led_cen,ind = k_means(np.vstack((led_pix_x,led_pix_y)).T,2)[0:2]
        # Check if single or two blobs 
        if np.any(np.abs(np.diff(led_cen,axis=0)) > inter_blob_dist_th):
            # # Compute distance to edges find the blob that is closer to any one
            # # of the 4 edges and remove it
            # dist_to_edge = []
            blob_size = []
            for iBlob,xy in enumerate(led_cen):
               x,y = xy[0],xy[1]
               # dist_to_edge.append(np.min(np.abs([x,max_x-x,y,max_y-y])))
               blob_size.append(np.count_nonzero(ind==iBlob))
            # edge_blob_ind = np.argmin(dist_to_edge)
            bad_blob_ind = np.argmin(blob_size)
            # Remove edge blob's spots in the image
            sel_r = led_pix_rc[0][np.nonzero(ind==bad_blob_ind)]
            sel_c = led_pix_rc[1][np.nonzero(ind==bad_blob_ind)]  
            
            gs[sel_r,sel_c] = 0
            rc = np.nonzero(gs)
  
    plt.subplot(1,3,3)
    plt.imshow(gs,cmap='gray')
    
    # LED pixels after cleaning up
    led_pix_rc = np.nonzero(gs)  
    led_cen_rc = np.array([[np.median(x)] for x in led_pix_rc])
    led_cen_xy = np.flipud(led_cen_rc)
    led_pix_x,led_pix_y = led_pix_rc[1],led_pix_rc[0]
    
    cvm = np.cov(led_pix_x,led_pix_y)
    eg_val,eg_vec = np.linalg.eig(cvm)
    e1,e2 = eg_vec[:,0],eg_vec[:,1]
    
    s1 = eg_val[0]/np.sum(eg_val)
    s2 = eg_val[1]/np.sum(eg_val)
    
    e1p,e2p = e1*s1*150,e2*s2*150 # scale the eigen vectors for plotting
    
    plt.subplot(1,3,1)
    plt.plot([0,e1p[0]]+led_cen_xy[0],[0,e1p[1]]+led_cen_xy[1],color='r',linewidth=0.5)
    plt.plot([0,e2p[0]]+led_cen_xy[0],[0,e2p[1]]+led_cen_xy[1],color='b',linewidth=0.5)
    plt.show()
    
    # Roation
    # Project the mouse center on to the line orthogonal to the LED long axis
    # This will the the second eigen vector
    evs = eg_vec[:,[np.argmin(eg_val)]]    
    P = np.matmul(evs,evs.T) # a.T*a is one so no need for denominator of the projection matrix
    
    # Center mouse position to the LED center
    mouse_cen_xy_led_coord = mouse_cen_xy-led_cen_xy   
    proj_mouse_cen_xy_led_coord = np.matmul(P,mouse_cen_xy_led_coord)    
    
    proj_mouse_cen_xy = proj_mouse_cen_xy_led_coord+led_cen_xy
    # For plotting purpose, add mean 
    plt.plot(proj_mouse_cen_xy[0,0],proj_mouse_cen_xy[1,0],marker='o',
             color='y',markersize=1)
    # Rotate this point to 270 deg so that mouse nose points up and mouse
    # midline is aligned to y-axis
    target_angle = np.deg2rad(90)
    mouse_angle = np.arctan2(proj_mouse_cen_xy_led_coord[1],proj_mouse_cen_xy_led_coord[0])
    angle_to_rotate = target_angle-mouse_angle
    rot = lambda x,t: np.matmul(np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]]),x)
    
    # Rotate the eigen vector and check
    evs_rot_led_coord = rot(evs,angle_to_rotate[0])
    evs_rot_xy = 50*evs_rot_led_coord+led_cen_xy
    plt.plot([led_cen_xy[0,0],evs_rot_xy[0,0]],[led_cen_xy[1,0],evs_rot_xy[1,0]])
    plt.title('mouse angle: %u, to_rotate: %u'%(np.rad2deg(mouse_angle),np.rad2deg(angle_to_rotate)))
    
    # Rotate mouse center and its projected location
    pmc_led_coord_rot = rot(proj_mouse_cen_xy_led_coord,angle_to_rotate[0])
    pmc_xy_rot = pmc_led_coord_rot+led_cen_xy
    plt.plot(pmc_xy_rot[0],pmc_xy_rot[1],marker='s',color='y',markersize=1)
    mc_led_coord_rot = rot(mouse_cen_xy_led_coord,angle_to_rotate[0])
    mc_xy_rot = mc_led_coord_rot+led_cen_xy
    plt.plot(mc_xy_rot[0],mc_xy_rot[1],marker='s',color='r',markersize=1)
      
    # Rotate every point by
    # Center all pixel locations
    rmat = np.zeros((2000,2000))
    for iRow in range(frame_size[1]):
        for iCol in range(frame_size[0]):
            iloc_led_coord_xy = np.array([[iCol]-led_cen_xy[1,0],[iRow]-led_cen_xy[0,0]])
            iloc_led_coord_xy_rot = rot(iloc_led_coord_xy,angle_to_rotate[0])
            # Add center again
            iloc_xy_rot = np.ceil(iloc_led_coord_xy_rot+led_cen_xy).astype(int)            
            rmat[iloc_xy_rot[1]+800,iloc_xy_rot[0]+800] = frame[iRow,iCol]
    plt.figure()
    plt.imshow(np.uint8(rmat),cmap='gray')
    plt.show()
            
            
    
    
    
    

