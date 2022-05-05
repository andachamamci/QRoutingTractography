# -*- coding: utf-8 -*-
"""
The training code for Q-routing tractography method.
Setting the rewards based on fODF (the costs as in shortest-paths method),
calculates the q-value for each action (moving in a direction).
This code is used to produce the results that are submiited to 
the 2nd round of the Iron Tract Challenge by team 13.
Andaç Hamamcı, Mert Yıldız
Medical Imaging Lab., Yeditepe Univ.
andac.hamamci@yeditepe.edu.tr
https://imagingyeditepe.github.io/
"""

import numpy as np
import nibabel as nib

'''
Read the input file.
nbh_pdf was calculated using the create_graph function of CATractography code.
create_graph function calculates the ODF in 26 directions in 3D, then scale maximum to 1 and 
takes the average for opposite directions to assure the symmetry. 
'''
DWIFILE_NAME = '../IronTrackChallenge/hcpl_graph.npz'

hcpl_graph=np.load(DWIFILE_NAME)
nbh_pdf=hcpl_graph['nbh_pdf']
nbh=hcpl_graph['nbh']

'''
Set the costs (R). Instead of maximizing the rewards in Q-learning framework, 
we try to find the paths which minimizes the costs. 
This is refered as Q-routing in the literature.
We set the costs as -ln(ODF) as in shortest-paths methods. 
So that minimizing the sum is maximizing the product of ODF's. 
'''
R=np.log(nbh_pdf)*-1

# Set a very high cost for actions to outside
R[-1,:,:,0]=100000
R[:,-1,:,1]=100000
R[-1,:,:,2]=100000
R[:,-1,:,2]=100000
R[-1,:,:,3]=100000
R[:,0,:,3]=100000
R[0,:,:,4]=100000
R[:,-1,:,4]=100000
R[0,:,:,5]=100000
R[:,0,:,6]=100000
R[0,:,:,7]=100000
R[:,0,:,7]=100000
R[:,:,0,8]=100000
R[0,:,:,8]=100000
R[:,:,-1,9]=100000
R[:,0,:,10]=100000
R[:,:,0,10]=100000
R[:,:,0,11]=100000
R[0,:,:,12]=100000
R[:,0,:,12]=100000
R[:,:,0,12]=100000
R[:,:,-1,13]=100000
R[0,:,:,13]=100000
R[:,-1,:,14]=100000
R[:,:,0,14]=100000
R[:,:,-1,15]=100000
R[:,0,:,15]=100000
R[-1,:,:,16]=100000
R[:,-1,:,16]=100000
R[:,:,0,16]=100000
R[0,:,:,17]=100000
R[:,-1,:,17]=100000
R[:,:,-1,17]=100000
R[-1,:,:,18]=100000
R[:,:,-1,18]=100000
R[:,-1,:,19]=100000
R[:,:,-1,19]=100000
R[-1,:,:,20]=100000
R[:,-1,:20]=100000
R[:,:,-1,20]=100000
R[-1,:,:,21]=100000
R[:,0,:,21]=100000
R[:,:,-1,21]=100000
R[-1,:,:,22]=100000
R[:,:,0,22]=100000
R[0,:,:,23]=100000
R[:,0,:,23]=100000
R[:,:,-1,23]=100000
R[0,:,:,24]=100000
R[:,-1,:,24]=100000
R[:,:,0,24]=100000
R[-1,:,:,25]=100000
R[:,0,:,25]=100000
R[:,:,0,25]=100000

'''
Initialize the Q-table
'''
Q=np.zeros(R.shape)
Q[R==100000]=100000

'''
Learning parameters for Q-learning
'''
discount = 1.0
lr=0.7

'''
Functions for Q-learning
'''
# This function returns all available actions in the state given as an argument
def available_actions(state):
    current_state_row = R[state[0],state[1],state[2],]
    av_act = np.where(current_state_row > -10000000000)[0]
    return av_act

# This function chooses at random which action to be performed within the range 
# of all the available actions.
def sample_next_action(available_act):
    next_action = int(np.random.choice(available_act,1))
    return next_action

# This function updates the Q matrix according to the path selected and the Q 
# learning algorithm
def update(current_state, action, lr, discount):
    
    cs=current_state + nbh[action,:]
    
    if cs[0] < 0 or cs[0] > Q.shape[0]-1 or cs[1] < 0 or cs[1] > Q.shape[1]-1 or cs[2] < 0 or cs[2] > Q.shape[2]-1:
        return
         
    max_index = np.where(Q[cs[0],cs[1],cs[2],] == np.min(Q[cs[0],cs[1],cs[2],]))[0]
    
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size = 1))
    else:
        max_index = int(max_index)
        
    max_value = Q[cs[0],cs[1],cs[2],max_index]
    
    # Q learning formula
    Q[current_state[0],current_state[1],current_state[2], action] = ((1-lr)*Q[current_state[0],current_state[1],current_state[2], action]+(lr)*(R[current_state[0],current_state[1],current_state[2], action] + discount * max_value))

''' 
Training
'''
# conv_sum is the variable to check the convergence
conv_sum=[]

for i in range(200000000):
    
    current_state = np.random.randint(0,Q.shape[0:3] )
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    update(current_state, action, lr, discount)
    if i%10000==0:
       conv_sum.append(np.sum(Q[Q<100000]))
    
''' Save the trained Q-table and convergence data to plot '''
np.savez_compressed('./Qtable_IronTrack_last', Q=Q)
np.savez_compressed('./conv_sum_training200mlyn', conv_sum=conv_sum)

