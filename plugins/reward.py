"""Functions for calculating the reward and done flag of the 
state transitions

episode is terminated when the target altitude layer is reached
reward is a penalty of:

 -0.25 if currently in intrusion
  0 otherwise"""

from bluesky import core, stack, traf, sim, tools, scr
from bluesky.tools.aero import ft, nm, fpm, Rearth, kts
import math as m
import plugins.functions as fn
import bluesky as bs


def calc_reward(state_,logstate):
    """ Calculate the reward and determine the done flag associated with 
    the current state
    
    input: new state, new logstate
    output: reward, done"""

    done = check_done(state_)
    reward = calc_reward(logstate)
    
    return reward, done
    
def check_done(state_):
    """ check if the episode is done by evaluating if the aircraft 
    has reached the target layer
    
    input: state_
    output: doneflag """

    headinglayers       = bs.settings.num_headinglayers
    
    basealtitude        = bs.settings.lower_alt
    maxaltitude         = bs.settings.upper_alt
    
    altperlayer         = (maxaltitude - basealtitude)/(headinglayers)
    
    if state_[0] < altperlayer: # state_[0] is the altitude difference with the target altitude (non-normalized state vector)
        return 1
    else:
        return 0

def calc_reward(logstate):
    """ calculate the reward for the state transition
    
    input: logstate
    output: reward """

    reward = 0

    if logstate[5] == 1: # logstate[5] is the intrusion boolean variable
        reward += -0.25
            
    return reward
