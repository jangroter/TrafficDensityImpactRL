# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:34:05 2021

@author: Jan
"""
from bluesky import core, stack, traf, sim, tools, scr
from bluesky.tools.aero import ft, nm, fpm, Rearth, kts
import math as m
import numpy as np
from bluesky.tools import geo
import bluesky as bs
import random


def haversine(lat1,lon1,lat2,lon2):
    R = Rearth
    dLat = m.radians(lat2 - lat1)
    dLon = m.radians(lon2 - lon1)
    lat1 = m.radians(lat1)
    lat2 = m.radians(lat2)

    a = m.sin(dLat/2)**2 + m.cos(lat1)*m.cos(lat2)*m.sin(dLon/2)**2
    c = 2*m.asin(m.sqrt(a))

    return R * c  

def bearing(lat1,lon1,lat2,lon2):
    #return m.atan2(m.cos(lat1)*m.sin(lat2)-m.sin(lat1)*m.cos(lat2)*m.cos(lon2-lon1),\
    #              m.sin(lon2-lon1)*m.cos(lat2))
    brg, _ = geo.kwikqdrdist(lat1, lon1, lat2, lon2)
    return brg

def checkinbounds(lat,lon):
    circlelat = bs.settings.controlarea_lat
    circlelon = bs.settings.controlarea_lon
    
    R = bs.settings.deliveryradius * nm
    
    distance = haversine(lat,lon,circlelat,circlelon)
    
    if distance < R:
        return True
    else:
        return False
       
def get_targetalt(i):
    currentalt = traf.alt[i]
    
    givecommand = random.randint(1,100)
    chance = bs.settings.altchangechance
      
    layer = get_layerfromalt(currentalt)
    if givecommand <= chance:              
        if layer <=  bs.settings.num_headinglayers / 2:
            new_layer = random.randint((bs.settings.num_headinglayers / 2)+1, bs.settings.num_headinglayers)
        else:
            new_layer = random.randint(1, bs.settings.num_headinglayers / 2)
            
        target_alt = get_altfromlayer(new_layer)
        
        return target_alt, 1, layer
    
    else:
        return currentalt, 0, layer
             
def get_layerfromalt(alt):
    alt = alt / ft
    num_layers = bs.settings.num_headinglayers
    alt_per_layer = (bs.settings.upper_alt - bs.settings.lower_alt)/num_layers
    
    layer = int(((alt - bs.settings.lower_alt ) // alt_per_layer) + 1)
    
    return layer
    
def get_altfromlayer(layer):    
    num_layers = bs.settings.num_headinglayers
    alt_per_layer = (bs.settings.upper_alt - bs.settings.lower_alt)/num_layers
    
    alt = (layer-1) * alt_per_layer + bs.settings.lower_alt + alt_per_layer/4
    
    return alt

def get_statesize():
    n_aircraft = bs.settings.num_aircraft
    state_start = 3
    state_per_ac = 8
    logstate_per_ac = 11
        
    state_size  = state_start + n_aircraft*state_per_ac + 1
    logstate_size = state_start + n_aircraft*logstate_per_ac + 1
    
    return state_size, state_start, logstate_size

def normalize_state(state):
    tempstate = state[:]
    normvec = np.loadtxt('normalize_vector.txt')
    n_aircraft = bs.settings.num_aircraft
    
    state_per_ac = 8

    for i in range(n_aircraft-1):
        # Create an normalization array equal to the size of the statevector
        # [-state_per_ac:] copies the vector entries corresponding to the intruder states
        normvec = np.concatenate((normvec,normvec[-state_per_ac:]))

    norm_state = (state - normvec[:,0])/normvec[:,1]

    return norm_state



