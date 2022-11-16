# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:06:36 2021

This script creates the state vector for the DRL algorithm for experiment 0 and 1

@author: Jan
"""

from bluesky import core, stack, traf, sim, tools, scr
from bluesky.tools.aero import ft, nm, fpm, Rearth, kts
from bluesky.tools import geo
import math as m
import bluesky as bs
import numpy as np
import functions as fn

## State Settings ##
searchradius = 0.3 * nm # m around ac searched
searchlayers = 4 
n_aircraft = bs.settings.num_aircraft

num_headinglayers = bs.settings.num_headinglayers

def get_state(idx, target_alt):
    state_size, statestart, logstate_size = fn.get_statesize()
    state = np.zeros(state_size) # Create state array of correct state length
    logstate = np.zeros(logstate_size)

    state[0] = logstate[0] = abs((target_alt*ft - traf.alt[idx]) / ft) # Remaining climb or descend altitude
    state[1] = logstate[1] = abs(traf.vs[idx]) # Vertical speed
    state[2] = logstate[2] = traf.tas[idx] # True air speed
    state[3] = logstate[3] = get_delta_heading(idx, num_headinglayers) # Heading difference with current layer

    aircraftinbounds = get_aircraftinbounds(idx, searchradius, searchlayers, target_alt)
    sortedaircraft = sort_aircraft(aircraftinbounds, idx, n_aircraft)

    state, logstate, int_idx = append_ac_state(statestart, state, n_aircraft, idx, sortedaircraft, logstate)

    return state, logstate, int_idx

def get_delta_heading(idx, num_headinglayers):
    layer     = fn.get_layerfromalt(traf.alt[idx])
    hdglayer = layer*2*(360/num_headinglayers)
    if hdglayer >= 360:
        hdglayer = hdglayer - 360
    hdgdif   = hdglayer - traf.hdg[idx]
    hdgdif   = (hdgdif + 180) % 360 - 180
    return hdgdif

def get_aircraftinbounds(idx, searchradius, searchlayers, target_alt):
    aircraftinbounds = []
    ownlayer    = fn.get_layerfromalt(traf.alt[idx])
    targetlayer = fn.get_layerfromalt(target_alt*ft)   
    for j_id in traf.id:
        include     = False
        j = traf.id2idx(j_id)
        if j != idx and j not in aircraftinbounds:  
            intlayer    = fn.get_layerfromalt(traf.alt[j])
            if targetlayer <= ownlayer:
                if intlayer in range(ownlayer-searchlayers,ownlayer+1):
                    include = True
            else:
                if intlayer in range(ownlayer, ownlayer + searchlayers):
                    include = True
            distance_ij = haversine(traf.lat[idx],traf.lon[idx],traf.lat[j],traf.lon[j])
            if distance_ij < searchradius and include:
                aircraftinbounds.append(j)
                
    return aircraftinbounds
 
def sort_aircraft(aircraftinbounds, idx, n_aircraft):
    ac_tdcpa = []
    for j in aircraftinbounds:
        tcpa, tinconf, toutconf, dcpa, conflict = calc_tdcpa(idx, j)
        tcpa = min(tcpa, 600)
        temp = [j,tcpa,dcpa,conflict,tinconf,toutconf]
        ac_tdcpa.append(temp)
    ac_tdcpa = np.array(ac_tdcpa)
    
    if len(ac_tdcpa) > 0:
        ac_tdcpa = ac_tdcpa[(ac_tdcpa[:,1]>=0)|(ac_tdcpa[:,3]==1)]
        ac_tdcpa = ac_tdcpa[ac_tdcpa[:,1].argsort()]
        sortedaircraft = ac_tdcpa[(ac_tdcpa[:,3]*-1).argsort()]
        sortedaircraft = sortedaircraft[:n_aircraft]
    else:
        sortedaircraft = []
    return sortedaircraft
                
def append_ac_state(startnum, state, n_aircraft, idx, sortedaircraft, logstate):
    counter = 1
    counterlog = 1
    acid = [None] * n_aircraft
    int_idx = np.empty((0,3))
    for i in range(len(sortedaircraft)):
        start = startnum+counter
        startlog = startnum+counterlog

        j = int(sortedaircraft[i,0])

        heightdif = abs(traf.alt[j] - traf.alt[idx])
        tcpa = sortedaircraft[i,1]
        dcpa = sortedaircraft[i,2]
        conflict = sortedaircraft[i,3]

        du, dv = calc_relvel_ownshipframe(idx,j)
        dis = haversine(traf.lat[idx],traf.lon[idx], traf.lat[j],traf.lon[j])
        brg = bearing(traf.lat[idx],traf.lon[idx], traf.lat[j],traf.lon[j])

        brgrad = np.radians(brg)
        hdg = m.radians(traf.hdg[idx])
        
        dx = dis * np.sin(brgrad - hdg) 
        dy = dis * np.cos(brgrad - hdg)

        intrusion = 0
        if conflict == 1 and sortedaircraft[i,4] < 0:
            intrusion = 1

        if conflict:
            int_idx = np.vstack((int_idx,[j,tcpa,traf.id[j]]))

        tempstate = np.array([conflict,tcpa,dcpa,heightdif,du,dv,dx,dy])
        templogstate = np.array([traf.id[j],intrusion,conflict,tcpa,dcpa,heightdif,du,dv,dx,dy,dis])
        state[start:(start+len(tempstate))] = tempstate
        logstate[startlog:(startlog+len(templogstate))] = templogstate
        
        acid[i] = traf.id[j]

        counter += len(tempstate)
        counterlog += len(templogstate)

    return state, logstate, int_idx
    
# Haversine formula which return the distance between 2 lat,lon coordinates
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
    brg, dist   = geo.kwikqdrdist(lat1, lon1, lat2, lon2)
    return brg
    
def calc_tdcpa(own_idx, int_idx):
    horconf = 0
    verconf = 0  

    conflict    = 0     
    # Horizontal conflict calculations
    dvx, dvy = calc_relvel_cartesian(own_idx, int_idx)
    
    dis     = haversine(traf.lat[own_idx],traf.lon[own_idx], traf.lat[int_idx],traf.lon[int_idx])
    brg     = bearing(traf.lat[own_idx],traf.lon[own_idx], traf.lat[int_idx],traf.lon[int_idx])
    
    brgrad = np.radians(brg)
    dx = dis * np.sin(brgrad) 
    dy = dis * np.cos(brgrad)  

    # print('own dx,dy:',dx,dy)
    # print('own dxv,dvy:',dvx,dvy)
    
    dv2 = dvx * dvx + dvy * dvy
    vrel = np.sqrt(dv2)
    
    if abs(dv2) < 1e-6:
        dv2 = 1e-6          # limit lower absolute value
    
    tcpa = -(dvx * dx + dvy * dy) / dv2
    dcpa = m.sqrt(abs(dis * dis - tcpa * tcpa * dv2))

    # print('own tcpa:', tcpa)
    
    pzradius = bs.settings.asas_pzr * nm
    
    if dcpa < pzradius:
        dcpa2   = dcpa * dcpa
        R2      = pzradius * pzradius
        
        dxinhor = np.sqrt(R2 - dcpa2)  # half the distance travelled inzide zone
        dtinhor = dxinhor / vrel
        
        tinhor  = tcpa - dtinhor
        touthor = tcpa + dtinhor
        
        horconf = 1
    else:
        
        tinhor = 0
        touthor = 0
    # Vertical conflict calculations
    pzalt   = bs.settings.asas_pzh * ft
    
    dalt    = traf.alt[own_idx] - traf.alt[int_idx]
    dvs     = traf.vs[own_idx] - traf.vs[int_idx]
    
    if abs(dvs) < 1e-6:
        dvs = 1e-6          # limit lower absolute value
    
    # Check for passing through each others zone

    tcrosshi = (dalt + pzalt) / -dvs
    tcrosslo = (dalt - pzalt) / -dvs
    tinver = np.minimum(tcrosshi, tcrosslo)
    toutver = np.maximum(tcrosshi, tcrosslo)
    
        
    # find time in conflict and out conflict
    tinconf = 0
    toutconf = 0
    if horconf :
        tinconf     = max(tinver, tinhor)
        toutconf    = min(toutver, touthor)
        
        if tinconf <= toutconf and toutconf > 0:
            conflict = 1
            # tcpa    = max(tinconf,0.)
            
    # Dont discard AC with which still conflicts can occur even though 
    # Tcpa < 0 because of still being within PZ radius
    
    # if tcpa < 0 and abs(dis) < pzradius:
    #     tcpa = 0
        
    # return tcpa, dcpa, conflict
    if tinver > 600:
        tinver = 600
    if tinver < 0:
        tinver = 0
    if toutver > 600:
        toutver = 600
    if toutver < 0:
        toutver = 0
    
    return tcpa, tinconf, toutconf, dcpa, conflict

def calc_relvel_cartesian(own_idx,int_idx):
    own_heading = m.radians(traf.hdg[own_idx])
    int_heading = m.radians(traf.hdg[int_idx])
    
    own_vx = traf.tas[own_idx] * m.sin(own_heading)
    own_vy = traf.tas[own_idx] * m.cos(own_heading)
    
    int_vx = traf.tas[int_idx] * m.sin(int_heading)
    int_vy = traf.tas[int_idx] * m.cos(int_heading)
       
    dvx = int_vx - own_vx
    dvy = int_vy - own_vy
    
    return dvx, dvy
    
def calc_relvel_ownshipframe(own_idx,int_idx):
    own_heading = m.radians(traf.hdg[own_idx])
    int_heading = m.radians(traf.hdg[int_idx])
    
    du = traf.tas[own_idx] - traf.tas[int_idx] * m.cos(int_heading - own_heading)
    dv = traf.tas[int_idx] * m.sin(int_heading - own_heading)
    
    return du, dv

def get_tcpa(own_idx, int_idx):
    dvx, dvy = calc_relvel_cartesian(own_idx, int_idx)
    
    dis     = haversine(traf.lat[own_idx],traf.lon[own_idx], traf.lat[int_idx],traf.lon[int_idx])
    brg     = bearing(traf.lat[own_idx],traf.lon[own_idx], traf.lat[int_idx],traf.lon[int_idx])
    
    brgrad = np.radians(brg)
    dx = dis * np.sin(brgrad) 
    dy = dis * np.cos(brgrad)  
    
    dv2 = dvx * dvx + dvy * dvy
    vrel = np.sqrt(dv2)
    
    if abs(dv2) < 1e-6:
        dv2 = 1e-6          # limit lower absolute value

    tcpa = -(dvx * dx + dvy * dy) / dv2
    return tcpa