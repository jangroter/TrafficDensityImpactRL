""" Functions related to the state vector.
Main function: get_state(idx, target_alt) returns 
the state vector associated with the aircraft 'idx' 

Used by all experiments for both action decisions and 
logging purposes"""

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
    """ function that returns the state vector, logstate vector
    and array containing the idx of the intruders associated
    with the aircraft under 'idx'.

    input: idx, target_alt (feet)
    output: state (np.array), logstate (np.array), int_idx (np.array)"""

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
    """ get the heading difference with the altitude layer
    input: idx, num_headinlayers
    output: heading difference (degrees)"""

    layer     = fn.get_layerfromalt(traf.alt[idx])
    hdglayer = layer*2*(360/num_headinglayers)
    if hdglayer >= 360:
        hdglayer = hdglayer - 360
    hdgdif   = hdglayer - traf.hdg[idx]
    hdgdif   = (hdgdif + 180) % 360 - 180
    return hdgdif

def get_aircraftinbounds(idx, searchradius, searchlayers, target_alt):
    """ generate a list of aircraft that are within the disk defined by
    radius: searchradius and height: searchlayers centered at the 
    aircraft denoted by idx 

    input: idx, searchradius (meter), searchlayers (int), target_alt (feet)
    output: list of potential intruder indices"""

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
            distance_ij = fn.haversine(traf.lat[idx],traf.lon[idx],traf.lat[j],traf.lon[j])
            if distance_ij < searchradius and include:
                aircraftinbounds.append(j)
                
    return aircraftinbounds
 
def sort_aircraft(aircraftinbounds, idx, n_aircraft):
    """ Sort the aircraft in bounds based on time till closest
    point of approach and conflict boolean to have a priority.
    
    input: aircraftinbounds (list), idx, n_aircraft
    output: sortedaircraft (list)"""

    ac_tdcpa = []
    for j in aircraftinbounds:
        # calculate conflict parameters associated with ownship and intruder
        tcpa, tinconf, toutconf, dcpa, conflict = calc_tdcpa(idx, j)
        # clip time till closest point of approach
        tcpa = min(tcpa, 600)
        temp = [j,tcpa,dcpa,conflict,tinconf,toutconf]
        ac_tdcpa.append(temp)
    ac_tdcpa = np.array(ac_tdcpa)
    
    if len(ac_tdcpa) > 0:
        # only keep aircraft that have tcpa > 0 or are in conflict 
        # e.g. remove aircraft that are moving away
        ac_tdcpa = ac_tdcpa[(ac_tdcpa[:,1]>=0)|(ac_tdcpa[:,3]==1)]
        # sort on tcpa
        ac_tdcpa = ac_tdcpa[ac_tdcpa[:,1].argsort()]
        # sort on conflict
        sortedaircraft = ac_tdcpa[(ac_tdcpa[:,3]*-1).argsort()]
        # only return n_aircraft most important elements
        sortedaircraft = sortedaircraft[:n_aircraft]
    else:
        sortedaircraft = []
    return sortedaircraft
                
def append_ac_state(startnum, state, n_aircraft, idx, sortedaircraft, logstate):
    """ function to append the information of the intruding aircraft
    to the remainder of that state and logstate arrays
    
    input: startnum, state, n_aircraft, idx, sortedaircraft, logstate
    output: state, logstate, int_idx"""
    
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
        dis = fn.haversine(traf.lat[idx],traf.lon[idx], traf.lat[j],traf.lon[j])
        brg = fn.bearing(traf.lat[idx],traf.lon[idx], traf.lat[j],traf.lon[j])

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

def calc_tdcpa(own_idx, int_idx):
    """ Calculate the time till and distance at closest 
    point of approach between 2 aircraft
    
    input: own_idx, int_idx
    output: tcpa (seconds), tinconf (seconds), toutconf (seconds), dcpa (meters), conflict (boolean)"""

    horconf = 0
    verconf = 0  

    conflict    = 0     

    # Horizontal conflict calculations
    dvx, dvy = calc_relvel_cartesian(own_idx, int_idx)
    
    dis     = fn.haversine(traf.lat[own_idx],traf.lon[own_idx], traf.lat[int_idx],traf.lon[int_idx])
    brg     = fn.bearing(traf.lat[own_idx],traf.lon[own_idx], traf.lat[int_idx],traf.lon[int_idx])
    
    brgrad = np.radians(brg)
    dx = dis * np.sin(brgrad) 
    dy = dis * np.cos(brgrad)  

    dv2 = dvx * dvx + dvy * dvy
    vrel = np.sqrt(dv2)
    
    if abs(dv2) < 1e-6:
        dv2 = 1e-6          # limit lower absolute value
    
    tcpa = -(dvx * dx + dvy * dy) / dv2
    dcpa = m.sqrt(abs(dis * dis - tcpa * tcpa * dv2))
    
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
    """calculate relative velocity between own and intruder 
    in cartesian reference frame
    
    input: own_idx, int_idx
    output: dvx (m/s), dvy (m/s)"""

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
    """calculate relative velocity between own and intruder 
    in ownship heading reference frame
    
    input: own_idx, int_idx
    output: du (m/s), dv (m/s)"""
    
    own_heading = m.radians(traf.hdg[own_idx])
    int_heading = m.radians(traf.hdg[int_idx])
    
    du = traf.tas[own_idx] - traf.tas[int_idx] * m.cos(int_heading - own_heading)
    dv = traf.tas[int_idx] * m.sin(int_heading - own_heading)
    
    return du, dv

def get_tcpa(own_idx, int_idx):
    """ Determine time till closest point of approach between 2 aircraft
    
    input: own_idx, int_idx
    output: tcpa (seconds)"""
    
    dvx, dvy = calc_relvel_cartesian(own_idx, int_idx)
    
    dis     = fn.haversine(traf.lat[own_idx],traf.lon[own_idx], traf.lat[int_idx],traf.lon[int_idx])
    brg     = fn.bearing(traf.lat[own_idx],traf.lon[own_idx], traf.lat[int_idx],traf.lon[int_idx])
    
    brgrad = np.radians(brg)
    dx = dis * np.sin(brgrad) 
    dy = dis * np.cos(brgrad)  
    
    dv2 = dvx * dvx + dvy * dvy
    vrel = np.sqrt(dv2)
    
    if abs(dv2) < 1e-6:
        dv2 = 1e-6          # limit lower absolute value

    tcpa = -(dvx * dx + dvy * dy) / dv2
    return tcpa