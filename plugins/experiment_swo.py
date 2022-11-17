""" 
SWO Experiment file that runs the simulation scenario with a shortest way out
conflict resolution algorithm. By default aircraft with an altitude
command will fly to the required altitude with a vertical speed of 2.5 m/s and 
a horizontal speed of 10 m/s. The shortest way out algorithm inputs deviations
from these velocities to resolve the conflicts and returns to the default value
once the conflicting aircraft are passed.

TO RUN THIS EXPERIMENT change the enabled_plugins in the settings.cfg file to:

enabled_plugins = ['area', 'layered_trafgen', 'experiment_swo']
"""

from re import L
from bluesky import core, stack, traf, tools, settings 
from bluesky.tools.aero import ft, nm, fpm, Rearth, kts
from bluesky.tools import geo, areafilter
import bluesky as bs
import numpy as np
import math
import pandas as pd
from pathlib import Path

import plugins.SAC.sac_agent as sac
import plugins.state as st
import plugins.functions as fn
import plugins.reward as rw

timestep = 1.5

default_vz = 2.5/fpm
default_speed = 10/kts

onsetperiod = 600 # Number of seconds before experiment starts

n_aircraft = bs.settings.num_aircraft

# Standard way of initializing Plugins in BlueSky
def init_plugin():  
    experiment_swo = Experiment_swo()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'EXPERIMENT_SWO',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }
    # init_plugin() should always return a configuration dict.
    return config

# Main experiment class
class Experiment_swo(core.Entity):  
    def __init__(self):
        super().__init__()

        self.print = False

        self.controlac = 0 # int type variable to keep track of the number of ac controlled

        self.rewards = np.array([]) # array that stores the total return after each episode
        self.not_finished = 0

        self.logfile = None # Empty initializer for the logfile, will be set later
        self.lognumber = 0 # Counter for which logfile should be written
        self.init_logfile()

        with self.settrafarrays():
            self.insdel = np.array([], dtype=np.bool)

            self.targetalt = np.array([])  # Target altitude of the AC   
            self.control = np.array([])  # Is AC controlled by external algorithm
            self.first = np.array([])  # Is it the first time AC is called
            self.distinit = np.array([])  # Initial (vertical) distance from target state
            self.totreward = np.array([])  # Total reward this AC has accumulated
            self.call = np.array([])  # Number of times AC has been comanded
            self.acnum = np.array([]) # ith aircraft 

            self.confint = [] # callsigns of all current and old intruders that are still active

    def create(self, n=1):
        # Set default variables when aircraft are created
        # Assigns values to the empty traffic arrays defined in __init__()
        super().create(n)
        self.insdel[-n:] = False

        self.targetalt[-n:] = traf.alt[-n:]  
        self.control[-n:] = 0
        self.totreward[-n:] = 0
        self.first[-n:] = 1
        self.distinit[-n:] = 0
        self.call[-n:] = 0
        self.acnum[-n:] = 0
        
        self.confint[-n:] = [[]]

    # Main function that is looped every 'timestep' seconds (simulation time, not real time)
    @core.timed_function(name='experiment_swo', dt=timestep)
    def update(self):

        # Check if aircraft is still inside the simulation area, delete those that are not
        self.check_inside()
        
        # Loop through remaining aircraft
        for acid in traf.id:

            # Get respective index of the aircraft ID in the Traffic Arrays and update check their control value
            ac_idx = traf.id2idx(acid)
            self.update_AC_control(ac_idx)

            # If control value of the aircraft is 1, run a control loop for this timestep
            if self.control[ac_idx] == 1:
                self.call[ac_idx] += 1

                # Get new state for determining actions and state used for logging
                state_, logstate, int_idx = st.get_state(ac_idx, self.targetalt[ac_idx])
                
                # Set default values for the next state transition if it is the first 
                # time aircraft gets control command
                if self.first[ac_idx]:
                    self.reset_action(acid,ac_idx)
                    self.first[ac_idx] = 0
                    self.controlac += 1

                # If not first time step, use SWO algorithm with State to determine 
                # changes in velocity vector
                else:
                    # Get current velocity vector in cartesian coordinate system
                    v = np.array([traf.cas[ac_idx]*np.sin(np.deg2rad(traf.hdg[ac_idx])), traf.cas[ac_idx]*np.cos(np.deg2rad(traf.hdg[ac_idx])), traf.vs[ac_idx]])
                    v_update = False

                    # Loop through array of conflicts
                    for j, tLos, int_id in int_idx:
                        # Determine the change in velocity with the shortest way out algorithm
                        dv, _ = self.SWO(ac_idx, int(j), float(tLos))
                        
                        # Change the velocity by the required velocity change
                        v = v-dv

                        # Store intruder in temporary array for checking if intruder has passed later
                        if int_id not in self.confint[ac_idx]:
                            self.confint[ac_idx].append(int_id)
                        v_update = True
                    
                    # Calculate the reward for this state
                    reward, done = rw.calc_reward(state_, logstate)
                    self.totreward[ac_idx] += reward
                    
                    # Execute the action and return it for logging
                    action = self.do_action(acid,ac_idx,v,v_update)

                    # If done, return aircraft to cruise mode
                    if done:
                        self.control[ac_idx] = 0
                        self.rewards = np.append(self.rewards, self.totreward[ac_idx])
                        stack.stack(f'alt {acid} {self.targetalt[ac_idx]}')
                        self.print = True
                    
                    # If not done, check if aircraft can return to default values
                    # this is only done if conflist is empty
                    else:
                        self.update_conflist(ac_idx)
                        self.reset_action(acid,ac_idx)
                    
                    # Logging and printing for debuggin and tracking progress
                    if (len(self.rewards)-self.not_finished) % 50 == 0 and self.print == True:
                        self.print = False
                        print(np.mean(self.rewards))
                        print(f'completed {len(self.rewards)} flights, out of bounds for {self.not_finished} flights')

                    if len(self.rewards) == 5000:
                        stack.stack('QUIT')

                    
                    self.log(logstate,action,acid,ac_idx)

    def update_AC_control(self,ac_idx):   
        if self.first[ac_idx]:                
            # Checks if the AC is inside the bounds of the delivery area
            inbounds    = fn.checkinbounds(traf.lat[ac_idx], traf.lon[ac_idx])  
            if inbounds:
                # function to select target altitude of aircraft, 
                # 5% chance that it is different than current. 
                targetalt, control, layer = fn.get_targetalt(ac_idx)

                if bs.sim.simt > onsetperiod:
                    self.control[ac_idx] = control
                    self.targetalt[ac_idx] = targetalt                                 #ft
                    self.distinit[ac_idx] = (abs(targetalt*ft - traf.alt[ac_idx]))/ft #ft

                # Ensure that each aircraft only has one chance at getting
                # an altitude command, dont reset first if control == 1
                # because that information is still required in that case.
                if control == 0:
                    self.first[ac_idx] = 0

    def do_action(self,acid,ac_idx,newv,v_update):
        """ Execute the new velocity that is put out by the 
        SWO algorithm. Also store this information in an action
        array to be used for logging.
        
        input: self, acid, ac_idx, newv ([vx,vy,vz]), v_update (boolean)
        output: action """

        action = np.array([0,0,0])
        # if new velocities:
        if v_update:
            # calculate track angle and horizontal speed from vx and vy
            newtrack = (np.arctan2(newv[0],newv[1])*180/np.pi) %360
            newhs    = np.sqrt(newv[0]**2 + newv[1]**2) / kts
            newvs    = newv[2] / fpm

            # set horizontal speed
            stack.stack(f'SPD {acid} {newhs}')

            # vertical velocity close to zero cannot be set in bluesky
            # instead, stop vertical manoeuvre temporary if vertical speed too small
            if newvs < 20:
                target = traf.alt[ac_idx] / ft
                newvs = 0        
            
            # Determine if climb or descend based on target altitude
            # and current altitude
            elif self.targetalt[ac_idx]*ft < traf.alt[ac_idx]:
                target = 0               
            else:
                target = 5000   

            # Execute vertical speed and heading
            stack.stack(f'ALT {acid} {target} {newvs}')
            stack.stack(f'HDG {acid} {newtrack}')

            # Store changed velocities in action array
            action[0] = newvs
            action[1] = bs.traf.cas[ac_idx] - newhs
            if newtrack - bs.traf.hdg[ac_idx] > 180:
                action[2] = -1*((newtrack - bs.traf.hdg[ac_idx] + 180) % 360 - 180)
            elif newtrack - bs.traf.hdg[ac_idx] < -180:
                action[2] = ((newtrack - bs.traf.hdg[ac_idx] + 180) % 360 - 180)
            else:
                action[2] = newtrack - bs.traf.hdg[ac_idx]

        return action
    
    def check_inside(self):
        """ Check which aircraft are still inside the experiment area.
        Delete aircraft that are outside experiment area.
        Operates on the entire traffic array, not on single aircraft.
        """
        # Get aircraft that are inside the experiment area
        insdel = areafilter.checkInside('SIM_ENVIRONMENT', traf.lat, traf.lon, traf.alt)

        # Get index of to be deleted aircraft by comparing aircraft
        # that were inside previous timestep, but are not this timestep
        delidx = np.where(np.array(self.insdel) * (np.array(insdel) == False))[0]
        self.insdel = insdel

        # Delete aircraft to be deleted, storing information of aircraft that
        # were conducting vertical manoeuvres.
        if len(delidx) > 0:
            delcontr = np.where((self.control[delidx]))[0]
            if len(delcontr) > 0:
                self.rewards = np.append(self.rewards, self.totreward[delcontr])
                self.not_finished += len(delcontr)
            traf.delete(delidx)

    def update_conflist(self,ac_idx):
        """ Function to update the list of resolved conflicts
        this list exists to limit that aircraft show
        oscilatory behaviour during conflict resolution"""

        if self.confint[ac_idx]:
            for i in self.confint[ac_idx]:
                int_idx = bs.traf.id2idx(i)
                # get time till closest point of approach with other aircraft
                tcpa = st.get_tcpa(ac_idx,int_idx)
                # if tcpa < 0, aircraft has passed and we can return to default flight
                if tcpa < 0:
                    self.confint[ac_idx].remove(i)

    def reset_action(self,acid,ac_idx):
        """ Function to reset the aircraft to the
        default values"""
        if not self.confint[ac_idx]:
            stack.stack(f'SPD {acid} {default_speed}')

            if self.targetalt[ac_idx]*ft < traf.alt[ac_idx]:
                target = 0               
            else:
                target = 5000
                
            stack.stack(f'ALT {acid} {target} {default_vz}')
            
    def log(self,logstate,action,acid,ac_idx):
        """ Function used for logging the data"""
        data = [acid, self.acnum[ac_idx], self.call[ac_idx]] + list(logstate) + list(action)
        self.logfile.loc[len(self.logfile)] = data
        
        # write out logfile every 10000 state transitions and create new logfile to prevent RAM saturation
        # also saves the models to a pickle file to be used for testing
        if len(self.logfile) == 10000:
            lognumber = str(self.lognumber)    

            if self.lognumber == 0:
                path = bs.settings.experiment_path + '\\' + bs.settings.experiment_name
                Path(path).mkdir(parents=True, exist_ok=True)

            logsavename = bs.settings.experiment_path +'\\'+ bs.settings.experiment_name+ '\\'+ 'logdata_'+lognumber+'.csv'
            self.logfile.to_csv(logsavename)

            self.lognumber += 1
            self.init_logfile()

    def init_logfile(self):
        """ Initialize logfile with the correct header / column names"""
        header = ['acid','acnum','callnum','alt_dif','vz','vh','d_hdg']
        intruder_header = ['int_','intrusion_','conflict_','tcpa_','dcpa_','dalt_','du_','dv_','dx_','dy_','dis_']
        tail_header = ['dvz','dvh','dhdg']

        for i in range(n_aircraft):
            header_append = [s + str(i) for s in intruder_header]
            header = header + header_append

        header = header + tail_header

        self.logfile = pd.DataFrame(columns = header)

    def SWO(self, own_idx, int_idx, tLOS):
        """ Shortest way out algorithm used for conflict resolution
        in this experiment. The algorithm is based on the MVP implementation
        found in bluesky/traffic/asas/MVP but only vertical manoeuvering aircraft
        resolves instead of all conflicting aircraft"""

        tcpa = st.get_tcpa(own_idx,int_idx)

        pz = bs.settings.asas_pzr * nm
        hpz = bs.settings.asas_pzh * ft
        safety = bs.settings.asas_marh
        
        int_idx = int(int_idx)
        
        brg, dist = geo.kwikqdrdist(bs.traf.lat[own_idx], bs.traf.lon[own_idx],\
                                    bs.traf.lat[int_idx], bs.traf.lon[int_idx],)
        
        dist = dist * nm
        qdr = np.radians(brg)
        drel = np.array([np.sin(qdr)*dist, \
                        np.cos(qdr)*dist, \
                        bs.traf.alt[int_idx]-bs.traf.alt[own_idx]])
    
        # Write velocities as vectors and find relative velocity vector
        v1 = np.array([traf.tas[own_idx]*np.sin(np.deg2rad(traf.hdg[own_idx])), traf.tas[own_idx]*np.cos(np.deg2rad(traf.hdg[own_idx])), traf.vs[own_idx]])
        v2 = np.array([traf.tas[int_idx]*np.sin(np.deg2rad(traf.hdg[int_idx])), traf.tas[int_idx]*np.cos(np.deg2rad(traf.hdg[int_idx])), traf.vs[int_idx]])
        vrel = np.array(v2-v1)

        dcpa = drel + vrel*tcpa
        dabsH = np.sqrt(dcpa[0]*dcpa[0]+dcpa[1]*dcpa[1])
        
        iH = (pz * safety) - dabsH
        
        # Exception handlers for head-on conflicts
        # This is done to prevent division by zero in the next step
        if dabsH <= 10.:
            dabsH = 10.
            dcpa[0] = drel[1] / dist * dabsH
            dcpa[1] = -drel[0] / dist * dabsH
        
        # If intruder is outside the ownship PZ, then apply extra factor
        # to make sure that resolution does not graze IPZ
        if (pz * safety) < dist and dabsH < dist:
            # Compute the resolution velocity vector in horizontal direction.
            # abs(tcpa) because it bcomes negative during intrusion.
            erratum=np.cos(np.arcsin((pz * safety)/dist)-np.arcsin(dabsH/dist))
            dv1 = (((pz * safety)/erratum - dabsH)*dcpa[0])/(abs(tcpa)*dabsH)
            dv2 = (((pz * safety)/erratum - dabsH)*dcpa[1])/(abs(tcpa)*dabsH)
        else:
            dv1 = (iH * dcpa[0]) / (abs(tcpa) * dabsH)
            dv2 = (iH * dcpa[1]) / (abs(tcpa) * dabsH)
        
        # Vertical resolution------------------------------------------------------

        # Compute the  vertical intrusion
        # Amount of vertical intrusion dependent on vertical relative velocity
        iV = (hpz * safety) if abs(vrel[2])>0.0 else (hpz * safety)-abs(drel[2])

        # Exception handlers for same-alt conflicts
        # This is done to prevent division by zero in the next step
        drel[2] = np.where(abs(drel[2]) < 0.01, 0.01, drel[2])

        # Get the time to solve the conflict vertically - tsolveV
        tsolV = abs(drel[2]/vrel[2]) if abs(vrel[2])>0.0 else tLOS
        
        # If the time to solve the conflict vertically is longer than the look-ahead time,
        # because the the relative vertical speed is very small, then solve the intrusion
        # within tinconf
        if tsolV>50:
            tsolV = tLOS
            iV = (hpz * safety)
        
        # If already in intrusion and no vertical speed differences, get out of intrusion horizontally
        tsolV = 1000 if tsolV == 0 else tsolV
        dv3 = np.where(abs(vrel[2])>0.0,  (iV/tsolV)*(-vrel[2]/abs(vrel[2])), (iV/tsolV))
        dv = np.array([dv1,dv2,dv3])
        
        return dv, tsolV