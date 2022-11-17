""" 
Baseline Experiment file that runs the simulation scenario without any type
of conflict resolution model being enabled. By default aircraft with an altitude
command will fly to the required altitude with a vertical speed of 2.5 m/s and 
a horizontal speed of 10 m/s.

TO RUN THIS EXPERIMENT change the enabled_plugins in the settings.cfg file to:

enabled_plugins = ['area', 'layered_trafgen', 'experiment_base']
"""

from bluesky import core, stack, traf, tools, settings 
from bluesky.tools.aero import ft, nm, fpm, Rearth, kts
from bluesky.tools import geo, areafilter
import bluesky as bs
import numpy as np
import pandas as pd
from pathlib import Path

import plugins.parameters as pm

import plugins.state as st
import plugins.functions as fn
import plugins.reward as rw

timestep = pm.timestep
state_size, _, _ = fn.get_statesize()
action_size = 3

default_vz = pm.def_vz / fpm
default_speed = pm.cruise_speed / kts

onsetperiod = pm.onsetperiod # Number of seconds before experiment starts

n_aircraft = pm.num_aircraft

# Standard way of initializing Plugins in BlueSky
def init_plugin():
    experiment_base = Experiment_base()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'EXPERIMENT_BASE',
        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }
    # init_plugin() should always return a configuration dict.
    return config

# Main experiment class 
class Experiment_base(core.Entity):  
    def __init__(self):
        super().__init__()

        self.print = False
        self.controlac = 0 # int type variable to keep track of the number of ac controlled
        self.not_finished = 0 # number of aircraft that have exited the experiment area before reaching the target layer

        self.rewards = np.array([]) # array that stores the total return after each episode
        self.state_size = state_size
        self.action_size = action_size

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
        
        self.insdel[-n:] = False # Aircraft should not be deleted

        self.targetalt[-n:] = traf.alt[-n:]  # Set target altitude to current altitude
        self.control[-n:] = 0
        self.first[-n:] = 1
        self.distinit[-n:] = 0
        self.totreward[-n:] = 0
        self.call[-n:] = 0
        self.acnum[-n:] = 0
        
        self.confint[-n:] = [[]]

    # Main function that is looped every 'timestep' seconds (simulation time, not real time)
    @core.timed_function(name='experiment_base', dt=timestep)
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

                # Get state for determining actions and state used for logging
                state_, logstate, int_idx = st.get_state(ac_idx, self.targetalt[ac_idx])
                
                # If it is the first time aircraft is being assigned a control command, set the correct speeds
                if self.first[ac_idx]:
                    self.reset_action(acid,ac_idx)
                    self.first[ac_idx] = 0
                    self.controlac += 1

                # If it is not the first time aircraft is being assigned a control command:
                else:
                    # Determine the reward accompanying the new state (reward is independent from old state or action)
                    reward, done = rw.calc_reward(state_,logstate)
                    self.totreward[ac_idx] += reward
                    
                    # Set action (2.5 m/s vertical speed, no speed or heading changes)
                    # Constant values because of No Conflict Resolution in this experiment
                    action = np.array([2.5,0,0])

                    # If aircraft is in the assigned altitude layer:
                    if done:
                        # Disable control
                        self.control[ac_idx] = 0
                        # Append the Return to the rewards array
                        self.rewards = np.append(self.rewards, self.totreward[ac_idx])
                        # Set aircraft back to cruise by assigning the target altitude to the build in autopilot module
                        stack.stack(f'alt {acid} {self.targetalt[ac_idx]}')
                        self.print = True
                    
                    # Logging
                    self.log(logstate,action,acid,ac_idx)

                    # Some verbose and quiting simulation after 5000 completed vertical flights.
                    if len(self.rewards) % 50 == 0 and self.print == True:
                        self.print = False
                        print(np.mean(self.rewards))
                        print(f'completed {len(self.rewards)} flights')

                        if len(self.rewards) == 5000:
                            stack.stack('QUIT')

    def update_AC_control(self,ac_idx):   
        """ Decide if aircraft is assigned different altitude when entering delivery area.
        Only works if aircraft is in area for the first time"""

        if self.first[ac_idx]:                
            # Checks if the AC is inside the bounds of the delivery area
            inbounds    = fn.checkinbounds(traf.lat[ac_idx], traf.lon[ac_idx])               
            if inbounds:
                # function to select target altitude of aircraft, 
                # 5% chance that it is different than current.
                targetalt, control, layer = fn.get_targetalt(ac_idx)

                # Check if simulation has passed onsetperiod
                if bs.sim.simt > onsetperiod:
                    self.control[ac_idx] = control
                    self.targetalt[ac_idx] = targetalt                                 #ft
                    self.distinit[ac_idx] = (abs(targetalt*ft - traf.alt[ac_idx]))/ft #ft

                # Ensure that each aircraft only has one chance at getting
                # an altitude command, dont reset first if control == 1
                # because that information is still required in that case.
                if control == 0:
                    self.first[ac_idx] = 0

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

    def reset_action(self,acid,ac_idx):
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
        if len(self.logfile) == 10000:
            lognumber = str(self.lognumber)    

            if self.lognumber == 0:
                path = pm.experiment_path + '/' +pm.experiment_name
                Path(path).mkdir(parents=True, exist_ok=True)

            logsavename = Path(pm.experiment_path +'/'+ pm.experiment_name+ '/'+ 'logdata_'+lognumber+'.csv')
            self.logfile.to_csv(logsavename)

            self.lognumber += 1
            self.init_logfile()

    def init_logfile(self):
        """ Initialize logfile with the correct header / column names"""

        header = ['acid','acnum','callnum','alt_dif','vz','vh','d_hdg']
        intruder_header = ['int_','intrusion_','conflict_','tcpa_','dcpa_','dalt_','du_','dv_','dx_','dy_','dis_']
        tail_header = ['dvz','dvh','dhdg']

        # loop the intruder header part for the total number of intruders
        # considered in the state vector (n_aircraft)
        for i in range(n_aircraft):
            header_append = [s + str(i) for s in intruder_header]
            header = header + header_append

        header = header + tail_header

        self.logfile = pd.DataFrame(columns = header)
