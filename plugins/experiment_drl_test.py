""" 
DRL Experiment test file that runs the simulation scenario as a testing environment
for the Soft Actor Critic algorithm. Running this loads a model that is tested
until 5000 episodes are run. 

TO RUN THIS EXPERIMENT change the enabled_plugins in the settings.cfg file to:

enabled_plugins = ['area', 'layered_trafgen', 'experiment_drl_test']
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

import plugins.parameters as pm

import matplotlib.pyplot as plt

import plugins.SAC.sac_agent as sac
import plugins.state as st
import plugins.functions as fn
import plugins.reward as rw

timestep = pm.timestep
state_size, _, _ = fn.get_statesize()
action_size = 3

onsetperiod = pm.onsetperiod # Number of seconds before experiment starts

n_aircraft = pm.num_aircraft


# '\\' for windows, '/' for linux or mac
dir_symbol = '\\'
model_path = pm.experiment_path + dir_symbol + pm.experiment_name

# Standard way of initializing Plugins in BlueSky
def init_plugin():
    experiment_drl_test = Experiment_drl_test()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'EXPERIMENT_DRL_TEST',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config

# Main experiment class
class Experiment_drl_test(core.Entity):  
    def __init__(self):
        super().__init__()

        self.agent = sac.SAC(action_size,state_size,model_path,test=True) # Initialize the SAC algorithm in testing mode
        self.agent.load_models() # load the trained models

        self.print_value = 100
        self.print = False

        self.controlac = 0 # int type variable to keep track of the number of ac controlled

        self.rewards = np.array([]) # array that stores the total return after each episode
        self.plot_reward = np.array([]) # array that stores the rolling average return for plotting
        self.state_size = state_size
        self.action_size = action_size

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

            self.state = []    # Latest state information
            self.action = []    # Latest selected actions
            self.choice = []    # Latest intended actions (e.g no noise)

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

    # Main function that is looped every 'timestep' seconds (simulation time, not real time)
    @core.timed_function(name='experiment_drl_test', dt=timestep)
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
                staten_ = fn.normalize_state(np.array(state_))

                # Set some default values for the next state transition if it is the first 
                # time aircraft gets control command
                if self.first[ac_idx]:
                    self.first[ac_idx] = 0
                    self.acnum[ac_idx] = self.controlac
                    reward = 0 # Cant get a reward in the first step
                    done = 0
                    self.controlac += 1

                # If not first step, determine reward from new state and check done    
                else:
                    reward, done = rw.calc_reward(state_,logstate)
                    self.totreward[ac_idx] += reward

                # If done, let AC cruise again, remove from control
                if done:
                    self.control[ac_idx] = 0
                    self.rewards = np.append(self.rewards, self.totreward[ac_idx])
                    stack.stack(f'alt {acid} {self.targetalt[ac_idx]}')
                    self.print = True
                
                # If not done select and do action based on current state
                else:
                    action = self.agent.step(staten_)
                    self.do_action(action[0],acid,ac_idx)

                # Logging and printing for debugging and tracking progress
                if len(self.rewards) > self.print_value and self.print == True:
                    self.print_value += 100
                    self.print = False
                    print(np.mean(self.rewards), self.not_finished)
                    self.not_finished = 0

                    self.plot_reward = np.append(self.plot_reward,np.mean(self.rewards[-2000:]))

                    fig, ax = plt.subplots()
                    ax.plot(self.plot_reward, label='Rolling mean of 500 reward')
                    ax.set_yscale('symlog', linthresh=0.001)
                    fig.savefig('reward_mean.png')
                    plt.close(fig)

                if len(self.rewards) == 5000:
                        stack.stack('QUIT')

                self.log(logstate,action[0],acid,ac_idx)

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

    def do_action(self,action,acid,ac_idx):
        """ execute the different actions outputted by the SAC agent"""
        self.do_vz(action,acid,ac_idx)     
        self.do_vh(action,acid,ac_idx) 
        self.do_hdg(action,acid,ac_idx)

    def do_vz(self,action,acid,ac_idx):
        """ Translate the tanh output for vertical speed 
        to vertical speed commands"""
        vz = ((action[0] + 1.)*2.5) /fpm
        if vz < 20:
            target = traf.alt[ac_idx] / ft
            vz = 0        
        elif self.targetalt[ac_idx]*ft < traf.alt[ac_idx]:
            target = 0               
        else:
            target = 5000
        
        stack.stack(f'alt {acid} {target} {vz}') 

    def do_vh(self,action,acid,ac_idx):
        """ Translate the tanh output for horizontal speed 
        to horizontal speed commands """        
        vh = action[1]*2.5
        
        targetvelocity  = (traf.cas[ac_idx] + vh)/kts
        targetvelocity  = np.clip(targetvelocity,10,30)
        
        stack.stack(f'SPD {acid} {targetvelocity}')
    
    def do_hdg(self,action,acid,ac_idx):
        """ Translate the tanh output for heading
        to heading commands """ 
        heading = traf.hdg[ac_idx]
        hdg = action[2]*45 
        
        targetheading   = heading + hdg

        if targetheading < 0:
            targetheading = 360 + targetheading
        
        if targetheading > 360:
            targetheading = targetheading - 360
        
        stack.stack(f'HDG {acid} {targetheading}')

    def log(self,logstate,action,acid,ac_idx):
        """ Function used for logging the data"""
        data = [acid, self.acnum[ac_idx], self.call[ac_idx]] + list(logstate) + list(action)
        self.logfile.loc[len(self.logfile)] = data

        # write out logfile every 10000 state transitions and create new logfile to prevent RAM saturation
        if len(self.logfile) == 10000:
            lognumber = str(self.lognumber)    

            if self.lognumber == 0:
                path = model_path
                Path(path).mkdir(parents=True, exist_ok=True)
                path = model_path+dir_symbol+"model"
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

        for i in range(n_aircraft):
            header_append = [s + str(i) for s in intruder_header]
            header = header + header_append

        header = header + tail_header

        self.logfile = pd.DataFrame(columns = header)

