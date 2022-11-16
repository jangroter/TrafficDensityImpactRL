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
state_size, _, _ = fn.get_statesize()
action_size = 3

default_vz = 2.5/fpm
default_speed = 10/kts

onsetperiod = 600 # Number of seconds before experiment starts

n_aircraft = bs.settings.num_aircraft


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

class Experiment_base(core.Entity):  
    def __init__(self):
        super().__init__()

        self.print = False

        self.controlac = 0 # int type variable to keep track of the number of ac controlled

        self.not_finished = 0

        self.rewards = np.array([])
        self.state_size = state_size
        self.action_size = action_size

        self.logfile = None
        self.lognumber = 0
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

    @core.timed_function(name='experiment_base', dt=timestep)
    def update(self):
        self.check_inside()
        for acid in traf.id:
            ac_idx = traf.id2idx(acid)
            self.update_AC_control(ac_idx)

            if self.control[ac_idx] == 1:
                self.call[ac_idx] += 1
                state_, logstate, int_idx = st.get_state(ac_idx, self.targetalt[ac_idx])
                
                if self.first[ac_idx]:
                    self.reset_action(acid,ac_idx)
                    self.first[ac_idx] = 0
                    self.controlac += 1

                else:
                    reward, done = rw.calc_reward(state_,logstate)
                    self.totreward[ac_idx] += reward
                    
                    action = np.array([2.5,0,0])

                    if done:
                        self.control[ac_idx] = 0
                        self.rewards = np.append(self.rewards, self.totreward[ac_idx])
                        stack.stack(f'alt {acid} {self.targetalt[ac_idx]}')
                        self.print = True

                    else:
                        self.reset_action(acid,ac_idx)
                    
                    if len(self.rewards) % 50 == 0 and self.print == True:
                        self.print = False
                        print(np.mean(self.rewards))
                        print(f'completed {len(self.rewards)} flights')

                        if len(self.rewards) == 5000:
                            stack.stack('QUIT')


                    self.log(logstate,action,acid,ac_idx)

    def update_AC_control(self,ac_idx):   
        if self.first[ac_idx]:                
            # Checks if the AC is inside the bounds of the delivery area
            inbounds    = fn.checkinbounds(traf.lat[ac_idx], traf.lon[ac_idx])               
            if inbounds:
                
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

    def check_inside(self):
        insdel = areafilter.checkInside('SIM_ENVIRONMENT', traf.lat, traf.lon, traf.alt)
        delidx = np.where(np.array(self.insdel) * (np.array(insdel) == False))[0]
        self.insdel = insdel

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
        data = [acid, self.acnum[ac_idx], self.call[ac_idx]] + list(logstate) + list(action)
        self.logfile.loc[len(self.logfile)] = data

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
        header = ['acid','acnum','callnum','alt_dif','vz','vh','d_hdg']
        intruder_header = ['int_','intrusion_','conflict_','tcpa_','dcpa_','dalt_','du_','dv_','dx_','dy_','dis_']
        tail_header = ['dvz','dvh','dhdg']

        for i in range(n_aircraft):
            header_append = [s + str(i) for s in intruder_header]
            header = header + header_append

        header = header + tail_header

        self.logfile = pd.DataFrame(columns = header)
