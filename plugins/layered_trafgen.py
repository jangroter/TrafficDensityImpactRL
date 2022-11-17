""" traffic generation plugin
Plugin constantly manages how many aircraft 
should have been spawned based on the desired traffic 
density and time that has passed

Traffic spawns are randomly generated
"""
from bluesky import core, stack, traf  #, settings, navdb, sim, scr, tools
from bluesky.tools.aero import ft, nm, fpm, Rearth, kts
import bluesky as bs
import numpy as np
import area
import random
import math

def get_spawnrate(density,mean_speed):
    av_t_flight = (math.pi*circlerad*nm / 2)/(mean_speed*kts) #av_t_flight in [s], circlerad * nm converts to meters
    spawnrate = av_t_flight/(density*math.pi*(circlerad)**2) #traf_density in AC/NM so circle area in NM

    return spawnrate

# Default Values
circlelat = pm.controlarea_lat
circlelon = pm.controlarea_lon
circlerad = pm.controlarea_rad

speed = pm.cruise_speed
density = pm.traffic_density 
spawnrate = get_spawnrate(density,speed)

headinglayers = pm.num_headinglayers
degreesperheading = 2*(360/headinglayers)

basealtitude = pm.lower_alt
maxaltitude = pm.upper_alt
altperlayer = (maxaltitude - basealtitude)/(headinglayers)

def init_plugin():   
    # Configuration parameters

    layered_trafgen = Layered_trafgen()

    config = {
        # The name of your plugin
        'plugin_name':     'LAYERED_TRAFGEN',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config

class Layered_trafgen(core.Entity):

    def __init__(self):
        super().__init__()

        stack.stack(f'Circle sim_environment {circlelat} {circlelon} {circlerad}')
        # stack.stack('Area sim_environment')
        stack.stack(f'Pan {circlelat} {circlelon}')
        stack.stack('Zoom 4')

        self.aircraftnumber = 0

    @core.timed_function(name='layered_trafgen', dt=0.5)
    def update(self):
        req_acspawned = int(bs.sim.simt / spawnrate)
        spawn_defecit = req_acspawned - self.aircraftnumber
        for i in range(spawn_defecit):
            self.aircraftnumber += 1
            
            acid = str(self.aircraftnumber)
            heading = random.randint(0,359)
            altitude = self.get_altitude(heading)
            lat,lon = self.get_spawn_location(heading)
            
            stack.stack(f'CRE {acid} MAVIC {lat} {lon} {heading} {altitude} {speed}')

    def get_altitude(self, heading):
        layer           = random.randint(0,1)
        extraheight     = layer * (maxaltitude - basealtitude) / 2.0
        
        return basealtitude + ( altperlayer * (heading // degreesperheading)) + extraheight + altperlayer/4
    
    def get_spawn_location(self, heading):

        enterpoint = random.randint(-9999,9999)/10000
       
        bearing     = np.deg2rad(heading + 180) + math.asin(enterpoint)

        lat         = np.deg2rad(circlelat)
        lon         = np.deg2rad(circlelon)
        radius      = circlerad * 1.852
        
        latspawn    = np.rad2deg(self.get_spawn_latitude(bearing,lat, radius))
        lonspawn    = np.rad2deg(self.get_spawn_longitude(bearing, lon, lat, np.deg2rad(latspawn), radius))
        
        return latspawn, lonspawn
        
    def get_spawn_latitude(self,bearing,lat,radius):
        R = Rearth/1000.
        return math.asin( math.sin(lat)*math.cos(radius/R) +\
               math.cos(lat)*math.sin(radius/R)*math.cos(bearing))
        
    def get_spawn_longitude(self,bearing,lon,lat1,lat2,radius):
        R   = Rearth/1000.
        return lon + math.atan2(math.sin(bearing)*math.sin(radius/R)*\
                     math.cos(lat1),math.cos(radius/R)-math.sin(lat1)*math.sin(lat2))

    def get_exit_location(self, heading):
      
        bearing     = np.deg2rad(heading)

        lat         = np.deg2rad(circlelat)
        lon         = np.deg2rad(circlelon)
        radius      = circlerad * 1.852
        
        latexit    = np.rad2deg(self.get_spawn_latitude(bearing,lat, radius))
        lonexit    = np.rad2deg(self.get_spawn_longitude(bearing, lon, lat, np.deg2rad(latexit), radius))
        
        return latexit, lonexit

