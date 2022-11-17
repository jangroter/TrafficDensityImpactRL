"""This file contains the values necessary for running the experiments"""

### Simulation variables
timestep = 1.5 # seconds
onsetperiod = 600 # seconds

### Scenario variables
traffic_density = 50

# experiment name
experiment_name = 'NoResolution_50'
experiment_path = 'output'

# Indicate the logfile path
log_path = 'output/output'

# Separation requirements
# ASAS horizontal PZ margin [nm]
asas_pzr = 0.027

# ASAS vertical PZ margin [ft]
asas_pzh = 25.0

cruise_speed = 10 # m/s
def_vz = 2.5 # m/s

controlarea_lat = 48.86
controlarea_lon = 2.37
controlarea_rad = 1.62 # NM
deliveryradius = 1.35 # NM

num_headinglayers = 16
lower_alt = 200 # ft
upper_alt = 1000 # ft

# Chance of getting altitude change command
altchangechance = 5 # %

### State variables
# Area in which to look for potential conflicts [nm]
searchradius = 0.3
searchlayers = 4

# Number of aircraft to include in the statevector
num_aircraft = 3
