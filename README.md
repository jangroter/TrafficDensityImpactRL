# BlueSky - The Open Air Traffic Simulator

[![Open in Visual Studio Code](https://img.shields.io/static/v1?logo=visualstudiocode&label=&message=Open%20in%20Visual%20Studio%20Code&labelColor=2c2c32&color=007acc&logoColor=007acc)](https://open.vscode.dev/TUDelft-CNS-ATM/bluesky)
[![GitHub release](https://img.shields.io/github/release/TUDelft-CNS-ATM/bluesky.svg)](https://GitHub.com/TUDelft-CNS-ATM/bluesky/releases/)
![GitHub all releases](https://img.shields.io/github/downloads/TUDelft-CNS-ATM/bluesky/total?style=social)

[![PyPI version shields.io](https://img.shields.io/pypi/v/bluesky-simulator.svg)](https://pypi.python.org/pypi/bluesky-simulator/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/bluesky-simulator?style=plastic)
[![PyPI license](https://img.shields.io/pypi/l/bluesky-simulator?style=plastic)](https://pypi.python.org/pypi/bluesky-simulator/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/bluesky-simulator?style=plastic)](https://pypi.python.org/pypi/bluesky-simulator/)

BlueSky is meant as a tool to perform research on Air Traffic Management and Air Traffic Flows, and is distributed under the GNU General Public License v3.

The goal of BlueSky is to provide everybody who wants to visualize, analyze or simulate air
traffic with a tool to do so without any restrictions, licenses or limitations. It can be copied,
modified, cited, etc. without any limitations.

**Citation info:** J. M. Hoekstra and J. Ellerbroek, "[BlueSky ATC Simulator Project: an Open Data and Open Source Approach](https://www.researchgate.net/publication/304490055_BlueSky_ATC_Simulator_Project_an_Open_Data_and_Open_Source_Approach)", Proceedings of the seventh International Conference for Research on Air Transport (ICRAT), 2016.

## This Fork: "Analysing the Impact of Traffic Density on the Efficacy of Reinforcement Learning during Safe Vertical Manouevres."
The goal of this fork is to provide open-source, freely accessible source code and data for reproducing the results of the paper. To run the experiments and simulations, plugins have been developed, which can be found under:

    bluesky/plugins

To see how to enable a certain plugin open the respective file under the plugins folder. To set the traffic density or any other parameter, see the parameters.py file under the plugin folder.

The experimental scenario spawns aircraft at a pre-set traffic density at different altitude depending on their heading. Some of the aircraft can obtain a new altitude command which requires them to cross through different layers of cruising aircraft. During this activity intrusions or so called losses of separation can occur which is the main variable of interest in this research.

### experiment_base.py
This is the baseline experiment plugin. When this plugin is enabled, aircraft will directly start flying towards their new altitude without performing any conflict resolution manoeuvres to minimize the number of intrusions. This number of intrusions observed during these experiments provides a control for the other experiments.

### experiment_swo.py
This file extends on the baseline plugin by adding a shortest way out algorithm for resolving conflicts. This algorithm is based on the Modified Voltage Potential algorithm, an implementation of which can be found under bluesky/asas/MVP.py, however unlike MVP in this implementation only the aircraft that is manoeuvring vertically is responsible for resolving the conflicts. 

### experiment_drl.py & experiment_drl_test.py
These files extend on the baseline plugin by having the aircraft that have altitude command be controlled by a Deep Reinforcement Learning algorithm. The used algorithm is the Soft Actor Critic (SAC) algorithm. The experiment_drl.py file is enabled when a new model is to be trained based on a reward function that penalizes intrusions, this model is intermittently saved. The experiment_drl_test.py file is enabled to test a trained model without the exploration noise/randomness that is present in the model during training.

## BlueSky Releases
BlueSky is also available as a pip package, for which periodically version releases are made. You can find the latest release here:
https://github.com/TUDelft-CNS-ATM/bluesky/releases
The BlueSky pip package is installed with the following command:

    pip install bluesky-simulator[full]

Using ZSH? Add quotes around the package name: `"bluesky-simulator[full]"`. For more installation instructions go to the Wiki.

## BlueSky Wiki
Installation and user guides are accessible at:
https://github.com/TUDelft-CNS-ATM/bluesky/wiki

## Some features of BlueSky:
- Written in the freely available, ultra-simple-hence-easy-to-learn, multi-platform language
Python 3 (using numpy and either pygame or Qt+OpenGL for visualisation) with source
- Extensible by means of self-contained [plugins](https://github.com/TUDelft-CNS-ATM/bluesky/wiki/plugin)
- Contains open source data on navaids, performance data of aircraft and geography
- Global coverage navaid and airport data
- Contains simulations of aircraft performance, flight management system (LNAV, VNAV under construction),
autopilot, conflict detection and resolution and airborne separation assurance systems
- Compatible with BADA 3.x data
- Compatible wth NLR Traffic Manager TMX as used by NLR and NASA LaRC
- Traffic is controlled via user inputs in a console window or by playing scenario files (.SCN)
containing the same commands with a time stamp before the command ("HH:MM:SS.hh>")
- Mouse clicks in traffic window are use in console for lat/lon/heading and position inputs

## Contributions
BlueSky can be considered 'perpetual beta'. We would like to encourage anyone with a strong interest in
ATM and/or Python to join us. Please feel free to comment, criticise, and contribute to this project. Please send suggestions, proposed changes or contributions through GitHub pull requests, preferably after debugging it and optimising it for run-time performance.
