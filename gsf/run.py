#!/usr/bin/python3
# SPDX-License-Identifier: GPL-3.0-or-later

import sys, os
import gsf

"""
This is an example script showing how to run GalacticStructureFinder (GSF) either 
for one model only (fixed number_of_clusters), or in a loop to find the optimal 
number_of_clusters.

GSF separates the stellar particles in a simulated dark matter halo into a 
fixed number of components using Gaussian Mixture Models in an arbitrary, 
user-defined feature space. 

The only required input are three files containing the star, gas, and dark 
matter particles properties, in this order. The galaxy is assumed isolated, 
and centered, and the expected units are: Msun for masses, kpc for Cartesian
coordinates, and km/s for Cartesian velocities.
"""


## Define here spatial filters to cut the particle data. 
filters = None
## The 'property' has to be some available property (either directly from the input or derivable)
## If the filter 'type' is 'BandPass', you have to also give both the 'LowLimit' and 'UpLimit' values.
## If the filter 'type' is 'LowPass' or 'HighPass', you need to give the 'Limit' value.
## e.g. filter definition to select the particles within a 3D radius of 4 kpc
#filters = {1:{'type':'LowPass','property':'r3','Limit':4.},
           #'add_to_file_names':'inner_region'}
## e.g. filter definition to select the particles with 3D radius 0.1<r3<4 [kpc], normalized binding energies e<-0.5, and ages age>10 [Gyr]
#filters = {1:{'type':'BandPass','property':'r3','LowLimit':0.1,'UpLimit':4.},
           #2:{'type':'LowPass','property':'e','Limit':-0.5},
           #3:{'type':'HighPass','property':'age','Limit':10.},
           #'add_to_file_names':'old_bulge'}
## e.g. filter definition to select a ring with radial width 2 kpc and vertical width of 2 kpc at a 2D radius of 8 kpc
#filters = {1:{'type':'BandPass','property':'r2','LowLimit':7.,'UpLimit':9.},
           #2:{'type':'BandPass','property':'z','LowLimit':-1.,'UpLimit':1.},
           #'add_to_file_names':'solar_neighbourhood'}

# The directory where all output will be saved
out_dir = 'tests/sim1/'
# The three data files needed as input
file_star = 'tests/sim1/sim1.halo_1.star.dat'
file_gas = 'tests/sim1/sim1.halo_1.gas.dat'
file_dark = 'tests/sim1/sim1.halo_1.dark.dat'


doloop = False  # switch to True if you want to find the optimal number of components
if doloop:
    print('Run gsf in a loop to get the plot log Likelihood vs nk, and log Likelihood vs n_param.')
    print('The log(L) vs n_param will be used to do the model selection using the elbow method.')
    gsf.gsf_loop(file_star, file_gas, file_dark, varlist='jzjc,jpjc,e', out_dir=out_dir,
                 n_init=10, verbose=False, filters=filters)  
else:
    print('Run gsf only for what is supposed to be a reasonable number of components (e.g. 3) to generate the moments maps.') 
    number_of_clusters = 3
    gsf.gsf(file_star, file_gas, file_dark, varlist='jzjc,jpjc,e', number_of_clusters=number_of_clusters, out_dir=out_dir,
            n_init=10, plot=True, band=False, M2L=False, inclination=90., fov=100, verbose=False, filters=filters)
