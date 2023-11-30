#!/usr/bin/python3

import sys, os
import gsf

print('This script calls gsf given valid particle data for one halo.')
print('gsf.py is a collection of functions that cover the computation of the gravitational potential,')
print('a wrapper for the Gaussian Mixture Models of scikit-learn and various plotting options.')
print('The only required input are three files holding the stellar, gas and dark matter particle data for one halo ONLY.')
print('The only mandatory fields for the gas and dark matter data files are: mass[M_sun], x[kpc], y[kpc], z[kpc].')
print('For the stellar data file, the mandatory fields are: mass[M_sun], x[kpc], y[kpc], z[kpc], vx[km/s], vy[km/s], vz[km/s].')
print('If no optional argument is given, the code will look for 2 clusters in the kinematic stellar space of')
print('(jz/jc,jp/jc,binding_energy), and all output will be saved in the directory set by the out_dir arg')
print('To have an idea of the various arguments that can be set use verbose=True.')
print('The feature space can be any combination of mandatory fields, properties derived from the mandatory ones (as described in gsf.available_features()),')
print('or any other extra properties present in the stellar data file.')
print('If your desired feature does not exist in gsf.available_features(), you only need to add it to this function,')
print('and describe how it should be computed in gsf.generate_tmp_file().')
print('The runtime for the computation of the potential scales as ~N^2')
print('If you want to run this part of gsf in parallel, set the system variable OMP_NUM_THREADS first (e.g. from bash: export OMP_NUM_THREADS=6).')
print('In the case loop4optimaln=True, main_gsf loops over number_of_clusters between 1 and 15, ')
print('and plots the log Likelihood as a function of number_of_clusters, and the log Likelihood and the scree test statistics as functions of the number of free parameteres.')
print('In this case no moments maps figures are generated.')
print('By default loop4optimaln=False, and the code plots also the maps of the zeroth, first and second order moments.')

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

# The three data files needed as input
out_dir = '/data/aco1/project_victor/g5.22e12/'
file_star = out_dir+'g5.22e12.01024.halo_1.align_with_star.star.dat'
file_gas = out_dir+'g5.22e12.01024.halo_1.align_with_star.gas.dat'
file_dark = out_dir+'g5.22e12.01024.halo_1.align_with_star.dark.dat'


## Run gsf in a loop (loop4optimaln=True) to get the plot log Likelihood vs nk, and log Likelihood vs n_param.
## The log(L) vs n_param will be used to do the model selection using the CHull method (Ceulemans & Kiers 2006)
#gsf.main_gsf(file_star, file_gas, file_dark, out_dir=out_dir, eps=0.01,radius_align=None,
#             varlist=['jzjc','jpjc','e'], loop4optimaln=True,
#             covariance_type='full', whiten_data=True, n_init=100, 
#             plot=True, band=False, M2L=False, inclination=90., filters=filters, verbose=False)


# Run gsf only for what is supposed to be a reasonable number of components to generate the moments maps. 
# For this you need to let loop4optimaln to the default value loop4optimaln=False.
for number_of_clusters in [2,3]:
    gsf.main_gsf(file_star, file_gas, file_dark, out_dir=out_dir, eps=0.1,radius_align=None,
                 varlist=['x','y','z','vx','vy','vz'], loop4optimaln=False,number_of_clusters=number_of_clusters, covariance_type='full', 
                 whiten_data=True, n_init=100, plot=True, band=False, M2L=False, inclination=90., filters=filters, fov=100, verbose=False)

