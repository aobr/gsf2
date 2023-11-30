# Copyright (c) 2018--, Aura Obreja and the GalacticStructureFinder (gsf) contributors.
# GalacticStructureFinder decomposes simulated galaxies based on their stellar kinematics. 
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This program uses third-party libraries:
# - numpy (http://www.numpy.org/)
# - scipy (https://www.scipy.org/)
# - matplotlib (https://matplotlib.org/)
# - pynbody (https://github.com/pynbody/pynbody)
# - scikit-learn (http://scikit-learn.org/stable/)
# Please refer to the original licensing conditions for each of these third party software libraries. 
# 
# If you use this program or parts of it, please cite the original article presenting gsf: 
# Obreja, Maccio, Moster et al MNRAS 2018, "Introducing galactic structure finder: the 
# multiple stellar kinematic structures of a Milky Way mass galaxy" 2018MNRAS.477.4915O 
#  

import pickle, os, glob, pdb, time
import numpy as np
import scipy
import scipy.interpolate
import scipy.special
from scipy import linalg
from scipy.spatial import ConvexHull
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sklearn.preprocessing import scale
from sklearn.mixture import GaussianMixture as GMM
from functools import reduce
import PIL
from PIL import Image
import twobody
import gc

grav_const=4.302e-6
plt.switch_backend('agg')
#plt.ion()


#################################################################################################################################
############## functions controling the available features for clustering #######################################################
#################################################################################################################################

def available_features(verbose=True):
    if verbose:
        print('The available features are:')
        print('- Cartesian coordinates in kpc: x, y, z')
        print('- Cartesian velocities in km/s: vx, vy, vz')
        print('- Stellar age in Gyr: age')
        print('- Formation time in Gyr: tform')
        print('- Metallicity in mass fraction: Z')
        print('- Metallicity in Iron: FeH')
        print('- Alpha enhancement: OFe')
        print('- Specific angular momentum of the corresponding (same energy) circular orbit: jc')
        print('- Cartesian specific angular momentum: jx, jy, jz')
        print('- Specific angular momentum along a direction perpendicular to the z-axis: jp')
        print('- Circularity parameter: jzjc')
        print('- The normal to the circularity parameter: jpjc')
        print('- Normalized binding energy: e')
        print('- Specific kinetic energy: ke')
        print('- Specific potential energy: pe')
        print('- Radius in the equatorial plane: r2')
        print('- 3D radius: r3')
        print('- Height above the equatorial plane: height')
        print('- Radial velocity in the equatorial plane: vR')
        print('- Rotational velocity (in the equatorial plane): vphi')
        print('- sqrt(vz^2+vR^2): vnorot')
        print('- sqrt(vz^2+vx^2): vT')
        print('- orbital actions: JR, JPhi, JZ')
    return ['x','y','z','vx','vy','vz','age','Z','FeH','OFe','ke','pe',
            'jc','jz','jx','jy','jp','jzjc','jpjc','e','r2','r3','height','vR','vphi','vnorot','vT',
            'JR','JPhi','JZ']

def feature_labels(feature):
    label = feature
    if feature=='x': label=r"x [kpc]"
    if feature=='y': label=r"y [kpc]"
    if feature=='z': label=r"z [kpc]"
    if feature=='vx': label=r"v$_{\rm x}$ [km s$^{\rm -1}$]"
    if feature=='vy': label=r"v$_{\rm y}$ [km s$^{\rm -1}$]"
    if feature=='vz': label=r"v$_{\rm z}$ [km s$^{\rm -1}$]"
    if feature=='age': label=r"age [Gyr]"
    if feature=='tform': label=r"t$_{\rm form}$ [Gyr]"
    if feature=='Z': label=r"Z"
    if feature=='FeH': label=r"[Fe/H]"
    if feature=='OFe': label=r"[O/Fe]"
    if feature=='jc': label=r"j$_{\rm c}$ [kpc km s$^{\rm -1}$]"
    if feature=='jz': label=r"j$_{\rm z}$ [kpc km s$^{\rm -1}$]"
    if feature=='jx': label=r"j$_{\rm x}$ [kpc km s$^{\rm -1}$]"
    if feature=='jy': label=r"j$_{\rm y}$ [kpc km s$^{\rm -1}$]"
    if feature=='jp': label=r"j$_{\rm p}$ [kpc km s$^{\rm -1}$]"
    if feature=='jzjc': label=r"j$_{\rm z}$/j$_{\rm c}$"
    if feature=='jpjc': label=r"j$_{\rm p}$/j$_{\rm c}$"
    if feature=='e': label=r"e/max(|e|)"
    if feature=='ke': label=r"e$_{\rm kin}$ [km$^{\rm 2}$s$^{\rm -2}$]"
    if feature=='pe': label=r"e$_{\rm grav}$ [km$^{\rm 2}$s$^{\rm -2}$]"
    if feature=='r2': label=r"R [kpc]"
    if feature=='r3': label=r"r [kpc]"
    if feature=='height': label=r"|z| [kpc]"
    if feature=='vR': label=r"v$_{\rm R}$ [km s$^{\rm -1}$]"
    if feature=='vphi': label=r"v$_{\rm\phi}$ [km s$^{\rm -1}$]"
    if feature=='vnorot': label=r"$\rm\sqrt{v_{\rm R}^{\rm 2}+v_{\rm z}^{\rm 2}}$ [km s$^{\rm -1}$]"
    if feature=='vT': label=r"$\rm\sqrt{v_{\rm x}^{\rm 2}+v_{\rm z}^{\rm 2}}$ [km s$^{\rm -1}$]"
    if feature=='JR': label=r"$J_{\rm R}$ [kpc km s$^{\rm -1}$]"
    if feature=='JPhi': label=r"$J_{\rm\phi}$ [kpc km s$^{\rm -1}$]"
    if feature=='JZ': label=r"$J_{\rm Z}$ [kpc km s$^{\rm -1}$]"
    return label

def feature_range_and_nbin(feature):
    frange, fnbin = [0.,1.], 100
    if feature=='x': frange, fnbin = [-30.,30.], 150
    if feature=='y': frange, fnbin = [-30.,30.], 150
    if feature=='z': frange, fnbin = [-30.,30.], 150
    if feature=='vx': frange, fnbin = [-300.,300.], 300
    if feature=='vy': frange, fnbin = [-300.,300.], 300
    if feature=='vz': frange, fnbin = [-300.,300.], 300
    if feature=='age': frange, fnbin = [0.,14.], 140
    if feature=='tform': frange, fnbin = [0.,14.], 140
    if feature=='Z': frange, fnbin = [0.,0.06], 60
    if feature=='FeH': frange, fnbin = [-2.5,1.0], 60
    if feature=='OFe': frange, fnbin = [-0.5,0.5], 100
    if feature=='jc': frange, fnbin = [0.,1.e4], 100
    if feature=='jx': frange, fnbin = [-1.e4,1.e4], 200
    if feature=='jy': frange, fnbin = [-1.e4,1.e4], 200
    if feature=='jz': frange, fnbin = [-1.e4,1.e4], 200
    if feature=='jp': frange, fnbin = [0.,1.e4], 100
    if feature=='jzjc': frange, fnbin = [-1.2,1.2], 240
    if feature=='jpjc': frange, fnbin = [0.,1.2], 120
    if feature=='e': frange, fnbin = [-1.,0.], 100
    if feature=='r2': frange, fnbin = [0.,12.], 150
    if feature=='r3': frange, fnbin = [0.,30.], 150
    if feature=='height': frange, fnbin = [0.,5.0], 50
    if feature=='vR': frange, fnbin = [-300.,300.], 300
    if feature=='vphi': frange, fnbin = [-300.,300.], 300
    if feature=='vnorot': frange, fnbin = [0.,300.], 150
    if feature=='vT': frange, fnbin = [0.,300.], 150
    if feature=='JR': frange, fnbin = [0., 1500.], 150
    if feature=='JPhi': frange, fnbin = [-1000., 3000.], 400
    if feature=='JZ': frange, fnbin = [0., 1000.], 100
    return frange, fnbin

def selected_features(varlist, featurelist):
    good_tags = []
    for k in range(len(varlist)):
        if varlist[k] in featurelist: 
            good_tags.append(varlist[k])
    return good_tags


def select_optimal_model(nk,nparam,logL,figout=None):
    
    nk = np.array(nk)
    nparam = np.array(nparam)
    logL = np.array(logL)
    
    plt.close()
    fig = plt.figure(figsize=(7.0,4.0))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0])
    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)
    ax.set_xlabel(r"n$_{\rm param}$", fontsize=18)
    ax.set_ylabel(r"log(L)", fontsize=18)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2

    ax2 = ax.twinx()
    ax2.set_ylabel(r"st", fontsize=18,color='dodgerblue')
    plt.setp(ax2.get_yticklabels(), fontsize=18)
    ax2.yaxis.labelpad = 2
    ax2.tick_params(axis='y', labelcolor='dodgerblue')

    ax.scatter(nparam,logL,color='black',zorder=1)
    ax.plot(nparam,logL,color='black',zorder=1)
    points = (np.array([nparam,logL])).transpose()
    hull = ConvexHull(points)
    nparam_keep = points[hull.vertices,0]
    logL_keep = points[hull.vertices,1]
    logL_keep = logL_keep[np.argsort(nparam_keep)]
    nparam_keep = nparam_keep[np.argsort(nparam_keep)]
    ax.scatter(nparam_keep,logL_keep,color='violet',s=30,marker='*',zorder=2)
    ax.plot(nparam_keep,logL_keep,color='violet',zorder=2)
    j = np.arange(len(nparam_keep))
    nparam_select = []
    gain_select = []
    for jj in j[1:-1]:
        nparam_select.append(nparam_keep[jj])
        top = (logL_keep[jj]-logL_keep[jj-1])/(nparam_keep[jj]-nparam_keep[jj-1])
        bottom = (logL_keep[jj+1]-logL_keep[jj])/(nparam_keep[jj+1]-nparam_keep[jj])
        gain_select.append(top/bottom)
    ax2.scatter(np.array(nparam_select),np.array(gain_select),s=30,marker='*',color='dodgerblue')
    ax2.plot(np.array(nparam_select),np.array(gain_select),color='dodgerblue')
    nk_select = []
    for npa in nparam_select: nk_select.append(int(nk[nparam==npa]))
    srt = np.argsort(gain_select)
    
    nk_out = np.array(nk_select)
    nparam_out = np.array(nparam_select)
    scree_out = np.array(gain_select)
    
    ytext = 0.35
    for ll in range(len(gain_select)):
        ax2.text(0.80,ytext,r"%i"%np.array(nk_select)[srt][ll],color='black',fontsize=15,ha='center',va='center',transform=ax2.transAxes)
        ytext = ytext+0.07
    ax2.text(0.80,ytext,r"n$_{\rm k}$",color='black',fontsize=15,ha='center',va='center',transform=ax2.transAxes)
    ytext = 0.35
    for ll in range(len(gain_select)):
        ax2.text(0.88,ytext,r"%.1f"%np.array(gain_select)[srt][ll],color='dodgerblue',fontsize=15,ha='center',va='center',transform=ax2.transAxes)
        ytext = ytext+0.07
    ax2.text(0.88,ytext,r"st",color='dodgerblue',fontsize=15,ha='center',va='center',transform=ax2.transAxes)
    pos_max = np.argmax(np.array(gain_select))
    nk_1best = nk[nparam==nparam_select[pos_max]][0]
    gain_select.pop(pos_max)
    nparam_select.pop(pos_max)
    pos_max = np.argmax(np.array(gain_select))
    nk_2best = nk[nparam==nparam_select[pos_max]][0]
    if 7<nk_1best: nk_best = nk_2best
    else: nk_best = nk_1best

    ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=False, right=True)
    ax2.tick_params(axis='both', which='both', direction='in', bottom=False, top=False, left=True, right=False)
    gs.update(left=0.15,bottom=0.14,right=0.90, top=0.95, hspace=0.40, wspace=0.40)
    if figout is not None: plt.savefig(figout)
    plt.close()
    
    print('The scree test selected the model with %i Gaussians as most favoured.'%nk_best)

    srt = np.argsort(scree_out)

    return nk_out[srt][::-1], nparam_out[srt][::-1], scree_out[srt][::-1]

#################################################################################################################################
########## functions for the computation of the gravitational potential and the definition of the equatorial plane ##############
#################################################################################################################################

def star_potential(file_star,file_dark,file_gas,out_dir=None,eps=0.1,verbose=True):

    if verbose:
        print('-------------------------------------------------------------------------------------------------------------------------')
        print('This functions calls the Fortran90 parallel module twobody to compute the gravitational potential')
        print('Required input: the files containing the stellar, dark matter and gas particle data.')
        print('By default, this function uses a gravitational softening of 0.1 kpc. Change the eps arg to some other appropriate value.')
        print('The output file will be saved in the same directory as the stellar particle input file if the arg out_dir is None.')
    
    start0 = time.time()
    if out_dir is None: 
        fileout = file_star[:-4]+'.phi.dat'
    else: 
        aux = file_star.split('/')
        fileout = aux[-1]
        fileout = out_dir+fileout[:-4]+'.phi.dat'

    if os.path.isfile(fileout):
        print('The potential file already exists. Returning...')
        return fileout

    data_star = pickle.load(open(file_star,'rb'))
    ms = data_star['mass']
    xs = data_star['x']
    ys = data_star['y']
    zs = data_star['z']
    print('total number of stellar particles = %i'%len(ms))
    
    print('Computing the contribution of all the stars to the gravitational potential at all stellar particles positions...')
    start1 = time.time()
    phi_from_star = twobody.star_star_potential(ms,xs,ys,zs,eps)    
    end1 = time.time()
    print('... took %s'%secondsToStr(end1-start1))
    
    print('Computing the contribution of all the gas to the gravitational potential at all stellar particles positions...')
    data_gas = pickle.load(open(file_gas,'rb'))
    m = data_gas['mass']
    x = data_gas['x']
    y = data_gas['y']
    z = data_gas['z']
    print('total number of gas particles = %i'%len(m))
    start2 = time.time()
    phi_from_gas = twobody.star_other_potential(m,x,y,z,xs,ys,zs,eps)    
    end2 = time.time()
    print('... took %s'%secondsToStr(end2-start2))
    
    print('Computing the contribution of all the dark matter to the gravitational potential at all stellar particles positions...')
    data_dark = pickle.load(open(file_dark,'rb'))
    m = data_dark['mass']
    x = data_dark['x']
    y = data_dark['y']
    z = data_dark['z']
    print('total number of dark matter particles = %i'%len(m))
    start3 = time.time()
    phi_from_dark = twobody.star_other_potential(m,x,y,z,xs,ys,zs,eps)    
    end3 = time.time()
    print('... took %s'%secondsToStr(end3-start3))

    phi = -(phi_from_star+phi_from_gas+phi_from_dark)*grav_const
    
    f = open(fileout,'wb')
    pickle.dump({'phi':phi},f)
    f.close()
    del(f)
    end = time.time()
    
    print('Total runtime: %s'%secondsToStr(end-start0))
    print('-------------------------------------------------------------------------------------------------------------------------')

    del(m,x,y,z,ms,xs,ys,zs,phi_from_dark,phi_from_gas,phi_from_star,phi)
    gc.collect()

    return fileout


def com_Shrinking_sphere(data, coef_nlow = 0.1, verbose = True):
    # Added by Sara Ortega
    if verbose:
        print('-------------------------------------------------------------------------------------------------------------------------')
        print('This functions recalculates the centre of mass and velocities using the particles within an sphere containing 0.1 of the total mass.')
        print('Required input: the data containing the mass, velocities and positions.')
        print('By default, this function uses a coefficient of 0.1. Change the eps arg to some other appropriate value.')

    # Start from standard com:
    Mstar = np.sum(data['mass'])
    x_com = np.sum(np.multiply(data['x'],data['mass']))
    y_com = np.sum(np.multiply(data['y'],data['mass']))
    z_com = np.sum(np.multiply(data['z'],data['mass']))
    x_com /= Mstar
    y_com /= Mstar
    z_com /= Mstar
    
    # Initial arrays:
    xsh = data['x']
    ysh = data['y']
    zsh = data['z']
    msh = data['mass']
    
    # Shrinking sphere conditions 
    n_part = len(data['mass'])  
    nlow = coef_nlow*n_part
    
    # Get initial positions
    xpri = xsh - x_com
    ypri = ysh - y_com  
    zpri = zsh - z_com
    rpri = np.sqrt(np.square(xpri)+np.square(ypri)+np.square(zpri))
    rmax = np.amax(rpri)
    cont = 0
    no_it = 0
    na = len(rpri)
    
    # Shrink the sphere until the desired number of particles 
    while ((na>=nlow) & (na>=100)):
        a = np.where(rpri <= (0.975*rmax))
        na = len(a[0])
        if na > 0:
            xcut = xsh[a]
            ycut = ysh[a]
            zcut = zsh[a]
            
            mcut = msh[a]
            mtotal = np.sum(mcut)
            xcm0 = np.sum(xcut*mcut)/mtotal
            ycm0 = np.sum(ycut*mcut)/mtotal
            zcm0 = np.sum(zcut*mcut)/mtotal
                        
            xpri = xcut-xcm0
            ypri = ycut-ycm0
            zpri = zcut-zcm0
            rpri = np.sqrt(np.square(xpri)+np.square(ypri)+np.square(zpri))
            rmax = np.amax(rpri)
            xsh=xcut
            ysh=ycut
            zsh=zcut
            msh=mcut
            cont=cont+1
        else:
            print("No puedo iterar")  
            no_it = no_it+1
            
    # Recalculate positions with new com and calculate vcom
    xpri = data['x'] - xcm0
    ypri = data['y'] - ycm0
    zpri = data['z'] - zcm0
    rpri = np.sqrt(np.square(xpri)+np.square(ypri)+np.square(zpri))
    arpri = np.where(rpri < rmax)
    Mstar = np.sum(data['mass'][arpri])
    vx_com = np.sum(np.multiply(data['vx'][arpri],data['mass'][arpri]))
    vy_com = np.sum(np.multiply(data['vy'][arpri],data['mass'][arpri]))
    vz_com = np.sum(np.multiply(data['vz'][arpri],data['mass'][arpri]))
    vx_com /= Mstar
    vy_com /= Mstar
    vz_com /= Mstar 
    
    return xcm0, ycm0, zcm0, vx_com, vy_com, vz_com        
            

def midplane_potential(file_star,file_dark,file_gas,out_dir=None,eps=0.1,radius_align=None,verbose=True,*args, **kwargs):

    if verbose:
        print('--------------------------------------------------------------------------------------------------------------------------')
        print('This functions calls the Fortran90 parallel module twobody to compute the gravitational potential in the equatorial plane.')
        print('The z-axis perpendicular to this plane is defined by the angular momentum of star inside 0.1rvir.')
        print('Set the arg radius_align to override the 0.1rvir default. The value can be either in fraction of rvir or in kpc.')
        print('By default, this function uses a gravitational softening of 0.1 kpc. Change the eps arg to some other appropriate value.')
        print('Required input: the files containing the stellar, dark matter and gas particle data.')
        print('The output file will be saved in the same directory as the stellar particle input file if the arg out_dir is None.')

    start = time.time()
    if out_dir is None: 
        fileout = file_star[:-4]+'.midplane_potential.dat'
    else: 
        aux = file_star.split('/')
        fileout = aux[-1]
        fileout = out_dir+fileout[:-4]+'.midplane_potential.dat'

    if os.path.isfile(fileout):
        print('The midplane potential file already exists. Returning...')
        return fileout

    print('Make sure the positions and velocities are in the c.o.m. reference system of the stars')
    data = pickle.load(open(file_star,'rb'))
        
    x_com, y_com, z_com, vx_com, vy_com, vz_com = com_Shrinking_sphere(data) # use the Shrinking sphere method to get the position and velocity of com
    rstar = np.sqrt((data['x']-x_com)**2+(data['y']-y_com)**2+(data['z']-z_com)**2)

    print('Find out the virial radius as the maximum radius of dark matter particles')
    data_dark = pickle.load(open(file_dark,'rb'))
    rvir = np.max(np.sqrt((data_dark['x']-x_com)**2+(data_dark['y']-y_com)**2+(data_dark['z']-z_com)**2))

    print('Define the maximum and minimum radius radii for the particles to be considered for the definition of the equatorial plane')
    if radius_align is None: radius_align = 0.1*rvir
    else:
        if radius_align < 1.0: radius_align = radius_align*rvir
    rmin = 3.*eps

    print('Compute the angular momentum of stars with %.2f < r/kpc < %.2f'%(rmin,radius_align))
    ain = np.where((rstar>rmin)&(rstar<radius_align))
    Jx = np.sum(((data['y'][ain]-y_com)*(data['vz'][ain]-vz_com)-(data['z'][ain]-z_com)*(data['vy'][ain]-vy_com))*data['mass'][ain])
    Jy = np.sum(((data['z'][ain]-z_com)*(data['vx'][ain]-vx_com)-(data['x'][ain]-x_com)*(data['vz'][ain]-vz_com))*data['mass'][ain])
    Jz = np.sum(((data['x'][ain]-x_com)*(data['vy'][ain]-vy_com)-(data['y'][ain]-y_com)*(data['vx'][ain]-vx_com))*data['mass'][ain])

    R_matrix = calc_faceon_matrix(np.array([Jx,Jy,Jz]))

    # apply the rotation matrix to the stellar arrays
    pos = np.array([data['x']-x_com,data['y']-y_com,data['z']-z_com]) 
    pos = np.dot(R_matrix, pos).transpose() # rotated x/y/z is now pos[:,0/1/2]
    vel = np.array([data['vx']-vx_com,data['vy']-vy_com,data['vz']-vz_com]) 
    vel = np.dot(R_matrix, vel).transpose() # rotated vx/vy/vz is now vel[:,0/1/2]

    print('test that the rotation matrix works as intented...')
    rstar = np.sqrt(pos[:,0].flatten()**2+pos[:,1].flatten()**2+pos[:,2].flatten()**2)
    ain = np.where((rstar>rmin)&(rstar<radius_align))
    Jx = np.sum((pos[:,1].flatten()*vel[:,2].flatten() - pos[:,2].flatten()*vel[:,1].flatten())[ain]*data['mass'][ain])
    Jy = np.sum((pos[:,2].flatten()*vel[:,0].flatten() - pos[:,0].flatten()*vel[:,2].flatten())[ain]*data['mass'][ain])
    Jz = np.sum((pos[:,0].flatten()*vel[:,1].flatten() - pos[:,1].flatten()*vel[:,0].flatten())[ain]*data['mass'][ain])
    print('This ratio should be 1: Jz/J = %.4f'%(Jz/np.sqrt(Jx**2+Jy**2+Jz**2)))

    # apply the rotation matrix to the dark matter position arrays
    pos_d = np.array([data_dark['x']-x_com,data_dark['y']-y_com,data_dark['z']-z_com])
    pos_d = np.dot(R_matrix, pos_d).transpose() # rotated x/y/z is now pos_d[:,0/1/2]

    # apply the rotation matrix to the gas position arrays
    data_gas = pickle.load(open(file_gas,'rb'))
    pos_g = np.array([data_gas['x']-x_com,data_gas['y']-y_com,data_gas['z']-z_com]) 
    pos_g = np.dot(R_matrix, pos_g).transpose() # rotated x/y/z is now pos_g[:,0/1/2]

    # define a set of positions in the equatorial plane at which to construct the jc -- e mapping
    minR = 0.1*eps
    maxR = rvir
    nbins = 100
    bin_edges = np.logspace(np.log10(minR), np.log10(maxR), num=nbins+1)
    Rbins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    xin = np.concatenate((Rbins,-Rbins,np.zeros(len(Rbins)),np.zeros(len(Rbins))))
    yin = np.concatenate((np.zeros(len(Rbins)),np.zeros(len(Rbins)),Rbins,-Rbins))
    xin = xin.flatten()
    yin = yin.flatten()
    zin = np.zeros(len(xin))
    ni = len(xin)

    print('Constructing the jc-E mapping considering the halo in isolation and recomputing the potential... ')

    phi_dark = twobody.midplane_potential(data_dark['mass'],pos_d[:,0].flatten(),pos_d[:,1].flatten(),pos_d[:,2].flatten(),xin,yin,zin,eps)
    phi_gas = twobody.midplane_potential(data_gas['mass'],pos_g[:,0].flatten(),pos_g[:,1].flatten(),pos_g[:,2].flatten(),xin,yin,zin,eps)
    phi_star = twobody.midplane_potential(data['mass'],pos[:,0].flatten(),pos[:,1].flatten(),pos[:,2].flatten(),xin,yin,zin,eps)

    phi = -(phi_dark+phi_gas+phi_star)*grav_const
    pot_midplane = np.zeros(nbins)
    for j in range(nbins):
        pot_midplane[j] = (phi[j]+phi[j+nbins]+phi[j+2*nbins]+phi[j+3*nbins])*0.25

    vcirc2_dark = twobody.midplane_vcirc2(data_dark['mass'],pos_d[:,0].flatten(),pos_d[:,1].flatten(),pos_d[:,2].flatten(),xin,yin,zin,eps)
    vcirc2_gas = twobody.midplane_vcirc2(data_gas['mass'],pos_g[:,0].flatten(),pos_g[:,1].flatten(),pos_g[:,2].flatten(),xin,yin,zin,eps)
    vcirc2_star = twobody.midplane_vcirc2(data['mass'],pos[:,0].flatten(),pos[:,1].flatten(),pos[:,2].flatten(),xin,yin,zin,eps)
    vcirc2 = vcirc2_dark + vcirc2_gas + vcirc2_star 
    
    v_circ = np.zeros(nbins)
    for j in range(nbins):
        v_circ[j] = np.sqrt(grav_const*(vcirc2[j]+vcirc2[j+nbins]+vcirc2[j+2*nbins]+vcirc2[j+3*nbins])*0.25)
    j_circ = v_circ*Rbins
    bindingE = 0.5*(v_circ**2) + pot_midplane
    
    mvir = np.sum(data['mass']) + np.sum(data_dark['mass']) + np.sum(data_gas['mass'])
    vvir = np.sqrt(grav_const*np.float(mvir)/np.float(rvir))
    
    f = open(fileout,'wb')
    pickle.dump({'matrix':R_matrix,'rvir':rvir,'mvir':mvir,'vvir':vvir,'radius_align':radius_align,'eps':eps,
                 'pos_com':np.array([x_com,y_com,z_com]),'vel_com':np.array([vx_com,vy_com,vz_com]),
                 'R':Rbins,'v_circ':v_circ,'j_circ':j_circ,'bindingE':bindingE},f)
    f.close()
    del(f)
 
    end = time.time()
    print('Runtime of midplane_potential: %s'%secondsToStr(end-start))
    gc.collect()
    return fileout


#################################################################################################################################
#################### functions to create the input space and run the clustering #################################################
#################################################################################################################################




def compute_percentiles(file_auxiliary,file_dec,data_massw,data_lumw,soft=True):
    
    data_dec = pickle.load(open(file_dec,'rb'))
    #gmm_features = data_dec['features']
    p_label = data_dec['p_label']
    label = data_dec['label']
    iord_dec = data_dec['iord']
    srt = np.argsort(iord_dec)
    p_label = p_label[srt]
    label = label[srt]
    iord_dec = iord_dec[srt]
    indx = np.unique(data_dec['label'])
    nk = len(indx)

    data = pickle.load(open(file_auxiliary,'rb'))
    # check that data['iord'] and iord_dec are the same
    if len(data['iord'])!=len(iord_dec): 
        print('The iords in the file_dec and file_auxiliary have different dimensions. Returning...')
        return
    elif int(np.sum(data['iord']-iord_dec))!=0:
        print('The iords in the file_dec and file_auxiliary are not the same. Returning...')
        return

    mass = data['mass']
    try:
        luminosity = data['luminosity']
    except:
        luminosity = data['mass']
        print('There is no luminosity field in the tmp file. I will just assume M/L=1.')
        pass

    #get the available features from the *.tmp file
    features_in_tmp_file = get_list_of_tags_from_file(file_auxiliary)
    tags = []
    for feature in features_in_tmp_file:
        if feature not in ['iord','mass','luminosity','filters']: tags.append(feature) 

    if nk==1:
        for feature in tags:
            for p,percent in list(zip([0.16,0.50,0.84],['16','50','84'])):
                
                varm = percentile(data[feature], mass, percent=p)
                varl = percentile(data[feature], luminosity, percent=p)
                data_massw[nk][feature][percent] = np.array(varm)
                data_lumw[nk][feature][percent] = np.array(varl)
    else:
        for feature in tags:
            for p,percent in list(zip([0.16,0.50,0.84],['16','50','84'])): 
                varm = []
                varl = []
                for i in indx: 
                    if soft:
                        varm.append(percentile(data[feature], np.multiply(mass,np.ravel(np.array(p_label[:,i]).flatten())), percent=p))
                        varl.append(percentile(data[feature], np.multiply(luminosity,np.ravel(np.array(p_label[:,i]).flatten())), percent=p))
                    else:
                        varm.append(percentile(data[feature][label==i], mass[label==i], percent=p))
                        varl.append(percentile(data[feature][label==i], luminosity[label==i], percent=p))
                data_massw[nk][feature][percent] = np.array(varm)
                data_lumw[nk][feature][percent] = np.array(varl)

    # add also the total mass and total angular momentum for each component to data_massw, and the total luminosity for each component to data_lumw
    if nk==1:
        data_massw[nk]['total_mass'] = np.array([np.sum(mass)])
        data_lumw[nk]['total_luminosity'] = np.array([np.sum(luminosity)])
        data_massw[nk]['total_Jx'] = np.array([np.sum(np.multiply(mass,data['jx']))])
        data_massw[nk]['total_Jy'] = np.array([np.sum(np.multiply(mass,data['jy']))])
        data_massw[nk]['total_Jz'] = np.array([np.sum(np.multiply(mass,data['jz']))])
        va, vb, vc, a, b, c = inertia_shape(mass,data['x'],data['y'],data['z'])
        triax, ellipt, prol = shapes(a,b,c)
        data_massw[nk]['cm'] = [np.array([0.,0.,0.])]    
        data_massw[nk]['va'] = [va]
        data_massw[nk]['vb'] = [vb]
        data_massw[nk]['vc'] = [vc]
        data_massw[nk]['a'] = np.array([a])
        data_massw[nk]['b'] = np.array([b])
        data_massw[nk]['c'] = np.array([c])
        data_massw[nk]['triaxiality'] = np.array([triax])
        data_massw[nk]['ellipticity'] = np.array([ellipt])
        data_massw[nk]['prolateness'] = np.array([prol])
    else:
        varm = []
        varl = []
        varmx = []
        varmy = []
        varmz = []
        data_massw[nk]['cm'] = []
        data_massw[nk]['va'] = []
        data_massw[nk]['vb'] = []
        data_massw[nk]['vc'] = []
        var_a = []
        var_b = []
        var_c = []
        var_t = []
        var_e = []
        var_p = []
        for i in indx: 
            if soft:
                varm.append(np.sum(np.multiply(mass,np.ravel(np.array(p_label[:,i]).flatten()))))
                varl.append(np.sum(np.multiply(luminosity,np.ravel(np.array(p_label[:,i]).flatten()))))
                varmx.append(np.sum(np.multiply(data['jx'] ,np.multiply(mass,np.ravel(np.array(p_label[:,i]).flatten())))))
                varmy.append(np.sum(np.multiply(data['jy'] ,np.multiply(mass,np.ravel(np.array(p_label[:,i]).flatten())))))
                varmz.append(np.sum(np.multiply(data['jz'] ,np.multiply(mass,np.ravel(np.array(p_label[:,i]).flatten())))))
            
                cm = center_of_mass(np.multiply(mass,np.ravel(np.array(p_label[:,i]).flatten())),data['x'],data['y'],data['z'])
                va, vb, vc, a, b, c = inertia_shape(np.multiply(mass,np.ravel(np.array(p_label[:,i]).flatten())),data['x']-cm[0],data['y']-cm[1],data['z']-cm[2])
            else:
                varm.append(np.sum(mass[label==i]))
                varl.append(np.sum(luminosity[label==i]))
                varmx.append(np.sum(np.multiply(data['jx'][label==i],mass[label==i])))
                varmy.append(np.sum(np.multiply(data['jy'][label==i],mass[label==i])))
                varmz.append(np.sum(np.multiply(data['jz'][label==i],mass[label==i])))

                cm = center_of_mass(mass[label==i],data['x'][label==i],data['y'][label==i],data['z'][label==i])
                va, vb, vc, a, b, c = inertia_shape(mass[label==i],data['x'][label==i]-cm[0],data['y'][label==i]-cm[1],data['z'][label==i]-cm[2])

            triax, ellipt, prol = shapes(a,b,c) 
            data_massw[nk]['cm'].append(cm)
            data_massw[nk]['va'].append(va)
            data_massw[nk]['vb'].append(vb)
            data_massw[nk]['vc'].append(vc)
            var_a.append(a)
            var_b.append(b)
            var_c.append(c)
            var_t.append(triax)
            var_e.append(ellipt)
            var_p.append(prol)
            
        data_lumw[nk]['total_luminosity'] = np.array(varl)
        data_massw[nk]['total_mass'] = np.array(varm)
        data_massw[nk]['total_Jx'] = np.array(varmx)
        data_massw[nk]['total_Jy'] = np.array(varmy)
        data_massw[nk]['total_Jz'] = np.array(varmz)    
        data_massw[nk]['a'] = np.array(var_a)
        data_massw[nk]['b'] = np.array(var_b)
        data_massw[nk]['c'] = np.array(var_c)
        data_massw[nk]['triaxiality'] = np.array(var_t)
        data_massw[nk]['ellipticity'] = np.array(var_e)
        data_massw[nk]['prolateness'] = np.array(var_p)

    gc.collect()                
    return 


def get_list_of_tags_from_file(infile):
    data = pickle.load(open(infile,'rb'))
    keysList = list(data.keys())
    del(data)
    gc.collect()
    return keysList


def apply_filters(dataframe,filters):    
    propertiesList = list(dataframe.keys())
    filter_index = list(np.arange(1,len(filters)))
    for i in filter_index:
        if filters[i]['type']=='BandPass':
            keep = np.where((dataframe[filters[i]['property']]>filters[i]['LowLimit']) & (dataframe[filters[i]['property']]<filters[i]['UpLimit']))
        if filters[i]['type']=='LowPass':
            keep = np.where(dataframe[filters[i]['property']]<filters[i]['Limit'])
        if filters[i]['type']=='HighPass':
            keep = np.where(dataframe[filters[i]['property']]>filters[i]['Limit'])
        print('Applying filter: ',filters[i])
        for prop in propertiesList: 
            try:
                dataframe[prop] = dataframe[prop][keep]
            except:
                print('Property %s is not a valid particle array.'%prop)
                pass
    return
    
    
def generate_tmp_file(file_star, file_potential, file_midplane, out_dir=None, filters=None, verbose=True):
    
    # Generates a temporary file with all the available features, to speed up. This file is large and should be only created once, 
    # and delete at the end of a gsf run. The name of this file is [file_star].tmp and is saved in the out_dir directory if the arg out_dir is not None. 
    if verbose:
        print('-------------------------------------------------------------------------------------------------------------------------')
        print('This function generates an auxiliary file with all the features available in file_star, plus all the features that can be')
        print('derived from the mandatory properties.')
        print('Required input:')
        print('file_star = the file with the stellar particle data')
        print('file_potential = the file containing the gravitational potential of all the stars in the halo')
        print('file_midplane = the file containing the definition of the equatorial plane')
        print('out_dir = the path to the directory where the output file will be saved. If not given, the tmp file will be save in the same dir of file_star')

    start = time.time()
    if out_dir is None: 
        fileout = file_star[:-4]+'.tmp'
    else: 
        aux = file_star.split('/')
        fileout = aux[-1]
        fileout = out_dir+fileout[:-4]+'.tmp'

    if filters is None:
        if os.path.isfile(fileout):
            print('The file %s already exists.'%fileout)
            return fileout
        else:
            print('Output will be saved to %s.'%fileout)
    
    # figure out which features are available in the input file
    properties_available = get_list_of_tags_from_file(file_star)
    # initialize the list of all features that will be in the .tmp file
    list_of_features = []

    print('Load the midplane info needed (rotation matrix to get right orientation and jc-e mapping)')
    data_midplane = pickle.load(open(file_midplane,'rb'))
    j_circ = data_midplane['j_circ']
    bindingE = data_midplane['bindingE']
    bindingE, j_circ = transform2monotonic(bindingE,j_circ)
    R_matrix = data_midplane['matrix']
    pos_com = data_midplane['pos_com']
    vel_com = data_midplane['vel_com']

    print('Load the stellar data')
    data = pickle.load(open(file_star,'rb'))
    
    print('apply the rotation matrix to the stellar position and velocity arrays') 
    pos = np.array([data['x']-pos_com[0],data['y']-pos_com[1],data['z']-pos_com[2]]) 
    pos = np.dot(R_matrix, pos).transpose() # rotated x/y/z is now pos[:,0/1/2]
    vel = np.array([data['vx']-vel_com[0],data['vy']-vel_com[1],data['vz']-vel_com[2]]) 
    vel = np.dot(R_matrix, vel).transpose() # rotated vx/vy/vz is now vel[:,0/1/2]

    print('Put into arrays the mandatory properties: mass, x, y, z, vx, vy, vz...')
    mass = np.array(data['mass'])
    x = np.array(pos[:,0].flatten()) # positions in the correctly oriented reference frame
    y = np.array(pos[:,1].flatten())
    z = np.array(pos[:,2].flatten())
    vx = np.array(vel[:,0].flatten()) # velocities in the correctly oriented reference frame
    vy = np.array(vel[:,1].flatten())
    vz = np.array(vel[:,2].flatten())
    for var in ['mass', 'x', 'y', 'z', 'vx', 'vy', 'vz']: list_of_features.append(var)
    

    print('Load the potential energy (pe) at each stellar particle position...')
    data_phi = pickle.load(open(file_potential,'rb'))
    pe = data_phi['phi']
    
    print('Compute the properties that can be derived from the mandatory ones: jx, jy, jz, ke, jp, jc, jzjc, jpjc, e, r2, r3, height, vR, vphi, vnorot, vT...')
    jx = y*vz-z*vy
    jy = z*vx-x*vz
    jz = x*vy-y*vx
    ke = 0.5*(vx**2+vy**2+vz**2)
    jp = np.sqrt(jx**2+jy**2)
    energy = ke + pe
    energy[energy>0.] = np.max(energy[energy<0.])
    reduce_e = 10.**(-int(np.log10(max(abs(bindingE)))))
    reduce_jc = 10.**(-int(np.log10(max(j_circ))))
    interp = scipy.interpolate.InterpolatedUnivariateSpline(bindingE*reduce_e, j_circ*reduce_jc, k=1)
    rep = interp(energy*reduce_e)
    jc = rep/reduce_jc    
    jc[jc<=0.] = np.min(jc[jc>0.]) # make sure that no particles have negative interpolated jc
    jzjc = jz/jc
    jzjc[jzjc<-1.3] = -1.3 # crop all unreasonably low values of jzjc (in theory -1<=jz/jc<=1) 
    jzjc[jzjc>1.3] = 1.3 # crop all unreasonably high values of jzjc 
    jpjc = jp/jc
    jpjc[jpjc>1.3] = 1.3 # crop all unreasonably high values of jpjc (in theory jp/jc<=1) 
    e = energy/np.max(abs(energy))
    r2 = np.sqrt(x**2+y**2)
    r3 = np.sqrt(x**2+y**2+z**2)
    height = abs(z)
    vphi = jz/r2
    vR = (x*vx+y*vy)/r2
    vnorot = np.sqrt(vR**2+vz**2)
    vT = np.sqrt(vx**2+vz**2)
    for var in ['jx','jy','jz','ke','pe','jp','jc','jzjc','jpjc','e','r2','r3','height','vR','vphi','vnorot','vT']: list_of_features.append(var)
    
    diff_list = []
    for var in properties_available: 
        if var not in list_of_features: diff_list.append(var)
    
    print('If the index array does not exist in the input file, create it.')
    if 'iord' not in diff_list: 
        iord = np.arange(1,len(x)+1)
        list_of_features.append('iord')
    else: iord = data['iord']
    
    print('Create the dataframe with all the features in list_of_features.')
    dataframe = {'iord':iord,'mass':mass,'x':x,'y':y,'z':z,'vx':vx,'vy':vy,'vz':vz,'ke':ke,'pe':pe,
                 'jc':jc,'jz':jz,'jx':jx,'jy':jy,'jp':jp,'jzjc':jzjc,'jpjc':jpjc,'e':e,
                 'r2':r2,'r3':r3,'height':height,'vphi':vphi,'vR':vR,'vnorot':vnorot,'vT':vT}
    
    print('Check that diff_list contains only particle arrays...')
    to_remove = []
    for var in diff_list:
        if int(len(data['x'])-len(data[var]))>0: to_remove.append(var) 
    for item in to_remove:
            print('removing feature %s from the diff_list'%item)
            diff_list.remove(item)
    
    print('Add whatever extra features are available in the input file.')
    for var in diff_list:
        dataframe[var] = data[var]
        print('extra feature available: %s'%var)
        
    if filters is not None:
        print('Apply the filters to the data and adapt accordingly the name of the output file...')
        apply_filters(dataframe,filters)
        fileout = fileout[:-4]+'.'+filters['add_to_file_names']+'.tmp'
    
    dataframe['filters'] = filters

    print('Save the big file.')
    f = open(fileout,'wb')
    pickle.dump(dataframe,f)
    f.close()
    del(f)

    end = time.time()
    print('Runtime of generate_tmp_file: %s'%secondsToStr(end-start))

    gc.collect()
    
    return fileout

    

def GMM_input(tmp_file, varlist=['jzjc','jpjc','e'], trig_scaling=None, verbose=True):
    
    if verbose:
        print('-------------------------------------------------------------------------------------------------------------------------')
        print('This function computes the matrix [n_particles,n_features] to be fed to the clustering')
        print('By default it will use the 3D feature space [jz/jc,jp/jc,e/max(|e|)]')
        print('By default it will not apply the arctan scaling to the input features.')
        print('Setting any of the trig_scaling elements to True will result in the corresponding input feature')
        print('being scaled as arctan(feature/std(feature)). This might be a good idea for features with large dynamical ranges.')
        print('To change the input feature space, change the argument varlist.')
        print('Required input: tmp_file (generated by the function generate_tmp_file)')

    start = time.time()

    available_tags = available_features()
    print('Check if varlist is a subset of ',available_tags) 
    varlist_in_available_tags = selected_features(varlist,available_tags)

    print('Check which of the already selected features are actually available in the tmp file.')
    features_in_tmp_file = get_list_of_tags_from_file(tmp_file)
    good_tags = selected_features(varlist_in_available_tags,features_in_tmp_file)

    nf = len(good_tags)
    if nf < 1:
        print('Input feature space ',varlist,' not supported.') 
        print('If you want to use other variables that are not already considered, but are available in the tmp file,')
        print('you have to first add them in the functions: available_features (and to get the labeling correct also in the function feature_labels).')
        print('Returning...')
        return
 
    print('The input feature space for clustering is: ',good_tags)
    append2nameout = ''
    for string in good_tags: append2nameout = append2nameout+string

    if trig_scaling is not None: fileout = tmp_file[:-4]+('.gmm_on_%s'%append2nameout)+'_trigscal.dat'
    else: fileout = tmp_file[:-4]+('.gmm_on_%s'%append2nameout)+'.dat'

    print('The input matrix for clustering will be saved to %s'%fileout)
    
    if os.path.isfile(fileout):
        print('The file with the input matrix for clustering already exists. Returning...')
        return fileout
    
    print('Creating the input matrix for clustering from the tmp_file...')
    data = pickle.load(open(tmp_file,'rb'))
  
    ns = len(data['iord'])
    obs = np.ndarray(shape=(ns,nf), dtype=float)
    for k in range(len(good_tags)):
        obs[:,k] = data[good_tags[k]]
        aux_arr = obs[:,k].flatten()
        index = varlist.index(good_tags[k])
        if trig_scaling is not None:
            if tring_scaling['scale'][index]: 
                print('I am applying the arctan scaling to the feature %s'%good_tags[k])
                obs[:,k] = np.arctan(aux_arr/np.std(aux_arr)) 
                    
    f = open(fileout,'wb')
    pickle.dump({'feature_space':good_tags,'gmm_input':obs,'iord':data['iord'],'mass':data['mass'],'filters':data['filters'],
                 'trig_scaling':trig_scaling},f)
    f.close()
    del(f)    

    end = time.time()
    print('Runtime of GMM_input: %s'%secondsToStr(end-start))

    gc.collect()
    return fileout




#################################################################################################################################
###################################### the main function ########################################################################
#################################################################################################################################

def main_gsf(file_star, file_gas, file_dark, out_dir=None, eps=0.1,radius_align=None,
             varlist=['jzjc','jpjc','e'], trig_scaling=None, loop4optimaln=False,
             number_of_clusters=2, covariance_type='full', whiten_data=True, n_init=1, 
             plot=True, trig_scaling_plot=[False,False,False,False,False,False], band=False, M2L=False, inclination=90., fov=None, filters=None, verbose=True):

    if verbose:
        print('This is the main function of gsf.')
        print('The only required input are the files containing the 1) stellar particle data, 2) gas particle data, and 3) dark matter particle data.')
        print('This files have to be passed with their absolute path.')
        print('Optional input:')
        print('out_dir = the path to the output directory, where all the data files and plots will be saved.')
        print('If out_dir is not set, gsf will create a new directory output/')
        print('eps = the minimum gravitational softening in kpc.')
        print('radius_align = the 3D radius in kpc of the farthest particle to consider for orienting the galaxy.')
        print('varlist = the list of desired features on which to run the clustering. By default it uses the 3D space (jz/jc,jp/jc,energy).')
        print('To see all available features, call the function available_features()')
        print('loop4optimaln = Boolean setting whether the code should run for all nk from 1 to 15. By default it is False.')
        print('If loop4optimaln = True, the code will plot the log Likelihood as a function of number of clusters nk.')        
        print('number_of_clusters = the desired number of clusters (by default 2).')
        print('By default each clustering component is assumed to have it own general covariance matrix, arg covariance_type=full')
        print('The other possibilities for the covariance_type arg are diag, tied and spherical.')
        print('By default the input data is centered to the mean and scaled to unit variance. Set arg whiten_data=False to override the default.')
        print('By default, the Expectation-Maximization algorithm is run for up to 100 times or up until the convergence threshold is met (1.0e-03)')
        print('using a single k-means initialization (n_init=1).')
        print('By default it will also plot various diagnostics. Set arg plot=False if these plots are not to be saved.')
        print('The band arg sets the weighting in the moments maps. If band=False is passed, the code will use the particle mass.')
        print('The M2L arg sets the label of the 0th order moment map. If band=False is passed, the code will use the particle mass.')
        print('If M2L has a numerical value, it will be interpreted as the mass-to-light ratio, and used to compute a luminosity for each particle.')
        print('fov is the desired field-of-view in kpc. It defaults to 50 kpc.')
        print('If you want to use other args for the clustering algorithm, check the Gaussian Mixture webpage at http://scikit-learn.org/stable/')
        print('If you want to use only some of the particles, you can pass a dictionary to the variable filters. Check the run.py for details on how to construct it.')
        print('The boolean arg verbose allows for the descriptions of the functions to be printed.')


    if out_dir is None:
        os.system('mkdir output')
        out_dir = 'output/'

    start = time.time()
    
    print('Generate the files that the clustering algorithm needs...')
    file_potential = star_potential(file_star,file_dark,file_gas,out_dir=out_dir,eps=eps,verbose=verbose)
    file_midplane = midplane_potential(file_star,file_dark,file_gas,out_dir=out_dir,eps=eps,radius_align=radius_align,verbose=verbose)
    tmp_file = generate_tmp_file(file_star,file_potential,file_midplane,out_dir=out_dir,filters=filters,verbose=verbose)
    GMM_input_file = GMM_input(tmp_file,varlist=varlist,trig_scaling=trig_scaling,verbose=verbose)
    data_input = pickle.load(open(GMM_input_file,'rb'))
    effective_varlist = data_input['feature_space']

    #get the available features from the *.tmp file
    features_in_tmp_file = get_list_of_tags_from_file(tmp_file)
    tags = []
    for feature in features_in_tmp_file:
        if feature not in ['iord','mass','luminosity','filters']: tags.append(feature) 
    
    print('Features that will appear in the diagnostic file: ',tags)
    
    print('Run the clustering...')
    if loop4optimaln:
        # initialize two nested dictionaries to contain the 16th, 50th and 84th percentiles of all features for all clusters of all mixture models
        data_massw = {} # nested dictionary where the percentiles will be computed using mass weighting  
        data_lumw = {} # nested dictionary where the percentiles will be computed using luminosity weighting (if the luminosity is available in the tmp_file)
        for nk in np.arange(15)+1:
            data_massw[nk] = {}
            data_lumw[nk] = {}
            for feature in tags:
                data_massw[nk][feature] = {}
                data_lumw[nk][feature] = {}
                for percent in ['16','50','84']: 
                    data_massw[nk][feature][percent] = np.nan
                    data_lumw[nk][feature][percent] = np.nan
            gc.collect()
        BIC = []
        AIC = []
        nk = []
        num_param = []
        log_likelihood = []        
        for number_of_clusters in np.arange(15)+1:
            n_param, logL, bic, aic, file_dec = gmm_clustering(GMM_input_file, number_of_clusters = number_of_clusters, covariance_type = covariance_type,
                                                               whiten_data = whiten_data, n_init=n_init, plot=plot, trig_scaling_plot=trig_scaling_plot, verbose=verbose)
            BIC.append(bic)
            AIC.append(aic)
            log_likelihood.append(logL)
            nk.append(number_of_clusters)
            num_param.append(n_param)
            compute_percentiles(tmp_file,file_dec,data_massw,data_lumw)
            gc.collect()
        diagnostic_png = GMM_input_file[:-4]+'.scikit_gmm_'+covariance_type+'_logLvsnk.png'
        if whiten_data: diagnostic_png = diagnostic_png[:-13]+'_white_logLvsnk.png'
        plot_diagnostic(nk,BIC,log_likelihood,diagnostic_png)
        nk_selected, nparam_selected, scree_selected = select_optimal_model(nk,num_param,log_likelihood,figout=diagnostic_png[:-12]+'STvsNparam.png')
        
        diagnostic_file = diagnostic_png[:-12]+'diagnostics.dat'
        f = open(diagnostic_file,'wb')
        pickle.dump({'mixture_features':effective_varlist,'nk':np.array(nk),'num_param':np.array(num_param),
                     'log_likelihood':np.array(log_likelihood),'BIC':np.array(BIC),'AIC':np.array(AIC),
                     'mwp':data_massw,'lwp':data_lumw,
                     'CHull_model_selection':{'nk':nk_selected,'num_param':nparam_selected,'scree':scree_selected}},f)
        f.close()
        del(f)
        gc.collect()
    else: 
        n_param, logL, bic, aic, file_dec = gmm_clustering(GMM_input_file, number_of_clusters = number_of_clusters, covariance_type = covariance_type, 
                                                           whiten_data = whiten_data, n_init=n_init, plot=plot, trig_scaling_plot=trig_scaling_plot, verbose=verbose)    
        print('Plot the results as the zero, first and second order moments maps')
        plot_moment_maps(tmp_file, file_dec, inclination=inclination, band=band, M2L=M2L, fov=fov, verbose=verbose)
    
    finish = time.time()
    print('Total runtime: %s'%secondsToStr(finish-start))
    print('Finish')
    gc.collect()
    return 


