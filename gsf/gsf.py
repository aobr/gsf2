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
#################################### general purpose functions ##################################################################
#################################################################################################################################

def secondsToStr(t):
# Returns a string converting the time t in seconds to hh:mm:ss
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(t*1000,),1000,60,60]) 

def percentile(x, w, percent=0.5):
# Returns the percentile percent of array x for a given weight array w
    srt = np.argsort(x)
    cum = np.cumsum(w[srt])/np.sum(w)
    aux = x[srt]
    aux[np.argsort(abs(cum-percent))][0]
    return aux[np.argsort(abs(cum-percent))][0]
    
def ellipticity_from_moments(x,y,weights):
# Returns the ellipticity from a 2D map with 1D array coordinates x and y. 
# The weights is the array that should contain the mass or the flux.
# The maps is assumed to be centered. 
    r = np.sqrt(x**2+y**2)
    f = weights
    x = x[np.isfinite(f)]
    y = y[np.isfinite(f)]
    r = r[np.isfinite(f)]
    f = f[np.isfinite(f)]
    x = x[r>0]
    y = y[r>0]
    f = f[r>0]
    r = r[r>0]
    Mxx = np.sum(f*(x/r)**2)/np.sum(f)
    Myy = np.sum(f*(y/r)**2)/np.sum(f)
    Mxy = np.sum(f*(x*y)/r**2)/np.sum(f)
    Q = Mxx - Myy # = a-b/a+b cos(2phi)
    U = 2.*Mxy    # = a-b/a+b sin(2phi)
    assymetry = (1.-np.sqrt(Q**2+U**2))/(1.+np.sqrt(Q**2+U**2))
    ellipticity = 1.-assymetry
    return ellipticity

def transform2monotonic(x,y):
    # function to remove elements from equal length arrays x and y such that x is strictly monotonically increasing
    # first check for NaNs, and remove them
    y = y[np.logical_not(np.isnan(x))]
    x = x[np.logical_not(np.isnan(x))]
    if not np.all(x[:-1] <= x[1:]):
        print('The x array is not monotonically increasing. Get rid of the offending elements.')
        nx = len(x)
        n_off = 0
        while not np.all(x[:-1] <= x[1:]):
            i_ok = [0]
            for i in range(len(x) - 1): 
                if x[i+1] >= x[i]: i_ok.append(i+1)
                else: n_off += 1
            x = x[i_ok]
            y = y[i_ok]
        print('I discarded %i elements of a total of %i from the bindingE and j_circ arrays.'%(n_off,nx))     
    return x,y

def rotate_x(pv, angle):
    angle *= np.pi / 180
    rot = np.matrix([[1,      0,             0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle),  np.cos(angle)]])
    return np.matmul(rot,pv)

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


#################################################################################################################################
########################################### plotting functions ##################################################################
#################################################################################################################################


def physical_component_color(kname):
    color = 'black'
    if kname=='Thin disk': color='navy'
    if kname=='Thick disk': color='dodgerblue'
    if kname=='Disk': color='blue'
    if kname=='Counter rotating disk': color='darkcyan'
    if kname=='Bar': color='limegreen'
    if kname=='Spheroid': color='crimson'
    if kname=='Halo': color='magenta'
    if kname=='Classical bulge': color='brown'
    if kname=='Pseudo bulge': color='darkorange'
    if kname=='B/P bulge': color='violet'
    return color


def plot_clustering_results_in_1D(file_gmm,file_dec,filename_out,palette='rainbow'):
    
    print('Plotting the results as mass weighted distributions in the original feature space')

    data_gmm = pickle.load(open(file_gmm,'rb'))
    data_cluster = pickle.load(open(file_dec,'rb'))

    obs = data_gmm['gmm_input']
    features = data_gmm['feature_space']
    mass = data_gmm['mass']
    label = data_cluster['label']
    posterior_probability = data_cluster['p_label']
    covariance_type = data_cluster['covariance_type']
    gmeans = data_cluster['gmeans']
    gweight = data_cluster['gweight']
    gcovar = data_cluster['gcovar']

    indices = np.unique(label)
    indices = indices[np.argsort(indices)]
    nk = len(indices)
    nf = len(obs[0,:].flatten())
    ns = len(obs[:,0].flatten())

    color_scheme = plt.get_cmap(palette)
    cNorm  = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_scheme)
    colorzs = []
    if nk>1: deltac = 1./(nk-1)
    else: deltac = 1.
    for k in range(nk): colorzs.append(scalarMap.to_rgba(k*deltac))

    plt.close()
    if nf==2: fig, axs = plt.subplots(1,2, figsize=(7.0,3.5))
    if nf==3: fig, axs = plt.subplots(1,3, figsize=(10.5,3.5))
    if nf==4: fig, axs = plt.subplots(1,4, figsize=(14.0,3.5))
    if nf==5: fig, axs = plt.subplots(1,5, figsize=(17.5,3.5))
    if nf>5:
        print('Too many features. You need to adapt the function plot_clustering_results_in_1D. Returning...')
        return    
    fig.subplots_adjust(left=0.05,bottom=0.15,right=0.97,top=0.93,hspace=0.15,wspace=0.15)
    
    for ax, j in zip(axs.ravel(),np.arange(nf)):
        
        plt.setp(ax.get_xticklabels(),fontsize=12)
        plt.setp(ax.get_yticklabels(),fontsize=12)
        ax.xaxis.labelpad = 1
        ax.yaxis.labelpad = 1
        xlabel = feature_labels(features[j])
        ax.set_xlabel(xlabel,fontsize=18)
        if j==0: ax.set_ylabel(r"M$_{\rm *}$ [M$_{\rm\odot}$]",fontsize=18)
        frange, fnbin = feature_range_and_nbin(features[j])
        ax.set_xlim(frange)
        
        var = np.array(obs[:,j]).flatten()
        mass_total, bin_edges, binnumber = scipy.stats.binned_statistic(var, mass, statistic='sum', bins=fnbin, range=frange)
        xvar = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.plot(xvar,mass_total,color='lightgrey',ls='-',lw=3,zorder=-1)
        ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
        ymax = 1.1*np.max(mass_total)
        ax.set_ylim(0,ymax)
        
        xstr, horizontalalignment = 0.95, 'right'
        if xvar[mass_total==np.max(mass_total)]<0.5*(frange[1]-frange[0]):
            xstr, horizontalalignment = 0.05 , 'left'

        for k,indx in zip(range(nk),indices):
            massink, bin_edges, binnumber = scipy.stats.binned_statistic(var[label==indx], mass[label==indx], statistic='sum', bins=fnbin, range=frange)
            massk = np.multiply(mass,np.ravel(np.array(posterior_probability[:,indx]).flatten()))
            massinkp, bin_edges, binnumber = scipy.stats.binned_statistic(var, massk, statistic='sum', bins=fnbin, range=frange)
            ax.plot(xvar, massink, color=colorzs[k], ls='-', lw=1.5, zorder=k+1)
            ax.plot(xvar, massinkp, color=colorzs[k], ls='--', lw=1.5, zorder=k+1)
            median = percentile(var, massk, percent=0.5)
            strt = r"$x_{\rm 50}$=%.2f"%median
            ax.text(xstr,0.90-k*0.05,strt,color=colorzs[k],fontsize=12,horizontalalignment=horizontalalignment,verticalalignment='center',transform = ax.transAxes)
            
    plt.savefig(filename_out) 
    plt.close()
    gc.collect()
    return


def plot_clustering_results_in_2D(file_gmm,file_dec,filename_out,
                                  palette='rainbow',min_feature=None,max_feature=None,
                                  trig_scaling_plot=None):

    print('Plotting the results as mass weighted distributions in the original feature space')

    data_gmm = pickle.load(open(file_gmm,'rb'))
    data_cluster = pickle.load(open(file_dec,'rb'))

    obs = data_gmm['gmm_input']
    features = data_gmm['feature_space']
    mass = data_gmm['mass']
    # since the mass has nothing to do with the clustering, I will also not used it for the weighting
    mass = np.ones(len(mass))
    label = data_cluster['label']
    posterior_probability = data_cluster['p_label']
    #covariance_type = data_cluster['covariance_type']
    #gmeans = data_cluster['gmeans']
    #gweight = data_cluster['gweight']
    #gcovar = data_cluster['gcovar']

    indices = np.unique(label)
    indices = indices[np.argsort(indices)]
    nk = len(indices)
    nf = len(obs[0,:].flatten())
    ns = len(obs[:,0].flatten())
    nbin = 100

    color_scheme = plt.get_cmap(palette)
    cNorm  = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_scheme)
    colorzs = []
    if nk>1: deltac = 1./(nk-1)
    else: deltac = 1.
    for k in range(nk): colorzs.append(scalarMap.to_rgba(k*deltac))
    
    factor = 2.0           # size of one side of one panel
    lbdim = 0.4 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05         # w/hspace size
    plotdim = factor * nf + factor * (nf - 1.) * whspace
    dim = lbdim + plotdim + trdim

    plt.close()
    fig, axes = plt.subplots(nf, nf, figsize=(dim, dim))

    # Format the figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace)
    
    for i in range(nf):
        # The histrograms are plotted on the diagonal
        if nf==1:
            ax = axes
            ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=False, right=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            fig.subplots_adjust(left=0.1, bottom=0.2, right=tr, top=tr, wspace=whspace, hspace=whspace)
            continue
        ax = axes[i, i]
        if i<nf-1: ax.set_yticks([])
        if i==nf-1: ax.set_xticks([])
        ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=False, right=False)
        if i<nf-1: plt.setp(ax.get_xticklabels(), visible=False)
        if i==nf-1: 
            ax.tick_params(axis='both', which='both', direction='in', bottom=False, top=False, left=True, right=True)
            plt.setp(ax.get_yticklabels(), visible=False)

        x = obs[:,i].flatten()
        if trig_scaling_plot is not None:
            if trig_scaling_plot[i]: 
                scale_std = np.std(x)
                x = np.arctan(x/scale_std)
        if min_feature is not None: x_min=min_feature[i]
        else: x_min = np.min(x)
        if max_feature is not None: x_max=max_feature[i]
        else: x_max = np.max(x)
        x_range = [x_min,x_max]
        if i<nf-1: 
            ax.set_xlim(x_range)
        else: 
            ax.set_ylim(x_range)
        mass_hist, bin_edges, binnumber = scipy.stats.binned_statistic(x, mass, statistic='sum', bins=nbin, range=x_range)
        x_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        median = percentile(x, mass, percent=0.5)
        x_median = (median-x_range[0])/(x_range[1] - x_range[0])
        if trig_scaling_plot is not None:
            if trig_scaling_plot[i]: median = np.tan(median)*scale_std
        if i<nf-1: 
            ax.set_ylim(0,1.3*np.max(mass_hist))
            ax.plot(x_bins,mass_hist,color='lightgrey',ls='-',lw=3,zorder=-1)
            ax.text(x_median,0.90,r"%.2f"%median,color='lightgrey',horizontalalignment='center',verticalalignment='center',
                    rotation=90,transform = ax.transAxes)
        else: 
            ax.set_xlim(0,1.3*np.max(mass_hist))
            ax.plot(mass_hist,x_bins,color='lightgrey',ls='-',lw=3,zorder=-1)
            ax.text(0.90,x_median,r"%.2f"%median,color='lightgrey',horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)

        for k,indx in list(zip(range(nk),indices)):            
            massk = np.multiply(mass,np.ravel(np.array(posterior_probability[:,indx]).flatten()))
            mass_kp_hist, bin_edges, binnumber = scipy.stats.binned_statistic(x, massk, statistic='sum', bins=nbin, range=x_range)
            median = percentile(x, massk, percent=0.5)
            x_median = (median-x_range[0])/(x_range[1] - x_range[0])
            if trig_scaling_plot is not None:
                if trig_scaling_plot[i]: median = np.tan(median)*scale_std
            if i<nf-1: 
                ax.plot(x_bins,mass_kp_hist,color=colorzs[k],ls='-',lw=1.5,zorder=k+1)
                ax.text(x_median,0.90,r"%.2f"%median,color=colorzs[k],horizontalalignment='center',verticalalignment='center',
                        rotation=90,transform = ax.transAxes)
            else: 
                ax.plot(mass_kp_hist,x_bins,color=colorzs[k],ls='-',lw=1.5,zorder=k+1)
                ax.text(0.90,x_median,r"%.2f"%median,color=colorzs[k],horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
            
        # The 2D histograms are plotted on the lower left corner
        for j in range(nf):
            ax = axes[i, j]
            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif j == i:
                continue
            ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
            if i<nf-1: plt.setp(ax.get_xticklabels(), visible=False)
            else: ax.set_xlabel(feature_labels(features[j])) 
            if j>0: plt.setp(ax.get_yticklabels(), visible=False)
            else: ax.set_ylabel(feature_labels(features[i]))
            
            y = obs[:,i].flatten()
            if trig_scaling_plot is not None:
                if trig_scaling_plot[i]: 
                    scale_std_y = np.std(y)
                    y = np.arctan(y/scale_std_y)
            if min_feature is not None: y_min=min_feature[i]
            else: y_min = np.min(y)
            if max_feature is not None: y_max=max_feature[i]
            else: y_max = np.max(y)
            y_range = [y_min,y_max]
            
            x = obs[:,j].flatten()
            if trig_scaling_plot is not None:
                if trig_scaling_plot[j]: 
                    scale_std_x = np.std(x)
                    x = np.arctan(x/scale_std_x)            
            if min_feature is not None: x_min=min_feature[j]
            else: x_min = np.min(x)
            if max_feature is not None: x_max=max_feature[j]
            else: x_max = np.max(x)
            x_range = [x_min,x_max]
            range2d = (x_range,y_range)
            extent=[x_range[0],x_range[1],y_range[0],y_range[1]]
            for k,indx in list(zip(range(nk),indices)):            
                massk = np.multiply(mass,np.ravel(np.array(posterior_probability[:,indx]).flatten()))
                dens, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(x, y, massk, statistic='sum', bins=nbin, range=range2d)
                cs = ax.contour(dens.transpose(), 5, extent=extent, aspect='auto', origin='lower', linewidths=1, colors=[colorzs[k]], alpha=0.7)
                
                for level in cs.collections:
                    diam = []
                    for kp,path in reversed(list(enumerate(level.get_paths()))):
                        # go in reversed order due to deletions!

                        # include test for "smallness" of your choice here:
                        # I'm using a simple estimation for the diameter based on the
                        #    x and y diameter...
                        verts = path.vertices # (N,2)-shape array of contour line coordinates
                        diameter = np.max(verts.max(axis=0) - verts.min(axis=0))
                        diam.append(diameter)
                        # keep only the largest contour
                    diam = np.array(diam)
                    if len(diam)>1:
                        diammax = np.max(diam)
                        diam[diam<diammax] = -1
                        #print('diam:',diam)
                        ii = 0
                        for kp,path in reversed(list(enumerate(level.get_paths()))):
                            if diam[ii]<0: 
                                del(level.get_paths()[kp])
                            ii = ii+1
    
    plt.savefig(filename_out) 
    plt.close()
    gc.collect() 
    return


def plot_moment_maps(tmp_file, file_dec, inclination=90, band=False, M2L=False,
                     verbose=True, palette='rainbow', one_plot_only=True, label_soft=True, *args, **kwargs):

    if verbose:
        print('-------------------------------------------------------------------------------------------------------------------------')
        print('This function will create for each component found in file_dec a png figure with ')
        print('surface density, line-of-sight velocity and line-of-sight velocity maps')
        print('Required input:')
        print('tmp_file = the big temporary file with all the available info for stellar particles')
        print('file_dec = the file containing the result of the clustering algorithm.')
        print('By default the maps will be created in edge-on perspective: inclination=90.')
        print('By default the maps will be weighted with the particle masses.')
        print('If the arg band is True, and there is a luminosity feature in the tmp_file, the weighting will be done with luminosity instead.')
        print('If the arg mass-to-light M2L is given, the weighting will be done with the luminosities computed from the masses.')
        print('The M2L option will in practice change only the units in the surface density panel from Msun/pc^2 to Lsun/pc^2.')
        print('The band arg has precedence over M2L.')
        print('The one_plot_only option creates a mosaic with the moments for all components, instead of saving the individual figures.')
        print('The label_soft argument sets the type of label to use. By default the soft labels will be used to compute the maps.')
        print('The other optional arguments are the number of pixels per side nxny, and the dimension of the field of view fov.')
    
    nxny = kwargs.get('nxny', None) # number of pixels per side
    fov = kwargs.get('fov', None) # figure size in kpc
    kname = kwargs.get('kname', None) # list of component names

    start = time.time()

    print('Loading the results of the clustering from file_dec...')
    try:
        data_dec = pickle.load(open(file_dec,'rb'))
    except:
        print('Input file does not exist. You need to run gmm_clustering_for_stars first!')
        return
  
    label = data_dec['label']
    plabel = data_dec['p_label']
    iord_dec = data_dec['iord']
    srt = np.argsort(iord_dec)
    label = label[srt]
    plabel = plabel[srt]
    iord_dec = iord_dec[srt]

    print('The output figure(s) will be saved in the same dir as file_dec.')
    figure_name = file_dec[:-3]+('inclination_%i'%inclination)

    print('Read the tmp_file...')
    data = pickle.load(open(tmp_file,'rb'))

    if inclination==0:
        x = data['x']
        y = data['y']
        vz = data['vz']
    else: 
        pos = np.array([data['x'],data['y'],data['z']])
        pos = rotate_x(pos,inclination)
        x = np.array(pos[0,:]).flatten()
        y = np.array(pos[1,:]).flatten()
        vel = np.array([data['vx'],data['vy'],data['vz']])
        vel = rotate_x(vel,inclination)
        vz = np.array(vel[2,:]).flatten()

    weight = data['mass']
    sb_label = r"log($\rm\Sigma$/M$_{\rm\odot}$pc$^{\rm -2}$)"
    if band:
        try:
            weight = data['luminosity']
            sb_label = r"log($\rm\Sigma$/L$_{\rm\odot}$pc$^{\rm -2}$)"
            figure_name = figure_name+'_luminosity'
            print('The weighting will be done with the particle luminosities.')
        except:
            if M2L:
                weight = data['mass']/M2L
                sb_label = r"log($\rm\Sigma$/L$_{\rm\odot}$pc$^{\rm -2}$)"
                figure_name = figure+'_lumfromM2L'
                print('The weighting will be done with the particle luminosities, derived assuming a constant M/L=%.2f'%M2L)
            pass
    else:
        if M2L:
            weight = data['mass']/M2L
            figure_name = figure+'_lumfromM2L'
            sb_label = r"log($\rm\Sigma$/L$_{\rm\odot}$pc$^{\rm -2}$)"
            print('The weighting will be done with the particle luminosities, derived assuming a constant M/L=%.2f'%M2L)
        else:
            figure_name = figure_name+'_mass'
            print('The weighting will be done with the particle masses.')
    figure_name = figure_name+'_moments'
        
        
    if fov is None: 
        fov = 50. # kpc
        print('The box side is %i kpc'%fov)    
    lim = 0.5*fov
    if nxny is None:
        pixel_size_kpc = 0.5 # default pixel size is 500 pc
        nxny = int(2*fov/pixel_size_kpc)
    print('The images are %ix%i pixels'%(nxny,nxny))

    rangex = (-lim,lim)
    rangey = (-lim,lim)
    range2d = (rangex,rangey)
    extent=[rangex[0],rangex[1],rangey[0],rangey[1]]

    color_scheme = plt.get_cmap(palette)
    cNorm  = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_scheme)
        
    total_mass = np.sum(weight) 
    indices = np.unique(label)
    indices = indices[np.argsort(indices)]
    nk = len(indices)
    deltac = 1./(nk-1)

    if kname is None: 
        kname = []
        for indx in indices: kname.append('Component %i'%indx)

    kell = np.zeros(nk)
    for k,indx in list(zip(range(nk),indices)):
        ell = ellipticity_from_moments(x[label==indx],y[label==indx],weight[label==indx])
        kell[k] = ell

    plot_names = []
    b2a = []
    massf = []
    
    color = 'black'
    counts, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(x, y, weight, statistic='count', bins=nxny, range=range2d)
    sdens, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(x, y, weight, statistic='sum', bins=nxny, range=range2d)
    vlos, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(x, y, weight*vz, statistic='sum', bins=nxny, range=range2d)
    vlos = vlos/sdens
    vlos2, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(x, y, weight*vz**2, statistic='sum', bins=nxny, range=range2d)
    vlos2 = vlos2/sdens
    sigmalos = np.sqrt(vlos2-vlos**2)
    pixel_size_kpc = abs(xedges[1]-xedges[0])
    sdens = sdens/(pixel_size_kpc**2*1.0e6) # transform from solar units (M or L) per kpc^2 to solar units per pc^2
    mass_fraction = 1.
    massf.append(mass_fraction)
    ell = ellipticity_from_moments(x,y,weight)
    b2a.append(1.-ell)
    text = {'label':[('f=%.2f'%mass_fraction),('b/a=%.2f'%(1.-ell))]}
    plot_name = figure_name+'_component_all.png'
    plot_names.append(plot_name)
    # set limits for the color maps in all panels to be used for all components 
    #sdens[counts<6]=0. # it excludes too much of the map for low res sims
    sb_max = np.log10(np.max(sdens))
    sb_min = np.log10(np.min(sdens[sdens>0.]))
    sdens[sdens<10**sb_min] = np.nan # mask with NaN everything below 10**sb_min
    vlos[np.isnan(sdens)] = np.nan
    sigmalos[np.isnan(sdens)] = np.nan
    
    vlos_max = np.percentile(abs(vlos[np.logical_not(np.isnan(vlos))]),90,interpolation='midpoint')
    if inclination>30.: sigmalos_max = vlos_max #np.nanmax(sigmalos)
    else: sigmalos_max = np.nanmax(sigmalos)    # vlos_max*(2.-np.sin(inclination/180*np.pi))
        
    sigmalos_min = 0. #np.nanmin(sigmalos)
    plot_regular_maps(sdens,vlos,sigmalos,'All stars',extent,rangex,rangey,color,plot_name,sb_label,text=text,
                      sb_min=sb_min,sb_max=sb_max,vlos_max=vlos_max,sigmalos_min=sigmalos_min,sigmalos_max=sigmalos_max)
    
    for k,indx in zip(range(nk),indices):
        if label_soft:
            c_x = x
            c_y = y
            c_v = vz
            c_weight = np.multiply(weight,np.ravel(np.array(plabel[:,indx]).flatten()))        
        else:
            c_x = x[label==indx]
            c_y = y[label==indx]
            c_v = vz[label==indx]
            c_weight = weight[label==indx]
        mass_fraction = np.sum(c_weight)/total_mass
        massf.append(mass_fraction)
        ell = ellipticity_from_moments(c_x,c_y,c_weight)
        b2a.append(1.-ell)
        text = {'label':[('f=%.2f'%mass_fraction),('b/a=%.2f'%(1.-ell))]}
        color = scalarMap.to_rgba(k*deltac)
        counts, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(c_x, c_y, c_weight, statistic='count', bins=nxny, range=range2d)
        sdens, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(c_x, c_y, c_weight, statistic='sum', bins=nxny, range=range2d)
        vlos, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(c_x, c_y, c_weight*c_v, statistic='sum', bins=nxny, range=range2d)
        vlos = vlos/sdens
        vlos2, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(c_x, c_y, c_weight*c_v**2, statistic='sum', bins=nxny, range=range2d)
        vlos2 = vlos2/sdens
        sigmalos = np.sqrt(vlos2-vlos**2)
        pixel_size_kpc = abs(xedges[1]-xedges[0])
        sdens = sdens/(pixel_size_kpc**2*1.0e6) # transform from solar units (M or L) per kpc^2 to solar units per pc^2
        #sdens[counts<6]=0.
        plot_name = figure_name+('_component_%i.png'%indx)
        sdens[sdens<10**sb_min] = np.nan # mask with NaN everything below 10**sb_min
        vlos[np.isnan(sdens)] = np.nan
        sigmalos[np.isnan(sdens)] = np.nan
        plot_regular_maps(sdens,vlos,sigmalos,kname[k],extent,rangex,rangey,color,plot_name,sb_label,text=text,
                          sb_min=sb_min,sb_max=sb_max,vlos_max=vlos_max,sigmalos_min=sigmalos_min,sigmalos_max=sigmalos_max)
        plot_names.append(plot_name)

    if one_plot_only:
        print('I will merge all moments plots in one and erase the individual ones.')
        mosaic_name = plot_names[0][:-17]+'mosaic.png'
        srt = np.argsort(np.array(massf[1:]))[::-1]     # np.argsort(np.array(b2a[1:]))
        plot_names_ordered = []
        plot_names_ordered.append(plot_names[0]) # put the full galaxy on the top row
        for k in srt: plot_names_ordered.append(plot_names[k+1])
        print('The components are sorted by their mass fractions from bottom to top.')
        images = [ PIL.Image.open(i) for i in plot_names_ordered ]
        min_shape = sorted( [(np.sum(i.size), i.size ) for i in images])[0][1]
        mosaic = np.vstack( (np.asarray( i.resize(min_shape) ) for i in images ) )
        mosaic = PIL.Image.fromarray( mosaic)
        mosaic.save(mosaic_name)        
        for plot_name in plot_names: os.system('rm -rf '+plot_name)

    gc.collect()
    return 


def plot_regular_maps(sdens,vlos,sigmalos,figure_title,extent,rangex,rangey,color,plot_name,sb_label,text=None,
                      sb_min=None,sb_max=None,vlos_min=None,vlos_max=None,sigmalos_min=None,sigmalos_max=None):
    
    smd_color = plt.get_cmap('RdYlBu_r')
    vlos_color = plt.get_cmap('bwr')
    slos_color = plt.get_cmap('gnuplot2_r')

    plt.close()
    fig = plt.figure(figsize=(10.6,3.5))
    gs = gridspec.GridSpec(1, 3)
    
    ax1 = plt.subplot(gs[0])
    plt.setp(ax1.get_xticklabels(), fontsize=10, fontweight='bold')
    plt.setp(ax1.get_yticklabels(), fontsize=10, fontweight='bold')
    ax1.set_xlabel(r"[kpc]", fontsize=10, fontweight='bold')
    ax1.set_ylabel(r"[kpc]", fontsize=10, fontweight='bold')
    ax1.xaxis.labelpad = 2
    ax1.yaxis.labelpad = 2
    ax1.set_xlim(rangex)
    ax1.set_ylim(rangey)
    #ax1.set_facecolor('black')
    ax1.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    if sb_min is not None: vmin=sb_min
    else: vmin = np.log10(np.nanmin(sdens[sdens>0.]))
    if sb_max is not None: vmax=sb_max
    else: vmax = np.log10(np.nanmax(sdens[sdens>0.]))
    im1 = ax1.imshow(np.log10(sdens).transpose(), interpolation='nearest',extent=extent, aspect='auto', origin='lower',cmap=smd_color, vmin=vmin, vmax=vmax, zorder=1)
    cax1 = fig.add_axes([0.306, 0.15, 0.01, 0.78])
    cb1 = fig.colorbar(im1, cax=cax1,orientation="vertical")
    cb1.set_label(label=sb_label,fontsize=10,labelpad=2, fontweight='bold')
    cb1.ax.tick_params(axis='y',labelleft=False,direction='in',labelright=True,pad=1,size=2)
    for t in cb1.ax.get_yticklabels(): 
        t.set_fontsize(8)
        t.set_fontweight('bold')
    if text is not None:
        props = dict(boxstyle='round', facecolor='white', edgecolor='white', alpha=1.0)
        ytext = 0.90
        for string in text['label']:
            ax1.text(0.05,ytext,string,fontsize=10,fontweight='bold',horizontalalignment='left',verticalalignment='center',transform=ax1.transAxes, bbox=props)
            ytext = ytext-0.07

    ax2 = plt.subplot(gs[1])
    plt.setp(ax2.get_xticklabels(), fontsize=10, fontweight='bold')
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_xlabel(r"[kpc]", fontsize=10, fontweight='bold')
    ax2.xaxis.labelpad = 2
    ax2.yaxis.labelpad = 2
    ax2.set_xlim(rangex)
    ax2.set_ylim(rangey)
    #ax2.set_facecolor('black')
    ax2.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    if vlos_max is not None: vmin, vmax = -vlos_max, vlos_max
    else: vmin, vmax = -np.nanmax(abs(vlos)), np.nanmax(abs(vlos)) 
    im2 = ax2.imshow(vlos.transpose(), interpolation='nearest',extent=extent, aspect='auto', origin='lower',cmap=vlos_color, vmin=vmin, vmax=vmax, zorder=1)
    cax2 = fig.add_axes([0.618, 0.15, 0.01, 0.78])
    cb2 = fig.colorbar(im2, cax=cax2,orientation="vertical")
    cb2.set_label(label=r"v$_{\rm los}$ [km/s]",fontsize=10,labelpad=-5, fontweight='bold')
    cb2.ax.tick_params(axis='y',labelleft=False,direction='in',labelright=True,pad=1,size=2)
    for t in cb2.ax.get_yticklabels(): 
        t.set_fontsize(8)
        t.set_fontweight('bold')

    ax3 = plt.subplot(gs[2])
    plt.setp(ax3.get_xticklabels(), fontsize=10, fontweight='bold')
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax3.set_xlabel(r"[kpc]", fontsize=10, fontweight='bold')
    ax3.xaxis.labelpad = 2
    ax3.yaxis.labelpad = 2
    ax3.set_xlim(rangex)
    ax3.set_ylim(rangey)
    #ax3.set_facecolor('black')
    ax3.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    if sigmalos_min is not None: vmin=sigmalos_min
    else: vmin=np.nanmin(sigmalos) 
    if sigmalos_max is not None: vmax=sigmalos_max
    else: vmax=np.nanmax(sigmalos)
    im3 = ax3.imshow(sigmalos.transpose(), interpolation='nearest',extent=extent, aspect='auto', origin='lower',cmap=slos_color, vmin=vmin, vmax=vmax, zorder=1)
    cax3 = fig.add_axes([0.93, 0.15, 0.01, 0.78])
    cb3 = fig.colorbar(im3, cax=cax3,orientation="vertical")
    cb3.set_label(label=r"$\rm\sigma_{\rm los}$ [km/s]",fontsize=10,labelpad=2, fontweight='bold')
    cb3.ax.tick_params(axis='y',labelleft=False,direction='in',labelright=True,pad=1,size=2)
    for t in cb3.ax.get_yticklabels(): 
        t.set_fontsize(8)
        t.set_fontweight('bold')
    
    props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=1.0)
#    ax2.text(0.5,0.9,figure_title,fontweight='bold',color=color,fontsize=12,horizontalalignment='center',verticalalignment='bottom',transform=ax2.transAxes, bbox=props) 
    ax2.text(0.5,0.9,figure_title,fontweight='bold',color=physical_component_color(figure_title),fontsize=12,horizontalalignment='center',verticalalignment='bottom',transform=ax2.transAxes, bbox=props) 
    gs.update(left=0.05,bottom=0.15,right=0.93, top=0.93, hspace=0.22, wspace=0.22)
    plt.savefig(plot_name)
    plt.close() 
    gc.collect()
    return plot_name


def plot_diagnostic(nk,bic,logL,figout,nk_optimal=None):
    
    bic = np.array(bic)
    nk = np.array(nk)
    logL = np.array(logL)

    plt.close()
    fig = plt.figure(figsize=(7.0,4.0))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0])
    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)
    ax.set_xlabel(r"n$_{\rm k}$", fontsize=18)
    ax.set_ylabel(r"-log(L)", fontsize=18)
    ax.set_xlim(0,16)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    ax.scatter(nk,-logL,color='black')
    ax.plot(nk,-logL,color='black')
    if nk_optimal is not None: 
        ax.scatter(nk[nk==nk_optimal],-logL[nk==nk_optimal],marker='o',s=600,facecolor='orange',edgecolor='none',lw=2,alpha=1,zorder=-1)
    ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    gs.update(left=0.12,bottom=0.14,right=0.95, top=0.95, hspace=0.40, wspace=0.40)
    plt.savefig(figout)
    plt.close()    
    gc.collect()
    return


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


def calc_faceon_matrix(angmom_vec, up=[0.0, 1.0, 0.0]):
    # is exactly the same function as pynbody.analysis.angmom.calc_faceon_matrix()
    vec_in = np.asarray(angmom_vec)
    vec_in = vec_in / np.sum(vec_in ** 2).sum() ** 0.5
    vec_p1 = np.cross(up, vec_in)
    vec_p1 = vec_p1 / np.sum(vec_p1 ** 2).sum() ** 0.5
    vec_p2 = np.cross(vec_in, vec_p1)
    matr = np.concatenate((vec_p1, vec_p2, vec_in)).reshape((3, 3))
    # check if the matrix is orthogonal
    resid = np.dot(matr, np.asarray(matr).T) - np.eye(3)
    resid = (resid ** 2).sum()
    if resid > 1.e-8 or resid != resid: print('The rotation matrix is not orthogonal')
    return matr


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

def inertia_shape(mass,x,y,z):
    r = np.sqrt(x**2+y**2+z**2)
    Ixx = np.sum(np.multiply(mass,((y**2+z**2)/r**2)))
    Ixy = np.sum(np.multiply(mass,(x*y/r**2)))
    Ixz = np.sum(np.multiply(mass,(x*z/r**2)))
    Iyy = np.sum(np.multiply(mass,((x**2+z**2)/r**2)))
    Iyz = np.sum(np.multiply(mass,(y*z/r**2)))
    Izz = np.sum(np.multiply(mass,((x**2+y**2)/r**2)))
    massT = np.sum(mass)
    Inertia = np.array([[Ixx/massT,-Ixy/massT,-Ixz/massT],[-Ixy/massT,Iyy/massT,-Iyz/massT],[-Ixz/massT,-Iyz/massT,Izz/massT]])
    eig_values, eig_vector = linalg.eig(Inertia)
    aux = (np.array(eig_values)).real
    srt = np.argsort(aux)
    e1, e2, e3 = aux[srt]
    va = eig_vector[:,srt[0]]
    vb = eig_vector[:,srt[1]]
    vc = eig_vector[:,srt[2]]
    b2a = np.sqrt((e1+e3-e2)/(e3+e2-e1))
    c2a = np.sqrt((e1+e2-e3)/(e3+e2-e1))
    a = np.sqrt(5.*(e3+e2-e1)/2.)
    b = b2a*a
    c = c2a*a
    return va, vb, vc, a, b, c


def shapes(a,b,c):
    triaxiality = (1-b**2/a**2)/(1-c**2/a**2)
    ellipticity = (a**2-c**2)/(a**2+b**2+c**2)
    prolateness = (a**2+c**2-2.*b**2)/(a**2+b**2+c**2)
    return triaxiality, ellipticity, prolateness 


def center_of_mass(mass,x,y,z):
    m = np.sum(mass)    
    return np.array([np.sum(mass*x)/m,np.sum(mass*y)/m,np.sum(mass*z)/m])


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

def n_parameters(model):
    # This function was taken from sklearn, only changing how it is called.  
    """Return the number of free parameters in the model."""
    _, n_features = model.means_.shape
    if model.covariance_type == 'full':
        cov_params = model.n_components * n_features * (n_features + 1) / 2.
    elif model.covariance_type == 'diag':
        cov_params = model.n_components * n_features
    elif model.covariance_type == 'tied':
        cov_params = n_features * (n_features + 1) / 2.
    elif model.covariance_type == 'spherical':
        cov_params = model.n_components
    mean_params = n_features * model.n_components
    return int(cov_params + mean_params + model.n_components - 1)


def gmm_clustering(GMM_input_file, n_init=1, plot=True, trig_scaling_plot=None, verbose=True, *args, **kwargs):

    if verbose:
        print('-------------------------------------------------------------------------------------------------------------------------')
        print('This function calls scikit learn to run gaussian mixture models.')
        print('Required input: GMM_input_file')
        print('GMM_input_file = the file containing the feature space obs = np.array(shape(n_particles,n_features))')
        print('By default it will require the algorithm to search for 2 clusters.')
        print('If you want to run it with a different number of clusters, set the arg number_of_clusters.')
        print('By default it assumes each clustering component has it own general covariance matrix, arg covariance_type=full')
        print('By default the input data is centered to the mean and scaled to unit variance. Set arg whiten_data=False to override the default.')
        print('By default, the Expectation-Maximization algorithm is run for up to 100 times or up until the convergence threshold is met (1.0e-03)')
        print('using a single k-means initialization (n_init=1).')
        print('By default it will also plot the mass weighted 1D and the various 2D projected distributions.')
        print('Set arg plot=False if these plots are not to be saved.')
        print('If you want to use other args for the clustering algorithm, check the Gaussian Mixture webpage at http://scikit-learn.org/stable/')

    number_of_clusters = kwargs.get('number_of_clusters', None)
    whiten_data = kwargs.get('whiten_data', True)
    covariance_type = kwargs.get('covariance_type', None)
        
    try:
        data_gmm = pickle.load(open(GMM_input_file,'rb'))
    except:
        print('Input file %s does not exist! You need to run GMM_input first. Returning...'%GMM_input_file)
        return
    
    obs = data_gmm['gmm_input']
    features = data_gmm['feature_space']
    iord = data_gmm['iord']
    mass = data_gmm['mass']
    
    nf = len(obs[0,:].flatten())
    ns = len(obs[:,0].flatten())
    print('Number of data points: %i'%ns)
    print('Number of features: %i'%nf)
     
    filename_base = GMM_input_file[:-4]
    name_add = '_'
    white = ''

    if whiten_data: 
        inputdata = scale(obs)
        print('Centering data to the mean and component wise scale to unit variance. Output file names contain _white')
        white = '_white'
    else: inputdata = obs
        
    if number_of_clusters is None: number_of_clusters=2
    print('searching for %i clusters'%number_of_clusters)
  
    
    print('******************************************************************************************')
    print('Running Gaussian Mixture Model clustering in the parameter space of ',features)

    if covariance_type not in ['full','tied','diag','spherical']: covariance_type='full'

    filename_out=filename_base+'.scikit_gmm_'+covariance_type+'_'+("%i" % number_of_clusters)+'clusters'+white+'.dat'
    print('The result of the clustering will be saved to %s'%filename_out)
    
    if os.path.isfile(filename_out):
        print('The clustering file already exists. Returning...')
        if plot: plot_clustering_results_in_2D(GMM_input_file,filename_out,filename_out[:-4]+'.png',trig_scaling_plot=trig_scaling_plot)        
        gmm_out = pickle.load(open(filename_out,'rb'))
        return gmm_out['n_param'], gmm_out['logL'], gmm_out['bic'], gmm_out['aic'], filename_out

    st = time.time()
    aclus = GMM(n_components=number_of_clusters, covariance_type=covariance_type, n_init=n_init)
    aclus.fit(inputdata)
    centre = aclus.means_ # The means of the adjusted gaussians
    weight = aclus.weights_ # The mixing weights for each mixture component
    covar = aclus.covariances_ # Covariance parameters for each mixture component. The shape depends on covariance_type.
    label = aclus.predict(inputdata) # predicted labels
    logprob = aclus.score_samples(inputdata)
    posterior_probability = aclus.predict_proba(inputdata)
    logL = aclus.score(inputdata)
    bic = aclus.bic(inputdata)
    
    aic = aclus.aic(inputdata)
    converged = aclus.converged_
    n_param = n_parameters(aclus)
    
    elapsed_time = (time.time() - st)/60.
    print("Elapsed time running GMM: %.2f min" % elapsed_time)
        
    f = open(filename_out,'wb')
    pickle.dump({'input_file':GMM_input_file,'features':features,'runtime_min':elapsed_time,
                 'covariance_type':covariance_type,'n_init':n_init,'n_components':number_of_clusters,
                 'gmeans':centre,'gweight':weight,'gcovar':covar,'bic':bic,'logL':logL,
                 'label':label,'p_label':posterior_probability,'iord':iord,
                 'aic':aic,'converged':converged,'n_param':n_param},f)
    f.close()
    del(f)

    print('number of clusters: %i'%number_of_clusters)
    print('cluster means:      ',centre)
    print('cluster weights:    ',weight)
    print('covariance matrix:  ',covar)
    print('log Likelihood:     ',logL)
    print('BIC:                ',bic)
    print('AIC:                ',aic)
    print('converged:          ',converged)
    
    if plot: 
        plot_clustering_results_in_2D(GMM_input_file,filename_out,filename_out[:-4]+'.png',trig_scaling_plot=trig_scaling_plot)        
        #plot_clustering_results_in_1D(GMM_input_file,filename_out,filename_out[:-4]+'_1Dhist.png')
        
    gc.collect()
    return n_param, logL, bic, aic, filename_out


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


