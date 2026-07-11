# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2018--, Aura Obreja and the GalacticStructureFinder (gsf) contributors.

import pickle, gc, time, os
import numpy as np
import scipy
import scipy.interpolate
import scipy.stats
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.ticker import MultipleLocator
import PIL
from PIL import Image

from .domath import percentile, rotate_x, ellipticity_from_moments
from .features import feature_labels, feature_range_and_nbin


def physical_component_color(kname):
    """
    Associates a color to a named galaxy component. 
    """

    color = 'black'

    if kname=='Thin disk': color='navy'
    if kname=='Thick disk': color='dodgerblue'
    if kname=='Disk': color='blue'
    if kname=='Counter rotating disk': color='darkcyan'
    if kname=='Bar': color='limegreen'
    if kname=='Spheroid': color='crimson'
    if kname=='Halo': color='magenta'
    if kname=='Classical bulge': color='brown'
    if kname=='Disky bulge': color='darkorange'
    if kname=='B/P bulge': color='violet'

    return color


def plot_clustering_results_in_1D(file_gmm,file_dec,filename_out,palette='rainbow'):
    """
    Creates a figure with the distribution functions for all the features used for 
    clustering (available in file_gmm), split by the contributions of each mixture 
    component (available in file_dec). The third argument is the name of the saved
    figure with the full path.  
    """
    
    print('Plotting the results as mass weighted distributions in the original feature space.')

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
        
        var = np.array(obs[:,j]).flatten()
        if frange is None: frange = [np.nanmin(var),np.nanmax(var)]
        ax.set_xlim(frange)

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

    print('Plotting the clustering results as distributions in the original feature space.')

    data_gmm = pickle.load(open(file_gmm,'rb'))
    data_cluster = pickle.load(open(file_dec,'rb'))

    obs = data_gmm['gmm_input']
    features = data_gmm['feature_space']
    mass = data_gmm['mass']
    mass = np.ones(len(mass))   # do not weight with the mass
    label = data_cluster['label']
    posterior_probability = data_cluster['p_label']

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
                cs = ax.contour(dens.transpose(), 5, extent=extent, origin='lower', linewidths=1, colors=[colorzs[k]], alpha=0.7)

                # Keep only the largest connected segment at each contour level.
                # matplotlib >= 3.8 makes a ContourSet a single Collection whose
                # get_paths() returns one compound Path per level (disconnected
                # segments within a level are separated by MOVETO codes); the old
                # .collections attribute was removed in matplotlib 3.10. So we
                # split each level's compound Path and keep only the biggest piece.
                from matplotlib.path import Path as _MplPath
                pruned_paths = []
                for compound in cs.get_paths():
                    verts, codes = compound.vertices, compound.codes
                    if codes is None or len(verts) == 0:
                        pruned_paths.append(compound)
                        continue
                    starts = list(np.flatnonzero(codes == _MplPath.MOVETO))
                    bounds = starts + [len(codes)]
                    best_v, best_c, best_diam = None, None, -np.inf
                    for a, b in zip(bounds[:-1], bounds[1:]):
                        seg = verts[a:b]
                        if len(seg) == 0:
                            continue
                        diameter = np.max(seg.max(axis=0) - seg.min(axis=0))
                        if diameter > best_diam:
                            best_diam, best_v, best_c = diameter, verts[a:b], codes[a:b]
                    pruned_paths.append(compound if best_v is None else _MplPath(best_v, best_c))
                cs.set_paths(pruned_paths)
    
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
                figure_name = figure_name+'_lumfromM2L'
                print('The weighting will be done with the particle luminosities, derived assuming a constant M/L=%.2f'%M2L)
            pass
    else:
        if M2L:
            weight = data['mass']/M2L
            figure_name = figure_name+'_lumfromM2L'
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
    
    vlos_max = np.percentile(abs(vlos[np.logical_not(np.isnan(vlos))]),90,method='midpoint')
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
        mosaic = np.vstack( [np.asarray( i.resize(min_shape) ) for i in images ] )
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
    im1 = ax1.imshow(np.log10(sdens).transpose(), interpolation='nearest',extent=extent, aspect='auto', 
                     origin='lower',cmap=smd_color, vmin=vmin, vmax=vmax, zorder=1)
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
            ax1.text(0.05,ytext,string,fontsize=10,fontweight='bold',horizontalalignment='left',
                     verticalalignment='center',transform=ax1.transAxes, bbox=props)
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
    im2 = ax2.imshow(vlos.transpose(), interpolation='nearest',extent=extent, aspect='auto', origin='lower',
                     cmap=vlos_color, vmin=vmin, vmax=vmax, zorder=1)
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
    im3 = ax3.imshow(sigmalos.transpose(), interpolation='nearest',extent=extent, aspect='auto', origin='lower',
                     cmap=slos_color, vmin=vmin, vmax=vmax, zorder=1)
    cax3 = fig.add_axes([0.93, 0.15, 0.01, 0.78])
    cb3 = fig.colorbar(im3, cax=cax3,orientation="vertical")
    cb3.set_label(label=r"$\rm\sigma_{\rm los}$ [km/s]",fontsize=10,labelpad=2, fontweight='bold')
    cb3.ax.tick_params(axis='y',labelleft=False,direction='in',labelright=True,pad=1,size=2)
    for t in cb3.ax.get_yticklabels(): 
        t.set_fontsize(8)
        t.set_fontweight('bold')
    
    props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=1.0)
#    ax2.text(0.5,0.9,figure_title,fontweight='bold',color=color,fontsize=12,horizontalalignment='center',verticalalignment='bottom',transform=ax2.transAxes, bbox=props) 
    ax2.text(0.5,0.9,figure_title,fontweight='bold',color=physical_component_color(figure_title),fontsize=12,
             horizontalalignment='center',verticalalignment='bottom',transform=ax2.transAxes, bbox=props) 
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


def plot_bar_A2_profile(A2divA0, figname):
    """
    Two-panel bar diagnostic: the Fourier A2/A0 amplitude and the phase of the
    m=2 mode as a function of radius, for all the stars in the galaxy.
    """
    plt.close()
    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(2, 1)
    ax1 = plt.subplot(gs[0])
    ax1.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    ax1.plot(A2divA0['all binR'], A2divA0['all amplitude'], lw=4, zorder=1, color='lightgrey', label='all')
    ax1.set_ylabel(r"$A_{\rm 2}/A_{\rm 0}$", fontsize=24)
    ax1.set_ylim(-0.02, 1.0)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), fontsize=22)
    ax1.legend(loc=1, frameon=True, fontsize=22)
    ax1.xaxis.set_minor_locator(MultipleLocator())
    ax1.yaxis.set_minor_locator(MultipleLocator())
    ax2 = plt.subplot(gs[1])
    ax2.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    ax2.plot(A2divA0['all binR'], A2divA0['all phase'], lw=4, zorder=1, color='lightgrey')
    ax2.set_xlabel(r"$R$ [kpc]", fontsize=24)
    ax2.set_ylabel(r"$\psi$ [rad]", fontsize=24)
    ax2.set_ylim(-1.1*np.pi/2, 1.1*np.pi/2)
    plt.setp(ax2.get_xticklabels(), fontsize=22)
    plt.setp(ax2.get_yticklabels(), fontsize=22)
    ax2.xaxis.set_minor_locator(MultipleLocator())
    ax2.yaxis.set_minor_locator(MultipleLocator())
    gs.update(left=0.12, bottom=0.14, right=0.95, top=0.95, wspace=0.01, hspace=0.01)
    plt.savefig(figname, dpi=300)
    plt.close()
    return


def plot_phi_distributions(phi_recentered, component, figname, bar_id, bar_angle, kname=None, nphi=60):
    """
    Histogram of the recentered azimuthal angle phi for each component, colored
    and named following the classification conventions, with the bar component
    highlighted (filled) and the bar position angle annotated. Purely a
    diagnostic plot (no statistical test).
    """
    color_scheme = plt.get_cmap('rainbow')
    cNorm = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_scheme)

    indices = np.unique(component)
    deltac = 1./(len(indices)-1) if len(indices) > 1 else 1.

    plt.close()
    fig = plt.figure(figsize=(5, 3.5))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0])
    gs.update(left=0.15, bottom=0.15, right=0.95, top=0.95)
    ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)

    for ii in indices:
        xxx = np.copy(phi_recentered[component == ii])
        color = scalarMap.to_rgba(ii*deltac)
        label = 'C%i' % ii
        if kname is not None:
            color = physical_component_color(kname[ii])
            label = kname[ii]
        ax.hist(xxx, bins=nphi, alpha=1, label=label, histtype='step', color=color, lw=1.5)

    # Highlight the bar component with a filled histogram.
    ax.hist(phi_recentered[component == bar_id], bins=nphi, alpha=0.3, histtype='stepfilled',
            color=physical_component_color('Bar'), lw=1.5)

    ax.legend(loc=2, frameon=True)
    ax.set_xlabel(r"$\phi$ [$^{\rm o}$]")
    ax.set_ylabel(r"histogram")
    ftext = r"$\phi_{\rm bar}$($I_{\rm xy}$)=%.1f$^{\rm o}$" % bar_angle
    ax.text(0.95, 0.93, ftext, ha='right', va='center', transform=ax.transAxes)

    plt.savefig(figname, dpi=300)
    plt.close()
    return


def plot_faceon_surface_mass_density(tmp_file, dec_file, kname=None, max_sd=None, max_R=None,
                                     median_binding_energy=None, mass_fraction=None):
    """
    Face-on stellar surface mass density profile of the whole galaxy and of each
    component. If kname is given, components are colored and labelled following
    the classification conventions.
    """
    data_dec = pickle.load(open(dec_file, 'rb'))
    comp = data_dec['label']
    data_tmp = pickle.load(open(tmp_file, 'rb'))
    rxy = np.sqrt(data_tmp['x']**2 + data_tmp['y']**2)
    mass = data_tmp['mass']

    rlim = np.percentile(rxy, 90)
    if max_R is not None: rlim = max_R
    dr = 0.3
    nbins = int(rlim/dr)+1

    plt.close()
    fig = plt.figure(figsize=(4.0, 2.5))
    ax1 = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0.15, bottom=0.20, right=0.95, top=0.95)

    summass, bin_edges, _ = scipy.stats.binned_statistic(rxy, mass, statistic='sum', bins=nbins, range=(0, rlim))
    R = 0.5*(bin_edges[1:]+bin_edges[:-1])
    A = np.pi*(bin_edges[1:]**2-bin_edges[:-1]**2)*1.0e6
    smd = summass/A

    color_scheme = plt.get_cmap('rainbow')
    cNorm = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_scheme)
    colorzs = []
    if len(np.unique(comp)) > 1: deltac = 1./(len(np.unique(comp))-1)
    else: deltac = 1.
    for k in np.unique(comp): colorzs.append(scalarMap.to_rgba(k*deltac))

    ax1.plot(R, smd, zorder=-1, color='lightgrey', lw=2)
    loop_index = np.arange(len(np.unique(comp)))
    if median_binding_energy is not None: loop_index = np.argsort(median_binding_energy)
    for k in loop_index:
        summass_comp, bin_edges, _ = scipy.stats.binned_statistic(rxy[comp == k], mass[comp == k], statistic='sum', bins=nbins, range=(0, rlim))
        smd_comp = summass_comp/A
        color = colorzs[k]
        if kname is not None:
            color = physical_component_color(kname[k])
            label = kname[k]
            if mass_fraction is not None: label = kname[k]+(' (%.2f)' % mass_fraction[k])
            ax1.plot(R, smd_comp, color=color, lw=1.5, label=label)
        else:
            ax1.plot(R, smd_comp, color=color, lw=1)

    ax1.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    ax1.legend(loc=1, frameon=False)
    ax1.set_xlabel(r"R [kpc]")
    ax1.set_ylabel(r"$\Sigma_{\rm\bigstar}$ [M$_{\rm\odot}$pc$^{\rm -2}$]")
    if max_R is not None: ax1.set_xlim(0, max_R)
    else: ax1.set_xlim(0, 1.02*rlim)
    ax1.semilogy()
    ax1.xaxis.set_minor_locator(MultipleLocator())
    if max_sd is not None: ax1.set_ylim(1.0e-5*max_sd, max_sd)
    plt.savefig(dec_file[:-4]+'_faceon_smd.png', dpi=300)
    plt.close()
    return


def plot_2inclinations_moment_maps(tmp_file, file_dec, band=False, M2L=False,
                                   verbose=True, palette='rainbow', one_plot_only=True,
                                   label_soft=True, *args, **kwargs):
    """
    For each component, build a three-panel map (face-on surface density,
    edge-on surface density, edge-on line-of-sight velocity) and, if
    one_plot_only, merge them into a single mosaic ordered by mass fraction.
    """
    if verbose:
        print('-------------------------------------------------------------------------------------------------------------------------')
        print('This function will create for each component found in file_dec a png figure with ')
        print('surface density face-on, surface density edge-on, and line-of-sight velocity edge-on maps')
        print('Required input:')
        print('tmp_file = the big temporary file with all the available info for stellar particles')
        print('file_dec = the file containing the result of the clustering algorithm.')

    nxny = kwargs.get('nxny', None)  # number of pixels per side
    fov = kwargs.get('fov', None)    # figure size in kpc
    kname = kwargs.get('kname', None)  # list of component names

    start = time.time()

    print('Loading the results of the clustering from file_dec...')
    try:
        data_dec = pickle.load(open(file_dec, 'rb'))
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
    figure_name = file_dec[:-3]+'variant'

    print('Read the tmp_file...')
    data = pickle.load(open(tmp_file, 'rb'))

    x1 = data['x']
    y1 = data['y']

    inclination = 90.
    pos = np.array([data['x'], data['y'], data['z']])
    pos = rotate_x(pos, inclination)
    x = np.array(pos[0, :]).flatten()
    y = np.array(pos[1, :]).flatten()
    vel = np.array([data['vx'], data['vy'], data['vz']])
    vel = rotate_x(vel, inclination)
    vz = np.array(vel[2, :]).flatten()

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
                figure_name = figure_name+'_lumfromM2L'
                print('The weighting will be done with the particle luminosities, derived assuming a constant M/L=%.2f' % M2L)
            pass
    else:
        if M2L:
            weight = data['mass']/M2L
            figure_name = figure_name+'_lumfromM2L'
            sb_label = r"log($\rm\Sigma$/L$_{\rm\odot}$pc$^{\rm -2}$)"
            print('The weighting will be done with the particle luminosities, derived assuming a constant M/L=%.2f' % M2L)
        else:
            figure_name = figure_name+'_mass'
            print('The weighting will be done with the particle masses.')
    figure_name = figure_name+'_moments'

    if fov is None:
        fov = 50.  # kpc
        print('The box side is %i kpc' % fov)
    lim = 0.5*fov
    if nxny is None:
        pixel_size_kpc = 0.5  # default pixel size is 500 pc
        nxny = int(2*fov/pixel_size_kpc)
    print('The images are %ix%i pixels' % (nxny, nxny))

    rangex = (-lim, lim)
    rangey = (-lim, lim)
    range2d = (rangex, rangey)
    extent = [rangex[0], rangex[1], rangey[0], rangey[1]]

    color_scheme = plt.get_cmap(palette)
    cNorm = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_scheme)

    total_mass = np.sum(weight)
    indices = np.unique(label)
    indices = indices[np.argsort(indices)]
    nk = len(indices)
    deltac = 1./(nk-1)

    if kname is None:
        kname = []
        for indx in indices: kname.append('Component %i' % indx)

    kell_face = np.zeros(nk)
    kell = np.zeros(nk)
    for k, indx in list(zip(range(nk), indices)):
        ell = ellipticity_from_moments(x[label == indx], y[label == indx], weight[label == indx])
        kell[k] = ell
        ell = ellipticity_from_moments(x1[label == indx], y1[label == indx], weight[label == indx])
        kell_face[k] = ell

    plot_names = []
    b2a = []
    massf = []

    color = 'black'
    sdens, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(x, y, weight, statistic='sum', bins=nxny, range=range2d)
    vlos, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(x, y, weight*vz, statistic='sum', bins=nxny, range=range2d)
    vlos = vlos/sdens
    vlos2, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(x, y, weight*vz**2, statistic='sum', bins=nxny, range=range2d)
    vlos2 = vlos2/sdens
    pixel_size_kpc = abs(xedges[1]-xedges[0])
    sdens = sdens/(pixel_size_kpc**2*1.0e6)  # solar units per pc^2
    mass_fraction = 1.
    massf.append(mass_fraction)
    ell = ellipticity_from_moments(x, y, weight)
    b2a.append(1.-ell)
    text = {'label': [('f=%.2f' % mass_fraction), ('b/a=%.2f' % (1.-ell))]}
    plot_name = figure_name+'_2inc_component_all.png'
    plot_names.append(plot_name)
    sb_max = np.log10(np.max(sdens))
    sb_min = np.log10(np.min(sdens[sdens > 0.]))
    sdens[sdens < 10**sb_min] = np.nan
    vlos[np.isnan(sdens)] = np.nan

    vlos_max = np.percentile(abs(vlos[np.logical_not(np.isnan(vlos))]), 90, method='midpoint')

    sdens_face, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(x1, y1, weight, statistic='sum', bins=nxny, range=range2d)
    sdens_face = sdens_face/(pixel_size_kpc**2*1.0e6)
    sdens_face[sdens_face < 10**sb_min] = np.nan
    ell = ellipticity_from_moments(x1, y1, weight)
    text1 = {'label': [('b/a=%.2f' % (1.-ell))]}

    plot_regular_maps_2inclinations(sdens, vlos, sdens_face, 'All stars', extent, rangex, rangey, color, plot_name, sb_label,
                                    text=text, text1=text1, sb_min=sb_min, sb_max=sb_max, vlos_max=vlos_max)

    for k, indx in zip(range(nk), indices):
        if label_soft:
            c_x1 = x1
            c_y1 = y1
            c_x = x
            c_y = y
            c_v = vz
            c_weight = np.multiply(weight, np.ravel(np.array(plabel[:, indx]).flatten()))
        else:
            c_x1 = x1[label == indx]
            c_y1 = y1[label == indx]
            c_x = x[label == indx]
            c_y = y[label == indx]
            c_v = vz[label == indx]
            c_weight = weight[label == indx]
        mass_fraction = np.sum(c_weight)/total_mass
        massf.append(mass_fraction)
        ell = ellipticity_from_moments(c_x, c_y, c_weight)
        b2a.append(1.-ell)
        text = {'label': [('f=%.2f' % mass_fraction), ('b/a=%.2f' % (1.-ell))]}
        color = scalarMap.to_rgba(k*deltac)
        counts, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(c_x, c_y, c_weight, statistic='count', bins=nxny, range=range2d)
        sdens, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(c_x, c_y, c_weight, statistic='sum', bins=nxny, range=range2d)
        vlos, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(c_x, c_y, c_weight*c_v, statistic='sum', bins=nxny, range=range2d)
        vlos = vlos/sdens
        vlos2, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(c_x, c_y, c_weight*c_v**2, statistic='sum', bins=nxny, range=range2d)
        vlos2 = vlos2/sdens
        pixel_size_kpc = abs(xedges[1]-xedges[0])
        sdens = sdens/(pixel_size_kpc**2*1.0e6)
        plot_name = figure_name+('_2inc_component_%i.png' % indx)
        sdens[sdens < 10**sb_min] = np.nan
        vlos[np.isnan(sdens)] = np.nan

        ell = ellipticity_from_moments(c_x1, c_y1, c_weight)
        text1 = {'label': [('b/a=%.2f' % (1.-ell))]}
        sdens_face, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(c_x1, c_y1, c_weight, statistic='sum', bins=nxny, range=range2d)
        sdens_face = sdens_face/(pixel_size_kpc**2*1.0e6)
        sdens_face[sdens_face < 10**sb_min] = np.nan

        plot_regular_maps_2inclinations(sdens, vlos, sdens_face, kname[k], extent, rangex, rangey, color, plot_name, sb_label,
                                        text=text, text1=text1, sb_min=sb_min, sb_max=sb_max, vlos_max=vlos_max)

        plot_names.append(plot_name)

    if one_plot_only:
        print('I will merge all moments plots in one and erase the individual ones.')
        mosaic_name = plot_names[0][:-17]+'mosaic.png'
        srt = np.argsort(np.array(massf[1:]))[::-1]
        plot_names_ordered = []
        plot_names_ordered.append(plot_names[0])  # full galaxy on the top row
        for k in srt: plot_names_ordered.append(plot_names[k+1])
        print('The components are sorted by their mass fractions from bottom to top.')
        images = [PIL.Image.open(i) for i in plot_names_ordered]
        min_shape = sorted([(np.sum(i.size), i.size) for i in images])[0][1]
        mosaic = np.vstack([np.asarray(i.resize(min_shape)) for i in images])
        mosaic = PIL.Image.fromarray(mosaic)
        mosaic.save(mosaic_name)
        for plot_name in plot_names: os.system('rm -rf '+plot_name)

    gc.collect()
    return


def plot_regular_maps_2inclinations(sdens, vlos, sdens_face, figure_title, extent, rangex, rangey, color, plot_name, sb_label,
                                    text=None, text1=None, sb_min=None, sb_max=None, vlos_max=None):
    """
    Render one three-panel figure (edge-on surface density, edge-on
    line-of-sight velocity, face-on surface density) for a single component.
    """
    smd_color = plt.get_cmap('RdYlBu_r')
    vlos_color = plt.get_cmap('bwr')

    plt.close()
    fig = plt.figure(figsize=(10.6, 3.5))
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
    ax1.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    if sb_min is not None: vmin = sb_min
    else: vmin = np.log10(np.nanmin(sdens[sdens > 0.]))
    if sb_max is not None: vmax = sb_max
    else: vmax = np.log10(np.nanmax(sdens[sdens > 0.]))
    vmin_face, vmax_face = vmin, vmax
    im1 = ax1.imshow(np.log10(sdens).transpose(), interpolation='nearest', extent=extent, aspect='auto', origin='lower', cmap=smd_color, vmin=vmin, vmax=vmax, zorder=1)
    cax1 = fig.add_axes([0.306, 0.15, 0.01, 0.78])
    cb1 = fig.colorbar(im1, cax=cax1, orientation="vertical")
    cb1.set_label(label=sb_label, fontsize=10, labelpad=2, fontweight='bold')
    cb1.ax.tick_params(axis='y', labelleft=False, direction='in', labelright=True, pad=1, size=2)
    for t in cb1.ax.get_yticklabels():
        t.set_fontsize(8)
        t.set_fontweight('bold')
    if text is not None:
        props = dict(boxstyle='round', facecolor='white', edgecolor='white', alpha=1.0)
        ytext = 0.90
        for string in text['label']:
            ax1.text(0.05, ytext, string, fontsize=10, fontweight='bold', horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes, bbox=props)
            ytext = ytext-0.07

    ax2 = plt.subplot(gs[1])
    plt.setp(ax2.get_xticklabels(), fontsize=10, fontweight='bold')
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_xlabel(r"[kpc]", fontsize=10, fontweight='bold')
    ax2.xaxis.labelpad = 2
    ax2.yaxis.labelpad = 2
    ax2.set_xlim(rangex)
    ax2.set_ylim(rangey)
    ax2.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    if vlos_max is not None: vmin, vmax = -vlos_max, vlos_max
    else: vmin, vmax = -np.nanmax(abs(vlos)), np.nanmax(abs(vlos))
    im2 = ax2.imshow(vlos.transpose(), interpolation='nearest', extent=extent, aspect='auto', origin='lower', cmap=vlos_color, vmin=vmin, vmax=vmax, zorder=1)
    cax2 = fig.add_axes([0.618, 0.15, 0.01, 0.78])
    cb2 = fig.colorbar(im2, cax=cax2, orientation="vertical")
    cb2.set_label(label=r"v$_{\rm los}$ [km/s]", fontsize=10, labelpad=-5, fontweight='bold')
    cb2.ax.tick_params(axis='y', labelleft=False, direction='in', labelright=True, pad=1, size=2)
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
    ax3.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    im3 = ax3.imshow(np.log10(sdens_face).transpose(), interpolation='nearest', extent=extent, aspect='auto', origin='lower', cmap=smd_color, vmin=vmin_face, vmax=vmax_face, zorder=1)
    cax3 = fig.add_axes([0.93, 0.15, 0.01, 0.78])
    cb3 = fig.colorbar(im3, cax=cax3, orientation="vertical")
    cb3.set_label(label=sb_label, fontsize=10, labelpad=2, fontweight='bold')
    cb3.ax.tick_params(axis='y', labelleft=False, direction='in', labelright=True, pad=1, size=2)
    for t in cb3.ax.get_yticklabels():
        t.set_fontsize(8)
        t.set_fontweight('bold')
    if text1 is not None:
        props = dict(boxstyle='round', facecolor='white', edgecolor='white', alpha=1.0)
        ytext = 0.90
        for string in text1['label']:
            ax3.text(0.05, ytext, string, fontsize=10, fontweight='bold', horizontalalignment='left', verticalalignment='center', transform=ax3.transAxes, bbox=props)
            ytext = ytext-0.07

    props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=1.0)
    ax2.text(0.5, 0.9, figure_title, fontweight='bold', color=physical_component_color(figure_title), fontsize=12, horizontalalignment='center', verticalalignment='bottom', transform=ax2.transAxes, bbox=props)
    gs.update(left=0.05, bottom=0.15, right=0.93, top=0.93, hspace=0.22, wspace=0.22)
    plt.savefig(plot_name)
    plt.close()
    gc.collect()
    return plot_name
