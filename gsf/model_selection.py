import pickle, gc, time
import numpy as np
import scipy.interpolate
import scipy.special
from scipy import linalg
from scipy.spatial import ConvexHull
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx


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