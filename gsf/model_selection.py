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
from matplotlib.ticker import MultipleLocator


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

# ===========================================================================
# Data-driven model selection (st + modified ICL), replacing the old
# select_optimal_model() CHull scree above. The heavy per-particle arrays in
# the deco files are read one at a time by gather_deco_stats() and reduced to
# the few scalars each criterion needs.
# ===========================================================================


def gather_deco_stats(file_decs):
    """
    Read a list of decomposition (deco) files one at a time and return the
    statistics needed by all model-selection criteria, without keeping the
    large per-particle arrays in memory. Arrays are ordered by increasing
    number of components n.

    Returns a dict with:
      nf     : number of features
      N      : number of data points (particles)
      n      : array of number of components
      nparam : array of number of free GMM parameters
      lnL    : array of per-particle log-likelihoods
      EN     : array of fuzzy-classification entropies (Sum r*ln r, <= 0)
      files  : list of deco file paths, aligned with n
    """
    n_list, nparam_list, lnL_list, EN_list, files_list = [], [], [], [], []
    nf, N = None, None
    for f in file_decs:
        with open(f, 'rb') as fh:
            deco = pickle.load(fh)
        R = deco['p_label']                       # (N, K) responsibilities
        if N is None: N = R.shape[0]
        if nf is None: nf = deco['gmeans'].shape[1]
        R_safe = np.maximum(R, 1e-300)
        EN = np.sum(R*np.log(R_safe))             # <= 0
        n_list.append(deco['n_components'])
        nparam_list.append(deco['n_param'])
        lnL_list.append(deco['logL'])             # per particle
        EN_list.append(EN)
        files_list.append(f)
        del deco, R, R_safe
        gc.collect()

    n_arr = np.array(n_list)
    srt = np.argsort(n_arr)
    return {
        'nf': nf,
        'N': N,
        'n': n_arr[srt],
        'nparam': np.array(nparam_list)[srt],
        'lnL': np.array(lnL_list)[srt],
        'EN': np.array(EN_list)[srt],
        'files': [files_list[i] for i in srt],
    }


def compute_modified_icl(stats, lam=1.0):
    """
    Modified ICL of Equation 9: ICL*(lambda) = -ln(L) + 0.5*nparam*ln(N) - lambda*EN,
    evaluated for every model in stats. lambda=1 recovers the standard ICL.
    Returns an array aligned with stats['n'].
    """
    N = stats['N']
    neg_logL = -N*stats['lnL']
    bic_penalty = 0.5*stats['nparam']*np.log(N)
    return neg_logL + bic_penalty - lam*stats['EN']


def lambda_sweep(stats, lam_min=0.1, lam_max=100., n_lambda=1000, log_spacing=True):
    """
    Sweep the entropy weight lambda for the modified ICL and, at each lambda,
    record the optimal number of components. Returns the plateaus (contiguous
    lambda intervals over which a given n stays optimal), sorted by width.
    """
    K = stats['n']
    N = stats['N']
    logL_total = N*stats['lnL']
    EN = stats['EN']

    if log_spacing:
        lambdas = np.logspace(np.log10(lam_min), np.log10(lam_max), n_lambda)
    else:
        lambdas = np.linspace(lam_min, lam_max, n_lambda)

    neg_logL = -logL_total
    bic_pen = 0.5*stats['nparam']*np.log(N)
    icl_matrix = neg_logL[None, :] + bic_pen[None, :] - lambdas[:, None]*EN[None, :]
    Kopt = K[np.argmin(icl_matrix, axis=1)]

    plateaus = {}
    for k in np.unique(Kopt):
        lam_vals = lambdas[Kopt == k]
        plateaus[int(k)] = {'lam_start': lam_vals[0], 'lam_end': lam_vals[-1],
                            'width': lam_vals[-1]-lam_vals[0], 'n_points': int(np.sum(Kopt == k))}
    plateaus = dict(sorted(plateaus.items(), key=lambda x: -x[1]['width']))

    return {'lambdas': lambdas, 'Kopt': Kopt, 'K_values': K, 'icl_matrix': icl_matrix, 'plateaus': plateaus}


def modified_ICL(stats, figname=None, figtitle=None, Kmin=2, Kmax=100,
                 klm=2, lam_min=0.1, n_lambda=1000, log_spacing=True):
    """
    Modified-ICL stability analysis (Section 3.1.2). Computes the physically
    motivated lambda_max = |ln(L_klm)/EN_klm|, sweeps lambda, and derives the
    normalized plateau-width weight P(n) for each model order n in [Kmin,Kmax]
    (excluding n=1). Optionally saves the P(n) figure. Returns a dict with the
    ranking (models ordered by decreasing P(n)).
    """
    K = stats['n']
    N = stats['N']
    logL_total = N*stats['lnL']
    EN = stats['EN']

    if klm in list(K):
        ii = list(K).index(klm)
        lambda_max = abs(logL_total[ii]/EN[ii])
    else:
        lambda_max = 100.

    resultsl = lambda_sweep(stats, lam_min=lam_min, lam_max=lambda_max, n_lambda=n_lambda, log_spacing=log_spacing)
    plat = resultsl['plateaus']

    total_lambda_width = 0.
    for key in plat.keys():
        if key >= Kmin and key <= Kmax:
            total_lambda_width += plat[key]['width']
    Klist, deltafraclist = [], []
    for key in plat.keys():
        if key >= Kmin and key <= Kmax and total_lambda_width > 0:
            Klist.append(key)
            deltafraclist.append(plat[key]['width']/total_lambda_width)
    # order by decreasing P(n)
    order = np.argsort(deltafraclist)[::-1]
    Klist = [Klist[i] for i in order]
    deltafraclist = [deltafraclist[i] for i in order]

    if figname is not None:
        Roman_numbers = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV']
        plt.close()
        fig = plt.figure(figsize=(9.0, 6.0))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0])
        ax.set_xlabel(r"$n$", fontsize=24)
        ax.set_ylabel(r"$\mathcal{P}(n)$", fontsize=24)
        ax.yaxis.labelpad = -5
        plt.setp(ax.get_xticklabels(), fontsize=22)
        plt.setp(ax.get_yticklabels(), fontsize=22)
        ax.set_xlim(1.1, 15.9)
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
        xa = np.arange(15)+1
        da = 0.8
        prob = np.zeros(len(xa))
        for k, y in list(zip(Klist, deltafraclist)):
            if 1 <= k <= len(prob): prob[k-1] = y
        for pp, yy in list(zip(xa, prob)):
            ax.fill_between([pp-da/2, pp+da/2], [yy, yy], [0, 0], color='blue', alpha=0.5)
            if yy > 0.01:
                ax.text(pp, yy, f"{round(100*yy)}%", ha='center', va='bottom')
        ax.set_ylim(1/100, 150/100)
        ax.semilogy()
        if figtitle is not None: ax.set_title(figtitle, fontsize=24)
        ax.text(0.75, 0.9, r"Rank", color='black', fontsize=22, ha='center', va='center', transform=ax.transAxes)
        ax.text(0.88, 0.9, r"$n$", color='black', fontsize=22, ha='center', va='center', transform=ax.transAxes)
        ii = 0
        for k in Klist:
            if k <= 10:
                ii += 1
                if ii < 4:
                    ax.text(0.75, 0.9-0.1*ii, Roman_numbers[ii-1], color='black', fontsize=22, ha='center', va='center', transform=ax.transAxes)
                    ax.text(0.88, 0.9-0.1*ii, k, color='black', fontsize=22, ha='center', va='center', transform=ax.transAxes)
        gs.update(left=0.15, bottom=0.14, right=0.90, top=0.87)
        plt.savefig(figname, dpi=300)
        plt.close()

    return {'K': np.array(Klist), 'Pn': np.array(deltafraclist), 'plateaus': plat,
            'ranking': list(Klist), 'lambda_max': lambda_max}


def _st_ranking(stats, Kmin=2, Kmax=10):
    """
    Elbow/scree (st) ranking (Section 3.1.1): convex hull of ln(L) vs nparam,
    st statistic on the interior hull vertices, models ranked by decreasing st.
    """
    K = stats['n']
    logL = stats['N']*stats['lnL']
    n_param = stats['nparam']

    points = (np.array([n_param, logL])).transpose()
    hull = ConvexHull(points)
    npar_keep = points[hull.vertices, 0]
    logL_keep = points[hull.vertices, 1]
    logL_keep = logL_keep[np.argsort(npar_keep)]
    npar_keep = npar_keep[np.argsort(npar_keep)]
    j = np.arange(len(npar_keep))
    npar_sel, gain_sel = [], []
    for jj in j[1:-1]:
        npar_sel.append(npar_keep[jj])
        top = (logL_keep[jj]-logL_keep[jj-1])/(npar_keep[jj]-npar_keep[jj-1])
        bottom = (logL_keep[jj+1]-logL_keep[jj])/(npar_keep[jj+1]-npar_keep[jj])
        gain_sel.append(top/bottom)
    K_sel = [int(K[n_param == npa][0]) for npa in npar_sel]

    K_out = np.array(K_sel)
    scree_out = np.array(gain_sel)
    st_sort = np.argsort(scree_out)
    rank_st = [int(k) for k in np.flip(K_out[st_sort]) if Kmin <= k <= Kmax]
    return {'K': K_out, 'st': scree_out, 'rank_st': rank_st, 'nparam_sel': np.array(npar_sel)}


def st_model_selection(stats, figname=None, figtitle=None, Kmin=2, Kmax=10):
    """
    Elbow (st) model selection. Returns the st ranking and, if figname is given,
    saves the st-vs-n diagnostic figure.
    """
    res = _st_ranking(stats, Kmin=Kmin, Kmax=Kmax)
    K_out, scree_out, rank_st = res['K'], res['st'], res['rank_st']
    K = stats['n']
    n_param = stats['nparam']

    if figname is not None:
        xticks = np.arange(2, 15, 2)
        plt.close()
        fig = plt.figure(figsize=(9.0, 6.0))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0])
        ax.set_xlabel(r"$n$", fontsize=24)
        ax.set_ylabel(r"$st$", fontsize=24)
        ax.yaxis.labelpad = 10
        plt.setp(ax.get_xticklabels(), fontsize=22)
        plt.setp(ax.get_yticklabels(), fontsize=22)
        ax.set_xlim(0.5, 15.5)
        ax.set_xticks(xticks)
        srt = np.argsort(K_out)
        ax.scatter(K_out[srt], scree_out[srt], color='black')
        ax.plot(K_out[srt], scree_out[srt], color='black')
        ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.text(0.75, 0.9, r"Rank", color='black', fontsize=22, ha='center', va='center', transform=ax.transAxes)
        ax.text(0.88, 0.9, r"$n$", color='black', fontsize=22, ha='center', va='center', transform=ax.transAxes)
        romans = ['I', 'II', 'III']
        for rr in range(min(3, len(rank_st))):
            ax.text(0.75, 0.8-0.1*rr, romans[rr], color='black', fontsize=22, ha='center', va='center', transform=ax.transAxes)
            ax.text(0.88, 0.8-0.1*rr, r"%i" % rank_st[rr], color='black', fontsize=22, ha='center', va='center', transform=ax.transAxes)
        if figtitle is not None:
            ax.set_title(figtitle, fontsize=24)
        else:
            x2ticks = [n_param[K == x][0] for x in xticks if x in K]
            xticks_present = [x for x in xticks if x in K]
            ax2 = ax.twiny()
            ax2.set_xlabel(r"$n_{\rm par}$", fontsize=24)
            ax2.tick_params(axis='both', which='both', direction='in', bottom=False, top=True, left=False, right=False)
            ax2.set_xlim(0.5, 15.5)
            ax2.xaxis.set_minor_locator(MultipleLocator(1))
            ax2.set_xticks(xticks_present, x2ticks)
            plt.setp(ax2.get_xticklabels(), fontsize=22)
            ax2.xaxis.labelpad = 10
        gs.update(left=0.15, bottom=0.14, right=0.90, top=0.87)
        plt.savefig(figname, dpi=300)
        plt.close()

    return res


def compare_all_criteria(stats):
    """
    Print the top-ranked model order n for each criterion (BIC, AIC, ICL, st,
    modified-ICL). EN alone is intentionally not ranked (it is degenerate and
    trivially favours n=1). Returns a dict with the per-model criterion values
    and the st and mICL sub-results.
    """
    K = stats['n']
    N = stats['N']
    logL_total = N*stats['lnL']
    nparam = stats['nparam']

    BIC = -2*logL_total + nparam*np.log(N)
    AIC = -2*logL_total + 2*nparam
    ICL = compute_modified_icl(stats, lam=1.0)

    st_res = _st_ranking(stats)
    micl_res = modified_ICL(stats, figname=None)

    def _top_min(values, m=3):
        return [int(K[i]) for i in np.argsort(values)[:m]]

    print("\n--- Model-selection rankings (top 3, 1 = best) ---")
    print("  BIC : n = %s" % ", ".join(str(k) for k in _top_min(BIC)))
    print("  AIC : n = %s" % ", ".join(str(k) for k in _top_min(AIC)))
    print("  ICL : n = %s" % ", ".join(str(k) for k in _top_min(ICL)))
    print("  st  : n = %s" % ", ".join(str(k) for k in st_res['rank_st'][:3]))
    print("  mICL: n = %s" % ", ".join(str(k) for k in micl_res['ranking'][:3]))

    return {'K': K, 'BIC': BIC, 'AIC': AIC, 'ICL': ICL, 'st': st_res, 'mICL': micl_res}


def aggregate_model_selections(st_dict, mICL_dict, min_k=2, max_k=10, order_by='st'):
    """
    Combine the st and modified-ICL selections: take the intersection of the
    two rankings within [min_k, max_k], order it by decreasing st (order_by=
    'st') or decreasing lambda plateau width (order_by='mICL'), and return the
    best-ranked order n_opt. If the two selections do not intersect in range,
    fall back to the top st model (with a warning).
    """
    st_rank = [k for k in st_dict['rank_st'] if min_k <= k <= max_k]
    micl_rank = [k for k in mICL_dict['ranking'] if min_k <= k <= max_k]
    intersection = [k for k in st_rank if k in micl_rank]

    if len(intersection) == 0:
        if len(st_rank) > 0:
            n_opt = st_rank[0]
            print('WARNING: st and mICL selections do not intersect in [%i,%i]; '
                  'falling back to the top st model (n=%i).' % (min_k, max_k, n_opt))
        else:
            n_opt = None
            print('WARNING: no model selected in [%i,%i].' % (min_k, max_k))
        return n_opt

    if order_by == 'mICL':
        ordered = [k for k in micl_rank if k in intersection]
    else:
        ordered = [k for k in st_rank if k in intersection]
    return ordered[0]
