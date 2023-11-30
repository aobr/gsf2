import pickle, gc, os, time
from sklearn.preprocessing import scale
from sklearn.mixture import GaussianMixture as GMM

from domath import secondsToStr
from doplots import plot_clustering_results_in_2D, plot_clustering_results_in_1D


def n_parameters(model):
    """ 
    Returns the number of free parameters in the model. Function taken from sklearn, and 
    modified to be public. 
    """

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


def gmm_clustering(GMM_input_file, n_init=1, plot=True, trig_scaling_plot=None, 
                   verbose=True, *args, **kwargs):
    """
    Function that calls the Gaussian Mixture Models of sklearn on the data available in 
    the input file GMM_input_file.

    Parameters
    ----------
    n_init: int, default=1
        The Expectation-Maximization algorithm of sklearn runs for up to 100 times or until 
        the convergence threshold is met (1.0e-03), using n_init k-means initializations for
        the clusters. WARNING: you might need a larger value to get convergence!

    trig_scaling_plot : list of bool, default=None
        Setting any of the trig_scaling_plot elements to True will result in the corresponding 
        input feature being scaled as arctan(feature/std(feature)) for plotting only. This might 
        be a good idea for features with large dynamical ranges.

    number_of_clusters :  int, default=2
        Number of multi-dimensional Gaussians or galaxy components. 
        
    covariance_type : string, default='full'
        By default each clustering component is assumed to have its own general covariance 
        matrix ('full'). The other options are: 'tied', 'spherical' and 'diag'. For details, 
        check the sklearn documentation at https://scikit-learn.org/stable/modules/generated/
        
    whiten_data: bool, default=True
        Center the clustering feature space to the mean and scaled to unit variance, feature
        by feature. WARNING: do not set to False if the different features have (very) different 
        dynamical ranges! 

    verbose: bool, default=False
        Gives some useful information on how to find out which other extra arguments of the sklearn
        GaussianMixture can be used inside this functions. 

    """
    
    
    if verbose:
        print('-------------------------------------------------------------------------------------------------------------------------')
        print('If you want to use other args for the clustering algorithm, check the Gaussian Mixture webpage at http://scikit-learn.org/stable/')
        print('-------------------------------------------------------------------------------------------------------------------------')
 
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
    print(('Searching for %i clusters in the parameter space of '%number_of_clusters),features)

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
    logL = aclus.score(inputdata)   # log Likelihood
    bic = aclus.bic(inputdata)      # BIC
    
    aic = aclus.aic(inputdata)      # AIC
    converged = aclus.converged_    # bool 
    n_param = n_parameters(aclus)   # number of parameters (depends on the covariance_type)
    
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

    print('number of clusters: %i '%number_of_clusters)
    print('cluster means:      ',centre)
    print('cluster weights:    ',weight)
    print('covariance matrix:  ',covar)
    print('log Likelihood:     ',logL)
    print('BIC:                ',bic)
    print('AIC:                ',aic)
    print('converged:          ',converged)
    
    plot_clustering_results_in_2D(GMM_input_file,filename_out,filename_out[:-4]+'.png',trig_scaling_plot=trig_scaling_plot)        
    #plot_clustering_results_in_1D(GMM_input_file,filename_out,filename_out[:-4]+'_1Dhist.png')
        
    gc.collect()

    return n_param, logL, bic, aic, filename_out