#import click
#
#
#@click.command()
#def main():
#    click.echo("This is gsf's command line interface.")


import click

@click.command()
@click.argument('file_star', type=click.Path(exists=True, readable=True), nargs=1)
@click.argument('file_gas', type=click.Path(exists=True, readable=True), nargs=1)
@click.argument('file_dark', type=click.Path(exists=True, readable=True), nargs=1)
@click.option('--varlist', default='jzjc,jpjc,e', help='Comma separated names of the desired features on which to run the clustering.')
@click.option('--doloop', is_flag=False, help='Run gsf in a loop to get the plot log Likelihood vs nk, and log Likelihood vs n_param.')
@click.option('--out_dir', default=None, help='Path to the directory where all data should be saved. If None, gsf will create a new directory output/ in the running directory.')
@click.option('--number_of_clusters', default=2, help='Number of multi-dimensional Gaussians or galaxy components.')
@click.option('--eps', default=0.1, help='Gravitatonal softening in kpc.')
@click.option('--radius_align', default=None, help='The 3D radius of the farthest particle to consider for orienting the galaxy.')
@click.option('--trig_scaling', default=None, help='Setting any of the trig_scaling elements to True will result in the corresponding input feature being scaled as arctan(feature/std(feature)).')
@click.option('--covariance_type', default='full', help='By default each clustering component is assumed to have its own general covariance matrix (full).')
@click.option('--whiten_data', default=True, help='Center the clustering feature space to the mean and scaled to unit variance, feature by feature.')
@click.option('--n_init', default=1, help='The Expectation-Maximization algorithm of sklearn runs for up to 100 times or until the convergence threshold is met (1.0e-03), using n_init k-means initializations for the clusters.')
@click.option('--plot', is_flag=True, help='Plot the 1st, 2nd and 3rd order moment maps for all components and the full galaxy in one figure.')
@click.option('--band', is_flag=True, help='If the band is True, and there is a luminosity feature in the tmp_file, the moments maps figure will be done weighting with particle luminosity instead of particle mass.')
@click.option('--M2L', default=None, help='If the mass-to-light M2L is given, the weighting for the moemnts maps figure will be done with the luminosities computed from masses.')
@click.option('--inclination', default=90., help='Angle in degrees setting the image perspective.')
@click.option('--fov', default=None, help='Field-of-view value in kpc.')
@click.option('--verbose', default=True, help='Gives some useful information that can e.g. speed up the run.')
#@click.option('--filters', default=None, help='A dictionary with rules to select a subsample of the stellar particles for clustering using only LowPass, HighPass and BandPass on available or derivable features.')

def main(file_star, file_gas, file_dark, varlist='jzjc,jpjc,e', doloop=False, out_dir=None, number_of_clusters=2,
         eps=0.1, radius_align=None, trig_scaling=None, covariance_type='full', whiten_data=True, n_init=1, plot=True,
         band=False, M2L=False, inclination=90., fov=None, verbose=True):
    """
    This is the main function of GalacticStructureFinder (GSF)

    GSF separates the stellar particles in a simulated dark matter halo into a 
    fixed number of components using Gaussian Mixture Models in an arbitrary, 
    user-defined feature space. 
    
    The only required input are three files containing the star, gas, and dark 
    matter particles properties, in this order. The galaxy is assumed isolated, 
    and centered, and the expected units are: Msun for masses, kpc for Cartesian
    coordinates, and km/s for Cartesian velocities.  

    GalacticStructureFinder (GSF) either for one model only 
    (fixed number_of_clusters), or in a loop to find the optimal number_of_clusters.

    Parameters

    ----------
    file_star : string, required 
        Path to the file containing the star particles properties.

    file_gas : string, required
        Path to the file containing the gas particles properties.
        
    file_dark : string, required
        Path to the file containing the dark matter particles properties.
    
    varlist : string, default='jzjc,jpjc,e'
        Comma separated names of the desired features on which to run the clustering. 
        It can contain any combination of features either available in file_star or 
        derivable from those available, as listed in the function available_features(), 
        and constructed in the function generate_tmp_file(). Adding new features requires 
        altering the following functions: available_features (add the feature name as a new string), 
        generate_tmp_file (add the rule on how to compute the feature, if not directly 
        available in correct units in file_star), and optionally feature_labels (add the 
        nice latex formatting for the feature name). Any property available in the tmp_file
        generated by the function generate_tmp_file is a potentially valid feature. 

    doloop : bool, default=False
        Run gsf in a loop to get the plot log Likelihood vs nk, and log Likelihood vs n_param.
        The log(L) vs n_param will be used to do the model selection using the elbow method.

    out_dir : string, default=None
        Path to the directory where all data should be saved. If None, gsf will create 
        a new directory output/ in the running directory. 

    number_of_clusters :  int, default=2
        Number of multi-dimensional Gaussians or galaxy components. 

    eps : float, default=0.1
        Gravitatonal softening in kpc. Represents the minimum particle separation 
        that ensures non-divergent values for the gravitational potential and acceleration 
        at all particle positions. 

    radius_align : float, default=None
        The 3D radius of the farthest particle to consider for orienting the galaxy.
        Can be either in fraction of rvir or in kpc. If None, assumed to be 0.1*rvir. 
        
    trig_scaling : list of bool, default=None
        Setting any of the trig_scaling elements to True will result in the corresponding 
        input feature being scaled as arctan(feature/std(feature)). This might be a good idea 
        for features with large dynamical ranges.
        
    covariance_type : string, default='full'
        By default each clustering component is assumed to have its own general covariance 
        matrix ('full'). The other options are: 'tied', 'spherical' and 'diag'. For details, 
        check the sklearn documentation at https://scikit-learn.org/stable/modules/generated/
        
    whiten_data: bool, default=True
        Center the clustering feature space to the mean and scaled to unit variance, feature
        by feature. WARNING: do not set to False if the different features have (very) different 
        dynamical ranges! 
    
    n_init: int, default=1
        The Expectation-Maximization algorithm of sklearn runs for up to 100 times or until 
        the convergence threshold is met (1.0e-03), using n_init k-means initializations for
        the clusters. WARNING: you might need a larger value to get convergence!

    plot: bool, default=True
        Plot the 1st, 2nd and 3rd order moment maps for all components and the full galaxy in 
        one figure. If gsf is called in a loop to find the optimal number of components, setting 
        it to False is more efficient.  
        
    band: bool, default=False
        If the band is True, and there is a luminosity feature in the tmp_file, the moments maps
        figure will be done weighting with particle luminosity instead of particle mass. 
    
    M2L: float, default=None 
        If the mass-to-light M2L is given, the weighting for the moemnts maps figure will be done 
        with the luminosities computed from masses. In practice, the M2L option only changes the 
        units in the surface density panel from Msun/pc^2 to Lsun/pc^2. The band arg has precedence 
        over M2L.

    inclination: float, default=90.
        Angle in degrees setting the image perspective. Default is to view the galaxy edge-on. 
        Should be a float or int between 0 (face-on) and 90 (edge-on). 
    
    fov: float, default=None
        Field-of-view value in kpc. It defaults to 50 kpc internally. 
        
    verbose: bool, default=True
        Gives some useful information that can e.g. speed up the run. 


    Attributes
    ----------

    See Also
    --------

    Notes
    -----

    Examples
    --------
    >>> import numpy as np
    """    
    
    import gsf

    if doloop:
        print('Run gsf in a loop to get the plot log Likelihood vs nk, and log Likelihood vs n_param.')
        print('The log(L) vs n_param will be used to do the model selection using the elbow method.')
        gsf.gsf_loop(file_star, file_gas, file_dark, varlist=varlist, out_dir=out_dir, 
            eps=eps, radius_align=radius_align, trig_scaling=trig_scaling, covariance_type=covariance_type,
            whiten_data=whiten_data, n_init=n_init, plot=plot, verbose=verbose) 
    else:
        print('Run gsf only for what is supposed to be a reasonable number of components to generate the moments maps.') 
        gsf.gsf(file_star, file_gas, file_dark, varlist=varlist, number_of_clusters=number_of_clusters, out_dir=out_dir, 
            eps=eps, radius_align=radius_align, trig_scaling=trig_scaling, covariance_type=covariance_type,
            whiten_data=whiten_data, n_init=n_init, plot=plot, band=band, M2L=M2L, inclination=inclination, fov=fov, 
            verbose=verbose)


if __name__ == "__main__":
    main()
