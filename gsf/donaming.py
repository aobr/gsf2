# SPDX-License-Identifier: GPL-3.0-or-later

import pickle, os, gc
import numpy as np

from .domath import A2_profile, compute_bar_angle_from_2D_inertia_tensor
from .dosummary import compute_component_percentiles, make_latex_table
from .doplots import (plot_moment_maps, plot_2inclinations_moment_maps,
                      plot_faceon_surface_mass_density, plot_bar_A2_profile,
                      plot_phi_distributions)


def tag_components(tmp_file, file_dec, base_output_name=None, fov=80., min_A2A0=0.2,
                   make_plots=True, verbose=True):
    """
    Assign physical names to the kinematic components of a single decomposition
    model, write the tags to disk, and produce the LaTeX diagnostics table.

    Parameters
    ----------
    tmp_file : str
        The temporary file with all per-particle stellar information.
    file_dec : str
        The file containing the decomposition (clustering) result.
    base_output_name : str, optional
        Prefix (including path) for the diagnostic figures. Defaults to a name
        derived from `file_dec`.
    fov : float
        Field of view in kpc used for the moment maps.
    min_A2A0 : float
        Threshold on the peak of the A2/A0 profile above which the galaxy is
        considered barred.
    make_plots : bool
        If True, produce the diagnostic and moment-map figures.
    verbose : bool
        If True, print progress information.

    Returns
    -------
    tag_component : list of str
        The assigned name of each component, in component-index order.
    """
    if base_output_name is None:
        base_output_name = file_dec[:-4]+'.'

    # ----- Load the particle data and compute the azimuthal angle phi -----
    if verbose: print('Load the big file with everything.')
    data = pickle.load(open(tmp_file, 'rb'))
    cosphi = data['x']/data['r2']
    sinphi = data['y']/data['r2']

    phi = np.arcsin(sinphi)
    secondquadrant = ((sinphi >= 0) & (cosphi < 0))
    phi[secondquadrant] = np.pi-phi[secondquadrant]
    thirdquadrant = ((sinphi < 0) & (cosphi <= 0))
    phi[thirdquadrant] = abs(phi[thirdquadrant])
    forthquadrant = ((sinphi < 0) & (cosphi > 0))
    phi[forthquadrant] = np.pi+phi[forthquadrant]
    # this makes phi go from 0 to pi

    # ----- Fourier A2/A0 bar diagnostic for the whole galaxy -----
    if verbose: print('Compute the Fourier A2/A0 profile of the full galaxy.')
    A2_shell, Phase_shell, binR = A2_profile([data['x'], data['y'], data['z']], data['mass'], 40., 80, 20.)
    A2divA0 = {}
    A2divA0['all amplitude'] = np.copy(A2_shell)
    A2divA0['all phase'] = np.copy(Phase_shell)
    A2divA0['all binR'] = np.copy(binR)

    if make_plots:
        plot_bar_A2_profile(A2divA0, base_output_name+'bar_selection_via_maxA2ofR.png')

    # ----- Load the decomposition -----
    if verbose: print('Load the decomposition file.')
    data_dec = pickle.load(open(file_dec, 'rb'))
    component = data_dec['label']
    indices = np.unique(component)

    # Initialize classification
    tag_component = []
    for ii in range(len(indices)):
        tag_component.append('Unclassified')

    # ----- Mass-weighted percentiles and shape diagnostics for this model -----
    if verbose: print('Compute the mass-weighted percentiles.')
    resp, respg = compute_component_percentiles(tmp_file, file_dec, features=['r3', 'vphi', 'vz', 'e'], p=[16, 50, 84])

    # The parameters used for the classification (diskyness, normalized_extent,
    # distance_to_cm, ...) are now computed inside compute_component_percentiles
    # and returned in resp.
    diskyness = resp['diskyness']
    normalized_extent = resp['normalized_extent']
    distance_to_cm = resp['distance_to_cm']

    # ----- Classify the bar -----
    bar_id = None
    km = np.argmax(A2divA0['all amplitude'])
    R_maxA2divA0 = A2divA0['all binR'][km]
    if max(A2divA0['all amplitude']) > min_A2A0 and R_maxA2divA0 < respg['r3'][84]:
        # bar criterion fulfilled: among the bar-like (very negative e) components,
        # take the one with the smallest b/a (excludes puffy halos).
        idx = indices[resp['e'][50] < -0.5]
        b2a = resp['b'][resp['e'][50] < -0.5]/resp['a'][resp['e'][50] < -0.5]
        srt = np.argsort(b2a)
        bar_id = idx[srt][0]
        tag_component[bar_id] = 'Bar'

    # ----- Classify the disks -----
    component_is_disk = []
    for jj in indices:
        if diskyness[jj] >= 0.:
            if bar_id is None:
                component_is_disk.append(jj)
            elif jj != bar_id:
                component_is_disk.append(jj)

    disk_id = None
    thin_disk_id = None
    thick_disk_id = None
    counterrotating_disk_id = None
    if len(component_is_disk) == 1:
        disk_id = component_is_disk[0]
        tag_component[disk_id] = 'Disk'
    if len(component_is_disk) == 2:
        # option 1: thin + thick disk; option 2: disk + counter-rotating disk
        counter_rot_exists = False
        for jj in component_is_disk:
            if resp['vphi'][50][jj] < 0.:
                counter_rot_exists = True
                break
        if counter_rot_exists:  # option 2
            for jj in component_is_disk:
                if resp['vphi'][50][jj] < 0.:
                    counterrotating_disk_id = jj
                    tag_component[counterrotating_disk_id] = 'Counter rotating disk'
                else:
                    disk_id = jj
                    tag_component[disk_id] = 'Disk'
        else:  # option 1
            if diskyness[component_is_disk[0]] > diskyness[component_is_disk[1]]:
                thin_disk_id, thick_disk_id = component_is_disk[0], component_is_disk[1]
            else:
                thin_disk_id, thick_disk_id = component_is_disk[1], component_is_disk[0]
            tag_component[thin_disk_id] = 'Thin disk'
            tag_component[thick_disk_id] = 'Thick disk'
    if len(component_is_disk) > 2:
        for ll, jj in enumerate(component_is_disk):
            tag_component[jj] = 'Disk %i' % (ll+1)

    # ----- Classify the spheroids (bulge / halo / spheroid) -----
    component_is_spheroid = []
    for jj in indices:
        if diskyness[jj] <= 0.:
            if bar_id is None:
                component_is_spheroid.append(jj)
            elif jj != bar_id:
                component_is_spheroid.append(jj)

    spheroid_id = []
    halo_id = []
    bulge_id = []
    if len(component_is_spheroid) >= 1:
        for jj in component_is_spheroid:
            if normalized_extent[jj] <= 0.5:
                bulge_id.append(int(jj))
                tag_component[jj] = 'Bulge'
            else:
                if resp['e'][50][jj] > -0.5:
                    halo_id.append(int(jj))
                    tag_component[jj] = 'Halo'
                else:
                    spheroid_id.append(int(jj))
                    tag_component[jj] = 'Spheroid'

    # ----- Classical vs Disky bulge (only when there is no bar) -----
    # NOTE: the B/P (boxy/peanut) classification is intentionally left out for
    # now, as it is not part of the code paper.
    classical_id, disky_id = [], []
    if len(bulge_id) == 2 and bar_id is None:
        if diskyness[bulge_id[0]] < diskyness[bulge_id[1]]:
            tag_component[bulge_id[0]] = 'Classical bulge'
            tag_component[bulge_id[1]] = 'Disky bulge'
            classical_id.append(bulge_id[0])
            disky_id.append(bulge_id[1])
        else:
            tag_component[bulge_id[0]] = 'Disky bulge'
            tag_component[bulge_id[1]] = 'Classical bulge'
            classical_id.append(bulge_id[1])
            disky_id.append(bulge_id[0])

    # ----- Number multiple members of the same class -----
    # Order matters: the classical/disky relabelling must come after the generic
    # bulge numbering so it wins for the 2-bulge case.
    if len(spheroid_id) > 1:
        for ll, jj in enumerate(spheroid_id): tag_component[jj] = 'Spheroid %i' % (ll+1)
    elif len(spheroid_id) == 1:
        spheroid_id = spheroid_id[0]
        tag_component[spheroid_id] = 'Spheroid'
    else:
        spheroid_id = None
    if len(halo_id) > 1:
        for ll, jj in enumerate(halo_id): tag_component[jj] = 'Halo %i' % (ll+1)
    elif len(halo_id) == 1:
        halo_id = halo_id[0]
        tag_component[halo_id] = 'Halo'
    else:
        halo_id = None
    if len(bulge_id) > 1:
        for ll, jj in enumerate(bulge_id): tag_component[jj] = 'Bulge %i' % (ll+1)
    elif len(bulge_id) == 1:
        bulge_id = bulge_id[0]
        tag_component[bulge_id] = 'Bulge'
    else:
        bulge_id = None
    if len(classical_id) > 1:
        for ll, jj in enumerate(classical_id): tag_component[jj] = 'Classical bulge %i' % ll
    elif len(classical_id) == 1:
        classical_id = classical_id[0]
        tag_component[classical_id] = 'Classical bulge'
    else:
        classical_id = None
    if len(disky_id) > 1:
        for ll, jj in enumerate(disky_id): tag_component[jj] = 'Disky bulge %i' % ll
    elif len(disky_id) == 1:
        disky_id = disky_id[0]
        tag_component[disky_id] = 'Disky bulge'
    else:
        disky_id = None

    # ----- Classify infalling satellite galaxies -----
    component_is_satellite = []
    for jj in indices:
        if distance_to_cm[jj] > respg['r3'][84] and resp['e'][50][jj] > respg['e'][84] and resp['r3'][84][jj] < respg['r3'][84]:
            component_is_satellite.append(jj)

    satellite_id = None
    if len(component_is_satellite) == 1:
        satellite_id = component_is_satellite[0]
        tag_component[satellite_id] = 'Satellite'
    if len(component_is_satellite) > 1:
        satellite_id = []
        for ll, jj in enumerate(component_is_satellite):
            satellite_id.append(jj)
            tag_component[jj] = 'Satellite %i' % (ll+1)

    # ----- Number the leftover Unclassified components, if more than one -----
    unclassified_id = [jj for jj in range(len(tag_component)) if tag_component[jj] == 'Unclassified']
    if len(unclassified_id) > 1:
        for ll, jj in enumerate(unclassified_id):
            tag_component[jj] = 'Unclassified %i' % (ll+1)

    if verbose:
        print('---------------------------------------')
        for ll, nm in enumerate(tag_component): print('Component %i is a %s' % (ll, nm))
        print('---------------------------------------')

    # ----- Bar position angle + phi-distribution diagnostic plot -----
    if bar_id is not None:
        bar_angle = compute_bar_angle_from_2D_inertia_tensor(data['mass'][component == bar_id], data['x'][component == bar_id], data['y'][component == bar_id])
        if verbose: print('The bar position angle from the 2D inertia tensor [deg] = %i' % bar_angle)
        if make_plots:
            phi_recentered = phi/np.pi*180
            to_move = np.where(phi_recentered > bar_angle+90.)
            phi_recentered[to_move] -= 180.
            plot_phi_distributions(phi_recentered, component, base_output_name+'distributions_of_phi.png',
                                   bar_id, bar_angle, kname=tag_component)

    # ----- Moment maps and face-on surface mass density, with the names -----
    if make_plots:
        if verbose: print('Create the moment maps figure using the assigned component names.')
        plot_moment_maps(tmp_file, file_dec, inclination=90., verbose=False, fov=fov, label_soft=False, kname=tag_component)
        plot_2inclinations_moment_maps(tmp_file, file_dec, verbose=False, fov=fov, label_soft=True, kname=tag_component)
        if verbose: print('Plot the face-on surface mass density of all components.')
        plot_faceon_surface_mass_density(tmp_file, file_dec, kname=tag_component, max_sd=3.0e3, max_R=30.,
                                         median_binding_energy=resp['e'][50],
                                         mass_fraction=resp['total_mass']/respg['total_mass'])

    # ----- Save the tags -----
    file_out_tag = file_dec[:-4]+'_tags.dat'
    if verbose: print('Save the component tags to %s' % file_out_tag)
    with open(file_out_tag, 'wb') as f:
        pickle.dump({'tags': tag_component}, f)

    # ----- LaTeX summary table -----
    if verbose: print('Make the summary LaTeX table for this model.')
    make_latex_table(tmp_file, file_dec, file_dec[:-4]+"table.tex", kname=tag_component)

    gc.collect()
    return tag_component
