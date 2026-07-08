import os, pickle, gc
import numpy as np
from .features import get_list_of_tags_from_file
from .domath import percentile, inertia_shape, shapes, center_of_mass

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


def compute_component_percentiles(tmp_file, file_dec, soft=True, features=['r3', 'vphi', 'vz'], p=[16, 50, 84], save_to_file=True):
    """
    Mass-weighted percentiles and shape diagnostics for a SINGLE decomposition
    model. Unlike compute_percentiles (used inside the 1->15 model-selection
    loop, which fills nk-indexed dicts in place), this returns two dictionaries
    for one model:
      result        : per-component percentiles/shapes and the parameters used
                      to name the components (diskyness, normalized_extent, ...)
      result_global : the same quantities for the whole galaxy
    The naming/classification step (donaming.tag_components) relies on both.

    If save_to_file is True, the two dictionaries are pickled to
    file_dec[:-3]+'summary.dat'.
    """
    data_dec = pickle.load(open(file_dec, 'rb'))
    p_label = data_dec['p_label']
    label = data_dec['label']
    iord_dec = data_dec['iord']
    srt = np.argsort(iord_dec)
    p_label = p_label[srt]
    label = label[srt]
    iord_dec = iord_dec[srt]
    indx = np.unique(data_dec['label'])
    nk = len(indx)

    data = pickle.load(open(tmp_file, 'rb'))
    if len(data['iord']) != len(iord_dec):
        print('The iords in the file_dec and tmp_file have different dimensions. Returning...')
        return
    elif int(np.sum(data['iord']-iord_dec)) != 0:
        print('The iords in the file_dec and tmp_file are not the same. Returning...')
        return

    mass = data['mass']

    result = {}
    result_global = {}

    features_in_tmp_file = get_list_of_tags_from_file(tmp_file)
    tags = []
    for feature in features:
        if feature in features_in_tmp_file:
            tags.append(feature)

    for feature in tags:
        result[feature] = {}
        result_global[feature] = {}

    if nk == 1:
        for feature in tags:
            for pp in p:
                varm = percentile(data[feature], data['mass'], percent=pp/100.)
                result[feature][pp] = np.array(varm)
                result_global[feature][pp] = np.array(varm)
    else:
        for feature in tags:
            for pp in p:
                varm = percentile(data[feature], data['mass'], percent=pp/100.)
                result_global[feature][pp] = np.array(varm)
                varm = []
                for i in indx:
                    if soft:
                        varm.append(percentile(data[feature], np.multiply(data['mass'], np.ravel(np.array(p_label[:, i]).flatten())), percent=pp/100.))
                    else:
                        varm.append(percentile(data[feature][label == i], data['mass'][label == i], percent=pp/100.))
                result[feature][pp] = np.array(varm)

    result_global['total_mass'] = np.array(np.sum([data['mass']]))
    result_global['total_Jx'] = np.array([np.sum(np.multiply(data['mass'], data['jx']))])
    result_global['total_Jy'] = np.array([np.sum(np.multiply(data['mass'], data['jy']))])
    result_global['total_Jz'] = np.array([np.sum(np.multiply(data['mass'], data['jz']))])
    va, vb, vc, a, b, c = inertia_shape(mass, data['x'], data['y'], data['z'])
    triax, ellipt, prol = shapes(a, b, c)
    result_global['cm'] = [np.array([0., 0., 0.])]
    result_global['va'] = [va]
    result_global['vb'] = [vb]
    result_global['vc'] = [vc]
    result_global['a'] = np.array([a])
    result_global['b'] = np.array([b])
    result_global['c'] = np.array([c])
    result_global['triaxiality'] = np.array([triax])
    result_global['ellipticity'] = np.array([ellipt])
    result_global['prolateness'] = np.array([prol])

    if nk == 1:
        for stuff in ['total_mass', 'total_Jx', 'total_Jy', 'total_Jz', 'cm', 'va', 'vb', 'vc', 'a', 'b', 'c', 'triaxiality', 'ellipticity', 'prolateness']:
            result[stuff] = result_global[stuff]
    else:
        varm = []
        varmx = []
        varmy = []
        varmz = []
        result['cm'] = []
        result['va'] = []
        result['vb'] = []
        result['vc'] = []
        var_a = []
        var_b = []
        var_c = []
        var_t = []
        var_e = []
        var_p = []
        for i in indx:
            if soft:
                varm.append(np.sum(np.multiply(data['mass'], np.ravel(np.array(p_label[:, i]).flatten()))))
                varmx.append(np.sum(np.multiply(data['jx'], np.multiply(data['mass'], np.ravel(np.array(p_label[:, i]).flatten())))))
                varmy.append(np.sum(np.multiply(data['jy'], np.multiply(data['mass'], np.ravel(np.array(p_label[:, i]).flatten())))))
                varmz.append(np.sum(np.multiply(data['jz'], np.multiply(data['mass'], np.ravel(np.array(p_label[:, i]).flatten())))))
                cm = center_of_mass(np.multiply(data['mass'], np.ravel(np.array(p_label[:, i]).flatten())), data['x'], data['y'], data['z'])
                va, vb, vc, a, b, c = inertia_shape(np.multiply(data['mass'], np.ravel(np.array(p_label[:, i]).flatten())), data['x']-cm[0], data['y']-cm[1], data['z']-cm[2])
            else:
                varm.append(np.sum(data['mass'][label == i]))
                varmx.append(np.sum(np.multiply(data['jx'][label == i], data['mass'][label == i])))
                varmy.append(np.sum(np.multiply(data['jy'][label == i], data['mass'][label == i])))
                varmz.append(np.sum(np.multiply(data['jz'][label == i], data['mass'][label == i])))
                cm = center_of_mass(data['mass'][label == i], data['x'][label == i], data['y'][label == i], data['z'][label == i])
                va, vb, vc, a, b, c = inertia_shape(data['mass'][label == i], data['x'][label == i]-cm[0], data['y'][label == i]-cm[1], data['z'][label == i]-cm[2])

            triax, ellipt, prol = shapes(a, b, c)
            result['cm'].append(cm)
            result['va'].append(va)
            result['vb'].append(vb)
            result['vc'].append(vc)
            var_a.append(a)
            var_b.append(b)
            var_c.append(c)
            var_t.append(triax)
            var_e.append(ellipt)
            var_p.append(prol)

        result['total_mass'] = np.array(varm)
        result['total_Jx'] = np.array(varmx)
        result['total_Jy'] = np.array(varmy)
        result['total_Jz'] = np.array(varmz)
        result['a'] = np.array(var_a)
        result['b'] = np.array(var_b)
        result['c'] = np.array(var_c)
        result['triaxiality'] = np.array(var_t)
        result['ellipticity'] = np.array(var_e)
        result['prolateness'] = np.array(var_p)

    # --- Parameters used by donaming.tag_components to name the components ---
    # (everything except the Fourier A2/bar diagnostics, which stay in donaming)
    def _percomp(res, feat, pp):
        # per-component percentile array (works for nk==1 and nk>1)
        return np.atleast_1d(res[feat][pp])

    a_arr = np.atleast_1d(result['a'])
    c_arr = np.atleast_1d(result['c'])
    c2a = c_arr/a_arr

    # rotational support kappa = 3*Erot/Ekin - 2, using the hard labels
    if 'ke' in data and 'vphi' in data:
        Erot2Ekin = []
        for jj in indx:
            ekin = np.sum(data['mass'][label == jj]*data['ke'][label == jj])
            Erot2Ekin.append(0.5*np.sum(data['mass'][label == jj]*data['vphi'][label == jj]**2)/ekin)
        Erot2Ekin = np.array(Erot2Ekin)
        kappa = 3*Erot2Ekin-2
        xi = 2*(1-c2a)-1
        result['Erot2Ekin'] = Erot2Ekin
        result['diskyness'] = (kappa+xi)/2

    if 'r3' in tags:
        result['normalized_extent'] = _percomp(result, 'r3', 84)/np.atleast_1d(result_global['r3'][84])[0]

    if 'vphi' in tags and 'vz' in tags:
        result['vrot2sigmaz'] = _percomp(result, 'vphi', 50)/((_percomp(result, 'vz', 84)-_percomp(result, 'vz', 16))/2)

    distance_to_cm = []
    for jj in range(len(np.atleast_1d(indx))):
        cm = result['cm'][jj]
        distance_to_cm.append(np.sqrt(cm[0]**2+cm[1]**2+cm[2]**2))
    result['distance_to_cm'] = np.array(distance_to_cm)
    result['c2a'] = c2a

    if save_to_file:
        summary_file = file_dec[:-3]+'summary.dat'
        with open(summary_file, 'wb') as fs:
            pickle.dump({'result': result, 'result_global': result_global}, fs)

    del(data, data_dec, p_label, label, iord_dec, srt, varm)
    gc.collect()

    return result, result_global


def make_latex_table(tmp_file, file_dec, file_out, kname=None):
    """
    Write a LaTeX table of per-component diagnostics (mass, mass fraction, shape,
    kinematics, and mass-weighted percentiles of the available features) for a
    single decomposition model. `kname`, if given, supplies the column headers
    (component names); otherwise columns are labelled C0, C1, ... The optional
    'age' and 'Z' rows are only written if those fields exist in tmp_file.
    """
    data = pickle.load(open(tmp_file, 'rb'))
    print('Load the decontaminated deco file.')
    data_dec = pickle.load(open(file_dec, 'rb'))
    component = data_dec['label']

    nk = len(np.unique(component))

    names = []
    names.append('All')
    if kname is not None:
        for nm in kname: names.append(nm)
    else:
        for nm in range(nk): names.append('C%i' % nm)

    outfile = open(file_out, "w")
    outfile.write("\\begin{table}[h!]\n")
    outfile.write("\\centering\n")
    bstr = "\\begin{tabular}{cc"
    for jj in range(nk): bstr = bstr+r"c"
    bstr = bstr+"}\n"
    outfile.write(bstr)
    outfile.write("\\hline\n")

    bstr = "Property"
    bstr = bstr+" & All"
    if kname is not None:
        for jj in range(nk): bstr = bstr+" & %s" % kname[jj]
    else:
        for jj in range(nk): bstr = bstr+" & C%i" % jj
    outfile.write(bstr+"\\\\ \n")
    outfile.write("\\hline\n")

    mgal = np.sum(data['mass'])
    expo = np.floor(np.log10(mgal))
    bstr = "$M$ [10$^{\\rm %i}$M$_{\\rm\\odot}$]" % expo
    mant = mgal/10**expo
    bstr = bstr+" & %.2f" % mant
    for jj in range(nk):
        mant = np.sum(data['mass'][component == jj])/10**expo
        bstr = bstr+" & %.2f" % mant
    outfile.write(bstr+"\\\\ \n")

    bstr = "$f$=$\\frac{M}{M_{\\rm *}}$"
    bstr = bstr+" & 1.00"
    for jj in range(nk):
        bstr = bstr+" & %.2f" % (np.sum(data['mass'][component == jj])/mgal)
    outfile.write(bstr+"\\\\ \n")

    b2a = []
    flatness = []
    _, _, _, a, b, c = inertia_shape(data['mass'], data['x'], data['y'], data['z'])
    b2a.append(b/a)
    flatness.append(2.*(1-c/a)-1.)
    for jj in range(nk):
        _, _, _, a, b, c = inertia_shape(data['mass'][component == jj], data['x'][component == jj], data['y'][component == jj], data['z'][component == jj])
        b2a.append(b/a)
        flatness.append(2.*(1-c/a)-1.)

    bstr = "$b/a$"
    bstr = bstr+" & %.2f" % b2a[0]
    for jj in range(nk):
        bstr = bstr+" & %.2f" % b2a[jj+1]
    outfile.write(bstr+"\\\\ \n")

    bstr = "$\\xi$"
    bstr = bstr+" & %.2f" % flatness[0]
    for jj in range(nk):
        bstr = bstr+" & %.2f" % flatness[jj+1]
    outfile.write(bstr+"\\\\ \n")

    Erot2Ekin = []
    Erot2Ekin.append(0.5*np.sum(data['mass']*data['vphi']**2)/np.sum(data['mass']*data['ke']))
    for jj in range(nk):
        Erot2Ekin.append(0.5*np.sum(data['mass'][component == jj]*data['vphi'][component == jj]**2)/np.sum(data['mass'][component == jj]*data['ke'][component == jj]))
    bstr = "$\\kappa$"
    bstr = bstr+" & %.2f" % (3*Erot2Ekin[0]-2)
    for jj in range(nk):
        bstr = bstr+" & %.2f" % (3*Erot2Ekin[jj+1]-2)
    outfile.write(bstr+"\\\\ \n")

    bstr = "$\\rm\\sigma$($z$) [kpc]"
    sigmaz = 0.5*(percentile(data['z'], data['mass'], percent=0.84) - percentile(data['z'], data['mass'], percent=0.16))
    bstr = bstr+" & %.2f" % sigmaz
    for jj in range(nk):
        sigmaz = 0.5*(percentile(data['z'][component == jj], data['mass'][component == jj], percent=0.84) - percentile(data['z'][component == jj], data['mass'][component == jj], percent=0.16))
        bstr = bstr+" & %.2f" % sigmaz
    outfile.write(bstr+"\\\\ \n")

    bstr = "$\\rm\\sigma$($v_z$) [km/s]"
    sigmaz = 0.5*(percentile(data['vz'], data['mass'], percent=0.84) - percentile(data['vz'], data['mass'], percent=0.16))
    bstr = bstr+" & %i" % sigmaz
    for jj in range(nk):
        sigmaz = 0.5*(percentile(data['vz'][component == jj], data['mass'][component == jj], percent=0.84) - percentile(data['vz'][component == jj], data['mass'][component == jj], percent=0.16))
        bstr = bstr+" & %i" % sigmaz
    outfile.write(bstr+"\\\\ \n")

    bstr = "$\\rm\\sigma$($v_R$) [km/s]"
    sigmaz = 0.5*(percentile(data['vR'], data['mass'], percent=0.84) - percentile(data['vR'], data['mass'], percent=0.16))
    bstr = bstr+" & %i" % sigmaz
    for jj in range(nk):
        sigmaz = 0.5*(percentile(data['vR'][component == jj], data['mass'][component == jj], percent=0.84) - percentile(data['vR'][component == jj], data['mass'][component == jj], percent=0.16))
        bstr = bstr+" & %i" % sigmaz
    outfile.write(bstr+"\\\\ \n")

    def _pct_row(title, values, mass_arr, fmt="%.2f"):
        aux = [percentile(values, mass_arr, percent=0.16), percentile(values, mass_arr, percent=0.50), percentile(values, mass_arr, percent=0.84)]
        return (" & "+fmt+"$^{\\rm +"+fmt+"}_{\\rm -"+fmt+"}$") % (aux[1], aux[2]-aux[1], aux[1]-aux[0])

    bstr = "$r$ [kpc]"
    bstr = bstr + _pct_row("r", data['r3'], data['mass'])
    for jj in range(nk):
        bstr = bstr + _pct_row("r", data['r3'][component == jj], data['mass'][component == jj])
    outfile.write(bstr+"\\\\ \n")

    bstr = "$R$ [kpc]"
    bstr = bstr + _pct_row("R", data['r2'], data['mass'])
    for jj in range(nk):
        bstr = bstr + _pct_row("R", data['r2'][component == jj], data['mass'][component == jj])
    outfile.write(bstr+"\\\\ \n")

    bstr = "$v_{\\rm\\phi}$ [km/s]"
    bstr = bstr + _pct_row("vphi", data['vphi'], data['mass'], fmt="%i")
    for jj in range(nk):
        bstr = bstr + _pct_row("vphi", data['vphi'][component == jj], data['mass'][component == jj], fmt="%i")
    outfile.write(bstr+"\\\\ \n")

    if 'age' in data:
        bstr = "$age$ [Gyr]"
        bstr = bstr + _pct_row("age", data['age'], data['mass'])
        for jj in range(nk):
            bstr = bstr + _pct_row("age", data['age'][component == jj], data['mass'][component == jj])
        outfile.write(bstr+"\\\\ \n")

    if 'Z' in data:
        # Take the log AFTER computing the percentiles directly on Z, so that
        # particles with Z=0 do not turn into -inf before the percentile.
        def _z_dex_row(zvals, mass_arr):
            zp = np.array([percentile(zvals, mass_arr, percent=0.16),
                           percentile(zvals, mass_arr, percent=0.50),
                           percentile(zvals, mass_arr, percent=0.84)])
            aux = np.log10(zp/0.012)
            return " & %.2f$^{\\rm +%.2f}_{\\rm -%.2f}$" % (aux[1], aux[2]-aux[1], aux[1]-aux[0])
        bstr = "$Z$ [dex]"
        bstr = bstr + _z_dex_row(data['Z'], data['mass'])
        for jj in range(nk):
            bstr = bstr + _z_dex_row(data['Z'][component == jj], data['mass'][component == jj])
        outfile.write(bstr+"\\\\ \n")

    bstr = "$j_z/j_c$"
    bstr = bstr + _pct_row("jzjc", data['jzjc'], data['mass'])
    for jj in range(nk):
        bstr = bstr + _pct_row("jzjc", data['jzjc'][component == jj], data['mass'][component == jj])
    outfile.write(bstr+"\\\\ \n")

    bstr = "$j_p/j_c$"
    bstr = bstr + _pct_row("jpjc", data['jpjc'], data['mass'])
    for jj in range(nk):
        bstr = bstr + _pct_row("jpjc", data['jpjc'][component == jj], data['mass'][component == jj])
    outfile.write(bstr+"\\\\ \n")

    bstr = "$e$/max(|$e$|)"
    bstr = bstr + _pct_row("e", data['e'], data['mass'])
    for jj in range(nk):
        bstr = bstr + _pct_row("e", data['e'][component == jj], data['mass'][component == jj])
    outfile.write(bstr+"\\\\ \n")

    outfile.write("\\hline\n")
    outfile.write("\\end{tabular}\n")
    outfile.write("\\label{Tab:summary_components}\n")
    outfile.write("\\end{table}\n")
    outfile.close()
    return
