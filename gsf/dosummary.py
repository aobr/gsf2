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
