import os, gc, pickle, time
import numpy as np
import scipy
import scipy.interpolate
import scipy.special

from domath import transform2monotonic, secondsToStr
from dofilters import apply_filters



#################################################################################################################################
############## functions controling the available features for clustering #######################################################
#################################################################################################################################

def get_list_of_tags_from_file(infile):

    data = pickle.load(open(infile,'rb'))
    keysList = list(data.keys())
    del(data)

    gc.collect()
    return keysList


def selected_features(varlist, featurelist):

    good_tags = []
    for k in range(len(varlist)):
        if varlist[k] in featurelist: 
            good_tags.append(varlist[k])

    return good_tags


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