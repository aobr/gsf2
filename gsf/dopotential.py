import pickle, time, os, gc
import numpy as np

import _twobody as twobody
from domath import secondsToStr, calc_faceon_matrix

grav_const=4.302e-6

#################################################################################################################################
########## Functions for the computation of the gravitational potential and the definition of the equatorial plane ##############
#################################################################################################################################


def star_potential(file_star, file_dark, file_gas, out_dir=None, eps=0.1):
    """
    Calls functions from twobody to compute the gravitational potential at all stellar
    particle positions. 
    
    Parameters
    ----------
    file_star, file_dark, file_gas : str
        Filenames of the input simulated halo

    out_dir : str, default=None
        Path to the directory where all data should be saved. If None, the output file will 
        be saved in the same directory as the stellar particle input file.

    eps : float, default=0.1
        Gravitatonal softening in kpc. Represents the minimum particle separation 
        that ensures non-divergent values for the gravitational potential and acceleration 
        at all particle positions. 

    Returns
    -------
    fileout : str
        Name of the output file containing the gravitational potential for each star.         
    """
    
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


def com_Shrinking_sphere(data, coef_nlow = 0.1):
    # Function added by Sara Ortega
    """
    Apply the shrinking sphere method to recenter the position and velocities in data
    
    Parameters
    ----------
    coef_nlow : float, default=0.1
        Sets the fraction of the total mass that gives the sphere in which to apply the algorithm.
        E.g. the centre of mass and velocities is calculated using the particles within an sphere 
        containing 10 percent of the total mass.  

    Returns
    -------
    xcm0, ycm0, zcm0, vx_com, vy_com, vz_com : floats
        Position and velocity of the center of mass.        
    """

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
            

def midplane_potential(file_star, file_dark, file_gas, out_dir=None, eps=0.1, 
                       radius_align=None, *args, **kwargs):
    """
    Calls functions from twobody to compute the gravitational potential in the 
    equatorial plane of the galaxy. 
    
    Parameters
    ----------
    file_star, file_dark, file_gas : str
        Filenames of the input simulated halo

    out_dir : str, default=None
        Path to the directory where all data should be saved. If None, the output file will 
        be saved in the same directory as the stellar particle input file.

    eps : float, default=0.1
        Gravitatonal softening in kpc. Represents the minimum particle separation 
        that ensures non-divergent values for the gravitational potential and acceleration 
        at all particle positions. 

    radius_align : float, default=None
        The 3D radius of the farthest particle to consider for orienting the galaxy.
        Can be either in fraction of rvir or in kpc. If None, assumed to be 0.1*rvir. 

    Returns
    -------
    fileout : str
        Name of the output file containing the description of the equatorial plane
        (e.g. radial dependencies of the gravitational potential, binding energy, 
        and circular velocity ).         
    """

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