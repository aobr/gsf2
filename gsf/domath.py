import time
import numpy as np
import scipy
from scipy import linalg
from functools import reduce


def secondsToStr(t):
    """ 
    Returns a string converting the time t in seconds to hh:mm:ss
    """
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(t*1000,),1000,60,60]) 


def percentile(x, w, percent=0.5):
    """ 
    Returns the percentile percent of array x for a given weight array w
    """
    srt = np.argsort(x)
    cum = np.cumsum(w[srt])/np.sum(w)
    aux = x[srt]
    aux[np.argsort(abs(cum-percent))][0]
    
    return aux[np.argsort(abs(cum-percent))][0]
    

def ellipticity_from_moments(x,y,weights):
    """ 
    Returns the ellipticity from a 2D map with 1D array coordinates x and y. 
    The weights is the array that should contain the pixels' masses or luminosities.
    The map is assumed to be centered. 
    """

    r = np.sqrt(x**2+y**2)
    f = weights
    x = x[np.isfinite(f)]
    y = y[np.isfinite(f)]
    r = r[np.isfinite(f)]
    f = f[np.isfinite(f)]
    x = x[r>0]
    y = y[r>0]
    f = f[r>0]
    r = r[r>0]
    Mxx = np.sum(f*(x/r)**2)/np.sum(f)
    Myy = np.sum(f*(y/r)**2)/np.sum(f)
    Mxy = np.sum(f*(x*y)/r**2)/np.sum(f)
    Q = Mxx - Myy # = a-b/a+b cos(2phi)
    U = 2.*Mxy    # = a-b/a+b sin(2phi)
    assymetry = (1.-np.sqrt(Q**2+U**2))/(1.+np.sqrt(Q**2+U**2))
    ellipticity = 1.-assymetry

    return ellipticity


def transform2monotonic(x,y):
    """ 
    Function to remove elements from equal length arrays x and y such that x is strictly monotonically 
    increasing. It first checks for NaNs in the x array, and removes them. 
    """
    
    y = y[np.logical_not(np.isnan(x))]
    x = x[np.logical_not(np.isnan(x))]
    
    if not np.all(x[:-1] <= x[1:]):
        print('The x array is not monotonically increasing. Get rid of the offending elements.')
        nx = len(x)
        n_off = 0
        while not np.all(x[:-1] <= x[1:]):
            i_ok = [0]
            for i in range(len(x) - 1): 
                if x[i+1] >= x[i]: i_ok.append(i+1)
                else: n_off += 1
            x = x[i_ok]
            y = y[i_ok]
        print('I discarded %i elements of a total of %i from the bindingE and j_circ arrays.'%(n_off,nx))     
    
    return x,y


def rotate_x(pv, angle):
    """ 
    Rotate Cartesian coordinate or velocity array of particles by angle angle around the x-axis.  
    """

    angle *= np.pi / 180    # transform deg to rad
    rot = np.matrix([[1,      0,             0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle),  np.cos(angle)]])
    
    return np.matmul(rot,pv)


def calc_faceon_matrix(angmom_vec, up=[0.0, 1.0, 0.0]):
    """ 
    Calculates the 3D rotation matrix that reorients the coordinate system with the z-axis 
    along the input vector angmom_vec. This function was taken from the pynbody package at
    https://pynbody.github.io/pynbody/
    """

    vec_in = np.asarray(angmom_vec)
    vec_in = vec_in / np.sum(vec_in ** 2).sum() ** 0.5
    vec_p1 = np.cross(up, vec_in)
    vec_p1 = vec_p1 / np.sum(vec_p1 ** 2).sum() ** 0.5
    vec_p2 = np.cross(vec_in, vec_p1)
    matr = np.concatenate((vec_p1, vec_p2, vec_in)).reshape((3, 3))
    # check if the matrix is orthogonal
    resid = np.dot(matr, np.asarray(matr).T) - np.eye(3)
    resid = (resid ** 2).sum()
    if resid > 1.e-8 or resid != resid: print('The rotation matrix is not orthogonal')

    return matr


def inertia_shape(mass,x,y,z):
    """ 
    Calculates the inertia tensor of the swarm of particles with masses mass and 
    Cartesian coordinates x,y,z, to return its ordered eigenvectors va,vb,vc,
    and eigenvalues a>b>c.  
    """

    r = np.sqrt(x**2+y**2+z**2)
    Ixx = np.sum(np.multiply(mass,((y**2+z**2)/r**2)))
    Ixy = np.sum(np.multiply(mass,(x*y/r**2)))
    Ixz = np.sum(np.multiply(mass,(x*z/r**2)))
    Iyy = np.sum(np.multiply(mass,((x**2+z**2)/r**2)))
    Iyz = np.sum(np.multiply(mass,(y*z/r**2)))
    Izz = np.sum(np.multiply(mass,((x**2+y**2)/r**2)))
    massT = np.sum(mass)
    Inertia = np.array([[Ixx/massT,-Ixy/massT,-Ixz/massT],
                        [-Ixy/massT,Iyy/massT,-Iyz/massT],
                        [-Ixz/massT,-Iyz/massT,Izz/massT]])
    eig_values, eig_vector = linalg.eig(Inertia)
    aux = (np.array(eig_values)).real
    srt = np.argsort(aux)
    e1, e2, e3 = aux[srt]
    va = eig_vector[:,srt[0]]
    vb = eig_vector[:,srt[1]]
    vc = eig_vector[:,srt[2]]
    b2a = np.sqrt((e1+e3-e2)/(e3+e2-e1))
    c2a = np.sqrt((e1+e2-e3)/(e3+e2-e1))
    a = np.sqrt(5.*(e3+e2-e1)/2.)
    b = b2a*a
    c = c2a*a
    
    return va, vb, vc, a, b, c


def shapes(a,b,c):
    """ 
    Calculates the trixiality, ellipticity and prolateness from the 
    eigenvalues of the inertia tensor a>b>c.  
    """

    triaxiality = (1-b**2/a**2)/(1-c**2/a**2)
    ellipticity = (a**2-c**2)/(a**2+b**2+c**2)
    prolateness = (a**2+c**2-2.*b**2)/(a**2+b**2+c**2)

    return triaxiality, ellipticity, prolateness 


def center_of_mass(mass,x,y,z):
    """ 
    Calculates the center of mass for the swarm of particles with masses mass.  
    """

    m = np.sum(mass)    
    
    return np.array([np.sum(mass*x)/m,np.sum(mass*y)/m,np.sum(mass*z)/m])

def A2_profile(Stars_Pos, Mstar, boxsize, Nbins, rmax):
    """
    Fourier A2/A0 amplitude and phase of the m=2 mode as a function of 3D
    radius, used as a bar diagnostic. The galaxy is assumed already face-on.
    Returns (A2_shell, Phase_shell, rbins) restricted to bins with enough
    particles.
    """
    rmin = 0.0
    Nperbinmin = 100
    A2_shell = np.zeros(Nbins)
    Phase_shell = np.zeros(Nbins)

    dx = Stars_Pos[0]
    dy = Stars_Pos[1]
    dz = Stars_Pos[2]

    dummy = np.where(dx < -boxsize/2)[0]
    dx[dx > boxsize/2] = dx[dx > boxsize/2] - boxsize
    dx[dummy] = dx[dummy] + boxsize

    dummy = np.where(dy < -boxsize/2)[0]
    dy[dy > boxsize/2] = dy[dy > boxsize/2] - boxsize
    dy[dummy] = dy[dummy] + boxsize

    dummy = np.where(dz < -boxsize/2)[0]
    dz[dz > boxsize/2] = dz[dz > boxsize/2] - boxsize
    dz[dummy] = dz[dummy] + boxsize

    distance = np.sqrt(dx*dx + dy*dy + dz*dz)
    index = np.argsort(distance)
    distance = distance[index]
    Mstar_tmp = Mstar[index]
    dx = dx[index]/distance
    dy = dy[index]/distance
    dz = dz[index]/distance

    nbin_tmp = ((distance-rmin) / (rmax-rmin) * Nbins).astype(int)
    rbins = np.zeros(Nbins)
    Nperbin = np.zeros(Nbins, dtype=int)
    for k in range(Nbins):
        ndx_tmp = np.where((nbin_tmp == k))[0]
        Mshell = np.sum(Mstar_tmp[ndx_tmp])
        rbins[k] = np.mean(distance[ndx_tmp])
        Nperbin[k] = len(distance[ndx_tmp])
        costheta = dx[ndx_tmp]
        sintheta = dy[ndx_tmp]
        Am = np.nansum(2 * costheta*sintheta * Mstar_tmp[ndx_tmp])
        Bm = np.nansum((costheta*costheta - sintheta*sintheta) * Mstar_tmp[ndx_tmp])
        A2 = np.sqrt(Am*Am + Bm*Bm)/Mshell
        A2_shell[k] = A2
        Phase2 = 0.5 * np.arctan2(Am, Bm)
        Phase_shell[k] = Phase2

    return A2_shell[Nperbin > Nperbinmin], Phase_shell[Nperbin > Nperbinmin], rbins[Nperbin > Nperbinmin]


def inertia_shape_2D(mass, x, y):
    """
    2D (projected) reduced inertia tensor eigen-decomposition. Returns the two
    eigenvectors and eigenvalues (sorted ascending), used to derive a bar
    position angle.
    """
    r = np.sqrt(x**2 + y**2)
    x = x[r > 0]
    y = y[r > 0]
    mass = mass[r > 0]
    r = r[r > 0]
    Ixx = np.sum(np.multiply(mass, ((y**2)/r**2)))
    Ixy = np.sum(np.multiply(mass, (x*y/r**2)))
    Iyy = np.sum(np.multiply(mass, ((x**2)/r**2)))
    massT = np.sum(mass)
    Inertia = np.array([[Ixx/massT, -Ixy/massT], [-Ixy/massT, Iyy/massT]])
    eig_values, eig_vector = linalg.eig(Inertia)
    aux = (np.array(eig_values)).real
    srt = np.argsort(aux)
    e1, e2 = aux[srt]
    v1 = eig_vector[:, srt[0]]
    v2 = eig_vector[:, srt[1]]
    return v1, v2, e1, e2


def compute_bar_angle_from_2D_inertia_tensor(m_bar, x_bar, y_bar):
    """
    Position angle (degrees) of the bar, derived from the 2D inertia tensor of
    the bar particles.
    """
    v1, v2, e1, e2 = inertia_shape_2D(m_bar, x_bar, y_bar)
    bar_angle = np.arctan(v1[1]/v1[0]) * 180/np.pi
    return bar_angle
