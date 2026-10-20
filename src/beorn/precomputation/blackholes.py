"""
Black Hole Evolution and Emission
"""
import logging
logger = logging.getLogger(__name__)

#General imports
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad,cumulative_trapezoid,trapezoid
import matplotlib.pyplot as plt

#seed random
rng = np.random.default_rng(seed=0)

#beorn
from ..cosmo import Hubble #Hubble parameter [yr-1]
from ..astro import f_star_Halo
from ..structs.parameters import Parameters
from ..constants import h_eV_sec, eV_per_erg,h__,Hz_per_eV
from .massaccretion import mass_accretion

#constants to add
A = 2*1e-8

#Cosmological functions
def dtdz(z,parameters):
    return 1/((1+z)*Hubble(z,parameters))
def dzdt(z,parameters):
    return (1+z)*Hubble(z,parameters)

def z_to_t(zs,parameters):
    if not isinstance(zs,np.ndarray):
        return -quad(dtdz, zs,np.inf,args=(parameters))[0]
    return np.array([-quad(dtdz, z,np.inf)[0] for z in zs])

def t_to_z(ts,parameters):
    if not isinstance(ts,np.ndarray):
        return brentq(lambda z: z_to_t(z,parameters)-ts, -1, 3000)
    return np.array([brentq(lambda z: z_to_t(z,parameters)-t, -1, 3000) for t in ts])


#Seeding mechanism
def stellar_seed_mass_eq(z,M0,M_seed,parameters):
    #TODO 
    """Compute difference between total stellar mass and black hole seed mass
    
    Args:
        z (np.ndarray/float): redshift at which to evaluate difference
        M0 (float): descendant halo mass
        M_seed (float): Black hole seed mass

    Returns:
        (np.ndarray/float)
    """
    z0 = 5
    zs = np.linspace(z0,z,100)
    mval = mass_accretion(parameters,zs,np.array([M0]),np.array([0.785]))[0]
    fval = (f_star_Halo(parameters,mval).squeeze())[-1]
    Ob,Om = parameters.cosmology.Ob,parameters.cosmology.Om
    return Ob/Om*fval*mval.squeeze()[-1]-M_seed

def duty_cycle(M_BH,f0,gamma):
    #Compute the duty cycle of black hole accretion as a function of black hole mass.
    #TODO still constant, find appropriate values for f0 and gamma
    return f0
    #return f0/(1+(1e6/M_BH)**gamma)

def z_seed(M0,M_seed,parameters):
    funct = lambda z: stellar_seed_mass_eq(z,M0,M_seed,parameters)
    try:
        z_seed_solution = brentq(funct, 4, 32)
    except:
        return 5
    return z_seed_solution

def dN_dzdMp(Mh,Mp,z):
    #Compute the number of mergers per unit redshift and per unit progenitor mass
    #TODO put these constants in Parameters
    alpha,beta,gamma,eta = 0.133,-1.995,0.263,0.0993
    A,xi_ = 0.0104,9.72e-3
    
    return A*(Mh/1e12)**alpha*(Mp/Mh)**beta*np.exp((Mp/(Mh*xi_))**gamma)*(1+z)**eta #fakhouri 2010

def p_merge(Mh,Mp,xi,p):
    """Model Black hole mergers
    
    Args:
        Mh (float): descendant halo mass
        Mp (float): progenitor descendant halo mass
        xi (float): ratio of masses above which the black holes can merge
        p (float): probability of black holes to merge given their halos merge

    Returns:
        (float)
    """
    if Mp/Mh >xi:
        return p
    else:
        return 0.0

def get_scattered_parameters(parameters,N):

    f0s = rng.normal(parameters.source.f0,parameters.source.f0_sig,N)
    f0s = np.maximum(f0s,0)
    Mseeds = rng.normal(parameters.source.M_seed,parameters.source.M_seed_sig,N)
    Mseeds = np.maximum(Mseeds,0)
    ps = rng.normal(parameters.source.p_merge,parameters.source.p_merge_sig,N)
    ps = np.maximum(ps,0)
    xis = rng.normal(parameters.source.xi,parameters.source.xi_sig,N)
    xis = np.maximum(xis,0)

    return f0s,Mseeds,ps,xis



"""def black_hole_mass_iterative(zn,Mh0,parameters):
    Compute the black hole mass evolution for different halo mass and redshift bins
    
    Args:
        zn (np.ndarray): redshift bins
        Mh0 (np.ndarray): descendent halo masses

    Returns:
        numpy.ndarray: Black Hole mass growth in redshift and descendant halo mass bins
        shape (len(Mh0),len(zn))

    #parameters
    Mseed = parameters.source.M_seed
    f0 = parameters.source.f0
    p = parameters.source.p_merge
    xi = parameters.source.xi

    Mh = np.sort(Mh0)
    Mh = Mh[::-1]

    #solve seeding equation for every halo
    z_seeds = np.array([z_seed(m,Mseed,parameters) for m in Mh])
    Dz = np.abs(zn[1]-zn[0])
    
    #initiate array
    M_BH = np.zeros((len(Mh),len(zn)))
    M_BH_dot = np.zeros((len(Mh),len(zn)))
    #print(z_seeds)
    #seeding 
    for i,zseed in enumerate(z_seeds):
        index = np.digitize(zseed,zn)
        if index != len(zn):
            M_BH[i][index] = Mseed
    
    #loop through z bins
    for n in range(1,len(zn)):
        #loop through M bins
        for l in range(len(Mh)-1,-1,-1):

            M_BH[l][n] += M_BH[l][n-1]

            #eddington accretion from the previous z bin
            gas_accr = A*duty_cycle(M_BH[l][n-1],f0,0)*M_BH[l][n-1]*dtdz(zn[n],parameters)

            #mergers from smaller mass halos
            mergers = 0.0
            for i in range(len(Mh)-1,l,-1):
                
                mergers +=  np.abs((Mh[i]-Mh[i-1]))/Mh[l]*dN_dzdMp(Mh[l],Mh[i],zn[n]) * M_BH[i][n] * p_merge(Mh[l],Mh[i],xi,p)
            
            #add both contributions
            delta_M = gas_accr+ mergers
            
            #add them times the redshift step
            M_BH[l][n] += delta_M * Dz

            #save change
            M_BH_dot[l][n] += delta_M * dzdt(zn[n],parameters)

    return M_BH,M_BH_dot,Mh"""

def black_hole_mass_iterative(zn,Mh0,parameters):
    #NOW WITH SCATTER
    """Compute the black hole mass evolution for different halo mass and redshift bins
    
    Args:
        zn (np.ndarray): redshift bins
        Mh0 (np.ndarray): descendent halo masses

    Returns:
        numpy.ndarray: Black Hole mass growth in redshift and descendant halo mass bins
        shape (len(Mh0),len(zn))
    """
    #parameters
    Mseed = parameters.source.M_seed
    f0 = parameters.source.f0
    p = parameters.source.p_merge
    xi = parameters.source.xi
    if isinstance(Mh0,np.ndarray):
        f0s,Mseeds,ps,xis = get_scattered_parameters(parameters,Mh0.size)
    elif isinstance(Mh0,list):
        f0s,Mseeds,ps,xis = get_scattered_parameters(parameters,len(Mh0))
    else:
        f0s,Mseeds,ps,xis = get_scattered_parameters(parameters,1)
    

    Mh = np.sort(Mh0)
    Mh = Mh[::-1]

    #solve seeding equation for every halo
    z_seeds = np.array([z_seed(Mh[i],Mseeds[i],parameters) for i in range(len(Mh))])
    Dz = np.abs(zn[1]-zn[0])
    
    #initiate array
    M_BH = np.zeros((len(Mh),len(zn)))
    M_BH_dot = np.zeros((len(Mh),len(zn)))
    #print(z_seeds)
    #seeding 
    for i,zseed in enumerate(z_seeds):
        index = np.digitize(zseed,zn)
        if index != len(zn):
            M_BH[i][index] = Mseeds[i]
    
    #loop through z bins
    for n in range(1,len(zn)):
        #loop through M bins
        for l in range(len(Mh)-1,-1,-1):

            M_BH[l][n] += M_BH[l][n-1]

            #eddington accretion from the previous z bin
            gas_accr = A*duty_cycle(M_BH[l][n-1],f0s[l],0)*M_BH[l][n-1]*dtdz(zn[n],parameters)

            #mergers from smaller mass halos
            mergers = 0.0
            for i in range(len(Mh)-1,l,-1):
                
                mergers +=  np.abs((Mh[i]-Mh[i-1]))/Mh[l]*dN_dzdMp(Mh[l],Mh[i],zn[n]) * M_BH[i][n] * p_merge(Mh[l],Mh[i],xis[l],ps[l])
            
            #add both contributions
            delta_M = gas_accr+ mergers
            
            #add them times the redshift step
            M_BH[l][n] += delta_M * Dz

            #save change
            M_BH_dot[l][n] += delta_M * dzdt(zn[n],parameters)

    return M_BH,M_BH_dot,Mh


def eps_xray_BH(nu_, parameters: Parameters,Mdot_BH,type):
    """Spectral distribution of X-ray emission per unit BH accretion.
    Args:
        nu_ (float or array): Photon frequency in Hz.
        parameters (Parameters): Model parameters container.

    Returns:
        float or array: Spectral photon emission rate per unit BH accretion.

    Notes:
        The implementation follows Eq. 15 and 16 of Felix's Thesis
    """
    #Hard coded parameters:
    k1 = -0.026
    k2 = 0.278
    c1 = 5.712
    c2 = 17.67
    beta = 0.24
    psi4375 = 0.73
    psi_min = 0.2
    psi_max = 0.84
    f_X = 1
    c_X = parameters.source.xray_normalisation #[erg/s yr/M_sun]
    e_r = 0.1 #radiative efficiency
    al_IMBH = 1.5 #spectral index for IMBH
    al_HMXB = 1.0 #spectral index for HMXB

    #calculate the obscured fraction of the AGN
    def obsc_fraction(L_X):
        return np.minimum(psi_max,np.maximum(psi_min,psi4375-beta*(np.log10(L_X)-43.75)))
    
    if type == "HMXB":
        #normalize for each redshift
        A = 1.0/trapezoid(nu_**(-al_HMXB),nu_,axis=0)
        #print(A*f_X*c_X/(h_eV_sec*nu_) * nu_**(-al_HMXB) * eV_per_erg)
        return A*f_X*c_X/(h_eV_sec*nu_) * nu_**(-al_HMXB) * eV_per_erg #[photons/Hz/s /(yr/Msun)]
    elif type == "IMBH":
        L_bol = e_r*Mdot_BH*3e8**2
        K_X = np.full_like(L_bol,1.0)
        mask = L_bol > 0.0
        K_X[mask] = c1*(L_bol[mask]/1e10)**k1 + c2*(L_bol[mask]/1e10)**k2
        L_X = L_bol/K_X
        
        psi = obsc_fraction(L_X)
        #normalize for each redshift
        A = 1.0/trapezoid(nu_**(-al_IMBH),nu_,axis=0)
        #print((e_r*3e10**2)/(K_X * h_eV_sec/eV_per_erg * nu_) * (1-psi) * A * nu_**(-al_IMBH) * 6.3e25)
        return (e_r*3e10**2)/(K_X * h_eV_sec/eV_per_erg * nu_) * (1-psi) * A * nu_**(-al_IMBH) * 6.3e25
