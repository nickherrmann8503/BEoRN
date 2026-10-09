#testing purposes
import time
#general imports
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
from scipy.optimize import fsolve,minimize,brentq, minimize_scalar,newton
from scipy.optimize.elementwise import find_root
from scipy.integrate import trapezoid, solve_ivp,quad,cumulative_trapezoid
from scipy.optimize import differential_evolution, basinhopping

#BEORN imports
from beorn.cosmo import Hubble #Hubble parameter [yr-1]
import beorn.structs.parameters
from beorn.astro import f_star_Halo
from beorn.constants import h_eV_sec, eV_per_erg,h__
from beorn.precomputation.massaccretion import mass_accretion,mass_accretion_derivative
import logging
#Astropy imports
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM,z_at_value

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FILE_ROOT = Path(".")
CACHE_ROOT = FILE_ROOT / "cache"
OUTPUT_ROOT = FILE_ROOT / "output"

h = 0.673
z0 = 5 #reference redshift
#M0 = 1e11 #Solar masses, final descendant mass
Mseed = 100

#Physical parameters

Ob = 0.045
Om = 0.315
fstar_0 = 0.05
Mp = 2.5e11/h
al = 0.785 #alpha parameter
Mt = 1e8 #solar masses
#f = 0.1 #Eddington ratio
#A = (1-0.1) * (4*np.pi*6.67430e-11 * 1.6726219e-27)/(0.1*3e8 * 6.652458734e-29)/(365.25*24*3600) #Eddington accretion rate constant
A = 2*1e-8 #[/yr]
#print(0.1*0.9*4*np.pi*6.67*1e-11*1.67*1e-27/(0.1*3e8*6.652*1e-29)*365.25*24*3600)
eta = 1e-6

Nint = 321
Nz = 100
cosmo = FlatLambdaCDM(H0=67.3, Om0=Om)

#Simulation parameters
parameters = beorn.structs.Parameters()
parameters.source.f_st = 0.05
parameters.source.halo_mass_min = 1e4
parameters.source.g1 = 0.49
parameters.source.g2 = -0.61
parameters.source.g3 = 1.4
parameters.source.g4 = 0
parameters.source.Mp = 2.5e11/h
parameters.source.Mt = 1e8

#Load artificial halo catalog for testing purposes
cache_handler = beorn.io.Handler(CACHE_ROOT)
loader = beorn.load_input_data.ArtificialHaloLoader(
    parameters,
    halo_count = 100,
)
Mh_0 = np.logspace(9,13,Nint) #evenly distributed halo masses for testing purposes
#print(Mh_0)
#rng = np.random.default_rng()
#Mh_0 = rng.choice(Mh_0,size=Nint,replace=True)


def dtdz(z):
    return 1/((1+z)*Hubble(z,parameters=parameters))
def dzdt(z):
    return (1+z)*Hubble(z,parameters=parameters)

def z_to_t(zs):
    if not isinstance(zs,np.ndarray):
        return -quad(dtdz, zs,np.inf)[0]
    return np.array([-quad(dtdz, z,np.inf)[0] for z in zs])

def t_to_z(ts):
    if not isinstance(ts,np.ndarray):
        return brentq(lambda z: z_to_t(z)-ts, -1, 3000)
    return np.array([brentq(lambda z: z_to_t(z)-t, -1, 3000) for t in ts])

def z_to_t(z):
    return cosmo.age(z).value*10**9

def t_to_z(t):
    return z_at_value(cosmo.age,t/10**9 *u.Gyr).value


def halo_mass(M0,z):
    zs = np.linspace(z0,z,Nint)
    return mass_accretion(parameters,zs,np.array([M0]),np.array([0.785]))[0].squeeze()[-1]
    #return M0 * np.exp(-al*(z-z0)) #in solar masses

def halo_mass_derivative(M0,z):
    zs = np.linspace(z0,z,Nint)
    return mass_accretion(parameters,zs,np.array([M0]),np.array([0.785]))[1].squeeze()[-1]
    #return halo_mass(M0,z) * al * ((1+z) * Hubble(z, parameters=parameters))

"""def z_seed(M_seed):
    #funct = lambda z_seed: star_formation_efficiency(M0, z_seed)*halo_mass(M0, z_seed) - Mseed #Seeding mechanism
    #funct = lambda z: f_star_Halo(parameters,mass_accretion(parameters,np.linspace(z0,z,100),np.array([M0]),np.array([0.785]))[0][-1])*mass_accretion(parameters,np.linspace(z0,z,100),np.array([M0]),np.array([0.785]))[0][-1]-Mseed
    funct = lambda z: stellar_seed_mass_eq(z,M_seed)
    z_seed_solution = fsolve(funct, 15.5)
    return z_seed_solution[0]"""

def stellar_seed_mass_eq(z,M0,M_seed):
    zs = np.linspace(z0,z,100)
    fval = (f_star_Halo(parameters,mass_accretion(parameters,zs,np.array([M0]),np.array([0.785]))[0]).squeeze())[-1]
    mval = mass_accretion(parameters,zs,np.array([M0]),np.array([0.785]))[0].squeeze()[-1]
    return Ob/Om*fval*mval-M_seed

def duty_cycle(M_BH,f0,gamma):
    #Compute the duty cycle of black hole accretion as a function of black hole mass.
    return f0
    return f0/(1+(1e6/M_BH)**gamma)

def quick_guess(M0):
    
    return (23-6)/np.log10(1e16/M0/2)*1.2+6
"""
def z_seed(M0,M_seed):
    funct = lambda z: stellar_seed_mass_eq(z,M0,M_seed)
    guess = (23-6)/np.log10(1e16/M0/3)*1.2+6#quick_guess(M0)
    z_seed_solution = minimize_scalar(funct,bounds=(5,30),method="bounded", options={"gtol": 1e-10, "maxiter": 10_000})
    return z_seed_solution.x"""

"""def z_seed(M0,M_seed):
    funct = lambda z: stellar_seed_mass_eq(z,M0,M_seed)
    #guess = (23-6)/np.log10(1e16/M0/3)*1.2+6#quick_guess(M0)
    z_seed_solution = find_root(funct,(5,25))#,args=[M0,M_seed])
    return z_seed_solution.x"""

def z_seed(M0,M_seed):
    funct = lambda z: stellar_seed_mass_eq(z,M0,M_seed)
    guess = (23-6)/np.log10(1e16/M0/3)*1.2+6#quick_guess(M0)
    z_seed_solution = brentq(funct, 4, 30)
    return z_seed_solution

def dN_dzdMp(Mh,Mp,z):
    #Compute the number of mergers per unit redshift and per unit progenitor mass
    alpha,beta,gamma,eta = 0.133,-1.995,0.263,0.0993
    A,xi_ = 0.0104,9.72e-3
    #M_h present in draft but not in cited paper
    return A*(Mh/1e12)**alpha*(Mp/Mh)**beta*np.exp((Mp/(Mh*xi_))**gamma)*(1+z)**eta

def p_merge(Mh,Mp,xi,p):
    #print(Mp/Mh)
    #return 1.0
    if Mp/Mh >xi:
        return p
    else:
        return 0.0

def display_bins(M_BH,Mh,zn,Mseed,f0,p,xi,title):
    x_edges = zn   # len = ncols + 1
    y_edges = Mh  # len = nrows + 1
    plt.pcolormesh(x_edges, y_edges, M_BH, cmap='viridis',norm=LogNorm())
    plt.colorbar(label=title)
    plt.xlabel("z")
    plt.ylabel(r"$M_h$")
    plt.yscale("log")
    plt.title(rf"$M_{{seed}}$={Mseed},f={f0},p={p},$\xi$={xi}in {len(zn)} z bins and {len(Mh)} Mh bins")
    plt.show()

def black_hole_mass_iterative(Mh0,Mseed,f0,p,xi):
    z_end = 5
    Mh = np.sort(Mh0)
    Mh = Mh[::-1] #sort in descending order
    """zs_seed = np.linspace(14,18,100)
    plt.plot(zs_seed,stellar_seed_mass_eq(zs_seed,1e12,100))
    plt.axvline(x=z_seed(1e12, 100), color='red', linestyle='--', linewidth=1)
    plt.title(r"Stellar seed mass equation: $M_{seed}=100M_dot$")
    plt.show()
    plt.plot(zs_seed,Mh)
    plt.show()
    print(z_seed(Mh[0], 100),z_seed(Mh[-1], 100),z_seed(Mh[50], 100))"""

    #solve seeding equation for every halo
    z_seeds = np.array([z_seed(m,Mseed) for m in Mh])
    z_0 = np.max(z_seeds)#first seeding 
    zn = np.linspace(z_0,z_end,Nz) #redshift bins (z_0,z_end=5)
    #print(zn)
    #print(Mh)
    Dz = np.abs(zn[1]-zn[0])
    Dm = Mh[0]-Mh[-1]
    #initiate array
    M_BH = np.zeros((len(Mh),len(zn)))
    M_mergers = np.zeros((len(Mh),len(zn)))
    M_accr = np.zeros((len(Mh),len(zn)))
    M_BH_dot = np.zeros((len(Mh),len(zn)))
    #seeding 
    for i,zseed in enumerate(z_seeds):
        index = np.digitize(zseed,zn)
        M_BH[i][index] = Mseed
    
    #show the array
    #display_bins(M_BH,Mh,zn)
    
    #loop through z and M bins
    index = [0,0,0]
    for n in range(1,len(zn)):
        for l in range(len(Mh)-1,-1,-1):

            M_BH[l][n] += M_BH[l][n-1]

            #eddington accretion from the previous z bin
            gas_accr = A*duty_cycle(M_BH[l][n-1],f0,0)*M_BH[l][n-1]*dtdz(zn[n])

            mergers = 0.0
            for i in range(len(Mh)-1,l,-1):
                
                mergers +=  np.abs((Mh[i]-Mh[i-1]))/Mh[l]*dN_dzdMp(Mh[l],Mh[i],zn[n]) * M_BH[i][n] * p_merge(Mh[l],Mh[i],xi,p)
            
            #add both contributions
            delta_M = gas_accr+ mergers

            #add them times the redshift step
            M_BH[l][n] += delta_M * Dz

            #save change
            M_mergers[l][n] += mergers * Dz
            M_accr[l][n] += gas_accr * Dz
            M_BH_dot[l][n] += delta_M * dzdt(zn[n])
            
            #display_bins(np.log10(M_BH),Mh,zn)
        #display_bins(np.log10(M_BH),Mh,zn,f0)
        #display_bins(DNDMDZ,Mh,zn)
        #display_bins(M_BH,Mh,zn,Mseed,f0,p,xi,r"$M_{BH}$")
        
    
    
    #display_bins(M_BH,Mh,zn,Mseed,f0,p,xi,r"$M_{BH}$")
    #display_bins(M_mergers,Mh,zn,Mseed,f0,p,xi,r"$M_{mergers}$")
    #display_bins(M_accr,Mh,zn,Mseed,f0,p,xi,r"$M_{eddington}$")
    return M_BH,M_BH_dot,Mh,z_0

def black_hole_mass(z,M0,f,eta,Mseed,zseed):
    
    ts = z_to_t(z)
    #calculate eddington exponential growth from z_seed to z
    edd = Mseed*np.exp(A*f*(ts-z_to_t(zseed)))

    ts= ts[::-1] #reverse order to integrate correctly
    def integrand(t):
        integr = np.exp(-A*f*t)*halo_mass_derivative(M0,t_to_z(t))#get M_h_dot(z)
        integr[np.where(t< z_to_t(zseed))] = 0 #shouldnt integrate before z_seed
        return integr
    #get the cumulative integral of the merger contribution
    mergers = eta*np.exp(A*f*ts)*cumulative_trapezoid(integrand(ts),ts,initial=0)
    mergers = mergers[::-1]#reverse back to match the order of z
    
    BH_mass = mergers+edd
    BH_mass[np.where(z>zseed)]= 0#exclude contributions before z_seed

    return BH_mass
    
def black_hole_mass_derivative(z,M0,f,eta,Mseed,zseed):

    #growth rate from mergers proportional to growth of halo mass
    mergers = eta*halo_mass_derivative(M0,z)

    #eddington growth rate proportional to current black hole mass
    edd = A*f*black_hole_mass(z,M0,f,eta,Mseed,zseed)
    
    BH_mdot = mergers+edd
    BH_mdot[np.where(z>zseed)]= 0

    return BH_mdot

def fstar(M):
    return 2*Ob/Om*0.05/((M/Mp)**(0.49)+(M/Mp)**(-0.61))*(1+(Mt/M)**(1.4))**(0)

def plot_evolution(BH_evolution,index):
    BH_mass,BH_mergers,BH_accr,halo_mass_bins,z0 = BH_evolution
    zs = np.linspace(z0,5,Nz)
    plt.plot(zs,BH_mass[index],label="BH mass")
    plt.plot(zs,BH_mergers[index],label= "mergers")
    plt.plot(zs,BH_accr[index],label = "accretion")
    plt.plot(zs,BH_mergers[index]+BH_accr[index],label = "accretion+mergers")
    plt.yscale('log')
    plt.xlim((8,30))
    plt.title(f"Halo descendant mass: {halo_mass_bins[index]:.2e}"+r"$M_\odot$")
    plt.xlabel("z")
    plt.ylabel(r"$M_\odot$")
    plt.legend()
    plt.show()

ms = np.logspace(7,13,20)
zs = np.linspace(5,20,Nint)

"""plt.plot(zs,Ob/Om*f_star_Halo(parameters,mass_accretion(parameters,zs,np.array([1e11/h]),np.array([0.785]))[0]).squeeze())
plt.plot(zs,fstar(mass_accretion(parameters,zs,np.array([1e11/h]),np.array([0.785]))[0].squeeze()))
plt.show()
plt.plot(ms,Ob/Om*f_star_Halo(parameters,ms/h))
plt.title(r"Star formation efficiency(Nick)")
plt.xlabel(r"Halo mass [$M_dot$]")
plt.ylabel(r"$f_tar$")
plt.xscale('log')
plt.yscale('log')
plt.show()
zseed = [z_seed(m, 100) for m in ms]
print(zseed)
print(f"Solution for z_seed: {zseed}")
plt.plot(ms,zseed)
plt.xscale('log')
plt.show()"""
"""
print(z_seed(1e8,100),z_seed(1e10,100),z_seed(1e12,100),z_seed(1e14,100))
for m in ms:
    plt.plot(zs,stellar_seed_mass_eq(zs,m,Mseed),label=f"M0={m:.2e}")
    plt.yscale('log')
    plt.axvline(x=z_seed(m, Mseed), color='red', linestyle='--', linewidth=1)
    print(quick_guess(m))
    plt.axvline(x=quick_guess(m), color='green', linestyle='-', linewidth=1)
    plt.title(f"{m:.2e}")
    plt.show()"""
"""Mh = np.logspace(8,14,Nint)
zn = np.linspace(5,20,Nint)
merged = np.array([[p_merge(mh,mp,0,1)]])
display_bins()"""

"""pmerging = np.zeros((Nint,Nint))
pratio = np.zeros((Nint,Nint))
for i in range(len(Mh_0)):
    for j in range(len(Mh_0)):
        pmerging[i][j] =p_merge(Mh_0[i],Mh_0[j],0.001,1.0)
        pratio[i][j] = np.log10(Mh_0[j]/Mh_0[i])
display_bins(pratio,np.log10(Mh_0),np.log10(Mh_0))
display_bins(pmerging,np.log10(Mh_0),np.log10(Mh_0))"""


def plot_f_dependence(evolution,index,fs):
    zs = np.linspace(evolution[0][4],5,Nz)
    for i in range(len(fs)):
        plt.plot(zs,evolution[i][0][index],label=f"f0 = {fs[i]}")
    plt.xlabel("z")
    plt.ylabel(r"$M_\odot$")
    plt.yscale("log")
    plt.title(r"$M_0=$"+f"{evolution[i][3][index]:.2e}"+r"$M_\odot/h$")
    plt.xlim((3,30))
    plt.legend()
    plt.show()

def plot_contributions(evolution,index,f):
    zs = np.linspace(evolution[0][4],5,Nz)
    plt.plot(zs,evolution[f][0][index],label=f"Total Mass")
    plt.plot(zs,evolution[f][1][index],label=f"mergers",color="red")
    plt.plot(zs,evolution[f][2][index],label=f"accretion")
    plt.xlabel("z")
    plt.ylabel(r"$M_\odot$")
    plt.yscale("log")
    plt.title(r"$M_0=$"+f"{evolution[f][3][index]:.2e}"+r"$M_\odot/h$")
    plt.xlim((3,30))
    plt.legend()
    plt.show()


def Parameter_comparison(Mh_0):
    start_time = time.time()
    fig, axes = plt.subplots(2, 4, figsize=(14, 10),layout="constrained")
    fig.suptitle(rf"Iterative BH evolution in a halo of descendant mass:$10^{{{int(np.log10(Mh_0[-1]))}}}M_\odot$")
    z_low,z_high = 5,27
    for ax_mass in axes[0]:
        
        ax_mass.set_ylabel(r"$M_{BH} [M_\odot/h]$")
        ax_mass.set_xlabel("z")
        ax_mass.set_xticks([5, 10, 15,  20,25])
        ax_mass.set_yscale("log")
        ax_mass.set_xlim((z_low,z_high))
        ax_mass.set_ylim((1e1,1e9))
        ax_mass.yaxis.set_major_locator(LogLocator(base=10, numticks=100))  # every decade
        ax_mass.grid(True, which="major", linestyle="--", alpha=0.5)

    for ax_dot in axes[1]:
            
        ax_dot.set_ylabel(r"$\dot{M}_{BH} [M_\odot/yr/h]$")
        ax_dot.set_xlabel("z")
        ax_dot.set_xticks([5,10,15,20,25])
        ax_dot.set_yscale("log")
        ax_dot.set_xlim((z_low,z_high))
        ax_dot.set_ylim((1e-7,1e-1))
        ax_dot.yaxis.set_major_locator(LogLocator(base=10, numticks=100))  # every decade
        ax_dot.grid(True, which="major", linestyle="--", alpha=0.5)

    fs = [0.0,0.1,0.2,0.3]
    Mseeds = [50,100,300,800]
    ps = [0.0,0.1,0.3,1.0]
    xis = [0.0,0.5,0.7,0.9]
    Evolution = [[0 for i in range(4)] for j in range(4)]
    for i in range(len(fs)):
        Evolution[0][i] = black_hole_mass_iterative(Mh_0,Mseeds[1],fs[i],ps[2],xis[1])
        Evolution[1][i] = black_hole_mass_iterative(Mh_0,Mseeds[i],fs[1],ps[2],xis[1])
        Evolution[2][i] = black_hole_mass_iterative(Mh_0,Mseeds[1],fs[1],ps[i],xis[1])
        Evolution[3][i] = black_hole_mass_iterative(Mh_0,Mseeds[1],fs[1],ps[2],xis[i])
    print(time.time()-start_time)
    
    #plot f dependence
    for f in range(len(fs)):
        zs = np.linspace(Evolution[0][f][3],5,Nz)
        axes[0][0].plot(zs,Evolution[0][f][0][0],label = f"f={fs[f]:.1f}")
    axes[0][0].legend()
    axes[0][0].set_title(r"$M_{seed}=100M_\odot,p=0.3,\xi= 0.5$")

    #plot Mseed dependence
    for m in range(len(Mseeds)):
        zs = np.linspace(Evolution[1][m][3],5,Nz)
        axes[0][1].plot(zs,Evolution[1][m][0][0],label = rf"M_seed={Mseeds[m]}$M_\odot$")
    axes[0][1].legend()
    axes[0][1].set_title(r"$f = 0.1,p=0.3,\xi= 0.5$")

    #plot p dependence
    for p in range(len(Mseeds)):
        zs = np.linspace(Evolution[2][p][3],5,Nz)
        axes[0][2].plot(zs,Evolution[2][p][0][0],label = f"p={ps[p]:.1f}")
    axes[0][2].legend()
    axes[0][2].set_title(r"$f = 0.1,M_{seed} = 100M_\odot,\xi = 0.5$")

    #plot xi dependence
    for xi in range(len(Mseeds)):
        zs = np.linspace(Evolution[3][xi][3],5,Nz)
        axes[0][3].plot(zs,Evolution[3][xi][0][0],label = f"xi={xis[xi]:.1f}")
    axes[0][3].legend()
    axes[0][3].set_title(r"$f = 0.1,M_{seed} = 100M_\odot,p= 0.3$")

    #M DOT
    #plot f dependence
    for f in range(len(fs)):
        zs = np.linspace(Evolution[0][f][3],5,Nz)
        axes[1][0].plot(zs,Evolution[0][f][1][0],label = f"f={fs[f]:.1f}")
    axes[1][0].legend()
    axes[1][0].set_title(r"$M_{seed}=100M_\odot,p=0.3,\xi= 0.5$")

    #plot Mseed dependence
    for m in range(len(Mseeds)):
        zs = np.linspace(Evolution[1][m][3],5,Nz)
        axes[1][1].plot(zs,Evolution[1][m][1][0],label = rf"M_seed={Mseeds[m]}$M_\odot$")
    axes[1][1].legend()
    axes[1][1].set_title(r"$f = 0.1,p=0.3,\xi= 0.5$")

    #plot p dependence
    for p in range(len(Mseeds)):
        zs = np.linspace(Evolution[2][p][3],5,Nz)
        axes[1][2].plot(zs,Evolution[2][p][1][0],label = f"p={ps[p]:.1f}")
    axes[1][2].legend()
    axes[1][2].set_title(r"$f = 0.1,M_{seed} = 100M_\odot,\xi = 0.5$")

    #plot xi dependence
    for xi in range(len(Mseeds)):
        zs = np.linspace(Evolution[3][xi][3],5,Nz)
        axes[1][3].plot(zs,Evolution[3][xi][1][0],label = f"xi={xis[xi]:.1f}")
    axes[1][3].legend()
    axes[1][3].set_title(r"$f = 0.1,M_{seed} = 100M_\odot,p= 0.3$")

    plt.show()

    #felix evolution plots
    fig, axes = plt.subplots(2, 5, figsize=(14, 10),layout="constrained")
    fig.suptitle(f"Iterative BH Evolution for different descendant halo masses"+r"$(M_{seed} = 100M_\odot,p=0.3,\xi=0.5)$")
    z_low,z_high = 5,27
    zs = np.linspace(Evolution[0][1][3],5,Nz)
    Mh_0 = Mh_0[::-1]
    for m in range(5):
        for f in range(4):
            
            axes[0][m].plot(zs,Evolution[0][f][0][m*(Nint//4)],label = f"f0 = {fs[f]}" )
            axes[0][m].set_title(rf"$M_h=10^{{{int(np.log10(Mh_0[m*(Nint//4)]))}}}M_\odot$")
            axes[1][m].plot(zs,Evolution[0][f][1][m*(Nint//4)],label = f"f0 = {fs[f]}")
            axes[0][m].set_yscale("log")
            axes[1][m].set_yscale("log")
            axes[0][m].set_xlabel("z")
            axes[0][m].set_ylabel(r"$M_{BH} [M_\odot/h]$")
            axes[1][m].set_xlabel("z")
            axes[1][m].set_ylabel(r"$\dot{M}_{BH} [M_\odot/yr/h]$")
            axes[0][m].set_ylim((1e1,1e8))
            axes[1][m].set_ylim(1e-9,1e-1)
    axes[0][0].legend()
    Mh_0 = Mh_0[::-1]
    plt.show()
    

"""zs = np.linspace(5,25)
M_h = 1e10
M_p = 1e8
z_ = 15
M_ps = np.logspace(8,10,100)
plt.plot(zs,dN_dzdMp(M_h,M_p,zs))
plt.yscale("log")
plt.show()
plt.plot(M_ps,dN_dzdMp(M_h,M_ps,z_))
plt.yscale("log")
plt.xscale("log")
plt.show()"""

#black_hole_mass_iterative(Mh_0,100,0.1,0.3,0.5)

Parameter_comparison(Mh_0)
"""plot_contributions(BH_evolution,0,1)
plot_f_dependence(BH_evolution,2,fs)"""



    



