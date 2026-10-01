import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from scipy.optimize import fsolve,minimize
from scipy.integrate import trapezoid, solve_ivp,quad,cumulative_trapezoid

from beorn.cosmo import Hubble #Hubble parameter [yr-1]
import beorn.structs.parameters
from beorn.astro import f_star_Halo
from beorn.precomputation.massaccretion import mass_accretion,mass_accretion_derivative
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM,z_at_value

h = 0.673
z0 = 5 #reference redshift
M0 = 1e11 #Solar masses, final descendant mass
Mseed = 100

#Physical parameters

Ob = 0.045
Om = 0.315
fstar_0 = 0.05
Mp = 2.5e11
al = 0.785 #alpha parameter
Mt = 1e8 #solar masses
#f = 0.1 #Eddington ratio
#A = (1-0.1) * (4*np.pi*6.67430e-11 * 1.6726219e-27)/(0.1*3e8 * 6.652458734e-29)/(365.25*24*3600) #Eddington accretion rate constant
A = 2*1e-8 #[/yr]
print(0.1*0.9*4*np.pi*6.67*1e-11*1.67*1e-27/(0.1*3e8*6.652*1e-29)*365.25*24*3600)
eta = 1e-6

Nint = 300
cosmo = FlatLambdaCDM(H0=67.3, Om0=Om)

parameters = beorn.structs.Parameters()
parameters.source.f_st = 0.05
parameters.source.halo_mass_min = 1e4
parameters.source.g1 = 0.49
parameters.source.g2 = -0.61
parameters.source.g3 = 1.4
parameters.source.g4 = -4
parameters.source.Mp = 1e10



def z_to_t(z):
    """Cosmic time (age of the Universe) at redshift z, in years."""
    #if not isinstance(z, np.ndarray):
    fooo = lambda z_: cosmo.age(z_).value*10**9
    return fooo(z)
        
    t = cumulative_trapezoid(1/(Hubble(z,parameters)*(1+z)),z)
    return -np.append(t,0)

def t_to_z(t):
    fooo = lambda t_:z_at_value(cosmo.age,t/10**9 *u.Gyr)
    return fooo(t).value


def halo_mass(M0,z):
    zs = np.linspace(z0,z,Nint)
    return mass_accretion(parameters,zs,np.array([M0]),np.array([0.785]))[0].squeeze()[-1]
    #return M0 * np.exp(-al*(z-z0)) #in solar masses

def halo_mass_derivative(M0,z):
    zs = np.linspace(z0,z,Nint)
    return mass_accretion(parameters,zs,np.array([M0]),np.array([0.785]))[1].squeeze()[-1]
    #return halo_mass(M0,z) * al * ((1+z) * Hubble(z, parameters=parameters))

"""def star_formation_efficiency(M0,z):
    Mh = halo_mass(M0,z)
    S = (1+(Mt/Mh)**gamma3)**gamma4
    return S * (2*Ob/Om) * fstar_0 /( (Mh/Mp)**gamma1 + (Mh/Mp)**gamma2 )

def gas_accretion_rate(M_BH,z,f):
    return f*A*M_BH"""

def merger_rate(z,eta):
    return eta* halo_mass_derivative(M0,z)

def foo(z,M_seed):
    zs = np.linspace(z0,z,100)
    fval = (f_star_Halo(parameters,mass_accretion(parameters,zs,np.array([M0]),np.array([0.785]))[0]).squeeze())[-1]
    mval = mass_accretion(parameters,zs,np.array([M0]),np.array([0.785]))[0].squeeze()[-1]
    return np.abs(fval*mval-M_seed)

"""def z_seed(M_seed):
    #funct = lambda z_seed: star_formation_efficiency(M0, z_seed)*halo_mass(M0, z_seed) - Mseed #Seeding mechanism
    #funct = lambda z: f_star_Halo(parameters,mass_accretion(parameters,np.linspace(z0,z,100),np.array([M0]),np.array([0.785]))[0][-1])*mass_accretion(parameters,np.linspace(z0,z,100),np.array([M0]),np.array([0.785]))[0][-1]-Mseed
    funct = lambda z: foo(z,M_seed)
    z_seed_solution = fsolve(funct, 15.5)
    return z_seed_solution[0]"""

def z_seed(M_seed):
    #funct = lambda z_seed: star_formation_efficiency(M0, z_seed)*halo_mass(M0, z_seed) - Mseed #Seeding mechanism
    #funct = lambda z: f_star_Halo(parameters,mass_accretion(parameters,np.linspace(z0,z,100),np.array([M0]),np.array([0.785]))[0][-1])*mass_accretion(parameters,np.linspace(z0,z,100),np.array([M0]),np.array([0.785]))[0][-1]-Mseed
    funct = lambda z: foo(z,M_seed)
    z_seed_solution = minimize(funct,16,bounds=[(14,18)])
    
    return z_seed_solution.x[0]

def black_hole_mass(z,M0,f,eta,Mseed,zseed):
    
    
    ts = z_to_t(z)
    #print(ts-z_to_t(zseed))
    edd = Mseed*np.exp(A*f*(ts-z_to_t(zseed)))#edd is correct checked for mergers= 0
    ts= ts[::-1] #reverse order to integrate correctly
    def integrand(t):
        integr = np.exp(-A*f*t)*halo_mass_derivative(M0,t_to_z(t))
        integr[np.where(t< z_to_t(zseed))] = 0
        return integr

    #problem was integrates it for all times ie int t_seed till t(z=7) and multiplies this at each
    #mergers = -eta*np.exp(A*f*ts)*trapezoid(integrand(ts),ts)
    mergers = -eta*np.exp(A*f*ts)*cumulative_trapezoid(integrand(ts),ts,initial=0)
    mergers = -mergers[::-1]
    #print("eddington:",edd)
    #print("mergers:",mergers)
    BH_mass = mergers+edd
    BH_mass[np.where(z>zseed)]= 0
    return BH_mass

"""def black_hole_mass(z,M0,f,eta,Mseed):

    def rhs(y,z):
        mergers = eta*halo_mass_derivative(M0,z)
        edd = f * A * y
        return (mergers+edd)*dt_dz(z)
    return solve_ivp(rhs,(z_seed(Mseed),z[-1]),[Mseed],method="LSODA",
                     rtol=1e-8, atol=1e-6, dense_output=True).sol(z).squeeze()
"""
    

def black_hole_mass_derivative(z,M0,f,eta,Mseed,zseed):
    mergers = eta*halo_mass_derivative(M0,z)
    edd = A*f*black_hole_mass(z,M0,f,eta,Mseed,zseed)
    
    BH_mdot = mergers+edd
    BH_mdot[np.where(z>zseed)]= 0
    return BH_mdot

"""print(f"Solution for z_seed: {z_seed()}")
#print("BH mass:",black_hole_mass(6,M0))
#print("BH mass accretion:" ,black_hole_mass_derivative(15,M0))
"""
zseed = round(z_seed(Mseed),8)
print(f"Solution for z_seed: {zseed}")
"""zs = np.concatenate([np.linspace(7,zseed,Nint),np.linspace(zseed+(20-zseed)/Nint,20,Nint)])
plt.plot(zs,black_hole_mass(zs,M0,0.1,1e-6,Mseed,zseed))
plt.show()
Ms = np.linspace(50,1000,100)
zseeds = np.zeros(len(Ms))
for i,M in enumerate(Ms):
    zseeds[i] = z_seed(M)

plt.plot(zs,foo(zs,500))
plt.show()
plt.plot(Ms,zseeds)
plt.show()"""
zs = np.concatenate([np.linspace(z0,zseed,Nint),np.linspace(zseed+(20-zseed)/Nint,20,Nint)])
#print(eta*halo_mass_derivative(M0,zs))
plt.plot(zs,eta*halo_mass_derivative(M0,zs),label="mergers")
plt.plot(zs,black_hole_mass_derivative(zs,M0,0.1,1e-6,100,zseed),label="tot")
plt.plot(zs,A*0.1*black_hole_mass(zs,M0,0.1,eta,Mseed,zseed),label="eddington")
plt.ylabel(r"$\dot{M}_{BH}[M_\odot/yr]$")
plt.xlabel("z")
plt.title(r"$f=0.1,\eta=1e-6,M_{seed}=100M_\odot$")
plt.legend()
plt.show()



#plt.plot(zs,z_to_t(zs))
#plt.show()
"""plt.plot(zs,black_hole_mass(zs,M0,0.1,1e-6,Mseed))
plt.legend()
plt.yscale('log')
plt.show()"""
z_low = 6
z_high = 20
plotb = False
if plotb:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4),layout="constrained")

    for ax in axes:
        ax.set_ylabel(r"$\dot{M}_{BH} [M_\odot/yr/h]$")
        ax.set_xlabel("z")
        ax.set_xticks([7.5, 10, 12.5, 15, 17.5, 20])
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=100))  # every decade
        ax.grid(True, which="major", linestyle="--", alpha=0.5)
                             
    zs = np.concatenate([np.linspace(z0,z_seed(100),Nint),np.linspace(z_seed(100)+(20-z_seed(100))/Nint,20,Nint)])
    for f in [0.01,0.1,0.3,0.5]:
        axes[0].plot(zs,black_hole_mass_derivative(zs,M0,f,1e-6,100,zseed),label = f"f={f:.2f}")
    axes[0].legend()
    axes[0].set_xlim((z_low,z_high))
    axes[0].set_title(r"$\eta= 1.0e-06,M_{seed}=100M_\odot$")
    
    for eta in [1e-7,1e-6,1e-5,1e-4]:
        axes[1].plot(zs,black_hole_mass_derivative(zs,M0,0.1,eta,100,zseed),label = r"$\eta=$"+f"{eta:.1e}")
    axes[1].legend()
    axes[1].set_xlim((z_low,z_high))
    axes[1].set_title(r"$f= 0.1,M_{seed}=100M_\odot$")
    
    for Mseed in [50,100,300,800]:
        zs = np.concatenate([np.linspace(z0,z_seed(Mseed),Nint),np.linspace(z_seed(Mseed)+(20-z_seed(Mseed))/Nint,20,Nint)])
        axes[2].plot(zs,black_hole_mass_derivative(zs,M0,0.1,1e-6,Mseed,z_seed(Mseed)),label = r"$M_{seed}$="+f"{Mseed:.2f}")
    axes[2].legend()
    axes[2].set_xlim((z_low,z_high))
    axes[2].set_title(r"$f=0.1,\eta= 1.0e-06$")

    plt.show()
