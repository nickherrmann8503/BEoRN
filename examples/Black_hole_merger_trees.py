import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from time import time

import toolscosmo
from toolscosmo import _merger_trees,merger_trees

#BEORN
from beorn.cosmo import Hubble #Hubble parameter [yr-1]
import beorn.structs.parameters
from beorn.astro import f_star_Halo
from beorn.constants import h_eV_sec, eV_per_erg,h__
from beorn.precomputation.massaccretion import mass_accretion,mass_accretion_derivative
from beorn.precomputation.blackholes import black_hole_mass_iterative,dN_dzdxi
from random import random

h=0.68

#PARAMETERS
parameters = beorn.structs.Parameters()

#cosmological
parameters.cosmology.Om  = 0.31
parameters.cosmology.Ob  = 0.045
parameters.cosmology.Ol  = 0.68
parameters.cosmology.h0  = 0.68

#Simulation parameters
parameters.source.f_st = 0.05
parameters.source.halo_mass_min = 1e4
parameters.source.g1 = 0.49
parameters.source.g2 = -0.61
parameters.source.g3 = 1.4
parameters.source.g4 = 0
parameters.source.Mp = 2.5e11/h
parameters.source.Mt = 1e8

#BH parameters
default_p = 1.0
default_xi = 0.1
default_f0 = 0.1 
default_Mseed = 100

parameters.source.IMBH_factor = 1.0
parameters.source.HMXB_factor = 1.0
parameters.source.p_merge = default_p
parameters.source.xi = default_xi
parameters.source.f0 = default_f0
parameters.source.M_seed = default_Mseed


param = toolscosmo.par('lcdm')
param.code.zmin = 5
param.code.kmax = 50 #500
param.code.Nrbin = 300
param.cosmo.Om = 0.25
param.cosmo.Or = 0.0
param.cosmo.Ob = 0.045
param.cosmo.h0 = 0.73
param.cosmo.ns = 1.0
param.cosmo.As = 2.089e-09 #None #
param.cosmo.s8 = 0.9
param.cosmo.solver = 'toolscosmo' #'astropy'
param.cosmo.weights_only = False
param.mf.dc = 1.686
param.file.ps = 'CLASSemu' #'CLASS' #'CAMB' #
param.file.ps = toolscosmo.calc_Plin(param)

"""fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(zn,Mh[0,0,:],label="BEoRN")
Mtree_sim = merger_trees.MergerTree(Mh_0[0],zn[-1],zn[0],1e6)
Mtree_sim.run()
Mtree_sim.plot_merger_history(ax=ax)
plt.show()"""

A = 2*1e-8

#Cosmological functions
def dtdz(z,parameters):
    return 1/((1+z)*Hubble(z,parameters))
def dzdt(z,parameters):
    return (1+z)*Hubble(z,parameters)

def virial_temperature(Mh,z,parameters):

    return 4.8e-3 * (Mh)**(2/3) * (1+z) * (parameters.cosmology.Om/0.3)**(1/3) * (0.59/0.59)

def cosmic_time(z, parameters):
    """Age of the universe at redshift z in yr (flat LCDM, no radiation). Works on arrays."""
    Om = parameters.cosmology.Om
    OL = parameters.cosmology.Ol
    H0_yr = 100.0*parameters.cosmology.h0 / 977.8e9          # km/s/Mpc -> 1/yr
    return 2 / (3 * H0_yr * np.sqrt(OL)) * np.arcsinh(np.sqrt(OL / Om) * (1 + z)**-1.5)


def get_merger_trees(descendant_halo_masses,redshifts):
    #redshifts in descending order
    trees = []
    for mass in descendant_halo_masses:
        Mtree_sim = merger_trees.MergerTree(mass,redshifts[-1],redshifts[0],mass*1e-7)
        #Mtree_sim = _merger_trees.ParkinsonColeHelly2008(param,mass,redshifts[-1],redshifts[0],mass*1e-6)
        Mtree_sim.run(verbose=False)
        trees.append(Mtree_sim)
    return np.array(trees)

def plot_trees(trees,descendant_halo_masses,redshifts):
    fig,ax = plt.subplots(figsize=(10, 7))
    Mh,Mhdot = mass_accretion(parameters,zn,descendant_halo_masses,np.array([0.5]))

    for i in range(len(descendant_halo_masses)):
        ax.plot(trees[i].main_branch_history['redshift'], trees[i].main_branch_history['mass_msun'], lw=2.5, label=f'Main Progenitor ({descendant_halo_masses[i]:.2e})')
        #ax.plot(trees[i].z_tree, (trees[i].M_tree), lw=2.5, label=f'Main Progenitor ({descendant_halo_masses[i]:.2e})')
        
        ax.plot(redshifts, Mh[i,0,:], lw=2.5, ls='--', label=f'BEoRN ({descendant_halo_masses[i]:.2e})')
        ax.set_yscale('log')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Halo Mass [M$_\\odot$]')
    ax.legend()
    plt.show()

def black_hole_evolution(tree,f0):
    #Compute the black hole mass evolution along the merger tree
    """z_tree = tree.main_branch_history['redshift']
    Mh_tree = tree.main_branch_history['mass_msun']"""
    z_tree = tree[0]
    Mh_tree = tree[1]
    Mbh_tree = np.full(len(z_tree), 0.0)
    print(z_tree)

    #go from high redshift to low redshift and compute the black hole mass at each bin
    for i in range(len(z_tree)-1,-1,-1):
        
        if i < len(z_tree) - 1 and Mbh_tree[i+1] > 0.0:
            Mbh_tree[i] =  Mbh_tree[i+1]*np.exp(A * f0 * (cosmic_time(z_tree[i], parameters)-cosmic_time(z_tree[i+1], parameters)))
        elif Mbh_tree[i] == 0.0 and Mh_tree[i] > 1e5 and 13 < z_tree[i] < 25:
            Mbh_tree[i] = 100.0 #dayal 150 and no later than z=13
    #print(Mbh_tree)
    return Mbh_tree

def compute_merger_tree(descendant_halo_mass,z_max,z0):
    print(z0)
    Mtree_sim = merger_trees.MergerTree(descendant_halo_mass,z0,z_max,np.max([descendant_halo_mass*1e-5,5000.0]))
    Mtree_sim.run(verbose=False)
    z_tree = Mtree_sim.main_branch_history['redshift']
    Mh_tree = Mtree_sim.main_branch_history['mass_msun']
    z_sub = Mtree_sim.subhalo_history['redshift']
    #print("z_sub",z_sub)
    M_sub = Mtree_sim.subhalo_history['mass_msun']
    whole_history = [z_tree, Mh_tree]
    for i in range(len(z_sub)):
        if M_sub[i] > 1e5:
            subhist = compute_merger_tree(M_sub[i],z_max,z_sub[i])
            whole_history.append(subhist)
        else:
            whole_history.append([z_sub[i], M_sub[i], None])
    return whole_history

def compute_merger_tree_and_BH(descendant_halo_mass,z_max,z0,f0,p,xi):
     #where f0 is eddington fraction and p is the merger probability

    Mtree_sim = merger_trees.MergerTree(descendant_halo_mass,z0,z_max,np.max([descendant_halo_mass*1e-5,1000000.0]))
    Mtree_sim.run(verbose=False)
    z_tree = np.array(Mtree_sim.main_branch_history['redshift'].tolist())
    Mh_tree = np.array(Mtree_sim.main_branch_history['mass_msun'].tolist()) 
    z_sub = np.array(Mtree_sim.subhalo_history['redshift'].tolist())
    M_sub = np.array(Mtree_sim.subhalo_history['mass_msun'].tolist())
    whole_history = [z_tree, Mh_tree]
   
    #BH main branch history
    Mbh_tree = np.full(len(z_tree), 0.0)
    BH_history = [z_tree, Mbh_tree]

    #get the mergers indexes 
    nearest_long = np.abs(z_sub[:, None] - z_tree[None, :]).argmin(axis=1)


    #go through the history of the halo from high redshift to low redshift
    for i in range(len(z_tree)-1,-1,-1):

        #either seed a BH or grow the BH from the previous redshift bin
        if i < len(z_tree) - 1 and Mbh_tree[i+1] > 0.0:
            Mbh_tree[i] =  Mbh_tree[i+1]*np.exp(A * f0 * (cosmic_time(z_tree[i], parameters)-cosmic_time(z_tree[i+1], parameters)))
        elif Mbh_tree[i] == 0.0 and Mh_tree[i] > 1e5 and 13 < z_tree[i]<25:
            Mbh_tree[i] = 100.0 #dayal 150 and no later than z=13
        
        #for a massive enough subhalo compute the history
        short_idx = np.where(nearest_long == i)[0]

        for j in short_idx:
            if M_sub[j] > 1e5:
                subhist = compute_merger_tree_and_BH(M_sub[j],z_max,z_sub[j],f0,p,xi)
                whole_history.append(subhist[0])
                #probability to merge
                if M_sub[j]/Mh_tree[i] > xi and np.random.random() < p: 
                    BH_history.append(subhist[1])
                    
                    Mbh_tree[i] += subhist[1][1][0]
            else:
                #no BH because lower mass than 10^5
                whole_history.append([z_sub[j], M_sub[j], None])
    BH_history[1] = Mbh_tree
    #print(Mbh_tree)
    #print("z_sub",z_sub)
    return whole_history, BH_history

def Black_hole_history(history,f0,p):
    #calculates the black hole mass history along the merger tree
    M_BH = np.zeros_like(history[0]) #z_tree

    return M_BH



def plot_merger_tree(whole_history,ax, main_progenitor):
    if main_progenitor:
        m = whole_history[1] >0.0
        ax.plot(whole_history[0][m], whole_history[1][m], lw=1.5, label=f'Main Progenitor',color='blue')
    else:
        m = whole_history[1] >0.0
        ax.plot(whole_history[0][m], whole_history[1][m], lw=1.5,ls="--")
    for i in range(2,len(whole_history)):
        if whole_history[i] is not None: #and random()< 0.1:
            plot_merger_tree(whole_history[i],ax, main_progenitor=False)
"""Mh_0 = np.logspace(9, 13, 1)
trees = get_merger_trees(Mh_0,zn)"""
"""Mtree_sim = _merger_trees.ParkinsonColeHelly2008(param,M_res= 1e6)
Mtree_sim.run(Mh_0[-1],zn[-1],zn[0])
trees = [Mtree_sim]"""

"""f0s = [0.0,0.1,0.3,0.5,1.0]
for f0 in f0s:
    Mbh = black_hole_evolution(trees[0],f0)
    plt.plot(trees[0].main_branch_history['redshift'], Mbh, label=f"f0={f0}")

plt.legend()
plt.yscale("log")
plt.show()
zn = np.linspace(25,5,41)
BH = np.zeros_like(zn)
BH[0] = 100.0
for i in range(1,len(zn)):  
    BH[i] = BH[i-1]*np.exp(A*1.0*(cosmic_time(zn[i], parameters)-cosmic_time(zn[i-1], parameters)))#dtdz((zn[i]+zn[i-1])/2,parameters)*(zn[i-1]-zn[i])
plt.plot(zn,BH,label="Analytical")
plt.yscale("log")
plt.show()

plot_trees(trees, Mh_0, zn)"""


zn = np.linspace(25,5,41)
target_halo_mass = 1e11
f0 = 0.2
p= 0.5
xi = 0.01
start_time = time()
hist = compute_merger_tree_and_BH(target_halo_mass,25,zn[-1],f0,p,xi)
print("Time taken to compute merger tree and BH history:", (time() - start_time) / 60, "minutes")
fig,ax = plt.subplots(1,2,figsize=(15, 7))
ax[0].set_yscale('log')
ax[1].set_yscale('log')

plot_merger_tree(hist[0],ax[0], main_progenitor=True)


"""BH = black_hole_evolution(hist[0],0.1)
plt.plot(hist[1][0],BH)
plt.show()"""

plot_merger_tree(hist[1],ax[1], main_progenitor=True)
parameters.source.f0 = f0
parameters.source.p_merge = p
parameters.source.xi = xi
Mh_0s = np.logspace(9,13,21)
MBH,MBHdot,Mh_fakhouri = black_hole_mass_iterative(zn,Mh_0s,parameters)

ax[1].plot(zn,MBH[0,:],label="previous",color='red')
ax[0].legend()
ax[1].legend()
ax[0].set_xlabel('Redshift')
ax[1].set_xlabel('Redshift')
ax[0].set_ylabel('[M$_\\odot$]')
ax[1].set_ylabel('[M$_\\odot$]')
ax[0].set_title('Halo Merger Tree')
ax[1].set_title('Black Hole Merger Tree')
plt.show()


plt.plot(hist[1][0],hist[1][1],label="BH mass")
plt.plot(hist[0][0],hist[0][1],label="Halo mass")

plt.yscale("log")
plt.ylabel("Mass [M$_\\odot$]")
plt.xlabel("Redshift")
Mh,Mhdot = mass_accretion(parameters,zn,np.array([target_halo_mass]),np.array([0.5]))
plt.plot(zn,Mh[0,0,:],label="BEoRN")

plt.plot(zn,Mh_fakhouri[0,:],label="fakhouri")
plt.legend()
plt.show()
