#File to compare global signals for different BH parameters
# %% [markdown]
# # BEoRN: Quick-start with Artificial Halos
# 
# This notebook is a self-contained introduction to BEoRN that requires **no external simulation data**. Instead of reading real cosmological simulations, it generates a simple set of artificial dark matter halos and evolves them with an idealised exponential mass-growth history. This makes it a fast and dependency-free way to explore the BEoRN workflow.
# 
# The notebook walks through the complete pipeline step by step:
# 1. Set astrophysical and simulation parameters
# 2. Create an artificial halo catalog
# 3. Pre-compute radial radiation profiles
# 4. Paint 3D signal maps across a range of redshifts
# 5. Visualise the resulting 21-cm brightness temperature maps and global history
# 
# > **Note:** The halo positions and masses used here are illustrative, not physically representative. For a realistic simulation use the `21cmfast_halos` notebook instead.
# 
# **Output:** Intermediate cached results are written to `./cache/` and final grids to `./output/`. Both are created automatically.

# %%
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
from pathlib import Path
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from beorn.precomputation.blackholes import black_hole_mass_iterative,stellar_seed_mass_eq
from beorn.mass_function import HaloMassFunction
from beorn.precomputation.massaccretion import mass_accretion
from beorn.astro import f_star_Halo
from scipy.interpolate import interp1d
from time import time

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
FILE_ROOT = Path(".")
CACHE_ROOT = FILE_ROOT / "cache"
OUTPUT_ROOT = FILE_ROOT / "output"

# %%
import beorn

import matplotlib as mpl
mpl.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.titlesize': 14,
})


# %% [markdown]
# ## 1. Set simulation and astrophysical parameters
# 
# Here we define the cosmology, the simulation volume, and the astrophysical source model that describes how the first galaxies emit radiation.
# 
# - **Simulation:** box size (`Lbox`) and grid resolution (`Ncell`) set the spatial dynamic range. The halo mass bins define the range over which radiation profiles are pre-computed.
# - **Source model:** the star formation efficiency (`f_st`), ionizing photon output (`Nion`), X-ray normalisation, and escape fraction control how strongly halos heat and ionize their surroundings. These are the parameters to explore when studying different reionization scenarios.
# 
# > For this illustrative run the source parameters are chosen to produce a clear visual signal, not to match observational constraints.

# %%
parameters = beorn.structs.Parameters()

## Simulation parameters, i.e. the allowed "dynamic range" of the simulation
parameters.simulation.cores = 4 

parameters.solver.halo_mass_bin_min = 1e8
parameters.solver.halo_mass_bin_max = 1e15
parameters.solver.halo_mass_nbin = 201
parameters.solver.halo_mass_accretion_alpha = np.array([0.785, 0.795])
parameters.simulation.Lbox = 100
parameters.simulation.Ncell = 128

# Source age: how far back to integrate the emission history.
# z_source_start: no sources before this redshift (replaces hardcoded z=35).
# t_source_age: cap the lookback to this many Myr (None = use z_source_start only).
parameters.solver.z_source_start = 35.0  # default
parameters.source.t_source_age   = None  # default (no age cap)

## Source parameters
# emission
parameters.source.Nion = 5000 * 3
parameters.source.energy_cutoff_min_xray = 500
parameters.source.energy_cutoff_max_xray = 2000
parameters.source.energy_min_sed_xray = 500
parameters.source.energy_max_sed_xray = 2000
parameters.source.alS_xray = 1.5
parameters.source.xray_normalisation = 3.4e40*3

# fesc
parameters.source.f0_esc = 0.2
parameters.source.pl_esc = 0

# fstar
parameters.source.f_st = 0.05
parameters.source.g1 = 0.49
parameters.source.g2 = -0.61
parameters.source.g3 = 1.4
parameters.source.g4 = -4
parameters.source.Mp = 1e10#2.5e11/0.68
parameters.source.Mt = 1e8

# Lyman-alpha
parameters.source.n_lyman_alpha_photons = 9690

# Minimum star forming halo
parameters.source.halo_mass_min = 1e4

#Black holes
default_p = 1.0
default_xi = 0.3
default_f0 = 0.1 
default_Mseed = 100

parameters.source.IMBH_factor = 1.0
parameters.source.HMXB_factor = 1.0
parameters.source.p_merge = default_p
parameters.source.xi = default_xi
parameters.source.f0 = default_f0
parameters.source.M_seed = default_Mseed



# ## 2. Create the halo catalog
# 
# The `ArtificialHaloLoader` generates a set of halos placed randomly in the box, whose masses grow exponentially with time following a simple analytical prescription. This removes the need for a full N-body or semi-numerical simulation while still producing a realistic-looking halo distribution for demonstration purposes.
# 
# We also initialise the radiation profile solver here. It will use the same redshift grid as the halo loader.

cache_handler = beorn.io.Handler(CACHE_ROOT)
loader = beorn.load_input_data.ArtificialHaloLoader(
    parameters,
    halo_count = 100,
)
recompute = False
#print(loader.redshifts)

def compute_no_BH_z_quantities(parameters,redshifts):
    print("computing no BH signal")
    parameters.source.IMBH_factor = 0.0
    parameters.source.HMXB_factor = 0.0
    parameters.source.p_merge = default_p
    parameters.source.xi = default_xi
    parameters.source.f0 = default_f0
    parameters.source.M_seed = default_Mseed

    output_handler = beorn.io.Handler(OUTPUT_ROOT)

    solver = beorn.precomputation.RadiationProfileSolver(parameters, redshifts)
    profiles = solver.get_or_compute_profiles(cache_handler,force_recompute=False)
    
    p = beorn.painting.PaintingCoordinator(
        parameters,
        loader=loader,
        output_handler=output_handler,
        force_recompute=False,
    )

    No_BH_z_quantities = p.paint_full(profiles)

    parameters.source.IMBH_factor = 1.0
    parameters.source.HMXB_factor = 0.0
    return No_BH_z_quantities


def main(BH_parameters):




    BH_profiles = {
    "f0": [],
    "M_seed": [],
    "p": [],
    "xi": []
    }

    output_handler = beorn.io.Handler(OUTPUT_ROOT)
    
    BH_multi_z_quantities = {
        "f0": [],
        "M_seed": [],
        "p": [],
        "xi": []
    }

    #compute profiles for different parameters
    for f in BH_parameters['f0']:
        print(f"computing profile for f0:{f}")
        parameters.source.f0 = f
        solver = beorn.precomputation.RadiationProfileSolver(parameters, loader.redshifts)
        profile = solver.get_or_compute_profiles(cache_handler,force_recompute=recompute)
        BH_profiles['f0'].append(profile)

        print(f"painting f0={f}")
        p = beorn.painting.PaintingCoordinator(
            parameters,
            loader=loader,
            output_handler=output_handler,
            force_recompute=recompute,
        )
        BH_multi_z_quantities["f0"].append(p.paint_full(profile))
    parameters.source.f0 = default_f0

    for M in BH_parameters['M_seed']:
        print(f"computing profile for M_seed:{M}")
        parameters.source.M_seed = M
        solver = beorn.precomputation.RadiationProfileSolver(parameters, loader.redshifts)
        profile = solver.get_or_compute_profiles(cache_handler,force_recompute=recompute)
        BH_profiles['M_seed'].append(profile)

        print(f"painting M_seed={M}")
        p = beorn.painting.PaintingCoordinator(
            parameters,
            loader=loader,
            output_handler=output_handler,
            force_recompute=recompute,
        )
        BH_multi_z_quantities["M_seed"].append(p.paint_full(profile))
    parameters.source.M_seed = default_Mseed

    for p_ in BH_parameters['p']:
        print(f"computing profile for p:{p_}")
        parameters.source.p_merge = p_
        solver = beorn.precomputation.RadiationProfileSolver(parameters, loader.redshifts)
        profile = solver.get_or_compute_profiles(cache_handler,force_recompute=recompute)
        BH_profiles['p'].append(profile)

        print(f"painting p={p_}")
        p = beorn.painting.PaintingCoordinator(
            parameters,
            loader=loader,
            output_handler=output_handler,
            force_recompute=recompute,
        )
        BH_multi_z_quantities["p"].append(p.paint_full(profile))
    parameters.source.p_merge = default_p

    for xi in BH_parameters['xi']:
        print(f"computing profile for xi:{xi}")
        parameters.source.xi = xi
        solver = beorn.precomputation.RadiationProfileSolver(parameters, loader.redshifts)
        profile = solver.get_or_compute_profiles(cache_handler,force_recompute=recompute)
        BH_profiles['xi'].append(profile)

        print(f"painting xi={xi}")
        p = beorn.painting.PaintingCoordinator(
            parameters,
            loader=loader,
            output_handler=output_handler,
            force_recompute=recompute,
        )
        BH_multi_z_quantities["xi"].append(p.paint_full(profile))
    parameters.source.xi = default_xi

    # %% [markdown]
    # ## 4. Inspect the halo distribution
    # 
    # As a quick sanity check, we project the halo positions onto the 2D plane to confirm that they are distributed throughout the simulation box. Each pixel shows the number of halos along the line of sight.


    """
    mesh = halo_catalog.to_mesh()
    stacked_mesh = np.sum(mesh, axis=0)

    plt.figure()
    plt.imshow(stacked_mesh, origin='lower', cmap='viridis')
    plt.colorbar(label='Halo count')
    plt.xlabel('[cells]')
    plt.ylabel('[cells]')
    plt.title(f'Projected halo distribution at z={z_target}')
    plt.show()"""

    # %% [markdown]
    # ## 5. Paint the 3D signal maps
    # 
    # BEoRN places the pre-computed radiation profiles around each halo and accumulates their contributions on the 3D grid. The result is a time-series of coeval cubes, each containing the ionization fraction, gas temperature, and Lyman-alpha coupling field at a given redshift. From these, the 21-cm brightness temperature $\delta T_b$ is computed.

    # %%
    
    """print("painting f0 dependent profiles")
    for i,profile in enumerate(BH_profiles['f0']):
        parameters.source.f0 = BH_params['f0'][i]
        p = beorn.painting.PaintingCoordinator(
            parameters,
            loader=loader,
            output_handler=output_handler,
            force_recompute=recompute,
        )
        BH_multi_z_quantities["f0"].append(p.paint_full(profile))
    parameters.source.f0 = default_f0

    print("painting M_seed dependent profiles")
    for i,profile in enumerate(BH_profiles['M_seed']):
        parameters.source.M_seed = BH_params['M_seed'][i]
        p = beorn.painting.PaintingCoordinator(
            parameters,
            loader=loader,
            output_handler=output_handler,
            force_recompute=recompute,
        )
        BH_multi_z_quantities["M_seed"].append(p.paint_full(profile))
    parameters.source.M_seed = default_Mseed

    print("painting p dependent profiles")
    for i,profile in enumerate(BH_profiles['p']):
        parameters.source.p_merge = BH_params['p'][i]
        p = beorn.painting.PaintingCoordinator(
            parameters,
            loader=loader,
            output_handler=output_handler,
            force_recompute=recompute,
        )
        BH_multi_z_quantities["p"].append(p.paint_full(profile))
    parameters.source.p_merge = default_p

    print("painting xi dependent profiles")
    for i,profile in enumerate(BH_profiles['xi']):
        parameters.source.xi = BH_params['xi'][i]
        p = beorn.painting.PaintingCoordinator(
            parameters,
            loader=loader,
            output_handler=output_handler,
            force_recompute=recompute,
        )
        BH_multi_z_quantities["xi"].append(p.paint_full(profile))
    parameters.source.xi = default_xi"""

    return BH_multi_z_quantities

def plot_dTb_Powerspectrum_parameter(BH_multi_z,BH_parameters,param = "f0"):

    #compute or get no_BH spectrum
    default = compute_no_BH_z_quantities(parameters,loader.redshifts)

    colors = ["purple","orange","red","blue","yellow","grey"]
    fig = plt.figure(constrained_layout=True, figsize=(18, 6))
    fig.suptitle(f"BH parameter comparison for {param}")
    gs = gridspec.GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    ax4 = fig.add_subplot(gs[:, 2])

    #draw default
    beorn.plotting.draw_dTb_signal(ax1, default,color= "black",label= f"no BH")
    beorn.plotting.draw_dTb_power_spectrum_of_z(ax4, default, parameters,color= "black",label= f"no BH")
    beorn.plotting.draw_Temp_signal(ax2, default,color= "black",label= f"no BH")
    beorn.plotting.draw_xHII_signal(ax3, default,color= "black",label= f"no BH")
    

    for i in range(len(BH_parameters[param])):
        beorn.plotting.draw_dTb_signal(ax1, BH_multi_z[param][i],color= colors[i],label= f"{param}={BH_parameters[param][i]}")
        beorn.plotting.draw_Temp_signal(ax2, BH_multi_z[param][i],color= colors[i],label= f"{param}={BH_parameters[param][i]}")
        beorn.plotting.draw_xHII_signal(ax3, BH_multi_z[param][i],color= colors[i],label= f"{param}={BH_parameters[param][i]}")
        beorn.plotting.draw_dTb_power_spectrum_of_z(ax4, BH_multi_z[param][i], parameters,color= colors[i],label= f"{param}={BH_parameters[param][i]}")
    ax4.set_ylim(1e-4,1e2)
    plt.legend()
    plt.show()

def plot_light_cone(BH_multi_z,i,param):


    lightcone = beorn.structs.Lightcone.build(
            parameters,
            BH_multi_z[param][i],
            quantity="Grid_dTb",
            
        )
    
    fig, ax = plt.subplots(figsize=(18, 6))
    beorn.plotting.plot_lightcone(lightcone, ax, "", slice_number=80)
    plt.show()

def plot_BHMF_z(redshifts,parameters):
    M   = np.logspace(8, 15, 60)
    halo_mass,halo_mass_derivative = mass_accretion(parameters,loader.redshifts,M,np.array([0.79]))
    M_BH,M_BH_dot,Mh = black_hole_mass_iterative(loader.redshifts,M,parameters)
    hmf = HaloMassFunction(parameters,"sheth_tormen")
    #plot halo mass function at target redshift
    """plt.plot(M,n,label=f"{target_redshift}")
    plt.plot(M,n2,label="15")
    plt.plot(M,n3,label="20")
    plt.ylim(1e-12,1e2)
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$M_\odot$")
    plt.ylabel(r"$\frac{dn}{dlnM}$")
    plt.title(f"Halo mass function")
    plt.show()"""
    #need to reverse the order because M_BH is in decreasing halo mass
    M_BH = np.flip(M_BH,0)
    M_BH_dot = np.flip(M_BH_dot,0)
    n_BH = np.full((len(redshifts),60,1),0.0)

    for i in range(len(redshifts)):

        z = redshifts[i]
        target_redshift = z
        i_target = np.where(loader.redshifts==z)[0]

        n   = hmf.dndlnm(halo_mass[:,0,i_target], z=target_redshift)
        #n_BH = np.full_like(n,0)
        mask = np.where(M_BH_dot[:,i_target] >= 0.0)
        n_BH[i,mask,0] = n[mask,0]*halo_mass_derivative[mask,0,i_target] / M_BH_dot[mask,i_target]  / halo_mass[mask,0,i_target] * M_BH[mask,i_target] 
        for j in range(len(M_BH_dot[:,0])):
            if n_BH[i,j,0] == np.inf:
                n_BH[i,j,0] = np.nan
        #n_BH = n* M_BH[:,i_target]
        #print(Mh)
        #print(M_BH[:,i_target],n_BH[i])
        print(n_BH[i,1,0])
        plt.plot(M_BH[:,i_target],n_BH[i],label=f"z={target_redshift}")
        #plt.plot(halo_mass[:,0,i_target],n,label=f"z={target_redshift}")
        #plt.plot(halo_mass[:,0,i_target],n,label=f"halo mass at {target_redshift}")
        """plt.scatter(M_BH[mask,i_target],n_BH[i,mask,0])#M_BH[mask,i_target]*M_BH_dot[mask,i_target])
        plt.yscale("log")
        plt.xscale("log")
        plt.show()"""
    plt.xlim(1e1,1e9)
    plt.ylim(1e-6,1e2)
    #plt.xlim(1e1,1e10)
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$M_\odot$")
    plt.ylabel(r"$\frac{dn_{BH}}{dlnM_{BH}}$")
    plt.title(f"BHMF for f0={parameters.source.f0}, Mseed={parameters.source.M_seed}, p={parameters.source.p_merge}, xi={parameters.source.xi}")

    
    return n_BH,M_BH[:,i_target].squeeze()

def plot_BHMF_para(params,target_redshift):
    M   = np.logspace(8, 15, 60)
    halo_mass,halo_mass_derivative = mass_accretion(params,loader.redshifts,M,np.array([0.79]))
    hmf = HaloMassFunction(params,"sheth_tormen")

    i_target = np.where(loader.redshifts==target_redshift)[0]

    #need to reverse the order because M_BH is in decreasing halo mass
    
    for para in params:
        #change this
        params.source.xi = para
        M_BH,M_BH_dot,Mh = black_hole_mass_iterative(loader.redshifts,M,params)
        M_BH = np.flip(M_BH,0)
        M_BH_dot = np.flip(M_BH_dot,0)

        

        n   = hmf.dndlnm(halo_mass[:,0,i_target], z=target_redshift)

        n_BH = np.full_like(n,0)
        mask = np.where(M_BH_dot[:,i_target] >= 0.0)
        n_BH[mask,0] = n[mask,0]*halo_mass_derivative[mask,0,i_target] / M_BH_dot[mask,i_target]  / halo_mass[mask,0,i_target] * M_BH[mask,i_target] 

        #change label
        plt.plot(M_BH[:,i_target],n_BH,label=f"xi={para}")
        

    plt.xlim(1e1,1e12)
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$M_\odot$")
    plt.ylabel(r"$\frac{dn_{BH}}{dlnM_{BH}}$")
    #change title
    plt.title(f"BHMF for z={target_redshift}, p={params.source.p_merge}, f0={params.source.f0}, Mseed={params.source.M_seed}")
    plt.show()

def compute_MBH_MSTAR(params):
    M   = np.logspace(8, 15, 25) 
    halo_mass,halo_mass_derivative = mass_accretion(params,loader.redshifts,M,np.array([0.79]))
    M_BH,M_BH_dot,Mh = black_hole_mass_iterative(loader.redshifts,M,params)
    
    Ob,Om = params.cosmology.Ob,params.cosmology.Om

    M_star = (Ob / Om) * f_star_Halo(params, halo_mass) * halo_mass
    i_target = np.where(loader.redshifts == 7.5)
    #i2_target = np.where(loader.redshifts == 9.0)
    M_BH = np.flip(M_BH,0)
    M_BH_dot = np.flip(M_BH_dot,0)
    return M_star[:,0,i_target],M_BH[:,i_target]

def plot_MBH_Mstar(params,z):
    M   = np.logspace(8, 15, 25)
    
    halo_mass,halo_mass_derivative = mass_accretion(params,loader.redshifts,M,np.array([0.79]))
    start=time()
    M_BH,M_BH_dot,Mh = black_hole_mass_iterative(loader.redshifts,M,params)
    #print(time()-start)
    Ob,Om = params.cosmology.Ob,params.cosmology.Om

    M_star = (Ob / Om) * f_star_Halo(params, halo_mass) * halo_mass
    i_target = np.where(loader.redshifts == z)
    i2_target = np.where(loader.redshifts == z+3)
    i3_target = np.where(loader.redshifts == z+1.5)
    M_BH = np.flip(M_BH,0)
    M_BH_dot = np.flip(M_BH_dot,0)
    #plt.scatter(np.log10(M_star[:,0,i2_target]),np.log10(M_BH[:,i2_target]),color="red",label="Nick")

    f_inter = interp1d(M_star[:,0,i2_target].squeeze(),M_BH[:,i2_target].squeeze(),kind="cubic", fill_value="extrapolate")
    plt.fill_between(np.log10(M_star[:,0,i_target].squeeze()), np.log10(M_BH[:,i_target].squeeze()), np.log10(f_inter(M_star[:,0,i_target].squeeze())), alpha=0.3,label=f"simulation (z={z}-{z+3})",color="red")
    plt.title(f"f0={params.source.f0:.2f}, Mseed={params.source.M_seed:.1f}, p={params.source.p_merge:.2f}, xi={params.source.xi:.2f}")
    plt.plot(np.log10(M_star[:,0,i_target].squeeze()), np.log10(M_BH[:,i_target].squeeze()),label=f"simulation(z={z})",color="red")
    #print(np.log10(M_BH[:,i_target].squeeze()), y_z9_on_x6,np.log10(M_star[:,0,i2_target].squeeze()))
    
BH_params = {
    "f0": [1.0],
    "M_seed": [],
    "p": [],
    "xi": []
}

def claude_plot(parameters):
    
    
    
    d = pd.read_csv(r"C:\Users\nick\Code\Thesis26\BEoRN\examples\mbh_mstar_6z9_digitized.csv", comment="#")
    
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    x = np.linspace(6, 12, 50)
    #ax.plot(x, 8.69 + 1.17 * (x - 11), "k-", lw=1, label="Kormendy & Ho (2013)")
    #ax.plot(x, 7.45 + 1.05 * (x - 11), "k--", lw=1, label="Reines & Volonteri (2015)")
    #for off in (0, 1, 2):  # M_BH/M_* = 1, 0.1, 0.01
        #ax.plot(x, x - off, ":", c="darkred", lw=1)
    
    for name, c, lab in [("THESAN-ZOOM", "k", "THESAN-ZOOM"),
                        ("TNG50_z3", "#4b91fd", "TNG50 (z=3)")]:
        s = d[d.dataset == name]
        ax.plot(s.log_mstar, s.log_mbh, c=c, lw=2.5, label=lab)
        e = s.dropna(subset=["log_mbh_lo"])
        ax.errorbar(e.log_mstar, e.log_mbh,
                    yerr=[e.log_mbh - e.log_mbh_lo, e.log_mbh_hi - e.log_mbh],
                    fmt="none", ecolor=c, capsize=3, lw=1)
    
    s = d[d.dataset == "JWST_AGN"]
    ax.scatter(s.log_mstar, s.log_mbh, s=40, c="#f7d624", ec="k", zorder=5, label="JWST AGN")
    s = d[d.dataset == "QSO"]
    ax.scatter(s.log_mstar, s.log_mbh, s=45, marker="^", c="grey", ec="k", zorder=5, label="QSOs")
    
    if len(sys.argv) > 1:
        me = pd.read_csv(sys.argv[1])
        ax.scatter(me.log_mstar, me.log_mbh, s=12, c="tab:green", alpha=0.6, zorder=4, label="this work")

    plot_MBH_Mstar(parameters,6.0)
    
    ax.set(xlim=(6, 12), ylim=(4, 10),
        xlabel=r"$\log_{10}(M_*\,[{\rm M_\odot}])$", ylabel=r"$\log_{10}(M_{\rm BH}\,[{\rm M_\odot}])$")
    ax.text(6.2, 9.6, r"$6 < z < 9$")
    ax.legend(fontsize=7, loc="lower right")
    
    plt.tight_layout()
    
    plt.show()

def log_prior(theta):
    f0,Mseed,p,xi= theta#,f0_sig,Mseed_sig,p_sig,xi_sig = theta
    if 0<f0<2 and 50< Mseed <2000 and 0<p<1 and 0<xi<1:#and 0<= f0_sig <0.2 and 0<= Mseed_sig<200 and 0.0 <=p_sig<0.3 and 0<= xi_sig < 0.3:
        return 0.0
    return -np.inf

def parameters_from(theta,parameters):
    parameters.source.f0 = theta[0]
    parameters.source.M_seed = theta[1]
    parameters.source.p_merge = theta[2]
    parameters.source.xi = theta[3]
    """parameters.source.f0_sig = theta[4]
    parameters.source.M_seed_sig = theta[5]
    parameters.source.p_merge_sig = theta[6]
    parameters.source.xi_sig = theta[7]"""
    return parameters

def log_likelyhood(theta,M_star_obs,M_BH_obs,sigma_x,sigma_y,parameters):
    prior = log_prior(theta)
    if not np.isfinite(prior):
        return -np.inf
    parameters = parameters_from(theta,parameters)
    #returns calculated MBH and MSTAR at redshift 7.5 for every halo
    M_star,M_BH = compute_MBH_MSTAR(parameters)
    M_star,M_BH = M_star.squeeze(),M_BH.squeeze()

    f_BH = interp1d(M_star,M_BH,kind="linear",
                 bounds_error=False,fill_value="extrapolate")
    f_Mstar = interp1d(M_BH,M_star,kind="linear",
                 bounds_error=False,fill_value="extrapolate")
    M_BH_sim = f_BH(M_star_obs)
    M_star_sim = f_Mstar(M_BH_obs)
    return prior-0.5 * np.sum(((M_BH_obs-M_BH_sim)/sigma_y)**2+((M_star_obs-M_star_sim)/sigma_x)**2)
                            

def mcmc(parameters):
    #get jwst data
    d = pd.read_csv(r"C:\Users\nick\Code\Thesis26\BEoRN\examples\mbh_mstar_6z9_digitized.csv", comment="#")
    s = d[d.dataset == "JWST_AGN"]
    M_star_obs,M_BH_obs = 10**s.log_mstar, 10**s.log_mbh
    sigma_x,sigma_y = np.full_like(M_BH_obs,1.0),np.full_like(M_BH_obs,1.0)
    #try to do mcmc on jwst data
    import emcee
    ndim = 4
    nwalkers = 32
    center = np.array([0.8,800,0.8,0.2])
    spread= np.array([0.1,100,0.1,0.1])
    p0 = center+spread* np.random.randn(nwalkers,ndim)
    from multiprocessing import Pool
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers,ndim,log_likelyhood,args=(M_star_obs,M_BH_obs,sigma_x,sigma_y,parameters),pool=pool)
        sampler.run_mcmc(p0,5000,progress=True)

    tau = sampler.get_autocorr_time(tol=0)
    samples = sampler.get_chain(discard=int(3*tau.max()), thin=int(tau.max()/2), flat=True)
    print(tau)
    print(samples)

    #claude
    labels = ["f0", "Mseed", "p","xi"]#"f0_sig","Mseed_sig","p_sig","xi_sig"] 
    chain = sampler.get_chain()              # shape (steps, walkers, ndim)
    fig, axes = plt.subplots(len(labels), 1, sharex=True, figsize=(8, 2*len(labels)))
    for i, ax in enumerate(axes):
        ax.plot(chain[:, :, i], "k", alpha=0.3)
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("step")

    # 2. Discard burn-in and thin, using the autocorrelation time
    tau = sampler.get_autocorr_time(tol=0)
    samples = sampler.get_chain(discard=int(3*tau.max()), thin=max(1, int(tau.max()/2)), flat=True)

    # 3. Corner plot
    corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84],
                show_titles=True, title_fmt=".3f")

    theta_pred = []
    # 4. Parameter estimates: median with 68% interval
    for i, name in enumerate(labels):
        lo, med, hi = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{name} = {med:.4f} +{hi-med:.4f} -{med-lo:.4f}")
        theta_pred.append(med)

    # 5. Posterior predictive check: model curves from random posterior draws over the data
    plt.figure()
    for theta in samples[np.random.randint(len(samples), size=100)]:
        parameters = parameters_from(theta,parameters)
        M_BH,M_star = compute_MBH_MSTAR(parameters)
        M_star,M_BH = M_star.squeeze(),M_BH.squeeze()
        x_sim, y_sim = M_star,M_BH
        plt.plot(x_sim, y_sim, "C1", alpha=0.1)
    plt.errorbar(M_star_obs, M_BH_obs, yerr=sigma_y, fmt="o", color="k")
    plt.xlabel("x"); plt.ylabel("y")
    plt.show()

    parameters= parameters_from(theta_pred,parameters)
    claude_plot(parameters)

if __name__ == "__main__":

    #
    #BH_multi_Z = main(BH_params)

    #Light cones
    #plot_light_cone(BH_multi_Z,3,"f0")

    #Power spectrum plots
    #plot_dTb_Powerspectrum_parameter(BH_multi_Z,BH_params,"f0")

    #Scatter plots
    """parameters.source.p_merge = 0.5
    #parameters.source.f0 = 0.5
    no_scatter_n,M_BH = plot_BHMF_z([6.0],parameters)
    #now with scatter
    #parameters.source.f0_sig = 0.1
    parameters.source.p_merge_sig = 0.3
    scatters = []
    mean_M = np.full((len(M_BH)),0.0)
    mean_n = np.full((len(M_BH)),0.0)
    diff = np.full((len(M_BH)),0.0)
    Nit = 20
    xs = []
    ys = []

    for n in range(Nit):
        print(n)
        scatters.append(plot_BHMF_z([6.0],parameters))
        #plt.plot(scatters[n][1],scatters[n][0][0,:,0],linestyle="--")
        mean_M += scatters[n][1]
        mean_n += scatters[n][0][0,:,0]
        print(scatters[n][0][0,:,0])
        print(scatters[n][1])
        xs = np.concatenate([xs,scatters[n][1]])
        ys = np.concatenate([ys,scatters[n][0][0,:,0]])
    print(xs.max(),xs.min())
    print(ys.max(),ys.min())
    xbins = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 20)
    ybins = np.logspace(np.log10(ys.min()+0.01), np.log10(ys.max()), 20)
    print(xbins)
    H, xe, ye = np.histogram2d(xs, ys,bins=[xbins, ybins])
    #H = H / H.sum(axis=1, keepdims=True)
    plt.scatter(xs,ys,s=2,alpha=0.3)
    #points from one halo mass bin 10
    x_halo,y_halo = [scatters[l][1][-20] for l in range(Nit)],[scatters[l][0][0,-20,0] for l in range(Nit)]
    #plt.scatter(x_halo,y_halo,s=3,alpha=0.3)
    #plt.ylim(1e-20,1e3)


    mean_M /= Nit
    mean_n /= Nit
    plt.plot(mean_M,mean_n,label="mean scattered")
    plt.plot(M_BH,no_scatter_n[0,:,0],label="no scatter")
    plt.legend()
    plt.xlabel(r"$M_{BH}[M_\odot]$")
    plt.ylabel(r"$\frac{dn_{BH}}{dlnM_{BH}}(Mpc/h)^{-3}$")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("z=6,f0=0.1, Mseed=100, xi=0.3")
    plt.show()"""

    #mcmc(parameters)

    #plt.title(f"BHMF for f0={parameters.source.f0}, Mseed={parameters.source.M_seed}, p={parameters.source.p_merge}, xi={parameters.source.xi} and f0 scatter {0.05}")
    #plt.show()
    print(loader.redshifts)
    #plot_BHMF_para([0.0,0.1,0.3,0.5,0.7],17)
    claude_plot(parameters_from([1.0,100,0.5,0.3],parameters))
    plot_BHMF_z([6.0,9.0,12.0,15.0,18.0],parameters_from([1.0,100,0.5,0.3],parameters))
    plt.show()
    Mh0 = np.logspace(8,15,50)

    halo_mass, halo_mass_derivative = mass_accretion(parameters,loader.redshifts,Mh0,np.array([0.79]))
    M_BH,M_BH_dot,Mh = black_hole_mass_iterative(loader.redshifts,Mh0,parameters)
    print(halo_mass_derivative)
    plt.plot(loader.redshifts,halo_mass_derivative[0][0])
    plt.plot(loader.redshifts,M_BH_dot[-1])
    plt.yscale("log")
    plt.show()




    
    
    

