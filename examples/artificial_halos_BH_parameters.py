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
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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
parameters.solver.halo_mass_nbin = 200
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
parameters.source.xray_normalisation = 3.4e40 * 3

# fesc
parameters.source.f0_esc = 0.2
parameters.source.pl_esc = 0

# fstar
parameters.source.f_st = 0.5
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
parameters.source.IMBH_factor = 1.0
parameters.source.p_merge = 1.0
parameters.source.xi = 0.3



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


def compute_no_BH_z_quantities(parameters,redshifts):
    print("computing no BH signal")
    parameters.source.IMBH_factor = 0.0
    parameters.source.HMXB_factor = 0.0

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
    #compute profiles for different parameters
    for f in BH_parameters['f0']:
        print(f"computing profile for f0:{f}")
        parameters.source.f0 = f
        solver = beorn.precomputation.RadiationProfileSolver(parameters, loader.redshifts)
        profiles = solver.get_or_compute_profiles(cache_handler,force_recompute=recompute)
        BH_profiles['f0'].append(profiles)
    parameters.source.f0 = 0.5

    for M in BH_parameters['M_seed']:
        print(f"computing profile for M_seed:{M}")
        parameters.source.M_seed = M
        solver = beorn.precomputation.RadiationProfileSolver(parameters, loader.redshifts)
        profiles = solver.get_or_compute_profiles(cache_handler,force_recompute=recompute)
        BH_profiles['M_seed'].append(profiles)
    parameters.source.M_seed = 100

    for p in BH_parameters['p']:
        print(f"computing profile for p:{p}")
        parameters.source.p_merge = p
        solver = beorn.precomputation.RadiationProfileSolver(parameters, loader.redshifts)
        profiles = solver.get_or_compute_profiles(cache_handler,force_recompute=recompute)
        BH_profiles['p'].append(profiles)
    parameters.source.p_merge = 0.3

    for xi in BH_parameters['xi']:
        print(f"computing profile for xi:{xi}")
        parameters.source.xi = xi
        solver = beorn.precomputation.RadiationProfileSolver(parameters, loader.redshifts)
        profiles = solver.get_or_compute_profiles(cache_handler,force_recompute=recompute)
        BH_profiles['xi'].append(profiles)
    parameters.source.xi = 0.5

    # %% [markdown]
    # ## 4. Inspect the halo distribution
    # 
    # As a quick sanity check, we project the halo positions onto the 2D plane to confirm that they are distributed throughout the simulation box. Each pixel shows the number of halos along the line of sight.

    # %%
    

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
    output_handler = beorn.io.Handler(OUTPUT_ROOT)

    
    BH_multi_z_quantities = {
        "f0": [],
        "M_seed": [],
        "p": [],
        "xi": []
    }
    
    for i,profile in enumerate(BH_profiles['f0']):
        parameters.source.f0 = BH_params['f0'][i]
        p = beorn.painting.PaintingCoordinator(
            parameters,
            loader=loader,
            output_handler=output_handler,
            force_recompute=recompute,
        )
        BH_multi_z_quantities["f0"].append(p.paint_full(profile))
    parameters.source.f0 = 0.1
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
    parameters.source.M_seed = 100

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
    parameters.source.p_merge = 0.3

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
    parameters.source.xi = 0.5



    return BH_multi_z_quantities

def plot_dTb_Powerspectrum_parameter(BH_multi_z,BH_parameters,param = "f0"):

    #compute or get no_BH spectrum
    default = compute_no_BH_z_quantities(parameters,loader.redshifts)

    colors = ["purple","orange","red","blue","yellow","grey"]
    fig = plt.figure(constrained_layout=True, figsize=(18, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    #draw default
    beorn.plotting.draw_dTb_signal(ax1, default,color= "black",label= f"no BH")
    beorn.plotting.draw_dTb_power_spectrum_of_z(ax2, default, parameters,color= "black",label= f"no BH")
            

    for i in range(len(BH_parameters[param])):
        beorn.plotting.draw_dTb_signal(ax1, BH_multi_z[param][i],color= colors[i],label= f"{param}={BH_parameters[param][i]}")
        beorn.plotting.draw_dTb_power_spectrum_of_z(ax2, BH_multi_z[param][i], parameters,color= colors[i],label= f"{param}={BH_parameters[param][i]}")
        
        #beorn.plotting.draw_dTb_power_spectrum_of_z(ax2, BH_multi_z["M_seed"][i], parameters,color= colors[i],label= f"{param}={BH_parameters[param][i]}")
    plt.title(f"{param}")
    plt.legend()
    plt.show()


BH_params = {
    "f0": [0.0,0.1,0.3,0.5],
    "M_seed": [100,300,800],
    "p": [0.1,0.3,0.5,0.7,1.0],
    "xi": [0.0,0.2,0.4,0.6]
}

if __name__ == "__main__":
    BH_multi_Z = main(BH_params)

    plot_dTb_Powerspectrum_parameter(BH_multi_Z,BH_params,"M_seed")