#Compute Radial profiles
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from beorn.structs import RadiationProfiles, Parameters
from beorn.structs.radiation_profiles import RadiationProfilesFStarGrid
from beorn.cosmo import T_adiab

import logging

from beorn.structs import radiation_profiles
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FILE_ROOT = Path(".")
CACHE_ROOT = FILE_ROOT / "cache"
OUTPUT_ROOT = FILE_ROOT / "output"

import beorn
print(beorn.__file__)

import matplotlib as mpl
mpl.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.titlesize': 14,
})

#SET PARAMETERS
parameters = beorn.structs.Parameters()

## Simulation grid — must match the THESAN data
parameters.simulation.Lbox  = 100   # comoving Mpc/h
parameters.simulation.Ncell = 128     # grid resolution for painting
parameters.simulation.cores = 4

## Merger tree settings
parameters.source.mass_accretion_lookback = 10   # lookback snapshots for alpha fit
parameters.source.alpha_fallback          = 'mean'  # fallback for halos not in tree

## Solver: redshift grid and alpha bins
# Covering z = 6–10 is sufficient for a single-snapshot tutorial at z≈7.4.
# Widen this range if you want to paint multiple snapshots.
parameters.solver.redshifts                 = np.arange(20.0, 5.5, -0.5)
parameters.solver.halo_mass_accretion_alpha = np.array([0.785, 0.795]) # 25 bins, 0–5
parameters.solver.halo_mass_bin_min         = 1.3e10 
# Upper limit must cover the most massive halo in the box at every snapshot.
# At z≈7.4 in THESAN-DARK 2 the most massive halos reach ~a few × 10¹³ M☉;
# halos above this limit fall outside all profile bins and raise an AssertionError.
parameters.solver.halo_mass_bin_max         = 2e11
parameters.solver.halo_mass_nbin            = 2

## Cosmology (Planck 2015, matching THESAN)
parameters.cosmology.Om  = 0.31
parameters.cosmology.Ob  = 0.045
parameters.cosmology.Ol  = 0.68
parameters.cosmology.h0  = 0.68

## Source model — full run
parameters.source.f_st = 0.05
parameters.source.halo_mass_min = 1e4
parameters.source.g1 = 0.49
parameters.source.g2 = -0.61
parameters.source.g3 = 1.4
parameters.source.g4 = 0
parameters.source.Mp = 2.5e11/0.68
parameters.source.Mt = 1e8

# X-ray
parameters.source.energy_cutoff_min_xray = 500
parameters.source.energy_cutoff_max_xray = 2000
parameters.source.energy_min_sed_xray    = 200
parameters.source.energy_max_sed_xray    = 2000
parameters.source.alS_xray               = 1.5
parameters.source.xray_normalisation     = 0.3 * 3.4e40

# Lyman-alpha
parameters.source.n_lyman_alpha_photons = 9690

# Ionisation
parameters.source.Nion    = 15000
parameters.source.f0_esc  = 0.2
parameters.source.pl_esc  = 0

start_time = time.time()
#Artificial halo catalog
cache_handler = beorn.io.Handler(CACHE_ROOT)
loader = beorn.load_input_data.ArtificialHaloLoader(
    parameters,
    halo_count = 1,
)


#Compute radial profiles
solvermain = beorn.precomputation.RadiationProfileSolver(parameters, loader.redshifts)
profiles = solvermain.get_or_compute_profiles(cache_handler,force_recompute=True)
mass_index = parameters.solver.halo_mass_nbin // 2
print(parameters.solver.halo_mass_bins)
print(f"Mass bin {mass_index}: {parameters.solver.halo_mass_bins[mass_index]:e}")
profile_redshifts = [13.0, 9.5, 6]
# Show three α values to highlight the impact of different accretion rates
profile_alphas = [ 0.79 ]

#Radiation profiles without BH
"""beorn.plotting.radiation_profiles.plot_1D_profiles(
    parameters,
    profiles,
    mass_index=-1,
    redshifts=profile_redshifts,
    alphas=profile_alphas,
    label=f"1D radiation profiles — mass bin {-1},M:{parameters.solver.halo_mass_bins[-1]:e}",
    fontsize=15,
)"""



def _select_fstar_index(profiles: RadiationProfiles | RadiationProfilesFStarGrid, f_st: float | None) -> int | None:
    """Return the nearest f_st grid index when plotting stochastic profiles."""
    if not isinstance(profiles, RadiationProfilesFStarGrid):
        return None
    if f_st is None:
        raise ValueError("f_st must be provided when plotting RadiationProfilesFStarGrid data.")
    return int(np.argmin(np.abs(profiles.f_st_grid[:] - f_st)))

def plot_temperature_profile(parameters: Parameters, profiles: RadiationProfiles,redshifts,label,alphas,f_st = None,mass_index = 0,no_BH = True,line_style = "-"):

    co_radial_grid = profiles.r_grid_cell[:]
    r_lyal_phys = profiles.r_lyal[:]
    zz = profiles.z_history[:]
    f_st_index = _select_fstar_index(profiles, f_st)

    n_z = len(redshifts)
    n_a = len(alphas)

    actual_z_vals    = []
    actual_alpha_vals = []

    cmap = 'plasma'
    _cmap = plt.get_cmap(cmap)
    colors_z = [_cmap(0.15 + 0.7 * i / max(n_z - 1, 1)) for i in range(n_z)]
    cmap = 'plasma'
    _cmap = plt.get_cmap(cmap)
    colors_z = [_cmap(0.15 + 0.7 * i / max(n_z - 1, 1)) for i in range(n_z)]
    #color legend
    handles = [
    Line2D([0], [0], color=colors_z[0], lw=2, label=f"z= {redshifts[0]}"),
    Line2D([0], [0], color=colors_z[1], lw=2, label=f"z= {redshifts[1]}"),
    Line2D([0], [0], color=colors_z[2], lw=2, label=f"z= {redshifts[2]}"),
    ]
    ax = plt.gca()
    leg1 = ax.legend(handles=handles,loc='upper center')
    ax.add_artist(leg1)
    #default legend
    handles = [
    Line2D([0], [0], color="lightgrey", lw=2, label=f"z= {redshifts[0]}"),
    Line2D([0], [0], color="grey", lw=2, label=f"z= {redshifts[1]}"),
    Line2D([0], [0], color="black", lw=2, label=f"z= {redshifts[2]}"),
    ]
    ax = plt.gca()
    leg3 = ax.legend(handles=handles,loc='lower right')
    ax.add_artist(leg3)

    for i, zi in enumerate(redshifts):
        ind_z = np.argmin(np.abs(zz - zi))
        z_val = zz[ind_z]
        actual_z_vals.append(z_val)

        for j, alpha_j in enumerate(alphas):
            ind_alpha = np.argmin(np.abs(parameters.solver.halo_mass_accretion_alpha - alpha_j))
            alpha_val = parameters.solver.halo_mass_accretion_alpha[ind_alpha]
            if i == 0:
                actual_alpha_vals.append(alpha_val)

            Mh_i = profiles.halo_mass_bins[mass_index, ind_alpha, ind_z]
            T_adiab_z = T_adiab(z_val, parameters)
            if isinstance(profiles, RadiationProfilesFStarGrid):
                Temp_profile = profiles.rho_heat[:, mass_index, ind_alpha, f_st_index, ind_z] + T_adiab_z
                x_HII_radius = profiles.R_bubble[mass_index, ind_alpha, f_st_index, ind_z]
                lyal_profile = profiles.rho_alpha[:, mass_index, ind_alpha, f_st_index, ind_z]
            else:
                Temp_profile = profiles.rho_heat[:, mass_index, ind_alpha, ind_z] + T_adiab_z
                x_HII_radius = profiles.R_bubble[mass_index, ind_alpha, ind_z]
                lyal_profile = profiles.rho_alpha[:, mass_index, ind_alpha, ind_z]

            x_HII_profile = np.zeros(len(co_radial_grid))
            x_HII_profile[co_radial_grid < x_HII_radius] = 1

            color = colors_z[i]
            if no_BH:
                color = ["lightgrey","grey","black"]
                plt.loglog(co_radial_grid / 0.68, Temp_profile, lw=1.7, color=color[i], ls=line_style,label = f"z={zi}, "+label)
            else:
                plt.loglog(co_radial_grid / 0.68, Temp_profile, lw=1.7, color=color, ls=line_style,label = f"z={zi}, "+label)
            
def parameter_comparison_temperature(redshifts,cache_handler,test_params):
    #change this
    param = "M_seed"
    if len(test_params) == 0:
        parameters.source.HMXB_factor = 0.0
        parameters.source.IMBH_factor = 0.0
        solvermain = beorn.precomputation.RadiationProfileSolver(parameters, redshifts)
        profiles = solvermain.get_or_compute_profiles(cache_handler,force_recompute=True)
        mass_index = parameters.solver.halo_mass_nbin // 2
        plot_temperature_profile(
                    parameters,
                    profiles,
                    redshifts=profile_redshifts,
                    alphas=profile_alphas,
                    mass_index = -1,
                    line_style="-",
                    label = f"no BH"
        )
    else:
        parameters.source.HMXB_factor = 0.0
        parameters.source.IMBH_factor = 1.0
        line_styles = ["-","--","-.",":"]
        for i,p in enumerate(test_params):
            #change this
            parameters.source.M_seed = p
            solvermain = beorn.precomputation.RadiationProfileSolver(parameters, redshifts)
            profiles = solvermain.get_or_compute_profiles(cache_handler,force_recompute=True)
            mass_index = parameters.solver.halo_mass_nbin // 2
            
            plot_temperature_profile(
                parameters,
                profiles,
                redshifts=profile_redshifts,
                alphas=profile_alphas,
                mass_index = -1,
                line_style=line_styles[i],
                label = f"f0={p}",
                no_BH=False
            )
        handles = [
        Line2D([0], [0],ls = "-", lw=2, label=f"{param} = {test_params[0]}"),
        Line2D([0], [0],ls = "--",  lw=2, label=f"{param} = {test_params[1]}"),
        Line2D([0], [0],ls = "-.", lw=2, label=f"{param} = {test_params[2]}"),
        Line2D([0], [0],ls = ":", lw=2, label=f"{param} = {test_params[3]}"),
        ]
        ax = plt.gca()
        leg2 = ax.legend(handles=handles,loc='upper right')
        ax.add_artist(leg2)
        
    
#default
parameter_comparison_temperature(loader.redshifts,cache_handler,[])


#for different values
f0s = [0.0,0.1,0.3,0.5]
Mseeds = [50,100,300,800]
ps = [0.0,0.1,0.3,1.0]
xis = [0.0,0.5,0.7,0.9]
parameter_comparison_temperature(loader.redshifts,cache_handler,Mseeds)

plt.title("Temperature profiles")
plt.xlim(2e-2, 1e2)
plt.ylim(0.8, 1e5)
plt.xlabel('r [cMpc]')
plt.ylabel(r'$T_k$ [K]')
plt.tick_params(axis='both')
print(f"Total time taken: {time.time() - start_time:.2f} seconds")
plt.show()