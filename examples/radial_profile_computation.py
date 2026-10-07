#Compute Radial profiles
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
parameters.source.f_st   = 1
parameters.source.Mp     = 1.088e+11  # pivot mass
parameters.source.Mt     = 1.0e+7
parameters.source.g1     = 0
parameters.source.g2     = 0
parameters.source.g3     = 4
parameters.source.g4     = -1
parameters.source.halo_mass_min = 1e+5

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
profiles = solvermain.get_or_compute_profiles(cache_handler)

mass_index = parameters.solver.halo_mass_nbin // 2
print(parameters.solver.halo_mass_bins)
print(f"Mass bin {mass_index}: {parameters.solver.halo_mass_bins[mass_index]:e}")
profile_redshifts = [13.0, 9.5, 6]
# Show three α values to highlight the impact of different accretion rates
profile_alphas = [ 0.79 ]

#Radiation profiles without BH
beorn.plotting.radiation_profiles.plot_1D_profiles(
    parameters,
    profiles,
    mass_index=-1,
    redshifts=profile_redshifts,
    alphas=profile_alphas,
    label=f"1D radiation profiles — mass bin {-1},M:{parameters.solver.halo_mass_bins[-1]:e}",
    fontsize=15,
)

print(f"Total time taken: {time.time() - start_time:.2f} seconds")
plt.show()