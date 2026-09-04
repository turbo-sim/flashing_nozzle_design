"""Turbine inlet temperature sensitivity analysis for a single fluid.

Unlike run_flashing_nozzle_sizing.py, this script does not size a fixed list
of fluids at one operating point. Instead, it takes a single fluid and calls
rig_sizing() in a loop over a sweep of absolute turbine inlet temperatures
(TIT), between the condensing temperature (plus a margin) and the fluid
critical temperature (minus a margin). It then plots the resulting T-s
diagrams overlaid on one figure, colored by TIT, and a grid of key result
quantities against TIT.

"""

import os
import yaml
import numpy as np
import jaxprop as jxp
import matplotlib as mpl
import matplotlib.pyplot as plt

from flashing_nozzle_design import functions as fn
from flashing_nozzle_design import graphics

graphics.set_plot_options(grid=True)

# =============================================================================
# Sensitivity sweep settings
# =============================================================================

FLUID_NAME = "n-Butane"

# Number of turbine inlet temperature points to evaluate.
N_POINTS = 7

# Margin [K] kept above the condensing temperature for the lowest TIT point.
TIT_MARGIN_LOW = 20.0

# Margin [K] kept below the critical temperature for the highest TIT point.
TIT_MARGIN_HIGH = 10.0

# --- Create output directory ---
OUT_DIR = "results_sensitivity"
os.makedirs(OUT_DIR, exist_ok=True)


if __name__ == "__main__":

    # Load configuration file (base parameters shared by every sweep point)
    with open("SNSF_sizing.yaml", "r") as fp:
        config = yaml.safe_load(fp)

    fluid = jxp.Fluid(name=FLUID_NAME, backend=config["fluid_backend"])

    # Sweep of absolute turbine inlet temperatures [K]
    TIT_min = config["condensing_temperature"] + TIT_MARGIN_LOW
    TIT_max = fluid.critical_point.T - TIT_MARGIN_HIGH
    TIT_values = np.linspace(TIT_min, TIT_max, N_POINTS)

    # Run rig_sizing() for each turbine inlet temperature in the sweep
    results = []
    converged_TIT = []
    for TIT in TIT_values:

        print(f"Running rig_sizing for TIT={TIT:.1f} K")
        data = dict(config)
        data["use_reduced_turbine_inlet_temperature"] = False
        data["turbine_inlet_temperature"] = TIT

        converged, result = fn.rig_sizing(fluid, data)
        if not converged:
            fn.print_fluid_status("FAIL", f"{FLUID_NAME} @ TIT={TIT:.1f} K", result)
            continue

        fn.print_fluid_status("OK", f"{FLUID_NAME} @ TIT={TIT:.1f} K")
        results.append(result)
        converged_TIT.append(TIT)

    # T-s diagram overlay, one curve per turbine inlet temperature, colored
    # by the Reds colormap with a legend (in degC) identifying each curve
    fn.plot_fluid_Ts(FLUID_NAME, results, OUT_DIR)

    # Grid of key result quantities against turbine inlet temperature.
    # Only nozzle-side quantities are shown (this rig has no turbine).
    panels = [
        {"key": "mass_flow_rate", "label": "Mass flow rate [kg/s]"},
        {"key": "condenser_power", "label": "Condenser power [kW]", "scale": 1e-3},
        {"key": "pump_power", "label": "Pump power [kW]", "scale": 1e-3},
        {"get": lambda r: r["states"]["p"][fn.i_turb_in], "label": "Nozzle inlet pressure [kPa]", "scale": 1e-3},
        {"key": "choking_velocity", "label": "Nozzle throat velocity [m/s]"},
        {"key": "nozzle_velocity", "label": "Nozzle exit velocity [m/s]"},
        {"key": "nozzle_mach_number", "label": "Nozzle Mach number [-]"},
        {"key": "throat_diameter", "label": "Nozzle throat diameter [mm]", "scale": 1e3},
        {"key": "throat_to_droplet_diameter_ratio", "label": "Throat-to-droplet diameter ratio (throat) [-]"},
        {"key": "throat_to_droplet_diameter_ratio_exit", "label": "Throat-to-droplet diameter ratio (exit) [-]"},
    ]
    fn.plot_sensitivity_subplots(
        converged_TIT,
        "Nozzle inlet temperature [K]",
        results,
        panels,
        OUT_DIR,
        f"sensitivity_{FLUID_NAME}",
        ncols=3,
    )

    # Show the plots
    plt.show()
