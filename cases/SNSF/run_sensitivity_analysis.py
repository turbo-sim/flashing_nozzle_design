"""Nozzle inlet temperature sensitivity analysis, for a list of fluids and
a list of nozzle-inlet-condition cases.

F every fluid listed  in SNSF_sizing.yaml, and for each case in the CASES
list defined below (e.g. different inlet subcoolingS, or different inlet
vapor qualities), it sweeps the nozzle inlet temperature (TIT) from just
above the condensing temperature up to just below that fluid's critical
temperature, sizing the test rig at every point in the sweep.

Two different places hold the settings this script needs:

    - Shared base parameters - the fluid list, condensing temperature,
      component efficiencies and losses, droplet breakup model, etc. - come
      from SNSF_sizing.yaml, exactly as in run_flashing_nozzle_sizing.py.

    - The sensitivity sweep itself - how many TIT points to evaluate, how
      close to the condensing/critical temperature to sweep, and which
      inlet-condition cases to compare - is controlled by the constants
      near the top of this script (N_POINTS, TIT_MARGIN_LOW,
      TIT_MARGIN_HIGH, CASES).

For each fluid, once every case has been evaluated, the script
produces two figures:

    - a temperature-entropy (T-s) diagram of the cycle for that fluid,
      colored by nozzle inlet temperature

    - summary of key result quantities plotted against nozzle inlet temperature

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

# Number of turbine inlet temperature points to evaluate, per case.
N_POINTS = 7

# Margin [K] kept above the condensing temperature for the lowest TIT point.
TIT_MARGIN_LOW = 20.0

# Margin [K] kept below the critical temperature for the highest TIT point.
TIT_MARGIN_HIGH = 10.0

# Nozzle-inlet-condition cases to compare, each swept over the same TIT
# range. "overrides" are applied on top of the base config for that case.
# Cases are drawn with linestyles "-", "--", "-.", ":" (in this order).
CASES = [
    {
        "label": "Subcooling 10 K",
        "overrides": {"nozzle_inlet_condition": "subcooling", "subcooling_heater_outlet": 10.0},
    },
    {
        "label": "Quality 0.0",
        "overrides": {"nozzle_inlet_condition": "quality", "turbine_inlet_quality": 0.0},
    },
    {
        "label": "Quality 0.1",
        "overrides": {"nozzle_inlet_condition": "quality", "turbine_inlet_quality": 0.1},
    },
]

# --- Create output directory ---
OUT_DIR = "results_sensitivity"
os.makedirs(OUT_DIR, exist_ok=True)


if __name__ == "__main__":

    # Load configuration file (base parameters shared by every fluid/case/sweep point)
    with open("SNSF_sizing.yaml", "r") as fp:
        base_config = yaml.safe_load(fp)

    case_labels = [case["label"] for case in CASES]

    for fluid_name in base_config["fluids"]:

        fluid = jxp.Fluid(name=fluid_name, backend=base_config["fluid_backend"])

        # Sweep of absolute turbine inlet temperatures [K] (same for every case)
        TIT_min = base_config["condensing_temperature"] + TIT_MARGIN_LOW
        TIT_max = fluid.critical_point.T - TIT_MARGIN_HIGH
        TIT_values = np.linspace(TIT_min, TIT_max, N_POINTS)

        # Run rig_sizing() over the TIT sweep for each case. Console output
        # is indented by nesting level: fluid, then inlet condition (case),
        # then temperature point.
        print(f"\n{fluid_name}")
        print("=" * len(fluid_name))

        results = []
        converged_TIT = []
        for case in CASES:
            print(f"  {case['label']}")

            case_results = []
            case_TIT = []
            for TIT in TIT_values:
                data = dict(base_config)
                data.update(case["overrides"])
                data["use_reduced_turbine_inlet_temperature"] = False
                data["turbine_inlet_temperature"] = TIT

                converged, result = fn.rig_sizing(fluid, data)
                if not converged:
                    fn.print_fluid_status("FAIL", f"    TIT = {TIT:6.1f} K", result)
                    continue

                fn.print_fluid_status("OK", f"    TIT = {TIT:6.1f} K")
                case_results.append(result)
                case_TIT.append(TIT)

            results.append(case_results)
            converged_TIT.append(case_TIT)

        # T-s diagram overlay, one curve per (case, turbine inlet temperature):
        # color encodes TIT (Reds colormap, shared across cases), linestyle
        # identifies the case, with a case legend.
        fn.plot_fluid_Ts(fluid_name, results[0], OUT_DIR, group_labels=case_labels)

        # Grid of key result quantities against turbine inlet temperature, one
        # line per case per panel. Only nozzle-side quantities are shown (this
        # rig has no turbine).
        panels = [
            {"key": "mass_flow_rate", "label": "Mass flow rate [kg/s]"},
            {"key": "condenser_power", "label": "Condenser power [kW]", "scale": 1e-3},
            {"key": "pump_power", "label": "Pump power [kW]", "scale": 1e-3},
            {"get": lambda r: r["states"]["p"][fn.i_turb_in], "label": "Nozzle inlet pressure [kPa]", "scale": 1e-3},
            {"get": lambda r: r["states"]["p"][fn.i_pump_in], "label": "Nozzle outlet pressure [kPa]", "scale": 1e-3},
            {"key": "choking_velocity", "label": "Nozzle throat velocity [m/s]"},
            {"key": "nozzle_velocity", "label": "Nozzle exit velocity [m/s]"},
            {"key": "nozzle_mach_number", "label": "Nozzle Mach number [-]"},
            {"key": "throat_diameter", "label": "Nozzle throat diameter [mm]", "scale": 1e3},
            # {"key": "throat_to_droplet_diameter_ratio_exit", "label": "Nozzle-to-droplet diameter ratio [-]"},
        ]
        fn.plot_sensitivity_subplots(
            converged_TIT,
            "Nozzle inlet temperature [K]",
            results,
            panels,
            OUT_DIR,
            f"{fluid_name}_sensitivity",
            ncols=3,
            group_labels=case_labels,
            title=fluid_name,
        )

    # Show the plots
    plt.show()
