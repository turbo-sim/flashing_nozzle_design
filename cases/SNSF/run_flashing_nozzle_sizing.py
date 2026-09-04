"""Single-point test rig sizing for a fixed list of fluids.

Unlike run_flashing_nozzle_calculation.py, this script performs no fluid
screening: it directly sizes the test rig for each fluid listed in
test_rig_case_single_point.yaml at one operating point (reduced turbine
inlet temperature, turbine inlet quality, condensing temperature) and plots
the resulting T-s diagram.
"""

import os
from datetime import datetime

import yaml
import jaxprop as jxp
import matplotlib.pyplot as plt

from flashing_nozzle_design import functions as fn
from flashing_nozzle_design import graphics

graphics.set_plot_options(grid=True)

# --- Create output directory ---
OUT_DIR = "results_v1"
os.makedirs(OUT_DIR, exist_ok=True)


if __name__ == "__main__":

    print("Start:", datetime.now())
    # Load configuration file
    with open("SNSF_sizing.yaml", "r") as fp:
        config = yaml.safe_load(fp)

    # Size the test rig for each fluid at the single operating point
    results = {}
    for fluid_name in config["fluids"]:
        fluid = jxp.Fluid(
            name=fluid_name,
            backend=config["fluid_backend"],
        )

        converged, result = fn.rig_sizing(fluid, config)
        if not converged:
            fn.print_fluid_status("FAIL", fluid.name, result)
            continue

        results[fluid_name] = result
        fn.plot_fluid_Ts(fluid_name, result, OUT_DIR, savefig=True)
        fn.print_results(result, fluid_name=fluid.name, out_dir=OUT_DIR)
        print()
        
    # Write CSV summary
    df = fn.results_to_dataframe(results)
    df.to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)

    print("End:", datetime.now())
    plt.show()
