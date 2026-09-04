"""Size a flashing-nozzle test rig at a single operating point, for each
fluid in a list.

For every fluid listed under `fluids` in SNSF_sizing.yaml, this script
evaluates one thermodynamic cycle (pump -> heater -> nozzle -> condenser)
at the single operating point defined by the rest of that file, and sizes
the nozzle throat needed to deliver the requested heater power. 

All the physical inputs that matter  are set in SNSF_sizing.yaml.
To try a different fluid list or operating point, edit that file; nothing
in this script itself needs to change.

For each fluid that converges, the script:

    - plots a T-s diagram of the cycle (saved as "<fluid>_Ts_diagram.png")

    - prints a human-readable report of the sizing results to the terminal,
      and saves the same report as a .txt file

    - adds a row to a combined "results.csv" summary across all fluids

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
OUT_DIR = "results_single_case"
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
