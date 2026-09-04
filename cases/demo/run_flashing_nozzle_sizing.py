import os
from pathlib import Path

import yaml
import jaxprop as jxp
import matplotlib.pyplot as plt

from flashing_nozzle_design import functions as fn
from flashing_nozzle_design import graphics

graphics.set_plot_options(grid=True)

# --- Script directory (for locating files regardless of cwd) ---
SCRIPT_DIR = Path(__file__).resolve().parent

# --- Create output directory (next to this script, not cwd) ---
OUT_DIR = SCRIPT_DIR / "results"
os.makedirs(OUT_DIR, exist_ok=True)


if __name__ == "__main__":

    # Load configuration file (relative to this script's location, not cwd)
    with open(SCRIPT_DIR / "nozzle_sizing.yaml", "r") as fp:
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

    plt.show()