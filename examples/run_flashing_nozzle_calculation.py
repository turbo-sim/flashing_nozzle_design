"""Run the EMPOWER preliminary test rig flashing-nozzle calculation.

Reproduces the workflow from EMPOWER_preliminary_test_rig_model_v3_ECOS.py,
using the calculation functions consolidated in
src/flashing_nozzle_design/functions.py and the plotting utilities in
src/flashing_nozzle_design/graphics.py.
"""

import os
import yaml
import datetime
import jaxprop as jxp
import matplotlib.pyplot as plt

from flashing_nozzle_design import functions as fn
from flashing_nozzle_design import graphics

graphics.set_plot_options(grid=True)

# --- Create output directory ---
OUT_DIR = "results_v5"
os.makedirs(OUT_DIR, exist_ok=True)


if __name__ == "__main__":

    print("Start:", datetime.datetime.now())
    # Load configuration file
    with open("test_rig_case.yaml", "r") as fp:
        config = yaml.safe_load(fp)
        fluids = config.get("fluids", jxp.CP.FluidsList())

    Nsuitable = 0
    Nconverged = 0
    # Loop over all fluids
    results = {}
    for fluid_name in fluids:

        # 1. Try to initialize the fluid
        try:
            fluid = jxp.Fluid(
                name=fluid_name,
                backend=config["backend"],
            )
        except Exception as e:
            fn.print_fluid_status("SKIP", fluid_name, f"Initialization failed ({e})")
            continue

        # 2. Screening checks (critical limits, purity, triple point, etc.)
        ok, message = fn.screen_fluid(fluid, config)
        if not ok:
            fn.print_fluid_status("SKIP", fluid.name, message)
            continue
        Nsuitable += 1
        # 3. Run rig sizing
        converged, result = fn.rig_sizing(fluid, config)
        if not converged:
            fn.print_fluid_status("FAIL", fluid.name, result)
            continue

        # 4. Success path
        results[fluid_name] = result
        fn.print_fluid_status("OK", fluid.name)
        Nconverged += 1
        # 5. Ts plot and parametric study
        if fluid_name in config["fluids_parametric"]:
            fn.plot_fluid_Ts(fluid_name, result, OUT_DIR, savefig=True)
            fn.parametric_study(fluid, config, result["A_throat"], OUT_DIR)

    print("Fluid screening: %d investigated, %d suitable, %d converged" % (len(fluids), Nsuitable, Nconverged))
    # Write YAML summary
    with open(os.path.join(OUT_DIR, "results.yaml"), "w") as fp:
        yaml.safe_dump(
            fn.to_builtin(results),
            fp,
            sort_keys=False,
            default_flow_style=False,
        )

    # Write CSV summary
    df = fn.results_to_dataframe(results)
    df = df.sort_values("A_throat_mm2", ascending=False)
    df.round(2).to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)
    
    print("End:", datetime.datetime.now())
    plt.show()
