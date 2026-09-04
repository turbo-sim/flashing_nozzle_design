"""Main calculations for the flashing nozzle design project.

Cycle, choking, droplet-size and test-rig sizing routines for the EMPOWER
preliminary test rig model (flashing two-phase nozzle feeding a turbine
expander in an organic Rankine cycle).

This module reads its input parameters using descriptive key names (see
examples/test_rig_case_renamed.yaml for the matching input file) and exports
its results using descriptive key names. All quantities, input and output,
are in base SI units (Pa, K, W, m, m2, m/s, kg/s, kg/(m2 s)) except turbine
rotational speed, which is reported in rpm rather than rad/s as a practical
convention for turbomachinery. Since every quantity follows this single
convention, unit suffixes are omitted from the names themselves.
functions_old.py keeps the original, terser naming for reference.
"""

import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import jaxprop as jxp
from scipy.optimize import brentq, minimize_scalar

from . import graphics

STATE_MAPPING = {
    "pump_inlet": 0,  # 1
    "pump_outlet": 1,  # 2
    "heater_outlet": 2,  # 3
    "turbine_inlet": 3,  # 4
    "nozzle_outlet": 4,  # 5
    "turbine_outlet": 5,  # 6
    "condenser_vapor": 6,  # 7
    "condenser_liquid": 7,  # 8
}
# Define shorthand for state indices
i_pump_in = STATE_MAPPING["pump_inlet"]
i_pump_out = STATE_MAPPING["pump_outlet"]
i_heater_out = STATE_MAPPING["heater_outlet"]
i_turb_in = STATE_MAPPING["turbine_inlet"]
i_nozzle_out = STATE_MAPPING["nozzle_outlet"]
i_turb_out = STATE_MAPPING["turbine_outlet"]
i_cond_liq = STATE_MAPPING["condenser_liquid"]


def to_builtin(obj):
    """
    Recursively convert NumPy types to native Python types
    so the object can be serialized (YAML / JSON).
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()

    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]

    return obj


def results_to_dataframe(result):
    """
    Convert rig sizing results to a flat pandas DataFrame
    (one row per fluid).

    All quantities are kept in base SI units (no k/m/M-style multiples).
    Unit conversions for display/reporting purposes are left to a future
    exporting/printing function, not implemented here.
    """

    rows = []
    for fluid_name, res in result.items():
        rows.append({
            "fluid_name": fluid_name,
            "ambient_pressure": res["ambient_pressure"],
            "maximum_pressure": max(res["states"]["p"]),
            "maximum_temperature": max(res["states"]["T"]),
            "throat_area": res["throat_area"],
            "number_of_nozzles": res["number_of_nozzles"],
            "throat_area_per_nozzle": res["throat_area_per_nozzle"],
            "throat_diameter": res["throat_diameter"],
            "nozzle_mach_number": res["nozzle_mach_number"],
            "turbine_speed": res["turbine_speed"],
            "droplet_diameter_throat": res["droplet_diameter_throat"],
            "throat_to_droplet_diameter_ratio": res["throat_to_droplet_diameter_ratio"],
            "droplet_diameter_exit": res["droplet_diameter_exit"],
            "throat_to_droplet_diameter_ratio_exit": res["throat_to_droplet_diameter_ratio_exit"],
        })

    return pd.DataFrame(rows)


def solve_subcooled_state(fluid, h_in, dT_subcool, p_min, p_max):
    """
    Compute a subcooled liquid state at fixed enthalpy.

    The function finds the pressure p such that a state with
    (h = h_in, p = p) has a temperature that is dT_subcool
    below the saturation temperature at the same pressure.

    This is typically used to model throttling or valve processes
    where enthalpy is conserved and a minimum subcooling margin
    is enforced.

    Parameters
    ----------
    fluid : jxp.Fluid
        Working fluid object.
    h_in : float
        Target specific enthalpy [J/kg].
    dT_subcool : float
        Subcooling below saturation temperature [K].
    p_min, p_max : float
        Pressure bracket for root finding [Pa].

    Returns
    -------
    state : jxp.State
        Subcooled thermodynamic state satisfying the constraint.
    """

    # Residual enforces: T(h_in, p) = T_sat(p) - dT_subcool
    # for p>P_crit, T_sat(p) = T(p,rho_sat)
    def residual(p):
        if p < fluid.critical_point.p:
            st_sat = fluid.get_state(jxp.PQ_INPUTS, p, 0.0)     # saturated liquid at p
        else:
            st_sat = fluid.get_state(jxp.DmassP_INPUTS, fluid.critical_point.rhomass, p)     # saturated liquid at p
        st = fluid.get_state(jxp.HmassP_INPUTS, h_in, p)   # candidate subcooled state
        return (st_sat.T - dT_subcool) - st.T

    try:
        p = brentq(residual, p_min, p_max, xtol=1e-4)
    except Exception:
        raise RuntimeError(
            f"Could not find subcooled throttling state for {fluid.name}"
        )

    return fluid.get_state(jxp.HmassP_INPUTS, h_in, p)


def find_choked_state(fluid, h0, s0, T_min):
    """
    Determine choked-flow conditions for a given stagnation state.

    The function computes the velocity v* that maximizes the
    one-dimensional mass flux G = rho * v under isentropic expansion
    from the stagnation state (h0, s0).

    This corresponds to sonic (choked) conditions in a nozzle or
    turbine throat.

    Parameters
    ----------
    fluid : jxp.Fluid
        Working fluid object.
    h0 : float
        Stagnation specific enthalpy [J/kg].
    s0 : float
        Stagnation specific entropy [J/(kg K)].
    T_min : float
        Minimum allowable temperature during expansion [K].

    Returns
    -------
    G_star : float
        Choked mass flux [kg/(m² s)].
    v_star : float
        Choked velocity [m/s].
    p_star : float
        Static pressure at choking [Pa].
    """

    # Objective: maximize rho * v  →  minimize its negative
    def neg_mass_flux(v):
        st = fluid.get_state(jxp.HmassSmass_INPUTS, h0 - 0.5 * v**2, s0)
        return -st.d * v

    # Check that the minimum temperature constraint is compatible
    st_min = fluid.get_state(jxp.SmassT_INPUTS, s0, T_min)
    if st_min.h > h0:
        raise ValueError(f"Turbine inlet temperature below minimum ({T_min} K)")

    # Maximum possible velocity from energy balance
    v_max = np.sqrt(2 * (h0 - st_min.h))

    # Bounded optimization for robustness
    try:
        sol = minimize_scalar(
            neg_mass_flux,
            bounds=(1.0, 0.99 * v_max),
            method="bounded",
        )
    except Exception:
        sol = minimize_scalar(
            neg_mass_flux,
            bounds=(1.0, 0.7 * v_max),
            method="bounded",
        )

    v_star = sol.x
    st_star = fluid.get_state(jxp.HmassSmass_INPUTS, h0 - 0.5 * v_star**2, s0)

    return st_star, st_star.d * v_star, v_star,


def compute_droplet_size(fluid, velocity, pressure, length, data):
    """
    Estimate droplet size and slip velocity at a given location along the
    nozzle, assuming uniform acceleration from rest over the given length.

    The model balances aerodynamic forces and surface tension using a
    Weber-number-based breakup criterion. It is applicable wherever the
    flow is reasonably described as liquid droplets carried by a
    continuous vapor phase (e.g. near the nozzle exit, at higher vapor
    quality); it is less applicable where the flow is still bubbly (e.g.
    near the throat, at low vapor quality, shortly after nucleation).

    Parameters
    ----------
    fluid : jxp.Fluid
        Working fluid object.
    velocity : float
        Flow velocity at the location of interest [m/s] (e.g. the choked
        throat velocity, or the nozzle exit velocity), used together with
        `length` to estimate the mean acceleration up to that point.
    pressure : float
        Static pressure at the location of interest [Pa], used to evaluate
        the saturated liquid/vapor properties there.
    length : float
        Distance over which the flow is assumed to accelerate uniformly
        from rest to `velocity` [m] (e.g. the convergent section length for
        the throat, or the full nozzle length for the exit).
    data : dict
        Model parameters, including:
        - weber_number_critical
        - droplet_drag_coefficient

    Returns
    -------
    D_drop : float
        Characteristic droplet diameter [m].
    v_slip : float
        Vapor-liquid slip velocity [m/s].
    """
    We = data["weber_number_critical"]
    Cd = data["droplet_drag_coefficient"]

    # Saturated liquid and vapor states at the location's pressure
    st_l = fluid.get_state(jxp.PQ_INPUTS, pressure, 0.0)
    st_v = fluid.get_state(jxp.PQ_INPUTS, pressure, 1.0)

    # Density difference driving breakup
    rholv = st_l.d - st_v.d

    # Mean axial acceleration from rest, over the given length
    acceleration = velocity**2 / (2 * length)

    # Weber-number-based droplet diameter
    D_drop = np.sqrt(
        3 * Cd * We * st_l.surface_tension / (4 * rholv * acceleration)
    )

    # Slip velocity from surface tension / inertia balance
    v_slip = np.sqrt(We * st_l.surface_tension / (st_v.d * D_drop))
    return D_drop, v_slip


def evaluate_cycle(fluid, data, TIT, TIQ, Tcond, A_throat=None):
    """
    Evaluate thermodynamic cycle for one operating condition.

    Returns
    -------
    dict with keys:
        - states
        - performance
    """

    # Ambient saturation pressure
    st_amb = fluid.get_state(jxp.QT_INPUTS, 0.0, data["ambient_temperature"])
    p_ambient = st_amb.p

    # 8 - Saturated liquid after condenser
    st_8 = fluid.get_state(jxp.QT_INPUTS, 0.0, Tcond)

    # 1 - Pump inlet
    st_1 = fluid.get_state(jxp.PT_INPUTS, st_8.p, st_8.T - data["subcooling_pump_inlet"])

    # 4 - Turbine inlet
    st_4 = fluid.get_state(jxp.QT_INPUTS, TIQ, TIT)

    # 3 - Heater outlet / throttle inlet
    p_min = st_1.p
    p_max = 2.0 * fluid.critical_point.p
    st_3 = solve_subcooled_state(fluid, st_4.h, data["subcooling_heater_outlet"], p_min, p_max)

    # 2 - Pump outlet
    p2 = st_3.p / (1.0 - data["pressure_loss_fraction_heater"])
    st_2s = fluid.get_state(jxp.PSmass_INPUTS, p2, st_1.s)
    h2 = st_1.h + (st_2s.h - st_1.h) / data["efficiency_pump"]
    st_2 = fluid.get_state(jxp.HmassP_INPUTS, h2, p2)

    # Sonic (choked) conditions at the nozzle/turbine throat
    st_star, G_star, v_star = find_choked_state(fluid, st_4.h, st_4.s, st_1.T)

    # 6 - Turbine outlet
    p6 = st_1.p / (1.0 - data["pressure_loss_fraction_condenser"])
    st_6s = fluid.get_state(jxp.PSmass_INPUTS, p6, st_4.s)
    dh_is = st_4.h - st_6s.h
    spouting_velocity = np.sqrt(2.0 * dh_is)
    h6 = st_4.h - dh_is * data["efficiency_turbine"]
    st_6 = fluid.get_state(jxp.HmassP_INPUTS, h6, p6)

    # 5 - Nozzle outlet (representative reaction)
    R = data["degree_of_reaction"]
    p5 = p6 + R * (st_4.p - p6)
    st_5s = fluid.get_state(jxp.PSmass_INPUTS, p5, st_4.s)
    h5 = st_4.h + (st_5s.h - st_4.h) * data["efficiency_nozzle"]
    st_5 = fluid.get_state(jxp.HmassP_INPUTS, h5, p5)
    nozzle_velocity = np.sqrt(2.0 * (st_4.h - st_5.h))
    nozzle_mach = nozzle_velocity / st_5.a

    # 7 - Saturated vapor in condenser
    st_7_sat = fluid.get_state(jxp.PQ_INPUTS, p6, 1.0)
    st_7 = st_7_sat if st_7_sat.h < st_6.h else st_6

    # pseudoqality at turbine outlet -> can be higher than 1 for superheated
    pseudoQ = (st_5.h - st_8.h) / (st_7_sat.h - st_8.h)

    # Cycle efficiency
    net_work = ((st_4.h - st_6.h) + (st_1.h - st_2.h))
    efficiency_cycle = net_work / (st_3.h - st_2.h)

    # Export states in order
    states = [st_1, st_2, st_3, st_4, st_5, st_6, st_7, st_8]

    def export(attr):
        return np.array([getattr(st, attr) if st is not None else np.nan for st in states])

    result = {
        "states": {
            "p": export("p"),
            "T": export("T"),
            "h": export("h"),
            "s": export("s"),
            "d": export("d"),
            "q": export("q"),
            "a": export("a"),
        },
        "ambient_pressure": p_ambient,
        "choking_pressure": st_star.p,
        "choking_velocity": v_star,
        "choking_mass_flux": G_star,
        "spouting_velocity": spouting_velocity,
        "nozzle_velocity": nozzle_velocity,
        "nozzle_mach_number": nozzle_mach,
        "efficiency_cycle": efficiency_cycle,
        "turbine_outlet_pseudo_quality": pseudoQ,
    }
    if A_throat:
        add_scale_to_result(result, A_throat)
    return result



# Reds colormap range used for sweep coloring below: the colormap's own low
# end is too close to white to read well, so colors are sampled from this
# narrower slice instead of the full [0, 1] range.
REDS_RANGE = (0.3, 0.9)


def reds_color(t):
    """Map a normalized value t in [0, 1] to the REDS_RANGE slice of the Reds colormap."""
    low, high = REDS_RANGE
    return mpl.cm.Reds(low + (high - low) * t)


def plot_fluid_Ts(fluid_name, results, out_dir, savefig=True, y_margin=20.0, labels=None):
    """
    Plot a T-s diagram for one operating condition, or overlay several.

    Backward compatible with functions.py's plot_fluid_Ts: pass a single
    result dict (as returned by rig_sizing()/evaluate_cycle()) to plot one
    cycle exactly as before (single blue line, no legend). Pass a list of
    result dicts to overlay several cycles on the same phase diagram: each
    curve's color comes from the Reds colormap, scaled by its nozzle inlet
    temperature (lighter red for cooler, darker red for hotter), with a
    legend (rather than a colorbar) showing "TIT = <value> degC" per curve.

    Parameters
    ----------
    fluid_name : str
    results : dict or list of dict
        A single result, or a list of results to overlay.
    out_dir : str
    y_margin : float, optional
        Margin [K] added below the condensing temperature and above the
        higher of the turbine inlet temperature(s) / critical temperature.
    labels : list of str, optional
        One legend label per entry in `results`, used only when `results`
        is a list. Defaults to "TIT = <value> degC" using the nozzle inlet
        temperature of each result.
    """
    single_result = isinstance(results, dict)
    result_list = [results] if single_result else list(results)

    if not single_result:
        # Color and (by default) label each curve by its nozzle inlet
        # temperature (the "turbine_inlet" state in this nozzle-only rig).
        T_inlet = [result["states"]["T"][i_turb_in] for result in result_list]
        norm = mpl.colors.Normalize(vmin=min(T_inlet), vmax=max(T_inlet))
        colors = [reds_color(norm(T)) for T in T_inlet]
        if labels is None:
            labels = [f"TIT = {T - 273.15:.0f} °C" for T in T_inlet]
        # Plot from the highest nozzle inlet temperature down to the lowest.
        plot_order = sorted(range(len(result_list)), key=lambda i: T_inlet[i], reverse=True)
    else:
        plot_order = [0]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlabel("Entropy [kJ/(kg K)]")
    ax.set_ylabel("Temperature [°C]")

    # Plot phase diagram
    fluid = jxp.Fluid(name=fluid_name)
    fluid.plot_phase_diagram(
        x_prop="s", y_prop="T", axes=ax, plot_quality_isolines=True
    )

    # Reference state for axis limits
    state0 = fluid.get_state(jxp.QT_INPUTS, 0.0, 273.15)

    for i in plot_order:
        result = result_list[i]
        states = result["states"]
        s = np.asarray(states["s"])
        T = np.asarray(states["T"])
        if single_result:
            ax.plot(s, T, marker="o", color="b", markersize=4)
        else:
            ax.plot(s, T, marker="o", color=colors[i], label=labels[i], markersize=4)

    if not single_result:
        ax.legend(loc="upper left", fontsize=9)

    # Temperature axis limits: below the (lowest) condensing temperature,
    # and above the (highest) turbine inlet temperature / the critical
    # temperature, so the full saturation dome is always visible.
    T_cond = min(result["states"]["T"][i_cond_liq] for result in result_list)
    T_tit_max = max(result["states"]["T"][i_turb_in] for result in result_list)
    y_bottom = (T_cond - y_margin) - 273.15
    y_top = (max(T_tit_max, fluid.critical_point.T) + y_margin) - 273.15

    # Adjust axes scale and limits
    graphics.scale_graphics_x(fig, scale=1e-3, mode="multiply")
    graphics.scale_graphics_y(fig, scale=-273.15, mode="add")
    ax.relim()
    ax.autoscale_view()
    ax.set_xlim(left=state0.s / 1e3)
    ax.set_ylim(bottom=y_bottom, top=y_top)
    fig.tight_layout(pad=1.0)

    if savefig:
        suffix = "" if single_result else "_sensitivity"
        fname = f"Ts_{fluid_name}{suffix}.svg"
        fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
        fname = f"Ts_{fluid_name}{suffix}.pdf"
        fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
        fname = f"Ts_{fluid_name}{suffix}.png"
        fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")

    return fig, ax


def plot_sensitivity_subplots(x_values, x_label, results, panels, out_dir, filename, ncols=4, savefig=True):
    """
    Plot several result quantities against a common sweep variable, one
    quantity per subplot, laid out in a grid.

    Parameters
    ----------
    x_values : array-like
        Sweep variable values (shared x-axis), same length/order as results.
    x_label : str
        Label for the (shared) x-axis.
    results : list of dict
        One result dict per sweep point (as returned by rig_sizing/
        evaluate_cycle), same length/order as x_values.
    panels : list of dict
        One entry per subplot, in the order the subplots are filled
        (row-major). Each entry needs a "label" (y-axis label) and either:
        - "key": a top-level key into each result dict, or
        - "get": a callable taking one result dict and returning the value
          (for nested values, e.g. a state property, or a derived quantity).
        An optional "scale" (default 1.0) is multiplied into the value,
        e.g. for unit conversions such as W -> kW or m -> mm.
    out_dir : str
    filename : str
        File name (without extension) used when savefig is True.
    ncols : int, optional
        Number of subplot columns; the number of rows follows from the
        number of panels.
    """

    x_values = np.asarray(x_values)
    n_panels = len(panels)
    nrows = int(np.ceil(n_panels / ncols))

    # Color each point by its sweep value using the Reds colormap (same
    # convention as the T-s diagram); the connecting line uses the darkest
    # shade in the series (highest sweep value).
    norm = mpl.colors.Normalize(vmin=x_values.min(), vmax=x_values.max())
    marker_colors = [reds_color(norm(x)) for x in x_values]
    line_color = "black"

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for i, panel in enumerate(panels):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        getter = panel.get("get", lambda result, key=panel.get("key"): result[key])
        scale = panel.get("scale", 1.0)
        y_values = np.array([getter(result) * scale for result in results])

        # Connecting line first (darkest red, no markers), so the
        # individually-colored markers plotted after it end up on top.
        ax.plot(x_values, y_values, color=line_color, zorder=1, linewidth=0.75)
        for x, y, color in zip(x_values, y_values, marker_colors):
            ax.plot(x, y, marker="o", color=color, linestyle="none", zorder=2)

        ax.set_xlabel(x_label)
        ax.set_ylabel(panel["label"])

    # Hide unused axes in the grid
    for i in range(n_panels, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].axis("off")

    fig.tight_layout(pad=1.0)

    if savefig:
        fig.savefig(os.path.join(out_dir, f"{filename}.svg"), bbox_inches="tight")
        fig.savefig(os.path.join(out_dir, f"{filename}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(out_dir, f"{filename}.png"), bbox_inches="tight")

    return fig, axes


def screen_fluid(fluid, config):
    """
    Perform basic thermodynamic feasibility checks.

    Returns
    -------
    ok : bool
    msg : str or None
        Error message if screening fails.
    """

    if jxp.coolprop.is_pure_substance(fluid._AS) is False:
        return False, "Not a pure substance"

    if fluid.critical_point.T < config["critical_temperature_min"]:
        return False, "Critical temperature too low"

    if fluid.critical_point.T > config["critical_temperature_max"]:
        return False, "Critical temperature too high"

    if fluid._AS.Ttriple() > config["triple_point_temperature_max"]:
        return False, "Triple temperature too high"

    if fluid.get_state(jxp.QT_INPUTS, 0.0, config["ambient_temperature"]).p < config["ambient_pressure_min"]:
        return False, "Ambient saturation pressure too low"

    return True, None


def add_scale_to_result(result, A_throat):
    mass_flow_rate = result["choking_mass_flux"] * A_throat
    h = result["states"]["h"]
    d = result["states"]["d"]
    # Energy flows
    heater_power = mass_flow_rate * (h[i_heater_out] - h[i_pump_out])
    condenser_power = mass_flow_rate * (h[i_turb_out] - h[i_pump_in])
    turbine_power = mass_flow_rate * (h[i_turb_in] - h[i_turb_out])
    pump_power = mass_flow_rate * (h[i_pump_out] - h[i_pump_in])

    result.update({
            "throat_area": A_throat,
            "mass_flow_rate": mass_flow_rate,
            "heater_power": heater_power,
            "condenser_power": condenser_power,
            "turbine_power": turbine_power,
            "pump_power": pump_power,
        })

    return None


def rig_sizing(fluid, data):
    """
    Perform test rig sizing for a single fluid.
    Returns empty dict if screening or cycle evaluation fails.
    """

    if data["use_reduced_turbine_inlet_temperature"]:
        TIT = data["turbine_inlet_temperature_reduced"] * fluid.critical_point.T
    else:
        TIT = data["turbine_inlet_temperature"]
    TIQ = data["turbine_inlet_quality"]
    Tcond = data["condensing_temperature"]
    try:
        converged = True
        result = evaluate_cycle(fluid, data, TIT, TIQ, Tcond)
    except Exception as e:
        converged = False
        result = str(e)
        return converged, result

    # Heater power density for throat sizing
    h = result["states"]["h"]
    d = result["states"]["d"]
    q_heater = h[i_heater_out] - h[i_pump_out]
    mass_flow_rate = data["heater_power_target"] / q_heater
    A_throat = mass_flow_rate / result["choking_mass_flux"]

    # Each of the N parallel nozzles carries an equal share of the total
    # throat area; its diameter follows from A_per_nozzle = pi/4 * D^2.
    number_of_nozzles = data["number_of_nozzles"]
    throat_area_per_nozzle = A_throat / number_of_nozzles
    D_throat = np.sqrt(4.0 * throat_area_per_nozzle / np.pi)

    # Convergent (inlet-to-throat) and total (inlet-to-exit) nozzle lengths,
    # both scaled from the throat diameter.
    L_convergent = data["convergent_length_to_throat_diameter_ratio"] * D_throat
    L_divergent = data["divergent_length_to_throat_diameter_ratio"] * D_throat
    L_total = L_convergent + L_divergent

    # Droplet size at the throat: the flow here is still low-quality/bubbly
    # (nucleation has just started), so this is a weaker application of the
    # droplet breakup model, but is kept for reference/comparison.
    v_star = result["choking_velocity"]
    p_star = result["choking_pressure"]
    D_drop_throat, v_slip_throat = compute_droplet_size(fluid, v_star, p_star, L_convergent, data)

    # Droplet size at the nozzle exit: by here the flow is at substantially
    # higher void fraction (more likely mist/droplet flow), so this is the
    # more physically appropriate application of the breakup model. The
    # acceleration is estimated over the entire nozzle (inlet to exit),
    # consistent with nozzle_velocity being computed from the same
    # (v=0) stagnation inlet reference as the throat velocity.
    v_exit = result["nozzle_velocity"]
    p_exit = result["states"]["p"][i_nozzle_out]
    D_drop_exit, v_slip_exit = compute_droplet_size(fluid, v_exit, p_exit, L_total, data)

    # add scale to the results
    add_scale_to_result(result, A_throat)

    # Turbine rotational speed
    dh_turb = h[i_turb_in] - h[i_turb_out]
    vol_flow = mass_flow_rate / d[i_turb_out]
    omega = data["turbine_specific_speed"] * dh_turb**0.75 / vol_flow**0.50
    result.update({
        "number_of_nozzles": number_of_nozzles,
        "throat_area_per_nozzle": throat_area_per_nozzle,
        "throat_diameter": D_throat,
        "droplet_diameter_throat": D_drop_throat,
        "throat_to_droplet_diameter_ratio": D_throat / D_drop_throat,
        "droplet_diameter_exit": D_drop_exit,
        "throat_to_droplet_diameter_ratio_exit": D_throat / D_drop_exit,
        "turbine_speed": omega * 30.0 / np.pi
        })
    if result["nozzle_mach_number"] < data["nozzle_mach_min"]:
        converged = False
        result = "Nozzle Mach lower than minimum"
    return converged, result


def print_fluid_status(status, fluid_name, message=None):
    """
    Print a single-line status message for one fluid.
    """

    status_fmt = f"[{status:<4}]"

    if message is None:
        print(f"{status_fmt} {fluid_name}")
        return

    # Sanitize message to a single line
    text = " ".join(str(message).split())

    # Trim to 120 characters
    if len(text) > 100:
        text = text[:97] + "..."

    print(f"{status_fmt} {fluid_name:<20}: {text}")


def parametric_study(fluid, config, A_th, out_dir, savefig=True):
    fluid_name = fluid.name
    TITs = np.linspace(*config["turbine_inlet_temperature_reduced_sweep"]) * fluid.critical_point.T
    TIQs = np.linspace(*config["turbine_inlet_quality_sweep"])
    for n, Tcond in enumerate(config["condensing_temperatures_sweep"]):
        Qij = np.zeros((TITs.shape[0], TIQs.shape[0])) + np.nan
        xij = Qij + 0.0
        Maij = Qij + 0.0
        Tij = Qij + 0.0
        Prij = Qij + 0.0
        sij = Qij + 0.0
        for i, TIT in enumerate(TITs):
            for j, TIQ in enumerate(TIQs):
                try:
                    result = evaluate_cycle(fluid, config, TIT, TIQ, Tcond, A_throat=A_th)
                    Qij[i, j] = result["heater_power"]/1000  # kW
                    Maij[i, j] = result["nozzle_mach_number"]
                    Tij[i, j] = result["states"]["T"][i_turb_in]
                    Prij[i, j] = result["states"]["p"][i_heater_out]/fluid.critical_point.p
                    sij[i, j] = result["states"]["s"][i_turb_in]
                    xij[i, j] = result["turbine_outlet_pseudo_quality"]
                except Exception as e:
                    print(e)
                    pass
        for Z, Zname, Zlevels in zip([Qij, Maij],
                                   ['Heater power [kW]', 'Mach'],
                                   [np.arange(30, 100, 10), np.arange(1.4, 2.3, 0.1)]):
            fig, ax = plt.subplots(figsize=(5, 4))

            ax.set_xlabel("Entropy [kJ/(kg K)]")
            ax.set_ylabel("Temperature [°C]")
            #ax.set_title(fluid.name+" "+Zname+" (Tcond=%.2f °C)"%(Tcond-273.15))

            # Plot phase diagram
            fluid_plot = jxp.Fluid(name=fluid_name)
            fluid_plot.plot_phase_diagram(
                x_prop="s", y_prop="T", axes=ax, plot_quality_isolines=False
            )
            cmap = mpl.cm.plasma
            norm = mpl.colors.BoundaryNorm(Zlevels, cmap.N, extend='both')
            contours = ax.contourf(sij, Tij, Z, levels=Zlevels, cmap=cmap, norm=norm, extend='both')
            cont1 = ax.contour(sij, Tij, Qij, levels=[90.0], colors='k', linewidths=2, linestyles=':')
            cont2 = ax.contour(sij, Tij, xij, levels=[1.0], colors='darkblue', linewidths=2, linestyles='-')
            cont3 = ax.contour(sij, Tij, Prij, levels=[1.0], colors='darkred', linewidths=2, linestyles='-.')
            line_90kW = Line2D([0], [0], color='k', lw=2, linestyle=':')
            line_Q5 = Line2D([0], [0], color='darkblue', lw=2, linestyle='-')
            line_Pcrit = Line2D([0], [0], color='darkred', lw=2, linestyle='--')
            ax.contourf(sij, Tij, Prij, levels=[1.0, 1e6], colors='white')
            ax.contourf(sij, Tij, Qij, levels=[90.0, 1e6], colors='white')
            graphics.scale_graphics_x(fig, scale=1e-3, mode="multiply")
            graphics.scale_graphics_y(fig, scale=-273.15, mode="add")
            ax.relim()
            ax.autoscale_view()
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             ax=ax,  # orientation='vertical',
             label=Zname)
            ax.set_xlim(left=np.nanmin(sij/1000)-0.01)
            ax.set_ylim(bottom=100, top=170)
            ax.legend(
                [line_90kW, line_Q5, line_Pcrit],
                ["W$_{heater}$ = 90 kW", "Q$_5$ = 1", "P$_3$ = P$_{crit}$"],
                loc='upper left',
            )
            fig.tight_layout(pad=1.0)
            if savefig:
                fname = f"Ts_{fluid_name}_{fig.number}.svg"
                fdir = os.path.join(out_dir, fname)
                fig.savefig(fdir, bbox_inches="tight")
                fname = f"Ts_{fluid_name}_{fig.number}.pdf"
                fdir = os.path.join(out_dir, fname)
                fig.savefig(fdir, bbox_inches="tight")

    return None


# Units for the scalar quantities of a rig_sizing()/evaluate_cycle() result,
# used by print_results() below. All quantities are in base SI units (see
# module docstring); "-" marks a dimensionless quantity. Ordered roughly
# from overall rig performance down to nozzle geometry/droplet sizing.
RESULT_UNITS = {
    "mass_flow_rate": "kg/s",
    "heater_power": "W",
    "condenser_power": "W",
    "pump_power": "W",
    "turbine_power": "W",
    "efficiency_cycle": "-",
    "ambient_pressure": "Pa",
    "choking_pressure": "Pa",
    "choking_velocity": "m/s",
    "choking_mass_flux": "kg/(m2 s)",
    "spouting_velocity": "m/s",
    "nozzle_velocity": "m/s",
    "nozzle_mach_number": "-",
    "turbine_outlet_pseudo_quality": "-",
    "number_of_nozzles": "-",
    "throat_area": "m2",
    "throat_area_per_nozzle": "m2",
    "throat_diameter": "m",
    "droplet_diameter_throat": "m",
    "throat_to_droplet_diameter_ratio": "-",
    "droplet_diameter_exit": "m",
    "throat_to_droplet_diameter_ratio_exit": "-",
    "turbine_speed": "rpm",
}

# Units for the per-state properties stored under result["states"], used by
# print_results() to print the thermodynamic-state table.
STATE_UNITS = {
    "p": "Pa",
    "T": "K",
    "h": "J/kg",
    "s": "J/(kg K)",
    "d": "kg/m3",
    "q": "-",
    "a": "m/s",
}

# Display-only unit conversions used by print_results() below: displayed
# value = raw * scale + offset, in the given unit. RESULT_UNITS/STATE_UNITS
# above still describe the true (SI) units of the values stored in a
# result dict; these overrides only affect how print_results() formats them
# for a human-readable report (e.g. Pa -> kPa, K -> degC).
RESULT_DISPLAY_OVERRIDES = {
    "heater_power": ("kW", 1e-3, 0.0),
    "condenser_power": ("kW", 1e-3, 0.0),
    "pump_power": ("kW", 1e-3, 0.0),
    "turbine_power": ("kW", 1e-3, 0.0),
    "ambient_pressure": ("kPa", 1e-3, 0.0),
    "choking_pressure": ("kPa", 1e-3, 0.0),
    "throat_area": ("mm2", 1e6, 0.0),
    "throat_area_per_nozzle": ("mm2", 1e6, 0.0),
    "throat_diameter": ("mm", 1e3, 0.0),
    "droplet_diameter_throat": ("um", 1e6, 0.0),
    "droplet_diameter_exit": ("um", 1e6, 0.0),
}

STATE_DISPLAY_OVERRIDES = {
    "p": ("kPa", 1e-3, 0.0),
    "T": ("degC", 1.0, -273.15),
    "h": ("kJ/kg", 1e-3, 0.0),
    "s": ("kJ/(kg K)", 1e-3, 0.0),
}

# Fixed decimal places used to print each state property in print_results(),
# so every value in a column lines up at the same width, including zeros
# (printed as e.g. "0.00", not "0").
STATE_DECIMALS = {
    "p": 2,
    "T": 2,
    "d": 3,
    "q": 4,
    "h": 2,
    "s": 4,
}


def print_results(result, fluid_name=None, out_dir=None, filename=None, to_terminal=True):
    """
    Format the results of a single operating point (as returned by
    rig_sizing() or evaluate_cycle()) as a human-readable report: one line
    per scalar quantity, with its name, value and unit, followed by a table
    of the thermodynamic states.

    Parameters
    ----------
    result : dict
        A single rig_sizing()/evaluate_cycle() result.
    fluid_name : str, optional
        Fluid name, shown in the report header if given.
    out_dir : str, optional
        If given, the report is also written to a text file in this
        directory (the directory is created if it does not exist).
    filename : str, optional
        File name (without extension) used when out_dir is given. Defaults
        to "results_<fluid_name>" if fluid_name is given, else "results".
    to_terminal : bool, optional
        If True (default), also print the report to the terminal.

    Returns
    -------
    text : str
        The formatted report text.
    """
    lines = []
    header = "Rig sizing results" + (f" - {fluid_name}" if fluid_name else "")
    lines.append(header)
    lines.append("=" * len(header))

    name_width = max(len(key) for key in RESULT_UNITS)
    for key, unit in RESULT_UNITS.items():
        if key not in result:
            continue
        if key in RESULT_DISPLAY_OVERRIDES:
            unit, scale, offset = RESULT_DISPLAY_OVERRIDES[key]
            value = result[key] * scale + offset
        else:
            value = result[key]
        lines.append(f"{key:<{name_width}} = {value:>14.6g} {unit}")

    if "states" in result:
        lines.append("")
        lines.append("Thermodynamic states")
        lines.append("-" * len("Thermodynamic states"))
        props = ["p", "T", "d", "q", "h", "s"]
        prop_units = [STATE_DISPLAY_OVERRIDES.get(prop, (STATE_UNITS[prop],))[0] for prop in props]
        lines.append(
            f"{'state':<16}" + "".join(f"{prop + ' [' + unit + ']':>18}" for prop, unit in zip(props, prop_units))
        )
        for i, state_name in enumerate(STATE_MAPPING):
            row = f"{state_name:<16}"
            for prop in props:
                raw = result["states"][prop][i]
                if prop in STATE_DISPLAY_OVERRIDES:
                    _, scale, offset = STATE_DISPLAY_OVERRIDES[prop]
                    value = raw * scale + offset
                else:
                    value = raw
                row += f"{value:>18.{STATE_DECIMALS[prop]}f}"
            lines.append(row)

    text = "\n".join(lines)

    if to_terminal:
        print(text)

    if out_dir is not None:
        if filename is None:
            filename = f"results_{fluid_name}" if fluid_name else "results"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{filename}.txt"), "w") as fp:
            fp.write(text + "\n")

    return text
