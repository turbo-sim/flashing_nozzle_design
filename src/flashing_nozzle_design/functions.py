"""Main calculations for the flashing nozzle design project.

Cycle, choking, droplet-size and test-rig sizing routines for the EMPOWER
preliminary test rig model (flashing two-phase nozzle feeding a turbine
expander in an organic Rankine cycle).
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
i_turb_out = STATE_MAPPING["turbine_outlet"]


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
    """

    rows = []
    for fluid_name, res in result.items():
        rows.append({
            "fluid": fluid_name,
            "Pamb_kPa": res["p_amb"] / 1e3,
            "Pmax_kPa": max(res["states"]["p"]) / 1e3,
            "Tmax_C": max(res["states"]["T"]) - 273.15,
            "A_throat_mm2": res["A_throat"] * 1e6,
            "Ma_noz": res["nozzle_mach"],
            "kRPM": res["RPM"]/1000,
            "D_drop_ratio": res["D_drop_ratio"],
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


def compute_droplet_size(fluid, v_star, p_star, L_conv, data):
    """
    Estimate droplet size and slip velocity at the choking location.

    The model balances aerodynamic forces and surface tension
    using a Weber-number-based breakup criterion in a converging
    nozzle.

    Parameters
    ----------
    fluid : jxp.Fluid
        Working fluid object.
    v_star : float
        Choked velocity [m/s].
    p_star : float
        Choked pressure [Pa].
    data : dict
        Model parameters, including:
        - convergent_length
        - Weber
        - droplet_drag_coefficient

    Returns
    -------
    D_drop : float
        Characteristic droplet diameter [m].
    v_slip : float
        Vapor-liquid slip velocity [m/s].
    """
    We = data["Weber"]
    Cd = data["droplet_drag_coefficient"]

    # Saturated liquid and vapor states at choking pressure
    st_l = fluid.get_state(jxp.PQ_INPUTS, p_star, 0.0)
    st_v = fluid.get_state(jxp.PQ_INPUTS, p_star, 1.0)

    # Density difference driving breakup
    rholv = st_l.d - st_v.d

    # Mean axial acceleration in the convergent section
    acceleration = v_star**2 / (2 * L_conv)

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
    st_amb = fluid.get_state(jxp.QT_INPUTS, 0.0, data["Tamb"])
    Pamb = st_amb.p

    # 8 - Saturated liquid after condenser
    st_8 = fluid.get_state(jxp.QT_INPUTS, 0.0, Tcond)

    # 1 - Pump inlet
    st_1 = fluid.get_state(jxp.PT_INPUTS, st_8.p, st_8.T - data["dT_subcool_1"])

    # 4 - Turbine inlet
    st_4 = fluid.get_state(jxp.QT_INPUTS, TIQ, TIT)

    # 3 - Heater outlet / throttle inlet
    p_min = st_1.p
    p_max = 2.0 * fluid.critical_point.p
    st_3 = solve_subcooled_state(fluid, st_4.h, data["dT_subcool_3"], p_min, p_max)

    # 2 - Pump outlet
    p2 = st_3.p / (1.0 - data["heater_pressure_loss_fraction"])
    st_2s = fluid.get_state(jxp.PSmass_INPUTS, p2, st_1.s)
    h2 = st_1.h + (st_2s.h - st_1.h) / data["eta_pump"]
    st_2 = fluid.get_state(jxp.HmassP_INPUTS, h2, p2)

    # Sonic conditions
    st_star, G_star, v_star = find_choked_state(fluid, st_4.h, st_4.s, st_1.T)

    # 6 - Turbine outlet
    p6 = st_1.p / (1.0 - data["condenser_pressure_loss_fraction"])
    st_6s = fluid.get_state(jxp.PSmass_INPUTS, p6, st_4.s)
    dh_is = st_4.h - st_6s.h
    spouting_velocity = np.sqrt(2.0 * dh_is)
    h6 = st_4.h - dh_is * data["eta_turb"]
    st_6 = fluid.get_state(jxp.HmassP_INPUTS, h6, p6)

    # 5 - Nozzle outlet (representative reaction)
    R = data["reaction"]
    p5 = p6 + R * (st_4.p - p6)
    st_5s = fluid.get_state(jxp.PSmass_INPUTS, p5, st_4.s)
    h5 = st_4.h + (st_5s.h - st_4.h) * data["eta_noz"]
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
        "p_amb": Pamb,
        "p_star": st_star.p,
        "v_star": v_star,
        "G_star": G_star,
        "spouting_velocity": spouting_velocity,
        "nozzle_velocity": nozzle_velocity,
        "nozzle_mach": nozzle_mach,
        "efficiency_cycle": efficiency_cycle,
        "turbine_outlet_pseudoquality": pseudoQ,
    }
    if A_throat:
        add_scale_to_result(result, A_throat)
    return result


def plot_fluid_Ts(fluid_name, result, out_dir, savefig=True):
    """
    Plot T-s diagram for all operating conditions of one fluid.

    Parameters
    ----------
    fluid_name : str
    result : dict
        results[fluid_name] from the rig sizing / cycle evaluation.
    """

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlabel("Entropy [kJ/(kg K)]")
    ax.set_ylabel("Temperature [°C]")
    #ax.set_title(fluid_name)

    # Plot phase diagram
    fluid = jxp.Fluid(name=fluid_name)
    fluid.plot_phase_diagram(
        x_prop="s", y_prop="T", axes=ax, plot_quality_isolines=True
    )

    # Reference state for axis limits
    state0 = fluid.get_state(jxp.QT_INPUTS, 0.0, 273.15)

    states = result["states"]
    s = np.asarray(states["s"])
    T = np.asarray(states["T"])
    ax.plot(s, T, marker="o", color='b')

    # Adjust axes scale and limits
    #ax.legend(loc="upper left", fontsize=12)
    graphics.scale_graphics_x(fig, scale=1e-3, mode="multiply")
    graphics.scale_graphics_y(fig, scale=-273.15, mode="add")
    ax.relim()
    ax.autoscale_view()
    ax.set_xlim(left=state0.s / 1e3)
    ax.set_ylim(bottom=20, top=180)
    fig.tight_layout(pad=1.0)

    if savefig:
        fname = f"Ts_{fluid_name}.svg"
        fdir = os.path.join(out_dir, fname)
        fig.savefig(fdir, bbox_inches="tight")
        fname = f"Ts_{fluid_name}.pdf"
        fdir = os.path.join(out_dir, fname)
        fig.savefig(fdir, bbox_inches="tight")


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

    if fluid.critical_point.T < config["Tcrit_min"]:
        return False, "Critical temperature too low"

    if fluid.critical_point.T > config["Tcrit_max"]:
        return False, "Critical temperature too high"

    if fluid._AS.Ttriple() > config["Ttrip_max"]:
        return False, "Triple temperature too high"

    if fluid.get_state(jxp.QT_INPUTS, 0.0, config["Tamb"]).p < config["Pmin"]:
        return False, "Ambient saturation pressure too low"

    return True, None


def add_scale_to_result(result, A_throat):
    mass_flow = result["G_star"] * A_throat
    h = result["states"]["h"]
    d = result["states"]["d"]
    # Energy flows
    Q_heater = mass_flow * (h[i_heater_out] - h[i_pump_out])
    Q_condenser = mass_flow * (h[i_turb_out] - h[i_pump_in])
    W_turbine = mass_flow * (h[i_turb_in] - h[i_turb_out])
    W_pump = mass_flow * (h[i_pump_out] - h[i_pump_in])

    result.update({
            "A_throat": A_throat,
            "mass_flow": mass_flow,
            "Q_heater": Q_heater,
            "Q_condenser": Q_condenser,
            "W_turbine": W_turbine,
            "W_pump": W_pump,
        })

    return None


def rig_sizing(fluid, data):
    """
    Perform test rig sizing for a single fluid.
    Returns empty dict if screening or cycle evaluation fails.
    """

    if data["use_TITr"]:
        TIT = data["TITr_MPP"] * fluid.critical_point.T
    else:
        TIT = data["TIT_MPP"]
    TIQ = data["TIQ_MPP"]
    Tcond = data["Tcond_MPP"]
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
    mass_flow = data["Q_heater"] / q_heater
    A_throat = mass_flow / result["G_star"]
    D_throat = data["D_throat_to_sqrt_A_throat"] * A_throat**0.5
    L_convergent = data["L_to_D_throat_ratio"] * D_throat
    v_star = result["v_star"]
    p_star = result["p_star"]
    D_drop, v_slip = compute_droplet_size(fluid, v_star, p_star, L_convergent, data)
    # add scale to the results
    add_scale_to_result(result, A_throat)

    # Turbine rotational speed
    dh_turb = h[i_turb_in] - h[i_turb_out]
    vol_flow = mass_flow / d[i_turb_out]
    omega = data["specific_speed"] * dh_turb**0.75 / vol_flow**0.5
    result.update({
        "D_drop_ratio": D_throat / D_drop,
        "RPM": omega * 30.0 / np.pi
        })
    if result["nozzle_mach"] < data["Ma_noz_min"]:
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
    TITs = np.linspace(*config["TITr_space"]) * fluid.critical_point.T
    TIQs = np.linspace(*config["TIQ_space"])
    for n, Tcond in enumerate(config["Tcond_values"]):
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
                    Qij[i, j] = result["Q_heater"]/1000  # kW
                    Maij[i, j] = result["nozzle_mach"]
                    Tij[i, j] = result["states"]["T"][i_turb_in]
                    Prij[i, j] = result["states"]["p"][i_heater_out]/fluid.critical_point.p
                    sij[i, j] = result["states"]["s"][i_turb_in]
                    xij[i, j] = result["turbine_outlet_pseudoquality"]
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
