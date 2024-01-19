import numpy as np
from openmdao.utils.assert_utils import assert_near_equal

from aviary.variable_info.variables import Aircraft, Dynamic, Mission


def report_cruise(prob, remote=False, file=None):
    variables = [
        ("mass", "lbm", "mass_initial", "mass_final"),
        ("alpha", "deg", "aero_initial.alpha", "aero_final.alpha"),
        ("CL", "", "aero_initial.CL", "aero_final.CL"),
        ("CD", "", "aero_initial.CD", "aero_final.CD"),
        (Dynamic.Mission.THRUST_TOTAL, "lbf", "prop_initial.thrust_net_total",
         "prop_final.thrust_net_total"),
        (Dynamic.Mission.FUEL_FLOW_RATE_TOTAL, "lbm/s",
         "fuel_flow_initial", "fuel_flow_final"),
        ("TSFC", "1/h", "prop_initial.sfc", "prop_final.sfc"),
    ]

    print(60 * "=", file=file)
    print("{:^60}".format("cruise performance"), file=file)
    print("{:<21}   {:>15}   {:>15}".format("variable", "initial", "final"), file=file)
    print(60 * "-", file=file)
    for name, unstr, pathi, pathf in variables:
        units = None if unstr == "" else unstr
        vali = prob.get_val(f"cruise.{pathi}", units=units, get_remote=remote)[0]
        valf = prob.get_val(f"cruise.{pathf}", units=units, get_remote=remote)[0]
        print(f"{name:<10} {unstr:>10} | {vali:15.4f} | {valf:15.4f}", file=file)
    print(60 * "=", file=file)


def report_fuelburn(prob, remote=False, file=None):
    traj_phases = {
        "traj": [
            "groundroll",
            "rotation",
            "ascent",
            "accel",
            "climb1",
            "climb2",
            "cruise",
            "desc1",
            "desc2",
        ]
    }

    print(40 * "=", file=file)
    print("{:^40}".format("fuel burn in each phase"), file=file)
    print(40 * "-", file=file)
    for traj, phaselist in traj_phases.items():
        for phase in phaselist:
            if phase == 'cruise':
                vals = prob.get_val(
                    f"{traj}.{phase}.timeseries.mass",
                    units="lbm",
                    indices=[-1, 0],
                    get_remote=remote,
                )
            else:
                vals = prob.get_val(
                    f"{traj}.{phase}.states:mass",
                    units="lbm",
                    indices=[-1, 0],
                    get_remote=remote,
                )
            diff = np.diff(vals, axis=0)[0, 0]
            print(f"{phase:<12} {diff:>8.2f} lbm", file=file)
    print(40 * "=", file=file)


def report_gasp_comparison(prob, rtol=0.1, remote=False, file=None):
    # values from GASP output of large single aisle 1 at constant altitude using (I think) turbofan_23k_1
    expected_vals = [
        (
            "Design GTOW (lb)",
            175400.0,
            prob.get_val(Mission.Design.GROSS_MASS, get_remote=remote)[0],
        ),
        (
            "Actual GTOW (lb)",
            175400.0,
            prob.get_val(Mission.Summary.GROSS_MASS, get_remote=remote)[0],
        ),
        (
            "OEM (lb)",
            96348.0,
            prob.get_val(Aircraft.Design.OPERATING_MASS, get_remote=remote)[0],
        ),
        # mass at end of descent
        (
            "final mass (lb)",
            137348.0,
            prob.get_val(Mission.Landing.TOUCHDOWN_MASS, get_remote=remote)[0],
        ),
        # block fuel, includes reserves
        (
            "block fuel (lb)",
            43049.6,
            prob.get_val(Mission.Summary.TOTAL_FUEL_MASS, get_remote=remote)[0],
        ),
        (
            "landing distance (ft)",
            2980.0,
            prob.get_val('landing.'+Mission.Landing.GROUND_DISTANCE,
                         get_remote=remote)[0],
        ),
        (
            "flight distance (NM)",
            3675.0,
            prob.get_val(
                "traj.desc2.timeseries.states:distance", units="NM", get_remote=remote
            )[-1][0],
        ),
        # ROC @ 49 seconds to get from 456.3 ft to 500 ft minus liftoff time
        (
            "ascent duration (s)",
            49 + (500 - 465.3) / (2674.4 / 60) - 31.2,
            prob.get_val("traj.ascent.t_duration", get_remote=remote)[0],
        ),
        # AEO takeoff table
        (
            "gear retraction time (s)",
            37.3,
            prob.get_val("h_fit.t_init_gear", get_remote=remote)[0],
        ),
        (
            "flaps retraction time (s)",
            47.6,
            prob.get_val("h_fit.t_init_flaps", get_remote=remote)[0],
        ),
        # block time minus taxi time, no final descent
        (
            "flight time (hr)",
            8.295 - 10 / 60,
            prob.get_val("traj.desc2.timeseries.time", units="h", get_remote=remote)[
                -1
            ][0],
        ),
    ]

    print(80 * "=", file=file)
    print("{:^80}".format("comparison with reference GASP Large Single Aisle 1"), file=file)
    print(
        "{:<28} | {:>15} | {:>15} | {:>10}".format(
            "variable", "GASP vals", "Aviary actual", "rel. err."
        ),
        file=file,
    )
    print(80 * "-", file=file)
    results = {
        item[0]: {
            "desired": item[1],
            "actual": item[2],
            "rerr": (item[2] - item[1]) / item[1],
        }
        for item in expected_vals
    }
    for label, desired, actual in expected_vals:
        mark = ""
        rerr = (actual - desired) / desired
        if abs(rerr) > rtol:
            mark = "X"
        try:
            assert_near_equal(actual, desired, tolerance=rtol)
        except ValueError:
            mark = "X"
        print(
            f"{label:<28} | {desired:15.4f} | {actual:15.4f} | {rerr:> .2e} {mark}",
            file=file,
        )
    print(80 * "=", file=file)
    return results


def report_benchmark_comparison(
    prob, rtol=0.1, remote=False, file=None, size_engine=False, base='GASP'
):
    # values from GASP output of large single aisle 1 at constant altitude using (I think) turbofan_23k_1

    no_size = [
        175310.9512,
        175310.9512,
        96457.2320,
        137455.2320,
        42853.7192,
        2615.5252,
        3675.0000,
        17.3416,
        38.2458,
        48.2047,
        8.1171,
    ]
    yes_size = [
        175539.0746,
        175539.0746,
        96587.0999,
        137585.1013,
        42951.9734,
        2568.6457,
        3675.0000,
        21.4275,
        37.9418,
        49.8459,
        8.1145,
    ]
    # no_size_yes_pyCycle = [
    #     175394.6488,
    #     175394.6488,
    #     95820.0936,
    #     136997.9090,
    #     43394.7398,
    #     2560.4481,
    #     3675.0000,
    #     10.0131,
    #     33.2441,
    #     40.2571,
    #     9.5767,
    # ]

    FLOPS_base = [
        173370.02714369,
        0.000001,  # not exactly zero to avoid divide by zero errors
        95569.5191,
        136425.51908379,
        39944.5081,
        7732.7314,
        3378.7,
    ]
    if size_engine:
        check_val = yes_size
    elif not size_engine:
        check_val = no_size
    # elif not size_engine and pyCycle_used:
    #     check_val = no_size_yes_pyCycle
    if base == 'FLOPS':
        check_val = FLOPS_base
        distance_name = 'traj.descent.timeseries.states:distance'
        landing_dist_name = Mission.Landing.GROUND_DISTANCE
    else:
        distance_name = "traj.desc2.timeseries.states:distance"
        landing_dist_name = 'landing.'+Mission.Landing.GROUND_DISTANCE

    expected_vals = [
        (
            "Design GTOW (lb)",
            check_val[0],
            prob.get_val(Mission.Design.GROSS_MASS, get_remote=remote)[0],
        ),
        (
            "Actual GTOW (lb)",
            check_val[1],
            prob.get_val(Mission.Summary.GROSS_MASS, get_remote=remote)[0],
        ),
        (
            "OEW (lb)",
            check_val[2],
            prob.get_val(Aircraft.Design.OPERATING_MASS, get_remote=remote)[0],
        ),
        # mass at end of descent
        (
            "final mass (lb)",
            check_val[3],
            prob.get_val(Mission.Landing.TOUCHDOWN_MASS, get_remote=remote)[0],
        ),
        # block fuel, includes reserves
        (
            "block fuel (lb)",
            check_val[4],
            prob.get_val(Mission.Summary.TOTAL_FUEL_MASS, get_remote=remote)[0],
        ),
        (
            "landing distance (ft)",
            check_val[5],
            prob.get_val(landing_dist_name,
                         get_remote=remote)[0],
        ),
        (
            "flight distance (NM)",
            check_val[6],
            prob.get_val(
                distance_name, units="NM", get_remote=remote
            )[-1][0],
        ),
    ]

    if base != 'FLOPS':
        GASP_vals = [
            # ROC @ 49 seconds to get from 456.3 ft to 500 ft minus liftoff time
            (
                "ascent duration (s)",
                check_val[7],
                prob.get_val("traj.ascent.t_duration", get_remote=remote)[0],
            ),
            # AEO takeoff table
            (
                "gear retraction time (s)",
                check_val[8],
                prob.get_val("h_fit.t_init_gear", get_remote=remote)[0],
            ),
            (
                "flaps retraction time (s)",
                check_val[9],
                prob.get_val("h_fit.t_init_flaps", get_remote=remote)[0],
            ),
            # block time minus taxi time, no final descent
            (
                "flight time (hr)",
                check_val[10],
                prob.get_val("traj.desc2.timeseries.time", units="h", get_remote=remote)[
                    -1
                ][0],
            ),
        ]
        expected_vals = expected_vals + GASP_vals

    print(80 * "=", file=file)
    print(
        "{:^80}".format(
            "comparison with reference value (not a validated value, just a point to check against)"
        ),
        file=file,
    )
    print(
        "{:<28} | {:>15} | {:>15} | {:>10}".format(
            "variable", "original vals", "Aviary actual", "rel. err."
        ),
        file=file,
    )
    print(80 * "-", file=file)
    for label, desired, actual in expected_vals:
        mark = ""
        rerr = (actual - desired) / desired
        if abs(rerr) > rtol:
            mark = "X"
        try:
            assert_near_equal(actual, desired, tolerance=rtol)
        except ValueError:
            mark = "X"
        print(
            f"{label:<28} | {desired:15.4f} | {actual:15.4f} | {rerr:> .2e} {mark}",
            file=file,
        )
    print(80 * "=", file=file)
