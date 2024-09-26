import numpy as np
import openmdao.api as om
from openmdao.utils.units import convert_units

from aviary.constants import GRAV_METRIC_GASP, GRAV_ENGLISH_LBM, MU_TAKEOFF
from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft, Dynamic

LBF_TO_N = convert_units(1.0, 'lbf', 'N')


class UnsteadySolvedEOM(om.ExplicitComponent):
    """
    This class provides the 2-degree of freedom equations of motion for a flight condition.
    Given velocity, thrust, lift, drag, and alpha, it computes normal force,
    fuselage pitch angle, load factor, seconds passed per each meter of range covered,
    rate of change of true airspeed.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare("ground_roll", types=bool, default=False,
                             desc="True if the aircraft is confined to the ground. "
                                  "Removes altitude rate as an output and adjust "
                                  "the TAS rate equation.")

    def setup(self):
        nn = self.options["num_nodes"]

        # Inputs

        self.add_input(
            Dynamic.Mission.VELOCITY, shape=nn, desc="true air speed", units="m/s"
        )

        # TODO: This should probably be declared in Newtons, but the weight variable
        # is really a mass. This should be resolved with an adapter component that
        # uses gravity.
        self.add_input("mass", shape=nn, desc="aircraft mass", units="lbm")
        self.add_input(Dynamic.Mission.THRUST_TOTAL, shape=nn,
                       desc=Dynamic.Mission.THRUST_TOTAL, units="N")
        self.add_input(Dynamic.Mission.LIFT, shape=nn,
                       desc=Dynamic.Mission.LIFT, units="N")
        self.add_input(Dynamic.Mission.DRAG, shape=nn,
                       desc=Dynamic.Mission.DRAG, units="N")
        add_aviary_input(self, Aircraft.Wing.INCIDENCE, val=0, units="rad")
        self.add_input("alpha", val=np.zeros(
            nn), desc="angle of attack", units="rad")

        if not self.options["ground_roll"]:
            self.add_input(Dynamic.Mission.FLIGHT_PATH_ANGLE, val=np.zeros(
                nn), desc="flight path angle", units="rad")
            self.add_input("dh_dr", val=np.zeros(
                nn), desc="d(alt)/d(range)", units="m/distance_units")
            self.add_input("d2h_dr2", val=np.zeros(
                nn), desc="d(climb_rate)/d(range)", units="1/distance_units")

        # Outputs
        self.add_output("dt_dr", shape=nn, units="s/distance_units",
                        desc="Seconds passed per each meter of range covered.",
                        tags=['dymos.state_rate_source:time', 'dymos.state_units:s'])
        self.add_output("normal_force", val=np.ones(nn), units="N",
                        desc="normal forces")
        self.add_output("fuselage_pitch", val=np.ones(nn), units="rad",
                        desc="fuselage pitch angle")
        self.add_output("load_factor", val=np.ones(nn), units="unitless",
                        desc="load factor")
        self.add_output("dTAS_dt", shape=nn, units="m/s**2",
                        desc="rate of change of true airspeed")

        if not self.options["ground_roll"]:
            self.add_output("dgam_dt", shape=nn, units="rad/s",
                            desc="rate of change of flight path angle", )
            self.add_output("dgam_dt_approx", shape=nn, units="rad/s",
                            desc="approximate rate of change of flight path angle "
                                 "based on dh_dr and d2h_dr2")

    def setup_partials(self):
        nn = self.options["num_nodes"]
        ar = np.arange(nn, dtype=int)
        ground_roll = self.options["ground_roll"]

        self.declare_partials(
            of="dt_dr", wrt=Dynamic.Mission.VELOCITY, rows=ar, cols=ar
        )

        self.declare_partials(of=["normal_force", "dTAS_dt"],
                              wrt=[Dynamic.Mission.THRUST_TOTAL, Dynamic.Mission.DRAG,
                                   "mass", Dynamic.Mission.LIFT],
                              rows=ar, cols=ar)

        self.declare_partials(of="normal_force", wrt="mass",
                              rows=ar, cols=ar, val=LBF_TO_N * GRAV_ENGLISH_LBM)

        self.declare_partials(of="normal_force", wrt=Dynamic.Mission.LIFT,
                              rows=ar, cols=ar, val=-1.0)

        self.declare_partials(of="load_factor", wrt=[Dynamic.Mission.LIFT, "mass", Dynamic.Mission.THRUST_TOTAL],
                              rows=ar, cols=ar)

        self.declare_partials(of=["dTAS_dt", "normal_force", "load_factor"],
                              wrt=[Aircraft.Wing.INCIDENCE])

        self.declare_partials(of=["normal_force", "dTAS_dt"],
                              wrt=["alpha"],
                              rows=ar, cols=ar)

        self.declare_partials(of="fuselage_pitch",
                              wrt=["alpha"],
                              rows=ar, cols=ar, val=1.0)

        self.declare_partials(of="fuselage_pitch",
                              wrt=Aircraft.Wing.INCIDENCE,
                              val=-1.0)

        self.declare_partials(of="load_factor", wrt=["alpha"],
                              rows=ar, cols=ar)

        if not ground_roll:
            self.declare_partials(of="dt_dr", wrt=Dynamic.Mission.FLIGHT_PATH_ANGLE,
                                  rows=ar, cols=ar)

            self.declare_partials(of=["dgam_dt", "dgam_dt_approx"],
                                  wrt=[Dynamic.Mission.LIFT, "mass", Dynamic.Mission.THRUST_TOTAL,
                                       Dynamic.Mission.DRAG, "alpha", Dynamic.Mission.FLIGHT_PATH_ANGLE],
                                  rows=ar, cols=ar)

            self.declare_partials(of=["normal_force", "dTAS_dt"],
                                  wrt=[Dynamic.Mission.FLIGHT_PATH_ANGLE],
                                  rows=ar, cols=ar)

            self.declare_partials(
                of=["dgam_dt"], wrt=[Dynamic.Mission.VELOCITY], rows=ar, cols=ar
            )

            self.declare_partials(of="load_factor", wrt=[Dynamic.Mission.FLIGHT_PATH_ANGLE],
                                  rows=ar, cols=ar)

            self.declare_partials(of=["dgam_dt", "dgam_dt_approx"],
                                  wrt=[Dynamic.Mission.LIFT, "mass",
                                       Dynamic.Mission.THRUST_TOTAL, "alpha", Dynamic.Mission.FLIGHT_PATH_ANGLE],
                                  rows=ar, cols=ar)

            self.declare_partials(of="fuselage_pitch",
                                  wrt=[Dynamic.Mission.FLIGHT_PATH_ANGLE],
                                  rows=ar, cols=ar, val=1.0)

            self.declare_partials(
                of=["dgam_dt_approx"],
                wrt=["dh_dr", "d2h_dr2", Dynamic.Mission.VELOCITY],
                rows=ar,
                cols=ar,
            )

            self.declare_partials(of=["dgam_dt_approx", "dgam_dt"],
                                  wrt=[Aircraft.Wing.INCIDENCE])

    def compute(self, inputs, outputs):
        tas = inputs[Dynamic.Mission.VELOCITY]
        thrust = inputs[Dynamic.Mission.THRUST_TOTAL]
        # convert to newtons  # TODO: change this to use the units conversion
        weight = inputs["mass"] * GRAV_ENGLISH_LBM * LBF_TO_N
        drag = inputs[Dynamic.Mission.DRAG]
        lift = inputs[Dynamic.Mission.LIFT]
        alpha = inputs["alpha"]

        i_wing = inputs[Aircraft.Wing.INCIDENCE]

        g = GRAV_METRIC_GASP
        m = weight / g

        if self.options["ground_roll"]:
            mu = MU_TAKEOFF
            gamma = 0.0
        else:
            mu = 0.0
            gamma = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]
            dh_dr = inputs["dh_dr"]
            d2h_dr2 = inputs["d2h_dr2"]

        alpha_i = alpha - i_wing

        cgam = np.cos(gamma)
        sgam = np.sin(gamma)

        calpha_i = np.cos(alpha_i)
        salpha_i = np.sin(alpha_i)

        tcai = thrust * calpha_i
        tsai = thrust * salpha_i

        dr_dt = tas * cgam

        normal_force = weight - lift - tsai

        load_factor = (lift + tsai) / (weight * cgam)

        outputs["dt_dr"] = 1.0 / dr_dt

        outputs["normal_force"] = normal_force

        outputs["dTAS_dt"] = (tcai - drag - weight *
                              sgam - mu * normal_force) / m

        outputs["fuselage_pitch"] = gamma - i_wing + alpha

        outputs["load_factor"] = load_factor

        if not self.options["ground_roll"]:
            outputs["dgam_dt"] = (tsai + lift - weight * cgam) / (m * tas)
            dgam_dr = d2h_dr2 / (dh_dr ** 2 + 1)
            outputs["dgam_dt_approx"] = dgam_dr * dr_dt

    def compute_partials(self, inputs, partials):
        ground_roll = self.options["ground_roll"]

        thrust = inputs[Dynamic.Mission.THRUST_TOTAL]
        # convert to newtons  # TODO: change this to use the units conversion
        weight = inputs["mass"] * GRAV_ENGLISH_LBM * LBF_TO_N
        drag = inputs[Dynamic.Mission.DRAG]
        lift = inputs[Dynamic.Mission.LIFT]
        tas = inputs[Dynamic.Mission.VELOCITY]
        i_wing = inputs[Aircraft.Wing.INCIDENCE]
        alpha = inputs["alpha"]

        if self.options["ground_roll"]:
            mu = MU_TAKEOFF
            gamma = 0.0
        else:
            mu = 0.0
            gamma = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]
            dh_dr = inputs["dh_dr"]
            d2h_dr2 = inputs["d2h_dr2"]

        alpha_i = alpha - i_wing

        g = GRAV_METRIC_GASP
        m = weight / g

        mtas = m * tas
        mtas2 = mtas * mtas

        cgam = np.cos(gamma)
        sgam = np.sin(gamma)

        calpha_i = np.cos(alpha_i)
        salpha_i = np.sin(alpha_i)

        dr_dt = tas * cgam

        drdot_dtas = cgam
        drdot_dgam = -tas * sgam

        tcai = thrust * calpha_i
        tsai = thrust * salpha_i

        _f = tcai - drag - weight * sgam - mu * (weight - lift - tsai)

        partials["dt_dr", Dynamic.Mission.VELOCITY] = -cgam / dr_dt**2

        partials["dTAS_dt", Dynamic.Mission.THRUST_TOTAL] = calpha_i / \
            m + salpha_i / m * mu
        partials["dTAS_dt", Dynamic.Mission.DRAG] = -1. / m

        partials["dTAS_dt", "mass"] = \
            GRAV_ENGLISH_LBM * (LBF_TO_N * (-sgam - mu) / m - _f / (weight/LBF_TO_N * m))

        partials["dTAS_dt", Dynamic.Mission.LIFT] = mu / m
        partials["dTAS_dt", "alpha"] = -tsai / m + mu * tcai / m
        partials["dTAS_dt", Aircraft.Wing.INCIDENCE] = tsai / m - mu * tcai / m

        partials["normal_force", Dynamic.Mission.THRUST_TOTAL] = -salpha_i

        partials["load_factor", Dynamic.Mission.LIFT] = 1 / (weight * cgam)
        partials["load_factor", Dynamic.Mission.THRUST_TOTAL] = salpha_i / \
            (weight * cgam)
        partials["load_factor", "mass"] = \
            - (lift + tsai) / (weight**2/LBF_TO_N * cgam) * GRAV_ENGLISH_LBM

        partials["normal_force", "alpha"] = -tcai
        partials["normal_force", Aircraft.Wing.INCIDENCE] = tcai
        partials["load_factor", Aircraft.Wing.INCIDENCE] = -tcai / (weight * cgam)

        partials["load_factor", "alpha"] = tcai / (weight * cgam)

        if not ground_roll:
            partials["dt_dr", Dynamic.Mission.FLIGHT_PATH_ANGLE] = -drdot_dgam / dr_dt**2

            partials["dTAS_dt", Dynamic.Mission.FLIGHT_PATH_ANGLE] = -weight * cgam / m

            partials["dgam_dt", Dynamic.Mission.THRUST_TOTAL] = salpha_i / mtas
            partials["dgam_dt", Dynamic.Mission.LIFT] = 1. / mtas
            partials["dgam_dt", "mass"] = \
                GRAV_ENGLISH_LBM * (LBF_TO_N*cgam / (mtas) - (tsai +
                                    lift + weight*cgam)/(weight**2 / LBF_TO_N/g * tas))
            partials["dgam_dt", Dynamic.Mission.FLIGHT_PATH_ANGLE] = m * \
                tas * weight * sgam / mtas2
            partials["dgam_dt", "alpha"] = m * tas * tcai / mtas2
            partials["dgam_dt", Dynamic.Mission.VELOCITY] = (
                -m * (tsai + lift - weight * cgam) / mtas2
            )
            partials["dgam_dt", Aircraft.Wing.INCIDENCE] = -m * tas * tcai / mtas2

            dgam_dr = d2h_dr2 / (dh_dr ** 2 + 1)
            ddgam_dr_ddh_dr = -2 * dh_dr * d2h_dr2 / (dh_dr**2 + 1)**2
            ddgam_dr_dd2h_dr2 = 1. / (dh_dr ** 2 + 1)

            partials["dgam_dt_approx", "dh_dr"] = dr_dt * ddgam_dr_ddh_dr
            partials["dgam_dt_approx", "d2h_dr2"] = dr_dt * ddgam_dr_dd2h_dr2
            partials["dgam_dt_approx", Dynamic.Mission.VELOCITY] = dgam_dr * drdot_dtas
            partials["dgam_dt_approx",
                     Dynamic.Mission.FLIGHT_PATH_ANGLE] = dgam_dr * drdot_dgam
            partials["load_factor", Dynamic.Mission.FLIGHT_PATH_ANGLE] = (
                lift + tsai) / (weight * cgam**2) * sgam
