"""
This file is a work in progress and is currently not functional. It is a builder that
will have the ability to build any of the 2DOF phases in Aviary, and should be used in
conjunction with a mission builder. When complete, this file is intended to be the only
phase builder for 2DOF phases.
"""

import dymos as dm
import openmdao.api as om

from aviary.variable_info.variables import Dynamic


class Phase2DOF:
    def __init__(
        self,
        aviary_options,
        initials=None,
        bounds=None,
        scalers=None,
        type=None,
        transcription=None,
        methods=None,
    ):
        self.aviary_options = aviary_options
        self.initials = initials
        self.bounds = bounds
        self.scalers = scalers
        self.type = type
        self.transcription = transcription
        self.methods = methods

    def pre_process(self):
        if self.type == None:
            raise om.AnalysisError(
                "You have not provided a type of phase. Providing a type of phase is"
                " required."
            )
        elif self.type not in ["takeoff", Dynamic.Mission.VELOCITY_RATE, "climb", "cruise", "descent"]:
            raise om.AnalysisError(
                f'You have provided an invalid phase type to the 2DOF phase builder.'
                ' Your phase type is {self.type} which is not on of ["takeoff", Dynamic.Mission.VELOCITY_RATE,'
                ' "climb", "cruise", "descent"].'
            )

        if (
            self.initials.get_val("fix_initial_mass")["value"]
            and self.initials.get_val("connect_initial_mass")["value"]
        ):
            raise om.AnalysisError(
                "You have both fixed and connected the initial phase mass, which is"
                " not allowed."
            )

        if (
            self.initials.get_val("fix_initial_time")["value"]
            and self.initials.get_val("connect_initial_time")["value"]
        ):
            raise om.AnalysisError(
                "You have both fixed and connected the initial phase time, which is not"
                " allowed."
            )

    def fix_phase_value(
        self, variable, units, val
    ):  # very thin wrapper around dymos to protect user from add parameter function

        self.phase.add_parameter(variable, opt=False, units=units, val=val)

    def build_phase(self):
        self.pre_process()

        self.phase = dm.Phase(
            ode_class=ODE2DOF,  # pass aero and prop methods here
            transcription=self.transcription,
            ode_init_kwargs=self.aviary_options,
        )

        self.phase.set_time_options(
            initial_bounds=self.bounds.get_val("initial_time_bounds")["value"],
            duration_bounds=self.bounds.get_val("duration_bounds")["value"],
            fix_initial=self.initials.get_val("fix_initial_time")["value"],
            input_initial=self.initials.get_val("connect_initial_time")["value"],
            units=self.bounds.get_val("initial_time_bounds")["units"],
            duration_ref=self.scalers.get_val("duration_ref")["value"],
            initial_ref=self.scalers.get_val("initial_time_ref")["value"],
        )

        self.phase.add_state(
            Dynamic.Mission.MASS,
            fix_initial=self.initials.get_val("fix_initial_mass")["value"],
            input_initial=self.initials.get_val("connect_initial_mass")["value"],
            fix_final=False,
            lower=self.bounds.get_val("mass_bounds")["value"][0],
            upper=self.bounds.get_val("mass_bounds")["value"][1],
            units=self.bounds.get_val("mass_bounds")["units"],
            rate_source=Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.Mission.MASS,
            ref=self.scalers.get_val("mass_ref")["value"],
            ref0=self.scalers.get_val("mass_ref0")["value"],
            defect_ref=self.scalers.get_val("mass_defect_ref")["value"],
        )
        self.phase.add_state(
            Dynamic.Mission.DISTANCE,
            fix_initial=self.initials.get_val("fix_initial_distance")["value"],
            input_initial=False,
            fix_final=False,  # should this be controllable?
            lower=self.bounds.get_val("distance_bounds")["value"][0],
            upper=self.bounds.get_val("distance_bounds")["value"][1],
            units=self.bounds.get_val("distance_bounds")["units"],
            rate_source=Dynamic.Mission.DISTANCE_RATE,
            targets=Dynamic.Mission.DISTANCE,
            ref=self.scalers.get_val("distance_ref")["value"],
            ref0=self.scalers.get_val("distance_ref0")["value"],
            defect_ref=self.scalers.get_val("distance_defect_ref")["value"],
        )

        if self.type == "takeoff" or self.type == "climb" or self.type == "descent":
            self.phase.add_state(
                Dynamic.Mission.ALTITUDE,
                fix_initial=self.initials.get_val("fix_initial_flight_condition")[
                    "value"
                ],
                input_initial=False,
                fix_final=False,
                lower=self.bounds.get_val("altitude_bounds")["value"][0],
                upper=self.bounds.get_val("altitude_bounds")["value"][1],
                units=self.bounds.get_val("altitude_bounds")["units"],
                rate_source=Dynamic.Mission.ALTITUDE_RATE,
                targets=Dynamic.Mission.ALTITUDE,
                ref=self.scalers.get_val("altitude_ref")["value"],
                ref0=self.scalers.get_val("altitude_ref0")["value"],
                defect_ref=self.scalers.get_val("altitude_defect_ref")["value"],
            )

        if self.type == "takeoff" or self.type == Dynamic.Mission.VELOCITY_RATE:
            self.phase.add_state(
                "TAS",
                fix_initial=self.initials.get_val(
                    "fix_initial_flight_condition")["value"],
                input_initial=False,
                fix_final=False,
                lower=self.bounds.get_val("TAS_bounds")["value"][0],
                upper=self.bounds.get_val("TAS_bounds")["value"][1],
                units=self.bounds.get_val("TAS_bounds")["units"],
                rate_source="TAS_rate",
                targets="TAS",
                ref=self.scalers.get_val("TAS_ref")["value"],
                ref0=self.scalers.get_val("TAS_ref0")["value"],
                defect_ref=self.scalers.get_val("TAS_defect_ref")["value"],
            )

        if self.type == "takeoff":
            self.phase.add_state(
                "angle_of_attack",  # alpha
                fix_initial=self.initials.get_val("fix_initial_angles")["value"],
                input_initial=False,
                fix_final=False,
                lower=self.bounds.get_val("angle_of_attack_bounds")["value"][0],
                upper=self.bounds.get_val("angle_of_attack_bounds")["value"][1],
                units=self.bounds.get_val("angle_of_attack_bounds")["units"],
                rate_source="angle_of_attack_rate",
                targets="angle_of_attack",
                ref=self.scalers.get_val("angle_of_attack_ref")["value"],
                ref0=self.scalers.get_val("angle_of_attack_ref0")["value"],
                defect_ref=self.scalers.get_val("angle_of_attack_defect_ref")["value"],
            )
            self.phase.add_state(
                Dynamic.Mission.FLIGHT_PATH_ANGLE,  # gamma
                fix_initial=self.initials.get_val("fix_initial_angles")["value"],
                input_initial=False,
                fix_final=False,
                lower=self.bounds.get_val("flight_path_angle_bounds")["value"][0],
                upper=self.bounds.get_val("flight_path_angle_bounds")["value"][1],
                units=self.bounds.get_val("flight_path_angle_bounds")["units"],
                rate_source=Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
                targets=Dynamic.Mission.FLIGHT_PATH_ANGLE,
                ref=self.scalers.get_val("angle_of_attack_ref")["value"],
                ref0=self.scalers.get_val("angle_of_attack_ref0")["value"],
                defect_ref=self.scalers.get_val("angle_of_attack_defect_ref")["value"],
            )

        self.phase.add_timeseries_output(
            Dynamic.Mission.MACH, output_name=Dynamic.Mission.MACH, units="unitless")
        self.phase.add_timeseries_output("EAS", output_name="EAS", units="unitless")
        self.phase.add_timeseries_output(
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL, output_name=Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL, units="unitless"
        )
        self.phase.add_timeseries_output(
            "fuselage_pitch", output_name="fuselage_pitch", units="unitless"
        )
        self.phase.add_timeseries_output(
            "angle_of_attack", output_name="angle_of_attack", units="unitless"
        )
        self.phase.add_timeseries_output(
            Dynamic.Mission.FLIGHT_PATH_ANGLE, output_name=Dynamic.Mission.FLIGHT_PATH_ANGLE, units="unitless"
        )
        self.phase.add_timeseries_output(
            "TAS_violation", output_name="TAS_violation", units="unitless"
        )
        self.phase.add_timeseries_output("TAS", output_name="TAS", units="unitless")
        self.phase.add_timeseries_output("CL", output_name="CL", units="unitless")
        self.phase.add_timeseries_output(
            Dynamic.Mission.THRUST_TOTAL, output_name=Dynamic.Mission.THRUST_TOTAL, units="unitless")
        self.phase.add_timeseries_output("CD", output_name="CD", units="unitless")
        self.phase.add_timeseries_output(
            Dynamic.Mission.LIFT, output_name=Dynamic.Mission.LIFT, units="unitless")
        self.phase.add_timeseries_output(
            "normal_force", output_name="normal_force", units="unitless"
        )

        return self.phase


def link_trajectory(phases=None, aviary_options=None):
    if phases == None:
        raise om.AnalysisError("You have not provided any phases to link_trajectory.")

    traj = dm.Trajectory()
    phase_names = []

    for phase in phases:
        num_usage = len([phase.type for check in phase_names if phase.type in check])
        new_name = phase.type + str(num_usage + 1)
        phase_names.append(new_name)

        traj.add_phase(new_name, phase, **aviary_options)

        if len(phase_names) > 1:
            traj.link_phases(
                phases=[phase_names[-1], phase_names[-2]],
                vars=linked_vars(phase_names[-1], phase_names[-2]),
            )

    return traj


def linked_vars(phase1, phase2):  # TODO: add other combinations of phases
    type1 = "".join((char for char in phase1 if not char.isdigit()))
    type2 = "".join((char for char in phase2 if not char.isdigit()))

    linked_vars = []

    if type1 == "takeoff" and type2 == Dynamic.Mission.VELOCITY_RATE:
        linked_vars = [Dynamic.Mission.DISTANCE, "time", Dynamic.Mission.MASS, "TAS"]
    elif type1 == Dynamic.Mission.VELOCITY_RATE and type2 == "climb":
        linked_vars = ["time", Dynamic.Mission.ALTITUDE,
                       Dynamic.Mission.MASS, Dynamic.Mission.DISTANCE]
    elif type1 == "climb" and type2 == "climb":
        linked_vars = ["time", Dynamic.Mission.ALTITUDE,
                       Dynamic.Mission.MASS, Dynamic.Mission.DISTANCE]
    elif type1 == "climb" and type2 == "cruise":
        linked_vars = []  # TODO: update this
    elif type1 == "cruise" and type2 == "descent":
        linked_vars = []  # TODO: update this
    elif type1 == "descent" and type2 == "descent":
        linked_vars = ["time", Dynamic.Mission.ALTITUDE,
                       "mass", Dynamic.Mission.DISTANCE]
    else:
        raise om.AnalysisError(
            "You have provided a phase order which is not yet supported."
        )

    return linked_vars
