"""
Define utilities for building phases.

Classes
-------
PhaseBuilderBase : the interface for a phase builder
"""

from abc import ABC
from collections import namedtuple

import dymos as dm
import openmdao.api as om
import numpy as np

from aviary.mission.flops_based.ode.takeoff_ode import TakeoffODE
from aviary.mission.flops_based.phases.balanced_field_trajectory import BalancedFieldPhaseBuilder
from aviary.mission.initial_guess_builders import InitialGuess
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Dynamic
from aviary.variable_info.functions import setup_trajectory_params
from aviary.api import Aircraft, Dynamic, Mission

from aviary.mission.phase_builder_base import PhaseBuilderBase


_require_new_initial_guesses_meta_data_class_attr_ = namedtuple(
    '_require_new_initial_guesses_meta_data_class_attr_', ()
)


class BalancedFieldTrajectoryBuilder(ABC):
    """
    Define the interface for a balanced field trajectory builder.

    Attributes
    ----------
    name : str ('_unknown phase_')
        object label

    core_subsystems : (None)
        list of SubsystemBuilderBase objects that will be added to the phase ODE

    user_options : OptionsDictionary (<empty>)
        state/path constraint values and flags

    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

    ode_class : type (None)
        advanced: the type of system defining the ODE

    transcription : "Dymos transcription object" (None)
        advanced: an object providing the transcription technique of the
        optimal control problem

    subsystem_options : dict (None)
        dictionary of parameters to be passed to the subsystem builders

    default_name : str
        class attribute: derived type customization point; the default value
        for name

    default_ode_class : type
        class attribute: derived type customization point; the default value
        for ode_class used by build_phase

    default_options_class : type
        class attribute: derived type customization point; the default class
        containing the phase options options_dictionary

    is_analytic_phase : bool (False)
        class attribute: derived type customization point; if True, build_phase
        will return an AnalyticPhase instead of a Phase

    num_nodes : int (5)
        class attribute: derived type customization point; the default value
        for num_nodes used by build_phase, only for AnalyticPhases

    Methods
    -------
    build_phase
    make_default_transcription
    """

    __slots__ = (
        'name',
        'core_subsystems',
        'external_subsystems',
        'subsystem_options',
        'user_options',
        'initial_guesses',
        'ode_class',
        'transcription',
        'is_analytic_phase',
        'num_nodes',
        'meta_data',
        '_traj',
        '_phases',
    )

    MappedPhase = namedtuple('MappedPhase', ('phase', 'phase_builder'))

    _initial_guesses_meta_data_ = _require_new_initial_guesses_meta_data_class_attr_()

    default_name = '_unknown phase_'

    default_ode_class = TakeoffODE
    default_options_class = om.OptionsDictionary

    default_meta_data = _MetaData
    # endregion : derived type customization points

    def __init__(
        self,
        name=None,
        core_subsystems=None,
        external_subsystems=None,
        user_options=None,
        initial_guesses=None,
        subsystem_options=None,
        num_nodes=10,
        meta_data=None,
        traj=None,
    ):
        if name is None:
            name = self.default_name

        self.name = name

        if core_subsystems is None:
            core_subsystems = []
        if external_subsystems is None:
            external_subsystems = []

        self.core_subsystems = core_subsystems
        self.external_subsystems = external_subsystems

        if subsystem_options is None:
            subsystem_options = {}

        self.subsystem_options = subsystem_options

        self.user_options = self.default_options_class(user_options)

        if initial_guesses is None:
            initial_guesses = AviaryValues()

        self.initial_guesses = initial_guesses

        self.ode_class = self.default_ode_class
        self.num_nodes = num_nodes

        if external_subsystems is None:
            external_subsystems = []

        self.external_subsystems = external_subsystems

        if meta_data is None:
            meta_data = self.default_meta_data

        self.meta_data = meta_data

        self._traj = traj
        self._phases = {}

    def build_trajectory(
        self, *, aviary_options: AviaryValues, model: om.Group = None, traj: dm.Trajectory = None
    ) -> dm.Trajectory:
        """
        Return a new trajectory for detailed takeoff analysis.

        Call only after assigning phase builders for required phases.

        Parameters
        ----------
        aviary_options : AviaryValues
            collection of Aircraft/Mission specific options

        model : openmdao.api.Group (None)
            the model handling trajectory parameter setup; if `None`, trajectory
            parameter setup will not be handled

        traj : dymos.Trajectory (None)
            the trajectory to update; if `None`, a new trajetory will be updated and
            returned

        Returns
        -------
        the updated trajectory; if the specified trajectory is `None`, a new trajectory
        will be updated and returned

        Notes
        -----
        Do not modify this object or any of its referenced data between the call to
        `build_trajectory()` and the call to `apply_initial_guesses()`, or the behavior
        is undefined, no diagnostic required.
        """

        if traj is None:
            self._traj = traj = dm.Trajectory(parallel_phases=False)
            model.add_subsystem('traj', self._traj)

        # We're adding a balance comp, so use auto-ordering to put
        # systems in best order.
        self._traj.options['auto_order'] = True

        if aviary_options is None:
            aviary_options = AviaryValues()

        common_user_options = AviaryValues()
        common_user_options.set_val('max_duration', val=100.0, units='s')
        common_user_options.set_val('time_duration_ref', val=60.0, units='s')
        common_user_options.set_val('distance_max', val=10000.0, units='ft')
        common_user_options.set_val('max_velocity', val=200.0, units='kn')

        tobl_nl_solver = om.NewtonSolver(
            solve_subsystems=True, maxiter=100, iprint=2, atol=1.0e-6, rtol=1.0e-6
        )
        tobl_nl_solver.linesearch = om.BoundsEnforceLS()

        common_user_options.set_val('nonlinear_solver', val=tobl_nl_solver)
        common_user_options.set_val('linear_solver', val=om.DirectSolver())

        common_initial_guesses = AviaryValues()
        common_initial_guesses.set_val('time', [0.0, 30.0], 's')
        common_initial_guesses.set_val('distance', [0.0, 4100.0], 'ft')
        common_initial_guesses.set_val('velocity', [0.01, 150.0], 'kn')
        common_initial_guesses.set_val('throttle', 1.0)
        common_initial_guesses.set_val('angle_of_attack', 0.0, 'deg')

        gross_mass_units = 'lbm'
        gross_mass = aviary_options.get_val(Mission.Design.GROSS_MASS, gross_mass_units)
        common_initial_guesses.set_val('mass', gross_mass, gross_mass_units)

        #
        # First phase: Brake release to engine failure
        #
        # TODO: would like to be able to do this -  user_options=balanced_field_user_options | {'terminal_condition': 'VEF'}
        user_options = common_user_options.deepcopy()
        user_options.set_val('terminal_condition', 'VEF')
        initial_guesses = common_initial_guesses.deepcopy()
        initial_guesses.set_val('dV1', 3.0, 'kn')
        initial_guesses.set_val('dVEF', 1.0, 'kn')
        takeoff_brake_release_to_engine_failure_builder = BalancedFieldPhaseBuilder(
            'takeoff_brake_release_to_engine_failure',
            core_subsystems=self.core_subsystems,
            subsystem_options=self.subsystem_options,
            user_options=user_options,
            initial_guesses=initial_guesses,
        )
        self._add_phase(takeoff_brake_release_to_engine_failure_builder, aviary_options)

        #
        # Second Phase: Engine Failure to Decision Speed
        #
        # TODO: dymos PicardShooting shouldn't require initial guesses if connecting an initial state value
        #       In that case just assume the state value holds through the phase as an initial guess.
        user_options = common_user_options.deepcopy()
        user_options.set_val('terminal_condition', 'V1')
        initial_guesses = common_initial_guesses.deepcopy()
        initial_guesses.set_val('time', [30.0, 31.0], 's')
        initial_guesses.set_val('distance', [4100.0, 4500.0], 'ft')
        initial_guesses.set_val('velocity', [150.0, 151], 'kn')
        initial_guesses.set_val('throttle', 0.5)  # TODO: Don't do this hack, decrement num engines
        initial_guesses.set_val('angle_of_attack', 0.0, 'deg')
        initial_guesses.set_val('dV1', 3.0, 'kn')

        takeoff_engine_failure_to_v1_builder = BalancedFieldPhaseBuilder(
            'takeoff_engine_failure_to_v1',
            core_subsystems=self.core_subsystems,
            subsystem_options=self.subsystem_options,
            user_options=user_options,
            initial_guesses=initial_guesses,
        )

        self._add_phase(takeoff_engine_failure_to_v1_builder, aviary_options)

        #
        # Third Phase: Decision Speed to Rejected Takeoff Stop
        #
        # TODO: dymos PicardShooting shouldn't require initial guesses if connecting an initial state value
        #       In that case just assume the state value holds through the phase as an initial guess.
        user_options = common_user_options.deepcopy()
        user_options.set_val('terminal_condition', 'STOP')
        user_options.set_val('friction_key', Mission.Takeoff.BRAKING_FRICTION_COEFFICIENT)
        initial_guesses = common_initial_guesses.deepcopy()
        initial_guesses.set_val('time', [31.0, 10.0], 's')
        initial_guesses.set_val('distance', [4500.0, 6500.0], 'ft')
        initial_guesses.set_val('velocity', [151.0, 5.0], 'kn')
        initial_guesses.set_val('throttle', 0.0)
        initial_guesses.set_val('angle_of_attack', 0.0, 'deg')

        takeoff_v1_to_roll_stop = BalancedFieldPhaseBuilder(
            'takeoff_v1_to_roll_stop',
            core_subsystems=self.core_subsystems,
            subsystem_options=self.subsystem_options,
            user_options=user_options,
            initial_guesses=initial_guesses,
        )

        self._add_phase(takeoff_v1_to_roll_stop, aviary_options)

        #
        # Fourth Phase: Decision Speed to Rotation Speed
        #
        user_options = common_user_options.deepcopy()
        user_options.set_val('terminal_condition', 'VR')
        initial_guesses = common_initial_guesses.deepcopy()
        initial_guesses.set_val('time', [36.0, 1.0], 's')
        initial_guesses.set_val('distance', [4500.0, 5000.0], 'ft')
        initial_guesses.set_val('velocity', [151.0, 155.0], 'kn')
        initial_guesses.set_val('throttle', 0.5)
        initial_guesses.set_val('angle_of_attack', 0.0, 'deg')

        takeoff_v1_to_vr = BalancedFieldPhaseBuilder(
            'takeoff_v1_to_vr',
            core_subsystems=self.core_subsystems,
            subsystem_options=self.subsystem_options,
            user_options=user_options,
            initial_guesses=initial_guesses,
        )

        self._add_phase(takeoff_v1_to_vr, aviary_options)

        #
        # Fifth Phase: Rotate to liftoff
        #
        user_options = common_user_options.deepcopy()
        user_options.set_val('terminal_condition', 'LIFTOFF')
        user_options.set_val('pitch_control', 'ALPHA_RATE_FIXED')
        initial_guesses = common_initial_guesses.deepcopy()
        initial_guesses.set_val('time', [37.0, 7.0], 's')
        initial_guesses.set_val('distance', [5000.0, 5500.0], 'ft')
        initial_guesses.set_val('velocity', [155.0, 160.0], 'kn')
        initial_guesses.set_val('throttle', 0.5)
        initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, [0.0, 14.0], 'deg')
        initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK_RATE, 2.5, 'deg/s')

        takeoff_vr_to_liftoff = BalancedFieldPhaseBuilder(
            'takeoff_vr_to_liftoff',
            core_subsystems=self.core_subsystems,
            subsystem_options=self.subsystem_options,
            user_options=user_options,
            initial_guesses=initial_guesses,
        )

        self._add_phase(takeoff_vr_to_liftoff, aviary_options)

        #
        # Sixth Phase: Liftoff to Climb Gradient
        #
        user_options = common_user_options.deepcopy()
        user_options.set_val('terminal_condition', 'CLIMB_GRADIENT')
        user_options.set_val('pitch_control', 'ALPHA_RATE_FIXED')
        user_options.set_val('climbing', True)
        initial_guesses = common_initial_guesses.deepcopy()
        initial_guesses.set_val('time', [37.0, 1.0], 's')
        initial_guesses.set_val('distance', [5000.0, 5500.0], 'ft')
        initial_guesses.set_val('velocity', [155.0, 160.0], 'kn')
        initial_guesses.set_val('throttle', 0.5)
        initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, [10.0, 10.0], 'deg')
        initial_guesses.set_val(Dynamic.Mission.FLIGHT_PATH_ANGLE, [0.0, 0.5], 'deg')
        initial_guesses.set_val(Dynamic.Mission.ALTITUDE, [0.0, 0.5], 'ft')
        initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK_RATE, 2.0, 'deg/s')

        takeoff_liftoff_to_climb_gradient = BalancedFieldPhaseBuilder(
            'takeoff_liftoff_to_climb_gradient',
            core_subsystems=self.core_subsystems,
            subsystem_options=self.subsystem_options,
            user_options=user_options,
            initial_guesses=initial_guesses,
        )

        self._add_phase(takeoff_liftoff_to_climb_gradient, aviary_options)

        #
        # Seventh Phase: Climb Gradient to Obstacle
        #
        user_options = common_user_options.deepcopy()
        user_options.set_val('terminal_condition', 'OBSTACLE')
        user_options.set_val('pitch_control', 'GAMMA_FIXED')
        user_options.set_val('climbing', True)
        initial_guesses = common_initial_guesses.deepcopy()
        initial_guesses.set_val('time', [37.0, 5.0], 's')
        initial_guesses.set_val('distance', [5000.0, 5500.0], 'ft')
        initial_guesses.set_val('velocity', [155.0, 160.0], 'kn')
        initial_guesses.set_val('throttle', 0.5)
        initial_guesses.set_val(Dynamic.Mission.FLIGHT_PATH_ANGLE, [0.1, 0.1], 'deg')
        initial_guesses.set_val(Dynamic.Mission.ALTITUDE, [0.5, 35.0], 'ft')

        takeoff_climb_gradient_to_obstacle = BalancedFieldPhaseBuilder(
            'takeoff_climb_gradient_to_obstacle',
            core_subsystems=self.core_subsystems,
            subsystem_options=self.subsystem_options,
            user_options=user_options,
            initial_guesses=initial_guesses,
        )

        self._add_phase(takeoff_climb_gradient_to_obstacle, aviary_options)

        #
        # Add a trajectory-level balance to satisfy inter-phase relationships
        #

        bal_comp = self._traj.add_subsystem('balance_comp', om.BalanceComp())

        # The first balance sets dVEF such that the elapsed time from engine failure to V1 is one second.
        bal_comp.add_balance('dVEF', lhs_name='t_react', rhs_val=1.0, eq_units='s', units='kn')
        self._traj.connect(
            'balance_comp.dVEF', 'takeoff_brake_release_to_engine_failure.parameters:dVEF'
        )
        self._traj.connect('takeoff_engine_failure_to_v1.t_duration', 'balance_comp.t_react')

        # The second balance sets dV1 such that the distance at the end of the abort roll and the distance at obstacle clearance are the same
        bal_comp.add_balance(
            'dV1',
            lhs_name='distance_abort',
            rhs_name='distance_obstacle',
            eq_units='ft',
            units='kn',
        )
        self._traj.connect(
            'balance_comp.dV1',
            [
                'takeoff_brake_release_to_engine_failure.parameters:dV1',
                'takeoff_engine_failure_to_v1.parameters:dV1',
            ],
        )
        self._traj.connect(
            'takeoff_climb_gradient_to_obstacle.final_states:distance',
            'balance_comp.distance_obstacle',
        )
        self._traj.connect(
            'takeoff_v1_to_roll_stop.final_states:distance', 'balance_comp.distance_abort'
        )

        # The third balance sets the trigger ratio of V/V_stall for rotation such that
        # the climb gradient is achieved with V/V_stall = 1.2 (V2)

        bal_comp.add_balance(
            'VR_trigger_ratio',
            lhs_name='v_over_v_stall_at_climb_gradient',
            rhs_val=1.2,
            eq_units='unitless',
            units='unitless',
        )

        self._traj.connect(
            'balance_comp.VR_trigger_ratio',
            [
                'takeoff_brake_release_to_engine_failure.parameters:VR_ratio',
                'takeoff_engine_failure_to_v1.parameters:VR_ratio',
                'takeoff_v1_to_vr.parameters:VR_ratio',
            ],
        )

        self._traj.connect(
            'takeoff_liftoff_to_climb_gradient.timeseries.v_over_v_stall',
            'balance_comp.v_over_v_stall_at_climb_gradient',
            src_indices=om.slicer[-1, ...],
        )

        # Add a newton solver to the balanced field trajectory

        self._traj.nonlinear_solver = om.NewtonSolver(
            solve_subsystems=True, maxiter=100, iprint=2, atol=1.0e-6, rtol=1.0e-6
        )
        self._traj.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        self._traj.linear_solver = om.DirectSolver()

        # Press to Takeoff Sequence
        self._traj.link_phases(
            [
                'takeoff_brake_release_to_engine_failure',
                'takeoff_engine_failure_to_v1',
                'takeoff_v1_to_vr',
                'takeoff_vr_to_liftoff',
                'takeoff_liftoff_to_climb_gradient',
                'takeoff_climb_gradient_to_obstacle',
            ],
            vars=['time', 'distance', 'velocity', 'mass'],
            connected=True,
        )

        self._traj.link_phases(
            ['takeoff_v1_to_vr', 'takeoff_vr_to_liftoff', 'takeoff_liftoff_to_climb_gradient'],
            vars=[Dynamic.Vehicle.ANGLE_OF_ATTACK],
            connected=True,
        )

        self._traj.link_phases(
            ['takeoff_liftoff_to_climb_gradient', 'takeoff_climb_gradient_to_obstacle'],
            vars=[Dynamic.Mission.ALTITUDE, Dynamic.Mission.FLIGHT_PATH_ANGLE],
            connected=True,
        )

        # Abort Sequence
        self._traj.link_phases(
            ['takeoff_engine_failure_to_v1', 'takeoff_v1_to_roll_stop'],
            vars=['time', 'distance', 'velocity', 'mass'],
            connected=True,
        )

        if model is not None:
            phase_names = list(self._phases.keys())

            # This is a level 3 method that uses the default subsystems.
            # We need to create parameters for just the inputs we have.
            # They mostly come from the low-speed aero subsystem.

            kwargs = {'method': 'low_speed'}

            # TODO: Why is get_parameters different for different subsystems?
            # Do without blindly indexing.
            aero_builder = self.core_subsystems[0]

            params = aero_builder.get_parameters(aviary_options, **kwargs)

            # takeoff introduces this one.
            params[Mission.Takeoff.LIFT_COEFFICIENT_MAX] = {
                'shape': (1,),
                'static_target': True,
            }

            ext_params = {}
            for phase in self._phases.keys():
                ext_params[phase] = params

            setup_trajectory_params(
                model, self._traj, aviary_options, phase_names, external_parameters=ext_params
            )

        return self._traj

    def apply_initial_guesses(self, prob: om.Problem, traj_name):  # , phase: dm.Phase):
        """Apply any stored initial guesses; return a list of guesses not applied."""
        not_applied = {}
        phase_builder: PhaseBuilderBase = None

        for phase, phase_builder in self._phases.values():
            tmp = phase_builder.apply_initial_guesses(prob, traj_name, phase)

            if tmp:
                not_applied[phase_builder.name] = tmp

        return not_applied

    def _extra_ode_init_kwargs(self):
        """Return extra kwargs required for initializing the ODE."""
        return {}

    def _add_phase(self, phase_builder: PhaseBuilderBase, aviary_options: AviaryValues):
        name = phase_builder.name
        phase = phase_builder.build_phase(aviary_options)

        self._traj.add_phase(name, phase)

        self._phases[name] = self.MappedPhase(phase, phase_builder)
