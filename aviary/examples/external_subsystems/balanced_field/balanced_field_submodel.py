"""
Group containing a submodel component with detailed landing.
"""
from copy import copy
import warnings

import openmdao.api as om

import aviary.api as av
from aviary.utils.preprocessors import preprocess_options
from aviary.variable_info.enums import EquationsOfMotion
from aviary.variable_info.variables import Aircraft, Mission, Settings


def create_balance_field_subprob(aviary_inputs, use_spoiler=False):

    subprob = create_prob(aviary_inputs, use_spoiler)

    comp = AviarySubmodelComp(
        problem=subprob,
        inputs=[
            Aircraft.Wing.AREA,
            Aircraft.Wing.ASPECT_RATIO,
            Aircraft.Wing.HEIGHT,
            Aircraft.Wing.SPAN,
            Mission.Takeoff.DRAG_COEFFICIENT_MIN,
            Mission.Takeoff.LIFT_COEFFICIENT_MAX,
            #('traj.takeoff_brake_release_to_engine_failure.states:mass', Mission.Summary.GROSS_MASS),
        ],
        outputs=[
            ('traj.takeoff_climb_gradient_to_obstacle.final_states:distance', 'distance_obstacle'),
        ]
    )

    return comp


def create_prob(aviary_inputs, use_spoiler=False):
    """
    Return a problem
    """
    aero_builder = av.CoreAerodynamicsBuilder(
        name='low_speed_aero', code_origin=av.LegacyCode.FLOPS
    )

    # fmt: off
    takeoff_subsystem_options = {
        'low_speed_aero': {
            'method': 'low_speed',
            'ground_altitude': 0.0,  # units='m'
            'angles_of_attack': [
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            ],  # units='deg'
            'lift_coefficients': [
                0.5178, 0.6, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25,
                1.35, 1.5, 1.6, 1.7, 1.8, 1.85, 1.9, 1.95,
            ],
            'drag_coefficients': [
                0.0674, 0.065, 0.065, 0.07, 0.072, 0.076, 0.084, 0.09,
                0.10, 0.11, 0.12, 0.13, 0.15, 0.16, 0.18, 0.20,
            ],
            'lift_coefficient_factor': 1.0,
            'drag_coefficient_factor': 1.0,
        }
    }
    # fmt: off

    # when using spoilers, add a few more options
    if use_spoiler:

        spoiler_drag = aviary_inputs.get_val(Mission.Takeoff.SPOILER_DRAG_COEFFICIENT)
        spoiler_lift = aviary_inputs.get_val(Mission.Takeoff.SPOILER_LIFT_COEFFICIENT)

        takeoff_subsystem_options = {
            'low_speed_aero': {
                **takeoff_subsystem_options['low_speed_aero'],
                'use_spoilers': True,
                'spoiler_drag_coefficient': spoiler_drag,
                'spoiler_lift_coefficient': spoiler_lift,
            }
        }

    # We also need propulsion analysis for takeoff and landing. No additional configuration
    # is needed for this builder
    engines = [av.build_engine_deck(aviary_inputs)]
    # Note that the aviary_inputs is already in a pre-processed state.
    prop_builder = av.CorePropulsionBuilder(engine_models=engines)

    balanced_field_user_options = av.AviaryValues()

    from aviary.utils.test_utils.default_subsystems import get_default_premission_subsystems
    from aviary.variable_info.functions import setup_model_options

    dto_build = av.BalancedFieldTrajectoryBuilder('balanced_field_traj',
                                                  core_subsystems=[aero_builder, prop_builder],
                                                  subsystem_options=takeoff_subsystem_options,
                                                  user_options=balanced_field_user_options)

    subprob = om.Problem()

    #ivc = om.IndepVarComp(Mission.Summary.GROSS_MASS, val=1.0, units='lbm')
    #subprob.model.add_subsystem('takeoff_mass_ivc', ivc, promotes=['*'])
    #subprob.model.connect(
    #    Mission.Summary.GROSS_MASS,
    #    "traj.takeoff_brake_release_to_engine_failure.states:mass",
    #    flat_src_indices=[0],
    #)

    # Instantiate the trajectory and add the phases
    traj = dto_build.build_trajectory(aviary_options=aviary_inputs, model=subprob.model)

    setup_model_options(subprob, aviary_inputs)

    # This is kind of janky, but we need these after the subproblem sets it up.
    subprob.aviary_inputs = aviary_inputs
    subprob.dto_build = dto_build

    return subprob


class AviarySubmodelComp(om.SubmodelComp):
    """
    We need to subclass so that we can set the initial conditions.
    """
    def setup(self):
        # suppress warnings:
        # "input variable '...' promoted using '*' was already promoted using 'aircraft:*'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', om.PromotionWarning)
            super().setup()


        sub = self._subprob
        av.set_aviary_initial_values(sub, sub.aviary_inputs)
        sub.dto_build.apply_initial_guesses(sub, 'traj')

        sub.final_setup()
