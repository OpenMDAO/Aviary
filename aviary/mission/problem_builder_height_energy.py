from aviary.mission.flops_based.phases.build_landing import Landing
from aviary.mission.flops_based.phases.build_takeoff import Takeoff
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.functions import wrapped_convert_units
from aviary.utils.process_input_decks import update_GASP_options, initialization_guessing
from aviary.variable_info.enums import AnalysisScheme
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.variables import Aircraft, Mission, Dynamic, Settings


class AviaryProblemBuilder_HE():
    """
    A Height-Energy specific builder that customizes AviaryProblem() for use with
    height energy phases.
    """

    def initial_guesses(self, prob, engine_builders):
        # TODO: This should probably be moved to the set_initial_guesses() method in AviaryProblem class
        # Defines how the problem should build it's initial guesses for load_inputs()
        # this modifies mass_method, initialization_guesses, and aviary_values

        aviary_inputs = prob.aviary_inputs
        prob.mass_method = aviary_inputs.get_val(Settings.MASS_METHOD)

        if prob.mass_method is LegacyCode.GASP:
            # Support for GASP mass methods with HE.
            aviary_inputs = update_GASP_options(aviary_inputs)

        if engine_builders is None:
            engine_builders = build_engine_deck(aviary_inputs)
        prob.engine_builders = engine_builders

        prob.initialization_guesses = initialization_guessing(
            aviary_inputs, prob.initialization_guesses, prob.engine_builders)

        # Deal with missing defaults in phase info:
        if prob.pre_mission_info is None:
            prob.pre_mission_info = {'include_takeoff': True,
                                     'external_subsystems': []}
        if prob.post_mission_info is None:
            prob.post_mission_info = {'include_landing': True,
                                      'external_subsystems': []}

        # Commonly referenced values
        aviary_inputs.set_val(
            Mission.Summary.GROSS_MASS,
            val=prob.initialization_guesses['actual_takeoff_mass'],
            units='lbm'
        )

        if 'target_range' in prob.post_mission_info:
            aviary_inputs.set_val(Mission.Summary.RANGE, wrapped_convert_units(
                prob.post_mission_info['target_range'], 'NM'), units='NM')
            prob.require_range_residual = True
            prob.target_range = wrapped_convert_units(
                prob.post_mission_info['target_range'], 'NM')
        else:
            prob.require_range_residual = False
            # still instantiate target_range because it is used for default guesses
            # for phase comps
            prob.target_range = aviary_inputs.get_val(
                Mission.Design.RANGE, units='NM')

    def phase_info_default_location(self, prob):
        # Set the location of the default phase info for the EOM if no phase_info is specified

        if prob.analysis_scheme is AnalysisScheme.COLLOCATION:
            from aviary.interface.default_phase_info.height_energy import phase_info

        return phase_info

    def add_takeoff_systems(self, prob):
        # Initialize takeoff options
        takeoff_options = Takeoff(
            airport_altitude=0.,  # ft
            num_engines=prob.aviary_inputs.get_val(Aircraft.Engine.NUM_ENGINES)
        )

        # Build and add takeoff subsystem
        takeoff = takeoff_options.build_phase(False)
        prob.model.add_subsystem(
            'takeoff',
            takeoff,
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['mission:*'],
        )

    def add_post_mission_takeoff_systems(self, prob):

        first_flight_phase_name = list(prob.phase_info.keys())[0]
        connect_takeoff_to_climb = not prob.phase_info[first_flight_phase_name][
            'user_options'].get('add_initial_mass_constraint', True)

        if connect_takeoff_to_climb:
            prob.model.connect(Mission.Takeoff.FINAL_MASS,
                               f'traj.{first_flight_phase_name}.initial_states:mass')
            prob.model.connect(Mission.Takeoff.GROUND_DISTANCE,
                               f'traj.{first_flight_phase_name}.initial_states:distance')

            control_type_string = 'control_values'
            if prob.phase_info[first_flight_phase_name]['user_options'].get(
                    'use_polynomial_control', True):
                if not use_new_dymos_syntax:
                    control_type_string = 'polynomial_control_values'

            if prob.phase_info[first_flight_phase_name]['user_options'].get(
                    'optimize_mach', False):
                # Create an ExecComp to compute the difference in mach
                mach_diff_comp = om.ExecComp(
                    'mach_resid_for_connecting_takeoff = final_mach - initial_mach')
                prob.model.add_subsystem('mach_diff_comp', mach_diff_comp)

                # Connect the inputs to the mach difference component
                prob.model.connect(Mission.Takeoff.FINAL_MACH,
                                   'mach_diff_comp.final_mach')
                prob.model.connect(
                    f'traj.{first_flight_phase_name}.{control_type_string}:mach',
                    'mach_diff_comp.initial_mach', src_indices=[0])

                # Add constraint for mach difference
                prob.model.add_constraint(
                    'mach_diff_comp.mach_resid_for_connecting_takeoff', equals=0.0)

            if prob.phase_info[first_flight_phase_name]['user_options'].get(
                    'optimize_altitude', False):
                # Similar steps for altitude difference
                alt_diff_comp = om.ExecComp(
                    'altitude_resid_for_connecting_takeoff = final_altitude - initial_altitude', units='ft')
                prob.model.add_subsystem('alt_diff_comp', alt_diff_comp)

                prob.model.connect(Mission.Takeoff.FINAL_ALTITUDE,
                                   'alt_diff_comp.final_altitude')
                prob.model.connect(
                    f'traj.{first_flight_phase_name}.{control_type_string}:altitude',
                    'alt_diff_comp.initial_altitude', src_indices=[0])

                prob.model.add_constraint(
                    'alt_diff_comp.altitude_resid_for_connecting_takeoff', equals=0.0)

    def add_landing_systems(self, prob):

        landing_options = Landing(
            ref_wing_area=prob.aviary_inputs.get_val(
                Aircraft.Wing.AREA, units='ft**2'),
            Cl_max_ldg=prob.aviary_inputs.get_val(
                Mission.Landing.LIFT_COEFFICIENT_MAX)  # no units
        )

        landing = landing_options.build_phase(False)

        prob.model.add_subsystem(
            'landing', landing, promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['mission:*'])

        last_flight_phase_name = list(prob.phase_info.keys())[-1]

        control_type_string = 'control_values'
        if prob.phase_info[last_flight_phase_name]['user_options'].get(
                'use_polynomial_control', True):
            if not use_new_dymos_syntax:
                control_type_string = 'polynomial_control_values'

        last_regular_phase = prob.regular_phases[-1]
        prob.model.connect(f'traj.{last_regular_phase}.states:mass',
                           Mission.Landing.TOUCHDOWN_MASS, src_indices=[-1])
        prob.model.connect(f'traj.{last_regular_phase}.{control_type_string}:altitude',
                           Mission.Landing.INITIAL_ALTITUDE,
                           src_indices=[0])

