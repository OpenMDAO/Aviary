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

        prob.mass_method = prob.aviary_inputs.get_val(Settings.MASS_METHOD)

        if prob.mass_method is LegacyCode.GASP:
            # Support for GASP mass methods with HE.
            prob.aviary_inputs = update_GASP_options(prob.aviary_inputs)

        if engine_builders is None:
            engine_builders = build_engine_deck(prob.aviary_inputs)
        prob.engine_builders = engine_builders

        prob.initialization_guesses = initialization_guessing(
            prob.aviary_inputs, prob.initialization_guesses, prob.engine_builders)

        # Deal with missing defaults in phase info:
        if prob.pre_mission_info is None:
            prob.pre_mission_info = {'include_takeoff': True,
                                     'external_subsystems': []}
        if prob.post_mission_info is None:
            prob.post_mission_info = {'include_landing': True,
                                      'external_subsystems': []}

        # Commonly referenced values
        prob.aviary_inputs.set_val(
            Mission.Summary.GROSS_MASS, val=prob.initialization_guesses['actual_takeoff_mass'], units='lbm')

        if 'target_range' in prob.post_mission_info:
            prob.aviary_inputs.set_val(Mission.Summary.RANGE, wrapped_convert_units(
                prob.post_mission_info['target_range'], 'NM'), units='NM')
            prob.require_range_residual = True
            prob.target_range = wrapped_convert_units(
                prob.post_mission_info['target_range'], 'NM')
        else:
            prob.require_range_residual = False
            # still instantiate target_range because it is used for default guesses for phase comps
            prob.target_range = prob.aviary_inputs.get_val(
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

