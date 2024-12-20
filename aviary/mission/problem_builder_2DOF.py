from aviary.variable_info.variables import Settings
# from aviary.utils.functions import wrapped_convert_units
# from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.enums import AnalysisScheme
from aviary.utils.process_input_decks import update_GASP_options
from aviary.utils.process_input_decks import initialization_guessing
# from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Mission, Dynamic, Settings
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.functions import create_opts2vals, add_opts2vals, wrapped_convert_units


class AviaryProblemBuilder_2DOF():
    """
    A 2DOF specific builder that customizes AviaryProblem() for use with 
     two degree of freedom phases.
    """

    def initial_guesses(self, prob, engine_builders):
        # TODO: This should probably be moved to the set_initial_guesses() method in AviaryProblem class
        # Defines how the problem should build it's initial guesses for load_inputs()
        # this modifies mass_method, initialization_guesses, and aviary_values

        prob.mass_method = prob.aviary_inputs.get_val(Settings.MASS_METHOD)

        if engine_builders is None:
            engine_builders = build_engine_deck(prob.aviary_inputs)
        prob.engine_builders = engine_builders

        prob.aviary_inputs = update_GASP_options(prob.aviary_inputs)

        prob.initialization_guesses = initialization_guessing(
            prob.aviary_inputs, prob.initialization_guesses, prob.engine_builders)

        prob.aviary_inputs.set_val(Mission.Summary.CRUISE_MASS_FINAL,
                                   val=prob.initialization_guesses['cruise_mass_final'], units='lbm')
        prob.aviary_inputs.set_val(Mission.Summary.GROSS_MASS,
                                   val=prob.initialization_guesses['actual_takeoff_mass'], units='lbm')

        # Deal with missing defaults in phase info:
        if prob.pre_mission_info is None:
            prob.pre_mission_info = {'include_takeoff': True,
                                     'external_subsystems': []}
        if prob.post_mission_info is None:
            prob.post_mission_info = {'include_landing': True,
                                      'external_subsystems': []}

        # Commonly referenced values
        prob.cruise_alt = prob.aviary_inputs.get_val(
            Mission.Design.CRUISE_ALTITUDE, units='ft')
        prob.mass_defect = prob.aviary_inputs.get_val('mass_defect', units='lbm')

        prob.cruise_mass_final = prob.aviary_inputs.get_val(
            Mission.Summary.CRUISE_MASS_FINAL, units='lbm')

        if 'target_range' in prob.post_mission_info:
            prob.target_range = wrapped_convert_units(
                prob.post_mission_info['post_mission']['target_range'], 'NM')
            prob.aviary_inputs.set_val(Mission.Summary.RANGE,
                                       prob.target_range, units='NM')
        else:
            prob.target_range = prob.aviary_inputs.get_val(
                Mission.Design.RANGE, units='NM')
            prob.aviary_inputs.set_val(Mission.Summary.RANGE, prob.aviary_inputs.get_val(
                Mission.Design.RANGE, units='NM'), units='NM')
        prob.cruise_mach = prob.aviary_inputs.get_val(Mission.Design.MACH)
        prob.require_range_residual = True

    def phase_info_default_location(self, prob):
        # Set the location of the default phase info for the EOM if no phase_info is specified

        if prob.analysis_scheme is AnalysisScheme.COLLOCATION:
            from aviary.interface.default_phase_info.two_dof import phase_info
        elif prob.analysis_scheme is AnalysisScheme.SHOOTING:
            from aviary.interface.default_phase_info.two_dof_fiti import phase_info, \
                phase_info_parameterization
            phase_info, _ = phase_info_parameterization(
                phase_info, None, prob.aviary_inputs)

        return phase_info
