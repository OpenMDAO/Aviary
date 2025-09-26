"""
Authors: Eliot Aretskin-Hariton, Kenneth Moore, Jatin Soni
Multi Mission Optimization Example using Aviary.

In this example, a monolithic optimization is created by instantiating two aviary groups
using using multiple add_aviary_group() calls. Once those groups are setup and all of their
phases are linked together, we then promote GROSS_MASS, RANGE, and wing SWEEP from each of
those sub-groups (prob.model.mission1 and prob.model.mission2) up to prob.model so
the optimizer can control them both with a single value. The fuel_burn results from each
of the mission1 and mission2 are summed and weighted to create the objective function.
"""

import copy as copy

import aviary.api as av
from aviary.models.missions.height_energy_default import phase_info
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.enums import ProblemType
from aviary.variable_info.variables import Aircraft, Mission, Settings


def multi_mission_example():
    # fly the same mission twice with two different passenger loads
    phase_info_mission1 = copy.deepcopy(phase_info)
    phase_info_mission2 = copy.deepcopy(phase_info)

    # get large single aisle values
    aviary_inputs_mission1 = get_flops_inputs('LargeSingleAisle2FLOPS')
    # loading a CSV - a new ' create vehicle function'

    aviary_inputs_mission2 = copy.deepcopy(aviary_inputs_mission1)

    # Due to current limitations in Aviary's ability to detect user input vs. default values,
    # the only way to set an aircraft to zero passengers is by setting
    # TOTAL_PAYLOAD_MASS = X CARGO_MASS + 0 PASSENGER_PAYLOAD_MASS.
    # This zeros out passenger and baggage mass.
    # Due to issue #610, setting PASSENGER_PAYLOAD_MASS = 0 will not work yet.
    # aviary_inputs_deadhead.set_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 4077, 'lbm')

    aviary_inputs_mission2.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, 1, 'unitless')
    aviary_inputs_mission2.set_val(Aircraft.CrewPayload.NUM_TOURIST_CLASS, 1, 'unitless')
    aviary_inputs_mission2.set_val(Aircraft.CrewPayload.NUM_BUSINESS_CLASS, 0, 'unitless')
    aviary_inputs_mission2.set_val(Aircraft.CrewPayload.NUM_FIRST_CLASS, 0, 'unitless')

    # merged_meta_data = av.merge_hierarchies(meta_data1, meta_data2)
    # merge all metata data hierarchies here

    prob = av.AviaryProblem(problem_type=ProblemType.MULTI_MISSION)
    # set constraints in the background. Currently works with every objective type except Range.
    # can accept meta_data = merged_meta_data

    prob.add_aviary_group('mission1', aircraft=aviary_inputs_mission1, mission=phase_info_mission1)
    # This method calls load_inputs(), check_and_preprocess_inputs(), and combines meta data.
    # This can only accept an AviaryValues, .csv are not accepted. You can pass engine_builders
    # and problem_configurator as inputs.

    prob.add_aviary_group('mission2', aircraft=aviary_inputs_mission2, mission=phase_info_mission2)
    # Load aircraft in second configuration for same mission

    prob.build_model()
    # combines four basic commands
    # prob.add_pre_mission_systems()
    # prob.add_phases()
    # prob.add_post_mission_systems()
    # prob.link_phases()

    prob.promote_inputs(
        ['mission1', 'mission2'],
        [
            (Mission.Design.GROSS_MASS, 'Aircraft1:GROSS_MASS'),
            (Mission.Design.RANGE, 'Aircraft1:RANGE'),
            (Aircraft.Wing.SWEEP, 'Aircraft1:SWEEP'),
        ],
    )
    # Links key design variables to ensure both aircraft are modelled the same:
    # Design gross mass sizes things like the landing gear
    # Design range sizing things like the avionics system

    prob.add_design_var_default(
        'Aircraft1:GROSS_MASS',
        lower=10.0,
        upper=900e3,
        units='lbm',
        default_val=100000,
    )
    prob.add_design_var_default(
        'Aircraft1:SWEEP',
        lower=23.0,
        upper=27.0,
        units='deg',
        default_val=25,
    )
    # This both adds the design variable AND sets the default value. This value can be over-written after-setup using set_val.

    prob.add_composite_objective(
        ('mission1', Mission.Summary.FUEL_BURNED, 2),
        ('mission2', Mission.Summary.FUEL_BURNED, 1),
        ref=1,
    )
    # Adds an objective where mission 1 is flown 2x more times than mission2
    # Alternative way that users could specify the same objective:
    # prob.add_composite_objective_adv(missions=['mission1', 'mission2'], mission_weights=[2,1], outputs=[Mission.Summary.FUEL_BURNED],  ref=1)
    # TODO: MULTI_MISSION cannot handle RANGE objectives correctly at the moment.

    # optimizer and iteration limit are optional provided here
    # Note: IPOPT needs more iters than SNOPT.
    prob.add_driver('IPOPT', max_iter=70)
    prob.add_design_variables()

    prob.setup()
    # combines 2 basic commands:
    # prob.setup()
    # prob.set_initial_guesses()

    # set_val on OpenMDAO design variables can be placed here and will over-write all other defaults

    prob.set_design_range(('mission1', 'mission2'), range='Aircraft1:RANGE')
    # Determines the maximum design_range from both missions and sets that as the design range for both missions
    # to ensure that the avionics system is designed similarly for both aircraft

    # TODO: how to handle "aircraft that the user says are the same but are not the same i.e. wing design is different"

    prob.run_aviary_problem()

    return prob


def print_mission_outputs(prob, outputs, mission_names):
    for var, units in outputs:
        for mission in mission_names:
            try:
                value = prob.get_val(name=f'{mission}.{var}', units=units)
                print(f'{mission}.{var} ({units}), {value}')
            except:
                print(f'{var} was unavailable. Perhapse it has been promoted to the problem level?')
        print(' ')


if __name__ == '__main__':
    prob = multi_mission_example()
    objective = prob.get_val('composite_objective', units=None)

    printoutputs = [
        (Mission.Design.GROSS_MASS, 'lbm'),
        (Aircraft.Design.EMPTY_MASS, 'lbm'),
        (Aircraft.LandingGear.MAIN_GEAR_MASS, 'lbm'),
        (Aircraft.LandingGear.NOSE_GEAR_MASS, 'lbm'),
        (Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO, 'unitless'),
        (Aircraft.Avionics.MASS, 'lbm'),
        (Aircraft.Furnishings.MASS, 'lbm'),
        (Aircraft.CrewPayload.PASSENGER_SERVICE_MASS, 'lbm'),
        (Mission.Summary.GROSS_MASS, 'lbm'),
        (Mission.Summary.FUEL_BURNED, 'lbm'),
        (Aircraft.CrewPayload.PASSENGER_MASS, 'lbm'),
        (Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, 'lbm'),
        (Aircraft.CrewPayload.CARGO_MASS, 'lbm'),
        (Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 'lbm'),
    ]

    print_mission_outputs(prob, printoutputs, ('mission1', 'mission2'))

    print('Objective Value (unitless): ', objective)
    print('Aircraft1:GROSS_MASS (lbm)', prob.get_val('Aircraft1:GROSS_MASS', units='lbm'))
    print('Aircraft1:SWEEP (deg)', prob.get_val('Aircraft1:SWEEP', units='deg'))

    # If you notice differences in Aircraft.Design.EMPTY_MASS, your aircraft are not
    # mirroring eachother and there is some difference in configuration between the two aircraft.
    # Aircraft.Design.EMPTY_MASS is the final dry mass summation from pre-mission.
    # You can use the following OpenMDAO commends below to list out and compare
    # the each individual mass from every subsystem on the aircraft
    # prob.model.mission1.list_vars(val=True, units=True, print_arrays=False)
    # prob.model.mission2.list_vars(val=True, units=True, print_arrays=False)
