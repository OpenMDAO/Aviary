from aviary.variable_info.enums import SpeedType
from aviary.subsystems.propulsion.propulsion_builder import CorePropulsionBuilder
from aviary.subsystems.geometry.geometry_builder import CoreGeometryBuilder
from aviary.subsystems.mass.mass_builder import CoreMassBuilder
from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
from aviary.variable_info.enums import LegacyCode
import dymos as dm
import openmdao.api as om

from aviary.mission.flops_based.phases.two_dof_phase import TwoDOFPhase
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
from aviary.utils.process_input_decks import create_vehicle
from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings
from aviary.utils.functions import get_path


FLOPS = LegacyCode.FLOPS


throttle_max = 1.0
throttle_climb = 0.956
throttle_cruise = 0.930
throttle_idle = 0.0

prop = CorePropulsionBuilder('core_propulsion', BaseMetaData)
mass = CoreMassBuilder('core_mass', BaseMetaData, FLOPS)
aero = CoreAerodynamicsBuilder('core_aerodynamics', BaseMetaData, FLOPS)
geom = CoreGeometryBuilder('core_geometry', BaseMetaData, FLOPS)

subsystem_options = {'core_aerodynamics':
                     {'method': 'low_speed',
                      'ground_altitude': 0.,  # units='m'
                      'angles_of_attack': [
                          0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                          6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                          12.0, 13.0, 14.0, 15.0],  # units='deg'
                      'lift_coefficients': [
                          0.5178, 0.6, 0.75, 0.85, 0.95, 1.05,
                          1.15, 1.25, 1.35, 1.5, 1.6, 1.7,
                          1.8, 1.85, 1.9, 1.95],
                      'drag_coefficients': [
                          0.0674, 0.065, 0.065, 0.07, 0.072, 0.076,
                          0.084, 0.09, 0.10, 0.11, 0.12, 0.13,
                          0.15, 0.16, 0.18, 0.20],
                      'lift_coefficient_factor': 1.,
                      'drag_coefficient_factor': 1.}}

default_premission_subsystems = [prop, geom, aero, mass]
default_mission_subsystems = [aero, prop]

phase_info = {
    'rotation': {
        'user_options': {
            'num_segments': 2,
            'fix_initial': True,
            'throttle_setting': throttle_max,
            'input_speed_type': SpeedType.TAS,
            'ground_roll': True,
            'clean': False,
            'initial_ref': (1.e3, 'm'),
            'initial_bounds': ((50., 5000.), 'm'),
            'duration_ref': (1.e3, 'm'),
            'duration_bounds': ((50., 2000.), 'm'),
            'control_order': 1,
            'order': 3,
            'opt': True,
        },
        'subsystem_options': subsystem_options,
        'initial_guesses': {
            'times': [(0., 1000.), 'm'],
            'TAS': [(0., 100.), 'm/s'],
            'mass': [(175.e3, 175.e3), 'lbm'],
            'altitude': [(0., 0.), 'ft'],
        },
    },
}

phase_name = 'rotation'
base_phase_options = phase_info[phase_name]

# We need to exclude some things from the phase_options that we pass down
# to the phases. Intead of "popping" keys, we just create new outer dictionaries.

phase_options = {}
for key, val in base_phase_options.items():
    phase_options[key] = val

phase_options['user_options'] = {}
for key, val in base_phase_options['user_options'].items():
    phase_options['user_options'][key] = val

prob = om.Problem(model=om.Group())

traj = prob.model.add_subsystem('traj', dm.Trajectory())

aviary_inputs, initial_guesses = create_vehicle(AviaryValues())

engine_inputs = AviaryValues()
engine_inputs.set_val(Aircraft.Engine.DATA_FILE, get_path(
    'models/engines/turbofan_22k.deck'))
engine_mass = 6293.8
engine_mass_units = 'lbm'
engine_inputs.set_val(Aircraft.Engine.MASS, engine_mass, engine_mass_units)
engine_inputs.set_val(
    Aircraft.Engine.REFERENCE_MASS,
    engine_mass,
    engine_mass_units)
scaled_sls_thrust = 22200.5
scaled_sls_thrust_units = 'lbf'
engine_inputs.set_val(
    Aircraft.Engine.SCALED_SLS_THRUST, scaled_sls_thrust, scaled_sls_thrust_units)
engine_inputs.set_val(
    Aircraft.Engine.REFERENCE_SLS_THRUST, scaled_sls_thrust, scaled_sls_thrust_units)
engine_inputs.set_val(Aircraft.Engine.THRUST_REVERSERS_MASS_SCALER, 0.0)
num_engines = 2
engine_inputs.set_val(Aircraft.Engine.NUM_ENGINES, num_engines)
num_fuselage_engines = 0
engine_inputs.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, num_fuselage_engines)
num_wing_engines = num_engines
engine_inputs.set_val(Aircraft.Engine.NUM_WING_ENGINES, num_wing_engines)
engine_inputs.set_val(Aircraft.Engine.WING_LOCATIONS, 0.289682918)
engine_inputs.set_val(Aircraft.Engine.SCALE_MASS, True)
engine_inputs.set_val(Aircraft.Engine.MASS_SCALER, 1.15)
engine_inputs.set_val(Aircraft.Engine.SCALE_PERFORMANCE, True)
engine_inputs.set_val(Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER, 1.0)
engine_inputs.set_val(Aircraft.Engine.SUPERSONIC_FUEL_FLOW_SCALER, 1.0)
engine_inputs.set_val(
    Aircraft.Engine.FUEL_FLOW_SCALER_CONSTANT_TERM, 0.0)
engine_inputs.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_LINEAR_TERM, 1.0)
engine_inputs.set_val(Aircraft.Engine.CONSTANT_FUEL_CONSUMPTION, 0.0, units='lb/h')
engine_inputs.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.0)
engine_inputs.set_val(Aircraft.Engine.GENERATE_FLIGHT_IDLE, True)
engine_inputs.set_val(Aircraft.Engine.IGNORE_NEGATIVE_THRUST, False)
engine_inputs.set_val(Aircraft.Engine.FLIGHT_IDLE_THRUST_FRACTION, 0.0)
engine_inputs.set_val(Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION, 1.0)
engine_inputs.set_val(Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION, 0.08)
engine_inputs.set_val(Aircraft.Engine.GEOPOTENTIAL_ALT, False)
engine_inputs.set_val(Aircraft.Engine.INTERPOLATION_METHOD, 'slinear')

# Create engine model
engine = EngineDeck(name='engine',
                    options=engine_inputs
                    )
preprocess_propulsion(aviary_inputs, [engine])

phase_object = TwoDOFPhase.from_phase_info(
    phase_name, phase_options, default_mission_subsystems, meta_data=BaseMetaData)

phase = phase_object.build_phase(aviary_options=aviary_inputs)

traj.add_phase(phase_name, phase)

phase.add_objective('time', loc='final', ref=1e3)

prob.driver = om.pyOptSparseDriver()
prob.driver.declare_coloring(show_sparsity=False, show_summary=False)

prob.driver.options["optimizer"] = "SNOPT"

prob.driver.opt_settings["Major optimality tolerance"] = 1e-6
prob.driver.opt_settings["Major feasibility tolerance"] = 1e-6
prob.driver.opt_settings["iSumm"] = 6
prob.driver.opt_settings["Major iterations limit"] = 50

prob.setup()

phase_object.apply_initial_guesses(prob, traj_name='traj', phase=phase)

dm.run_problem(prob, run_driver=True, simulate=False, make_plots=True)
