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
from aviary.utils.aviary_values import AviaryValues


FLOPS = LegacyCode.FLOPS


throttle_max = 1.0
throttle_climb = 0.956
throttle_cruise = 0.930
throttle_idle = 0.0

prop = CorePropulsionBuilder('core_propulsion', BaseMetaData)
mass = CoreMassBuilder('core_mass', BaseMetaData, FLOPS)
aero = CoreAerodynamicsBuilder('core_aerodynamics', BaseMetaData, FLOPS)
geom = CoreGeometryBuilder('core_geometry', BaseMetaData, FLOPS)

default_premission_subsystems = [prop, geom, aero, mass]
default_mission_subsystems = [aero, prop]

phase_info = {
    'groundroll': {
        'user_options': {
            'fix_initial': True,
            'num_segments': 3,
            'throttle_setting': throttle_max,
            'input_speed_type': SpeedType.TAS,
            'ground_roll': True,
            'clean': False,
            'initial_ref': (100., 'kn'),
            'initial_bounds': ((0., 500.), 'kn'),
            'duration_ref': (100., 'kn'),
            'duration_bounds': ((50., 1000.), 'kn'),
            'control_order': 1,
            'order': 3,
            'opt': True,
        }
    },
}


phase_name = 'groundroll'
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


phase_object = TwoDOFPhase.from_phase_info(
    phase_name, phase_options, default_mission_subsystems, meta_data=BaseMetaData)

phase = phase_object.build_phase(aviary_options=aviary_inputs)
