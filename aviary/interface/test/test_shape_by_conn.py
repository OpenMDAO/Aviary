from copy import deepcopy
import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

from aviary.core.aviary_problem import AviaryProblem
from aviary.models.external_subsystems.detailed_battery.battery_variables import Aircraft, Dynamic
from aviary.models.missions.energy_state_default import phase_info as energy_phase_info
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.utils.csv_data_file import read_data_file
from aviary.utils.develop_metadata import add_meta_data
from aviary.utils.named_values import NamedValues
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variable_meta_data import CoreMetaData
from aviary.variable_info.variables import Aircraft


@use_tempdirs
class TestShapebyConn(unittest.TestCase):
    def test_shape_bug(self):
        # Verifies that shape_by_conn variables no longer raise an exception during
        # setup.

        class ExtendedAircraft(Aircraft):
            """Aircraft data hierarchy with one new var."""

            class Wing(Aircraft.Wing):
                CG1 = 'aircraft:wing:cg1'
                CG2 = 'aircraft:wing:cg2'

        ExtendedMetaData = deepcopy(CoreMetaData)
        add_meta_data(
            ExtendedAircraft.Wing.CG1,
            units='ft',
            desc='CG of the wing.',
            default_value=1.0,
            meta_data=ExtendedMetaData,
        )

        add_meta_data(
            ExtendedAircraft.Wing.CG2,
            units='ft',
            desc='CG of the wing.',
            default_value=1.0,
            meta_data=ExtendedMetaData,
        )

        class SBC(om.ExplicitComponent):
            """This component has an input with shape_by_conn."""

            def setup(self):
                self.add_input(ExtendedAircraft.Wing.CG1, units='ft', shape_by_conn=True)
                self.add_input('cg_promote_me', units='ft', shape_by_conn=True)

                self.add_output('stuff', units='ft**2', copy_shape=ExtendedAircraft.Wing.CG1)

            def compute(self, inputs, outputs):
                pass

        class CGBuilder(SubsystemBuilder):
            _default_name = 'cg_sub'

            def build_post_mission(
                self,
                aviary_inputs=None,
                mission_info=None,
                subsystem_options=None,
                phase_mission_bus_lengths=None,
            ):
                grp = om.Group()
                grp.add_subsystem(
                    'sbc',
                    SBC(),
                    promotes_inputs=[
                        ExtendedAircraft.Wing.CG1,
                        ('cg_promote_me', ExtendedAircraft.Wing.CG2),
                    ],
                )
                return grp

        local_phase_info = deepcopy(energy_phase_info)

        prob = AviaryProblem(meta_data=ExtendedMetaData)

        prob.load_inputs(
            'validation_cases/validation_data/test_models/high_wing_single_aisle.csv',
            local_phase_info,
        )
        prob.load_external_subsystems([CGBuilder()])
        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.build_model()

        prob.setup()

        prob.set_val(ExtendedAircraft.Wing.CG1, np.ones((3, 4)))
        prob.set_val(ExtendedAircraft.Wing.CG2, np.ones((4, 2)))

        prob.final_setup()

    def test_shape_bug_two(self):
        # Verifies another shape_by_conn bug, where variables that are dynamic in some compoennts
        # and static in others need to have their value set.

        class PostComponent(om.ExplicitComponent):
            def setup(self):
                self.add_input(Aircraft.Design.DRAG_POLAR, shape_by_conn=True, units='unitless')
                add_aviary_input(
                    self, Aircraft.Design.LIFT_POLAR, shape_by_conn=True, units='unitless'
                )

                self.add_output('z')

        class PostSystem(om.Group):
            def setup(self):
                self.add_subsystem('c', PostComponent(), promotes=['*'])

        class PostBuilder(SubsystemBuilder):
            def build_post_mission(
                self,
                aviary_inputs=None,
                mission_info=None,
                subsystem_options=None,
                phase_mission_bus_lengths=None,
            ):
                return PostSystem()

        polar_file = 'models/large_single_aisle_1/aerodynamics_tables/large_single_aisle_1_aero_free_reduced_alpha.csv'

        phase_info = deepcopy(energy_phase_info)
        phase_info['pre_mission']['include_takeoff'] = False
        phase_info['post_mission']['include_landing'] = False

        data, _, _ = read_data_file(polar_file)
        ALTITUDE = data.get_val('Altitude', 'ft')
        MACH = data.get_val('Mach', 'unitless')
        ALPHA = data.get_val('Angle_of_Attack', 'deg')

        shape = (np.unique(ALTITUDE).size, np.unique(MACH).size, np.unique(ALPHA).size)
        CL = data.get_val('CL').reshape(shape)
        CD = data.get_val('CD').reshape(shape)

        ph_in = deepcopy(phase_info)

        aero_data = NamedValues()
        aero_data.set_val('altitude', ALTITUDE, 'ft')
        aero_data.set_val('mach', MACH, 'unitless')
        aero_data.set_val('angle_of_attack', ALPHA, 'deg')

        subsystem_options = {
            'method': 'tabular_cruise',
            'solve_alpha': True,
            'aero_data': aero_data,
            'connect_training_data': True,
        }
        ph_in['climb']['subsystem_options'] = {'aerodynamics': subsystem_options}
        ph_in['cruise']['subsystem_options'] = {'aerodynamics': subsystem_options}
        ph_in['descent']['subsystem_options'] = {'aerodynamics': subsystem_options}

        prob = AviaryProblem()

        prob.load_inputs(
            'validation_cases/validation_data/test_models/high_wing_single_aisle.csv',
            ph_in,
        )

        prob.model.aero_method = LegacyCode.GASP

        prob.load_external_subsystems([PostBuilder()])

        prob.aviary_inputs.set_val(Aircraft.Design.LIFT_POLAR, CL, units='unitless')
        prob.aviary_inputs.set_val(Aircraft.Design.DRAG_POLAR, CD, units='unitless')

        prob.check_and_preprocess_inputs()

        prob.build_model()

        prob.setup()
        prob.set_solver_print(0)
        prob.run_model()


if __name__ == '__main__':
    unittest.main()
