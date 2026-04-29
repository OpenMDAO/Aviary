from copy import deepcopy
import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

from aviary.core.aviary_problem import AviaryProblem
from aviary.utils.develop_metadata import add_meta_data
from aviary.models.external_subsystems.detailed_battery.battery_variables import Aircraft, Dynamic
from aviary.models.missions.energy_state_default import phase_info as energy_phase_info
from aviary.subsystems.subsystem_builder import SubsystemBuilder
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
            meta_data=CoreMetaData,
        )

        add_meta_data(
            ExtendedAircraft.Wing.CG2,
            units='ft',
            desc='CG of the wing.',
            default_value=1.0,
            meta_data=CoreMetaData,
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
            'subsystems/aerodynamics/flops_based/test/data/high_wing_single_aisle.csv',
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


if __name__ == '__main__':
    unittest.main()
