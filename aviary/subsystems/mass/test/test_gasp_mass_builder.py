import unittest

import openmdao.api as om
from openmdao.core.system import System

import aviary.api as av
from aviary.subsystems.mass.mass_builder import CoreMassBuilder
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
from aviary.variable_info.variables import Aircraft

GASP = LegacyCode.GASP


class TestGASPMassBuilderHybrid(av.TestSubsystemBuilder):
    """
    TestSubsystemBuilder for the CoreMassBuilder using GASP methods. SetUp used to provide required
    variables. Some methods are overriden so set_input_defaults can be called on the problem.
    """

    def setUp(self):
        self.subsystem_builder = CoreMassBuilder(
            'test_core_mass', meta_data=BaseMetaData, code_origin=GASP
        )
        self.aviary_values = av.AviaryValues()
        self.aviary_values.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, 3, units='unitless')
        self.aviary_values.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, 2, units='unitless')
        self.aviary_values.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, True, units='unitless')
        self.aviary_values.set_val(Aircraft.Wing.HAS_FOLD, True, units='unitless')
        self.aviary_values.set_val(Aircraft.Wing.HAS_STRUT, True, units='unitless')
        self.aviary_values.set_val(Aircraft.Engine.NUM_ENGINES, [1], units='unitless')
        self.aviary_values.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, 1)

    def test_check_pre_mission(self):
        if not hasattr(self, 'aviary_values'):
            self.aviary_values = AviaryValues()

        pre_mission_sys = self.subsystem_builder.build_pre_mission(
            aviary_inputs=self.aviary_values, subsystem_options={}
        )

        if pre_mission_sys is None:
            return

        group = om.Group()
        group.add_subsystem('pre_mission_sys', pre_mission_sys)
        prob = om.Problem(group)

        setup_model_options(prob, self.aviary_values)
        prob.model.set_input_defaults(
            f'pre_mission_sys.{Aircraft.Fuel.DENSITY}', val=6.687, units='lbm/galUS'
        )

        prob.setup()
        prob.final_setup()

        self.assertIsInstance(
            pre_mission_sys, System, 'The method should return an OpenMDAO System object.'
        )

        mass_names = self.subsystem_builder.get_mass_names()
        outputs = pre_mission_sys.list_outputs(out_stream=None, prom_name=True)

        for name in mass_names:
            mass_var_exists = any(name == output_[1]['prom_name'] for output_ in outputs)
            self.assertTrue(
                mass_var_exists, f"Mass variable '{name}' not found in the pre-mission model."
            )

    def test_check_design_variables(self):
        if not hasattr(self, 'aviary_values'):
            self.aviary_values = AviaryValues()

        design_vars = self.subsystem_builder.get_design_vars()

        pre_mission_sys = self.subsystem_builder.build_pre_mission(
            aviary_inputs=self.aviary_values, subsystem_options={}
        )

        if pre_mission_sys is None:
            return

        group = om.Group()
        group.add_subsystem('pre_mission', pre_mission_sys, promotes=['*'])
        prob = om.Problem(group)

        setup_model_options(prob, self.aviary_values)
        prob.model.set_input_defaults(
            f'pre_mission_sys.{Aircraft.Fuel.DENSITY}', val=6.687, units='lbm/galUS'
        )

        prob.setup()
        prob.final_setup()

        inputs = prob.model.list_inputs(out_stream=None, prom_name=True)

        for key, value in design_vars.items():
            # Check design variable existence
            design_var_exists = any(key == input_[1]['prom_name'] for input_ in inputs)
            self.assertTrue(design_var_exists, f"Design variable '{key}' not found in the model.")


class TestGASPMassBuilder(av.TestSubsystemBuilder):
    """
    TestSubsystemBuilder for the CoreMassBuilder using GASP methods. SetUp used to provide required
    variables. Some methods are overriden so set_input_defaults can be called on the problem.
    """

    def setUp(self):
        self.subsystem_builder = CoreMassBuilder(
            'test_core_mass', meta_data=BaseMetaData, code_origin=GASP
        )
        self.aviary_values = av.AviaryValues()
        self.aviary_values.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, 3, units='unitless')
        self.aviary_values.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, 2, units='unitless')
        self.aviary_values.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, False, units='unitless')
        self.aviary_values.set_val(Aircraft.Wing.HAS_FOLD, True, units='unitless')
        self.aviary_values.set_val(Aircraft.Wing.HAS_STRUT, True, units='unitless')
        self.aviary_values.set_val(Aircraft.Engine.NUM_ENGINES, [1], units='unitless')
        self.aviary_values.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, 1)

    def test_check_pre_mission(self):
        if not hasattr(self, 'aviary_values'):
            self.aviary_values = AviaryValues()

        pre_mission_sys = self.subsystem_builder.build_pre_mission(
            aviary_inputs=self.aviary_values, subsystem_options={}
        )

        if pre_mission_sys is None:
            return

        group = om.Group()
        group.add_subsystem('pre_mission_sys', pre_mission_sys)
        prob = om.Problem(group)

        setup_model_options(prob, self.aviary_values)
        prob.model.set_input_defaults(
            f'pre_mission_sys.{Aircraft.Fuel.DENSITY}', val=6.687, units='lbm/galUS'
        )

        prob.setup()
        prob.final_setup()

        self.assertIsInstance(
            pre_mission_sys, System, 'The method should return an OpenMDAO System object.'
        )

        mass_names = self.subsystem_builder.get_mass_names()
        outputs = pre_mission_sys.list_outputs(out_stream=None, prom_name=True)

        for name in mass_names:
            mass_var_exists = any(name == output_[1]['prom_name'] for output_ in outputs)
            self.assertTrue(
                mass_var_exists, f"Mass variable '{name}' not found in the pre-mission model."
            )

    def test_check_design_variables(self):
        if not hasattr(self, 'aviary_values'):
            self.aviary_values = AviaryValues()

        design_vars = self.subsystem_builder.get_design_vars()

        pre_mission_sys = self.subsystem_builder.build_pre_mission(
            aviary_inputs=self.aviary_values, subsystem_options={}
        )

        if pre_mission_sys is None:
            return

        group = om.Group()
        group.add_subsystem('pre_mission', pre_mission_sys, promotes=['*'])
        prob = om.Problem(group)

        setup_model_options(prob, self.aviary_values)
        prob.model.set_input_defaults(
            f'pre_mission_sys.{Aircraft.Fuel.DENSITY}', val=6.687, units='lbm/galUS'
        )

        prob.setup()
        prob.final_setup()

        inputs = prob.model.list_inputs(out_stream=None, prom_name=True)

        for key, value in design_vars.items():
            # Check design variable existence
            design_var_exists = any(key == input_[1]['prom_name'] for input_ in inputs)
            self.assertTrue(design_var_exists, f"Design variable '{key}' not found in the model.")


if __name__ == '__main__':
    unittest.main()
