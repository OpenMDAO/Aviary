import unittest
from importlib import import_module

import numpy as np
import openmdao.api as om
from openmdao.core.system import System

from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import setup_model_options


def skipIfMissingDependencies(builder):
    return unittest.skipIf(type(builder) is str, builder)


class TestSubsystemBuilderBase(unittest.TestCase):
    @staticmethod
    def import_builder(path_to_builder: str, base_package='aviary.examples.external_subsystems'):
        """
        Import a subsystem builder.

        This is intended to be used with skipIfMissingDependencies
        """
        try:
            package, method = path_to_builder.rsplit('.', 1)
            package_path, package_name = package.rsplit('.', 1)
            module_path = (
                '.'.join([path_to_builder, package_path]) if package_path else path_to_builder
            )
            module = import_module(package_name, module_path)
            builder = getattr(module, method)
        except ImportError:
            builder = 'Skipping due to missing dependencies'
        except AttributeError:
            builder = method + ' could not be imported from ' + base_package + '.' + package
        return builder

    def setUp(self):
        self.subsystem_builder = SubsystemBuilderBase()
        self.aviary_values = AviaryValues()

    def test_get_states(self):
        states = self.subsystem_builder.get_states()
        self.assertIsInstance(states, dict, msg='get_states should return a dictionary')

        for key, value in states.items():
            self.assertIsInstance(value, dict, msg=f"the value for '{key}' should be a dictionary")

            self.assertIsInstance(
                value['rate_source'],
                str,
                msg=f"the value for 'rate_source' key in '{key}' should be a string",
            )

    def test_get_linked_variables(self):
        linked_variables = self.subsystem_builder.get_linked_variables()
        self.assertIsInstance(linked_variables, list)
        for var in linked_variables:
            self.assertIsInstance(var, str)

    def test_get_pre_mission_bus_variables(self):
        bus_variables = self.subsystem_builder.get_pre_mission_bus_variables()

        # Check that a dictionary is returned
        self.assertIsInstance(
            bus_variables, dict, 'get_pre_mission_bus_variables should return a dictionary'
        )

        for name, values in bus_variables.items():
            # Check that the bus_variable has the required keys
            self.assertIn(
                'mission_name',
                values.keys(),
                f'Bus Variable "{name}" is missing the "mission_name" key',
            )
            self.assertIn(
                'units', values.keys(), f'Bus Variable "{name}" is missing the "units" key'
            )

            # Check that the values of the keys are the expected types (allow list)
            self.assertIsInstance(
                values['mission_name'],
                (str, list, type(None)),
                f'Bus Variable "{name}"\'s "mission_name" value should be a string or None',
            )
            self.assertIsInstance(
                values['units'], str, f'Bus Variable "{name}"\'s "units" value should be a string'
            )

    def test_build_pre_mission(self):
        if not hasattr(self, 'aviary_values'):
            self.aviary_values = AviaryValues()

        pre_mission_sys = self.subsystem_builder.build_pre_mission(aviary_inputs=self.aviary_values)

        if pre_mission_sys is not None:
            # Check that pre_mission_sys is an OpenMDAO System
            self.assertIsInstance(
                pre_mission_sys,
                System,
                msg='The returned object from `build_pre_mission` is not an OpenMDAO System.',
            )

    def test_build_mission(self, **kwargs):
        if not hasattr(self, 'aviary_values'):
            self.aviary_values = AviaryValues()
        # Test that the method returns an OpenMDAO System object
        mission_sys = self.subsystem_builder.build_mission(
            10, aviary_inputs=self.aviary_values, **kwargs
        )
        if mission_sys is not None:
            self.assertIsInstance(
                mission_sys, System, 'The method should return an OpenMDAO System object.'
            )

        with self.assertRaises(TypeError, msg='num_nodes argument missing from build_mission().'):
            self.subsystem_builder.build_mission(aviary_inputs=self.aviary_values)

    def test_get_constraints(self):
        constraints = self.subsystem_builder.get_constraints()

        # Check that a dictionary is returned
        self.assertIsInstance(constraints, dict, 'get_constraints should return a dictionary')

        for name, values in constraints.items():
            # Check that the constraint has the required keys
            self.assertIn('type', values.keys(), f'Constraint "{name}" is missing the "type" key')

            # Check that the values of the keys are the expected types
            self.assertIsInstance(
                values['type'], str, f'Constraint "{name}"\'s "type" value should be a string'
            )

    def test_get_design_vars(self):
        # Verify that the method returns a dictionary
        design_vars = self.subsystem_builder.get_design_vars()
        self.assertIsInstance(design_vars, dict, 'get_design_vars() should return a dictionary')

        # Verify that the keys in the dictionary are strings
        for key in design_vars.keys():
            self.assertIsInstance(
                key,
                str,
                'The keys in the dictionary returned by get_design_vars() should be strings',
            )

        # Verify that the values in the dictionary are also dictionaries
        for val in design_vars.values():
            self.assertIsInstance(
                val,
                dict,
                'The values in the dictionary returned by get_design_vars() should be dictionaries',
            )

        # Verify that the dictionaries have the correct keys
        for val in design_vars.values():
            self.assertIn(
                'units',
                val,
                "The dictionaries returned by get_design_vars() should have a 'units' key",
            )
            self.assertIn(
                'lower',
                val,
                "The dictionaries returned by get_design_vars() should have a 'lower' key",
            )
            self.assertIn(
                'upper',
                val,
                "The dictionaries returned by get_design_vars() should have an 'upper' key",
            )

    def test_get_parameters(self, **kwargs):
        if not hasattr(self, 'aviary_values'):
            self.aviary_values = AviaryValues()

        # Verify that the method returns a dictionary
        parameters = self.subsystem_builder.get_parameters(
            aviary_inputs=self.aviary_values, **kwargs
        )
        self.assertIsInstance(parameters, dict, 'get_parameters() should return a dictionary')

        # Verify that the keys in the dictionary are strings
        for key in parameters.keys():
            self.assertIsInstance(
                key,
                str,
                'The keys in the dictionary returned by get_parameters() should be strings',
            )

        # Verify that the values in the dictionary are also dictionaries
        for val in parameters.values():
            self.assertIsInstance(
                val,
                dict,
                'The values in the dictionary returned by get_parameters() should be dictionaries',
            )

        # Verify that the dictionaries have the correct keys
        for key, val in parameters.items():
            self.assertIn(
                'val',
                val,
                f"The dictionaries returned by get_parameters() should have a 'val' key for {key}",
            )
            self.assertIn(
                'units',
                val,
                f"The dictionaries returned by get_parameters() should have a 'units' key for {key}",
            )

    def test_get_initial_guesses(self):
        initial_guesses = self.subsystem_builder.get_initial_guesses()

        self.assertIsInstance(
            initial_guesses, dict, msg='get_initial_guesses() should return a dictionary'
        )
        for key, val in initial_guesses.items():
            self.assertIsInstance(val, dict, msg=f"The value for '{key}' should be a dictionary")
            self.assertIn(
                'val', val, msg=f"The value for '{key}' dictionary should have a 'val' key"
            )
            self.assertIn(
                'type', val, msg=f"The value for '{key}' dictionary should have a 'type' key"
            )

            guess_val = val['val']
            self.assertIsInstance(
                guess_val,
                (float, list, np.ndarray),
                msg=f"The value for '{key}' should be a float, list, or array",
            )

    def test_get_mass_names(self):
        # Instantiate a user-defined subsystem builder
        subsystem_builder = self.subsystem_builder

        # Ensure get_mass_names method returns a list
        self.assertIsInstance(
            subsystem_builder.get_mass_names(),
            list,
            'get_mass_names should return a list of mass names',
        )

        # Ensure each mass name is a string
        for mass_name in subsystem_builder.get_mass_names():
            self.assertIsInstance(mass_name, str, 'Each mass name in the list should be a string')

    def test_preprocess_inputs(self):
        if not hasattr(self, 'aviary_values'):
            self.aviary_values = AviaryValues()
        inputs = self.subsystem_builder.preprocess_inputs(self.aviary_values)
        self.assertIsInstance(
            inputs, AviaryValues, 'preprocess_inputs did not return an AviaryValues object'
        )

    def test_build_post_mission(self):
        # Perform post-mission operations
        if not hasattr(self, 'aviary_values'):
            self.aviary_values = AviaryValues()
        phase_info = {}
        phase_mission_bus_lengths = {'foo': 10, 'bar': 11}
        post_mission_sys = self.subsystem_builder.build_post_mission(
            self.aviary_values, phase_info, phase_mission_bus_lengths
        )

        if post_mission_sys is not None:
            # Check that post_mission_sys is an OpenMDAO system
            self.assertIsInstance(
                post_mission_sys, System, msg='post_mission_sys is not an OpenMDAO System.'
            )

    def test_define_order(self):
        order = self.subsystem_builder.define_order()
        self.assertIsInstance(order, list, 'define_order should return a list')

        for subsystem_name in order:
            self.assertIsInstance(
                subsystem_name, str, 'Each subsystem name in the list should be a string'
            )

    def test_get_outputs(self):
        outputs = self.subsystem_builder.get_outputs()
        self.assertIsInstance(outputs, list, 'get_outputs should return a list')

        for output_name in outputs:
            self.assertIsInstance(
                output_name, str, 'Each output name in the list should be a string'
            )

    def test_check_state_variables(self):
        if not hasattr(self, 'aviary_values'):
            self.aviary_values = AviaryValues()

        states = self.subsystem_builder.get_states()

        mission_sys = self.subsystem_builder.build_mission(
            num_nodes=5, aviary_inputs=self.aviary_values
        )

        if mission_sys is None:
            return

        group = om.Group()
        group.add_subsystem('mission', mission_sys, promotes=['*'])
        prob = om.Problem(group)

        setup_model_options(prob, self.aviary_values)

        prob.setup()
        prob.final_setup()

        inputs = prob.model.list_inputs(out_stream=None, prom_name=True)

        for key, value in states.items():
            if mission_sys is not None:
                # Check state variable existence
                state_var_exists = any(key == input[1]['prom_name'] for input in inputs)
                self.assertTrue(state_var_exists, f"State variable '{key}' not found in the model.")

    def test_check_pre_mission(self):
        if not hasattr(self, 'aviary_values'):
            self.aviary_values = AviaryValues()

        pre_mission_sys = self.subsystem_builder.build_pre_mission(self.aviary_values)

        if pre_mission_sys is None:
            return

        group = om.Group()
        group.add_subsystem('pre_mission_sys', pre_mission_sys)
        prob = om.Problem(group)

        setup_model_options(prob, self.aviary_values)

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

    def test_check_parameters(self, **kwargs):
        if not hasattr(self, 'aviary_values'):
            self.aviary_values = AviaryValues()

        parameters = self.subsystem_builder.get_parameters(
            aviary_inputs=self.aviary_values, **kwargs
        )

        mission_sys = self.subsystem_builder.build_mission(
            num_nodes=5, aviary_inputs=self.aviary_values
        )

        if mission_sys is None:
            return

        group = om.Group()
        group.add_subsystem('mission', mission_sys, promotes=['*'])
        prob = om.Problem(group)

        setup_model_options(prob, self.aviary_values)

        prob.setup()
        prob.final_setup()

        inputs = prob.model.list_inputs(out_stream=None, prom_name=True)

        for key, value in parameters.items():
            # Check parameter existence
            param_exists = any(key == input_[1]['prom_name'] for input_ in inputs)
            self.assertTrue(param_exists, f"Parameter '{key}' not found in the model.")

    def test_check_constraints(self):
        if not hasattr(self, 'aviary_values'):
            self.aviary_values = AviaryValues()

        constraints = self.subsystem_builder.get_constraints()

        mission_sys = self.subsystem_builder.build_mission(
            num_nodes=5, aviary_inputs=self.aviary_values
        )

        if mission_sys is None:
            return

        group = om.Group()
        group.add_subsystem('mission', mission_sys, promotes=['*'])
        prob = om.Problem(group)

        setup_model_options(prob, self.aviary_values)

        prob.setup()
        prob.final_setup()

        inputs = prob.model.list_inputs(out_stream=None, prom_name=True)
        outputs = prob.model.list_outputs(out_stream=None, prom_name=True)
        name = self.subsystem_builder.default_name

        for key, value in constraints.items():
            # Check constraint existence
            constraint_exists = (
                any(key == output[1]['prom_name'] for output in outputs)
                or any(key == input[1]['prom_name'] for input in inputs)
                or any(key == f'{name}.{output[1]["prom_name"]}' for output in outputs)
                or any(key == f'{name}.{input[1]["prom_name"]}' for input in inputs)
            )
            self.assertTrue(constraint_exists, f"Constraint '{key}' not found in the model.")

    def test_check_design_variables(self):
        if not hasattr(self, 'aviary_values'):
            self.aviary_values = AviaryValues()

        design_vars = self.subsystem_builder.get_design_vars()

        pre_mission_sys = self.subsystem_builder.build_pre_mission(aviary_inputs=self.aviary_values)

        if pre_mission_sys is None:
            return

        group = om.Group()
        group.add_subsystem('pre_mission', pre_mission_sys, promotes=['*'])
        prob = om.Problem(group)

        setup_model_options(prob, self.aviary_values)

        prob.setup()
        prob.final_setup()

        inputs = prob.model.list_inputs(out_stream=None, prom_name=True)

        for key, value in design_vars.items():
            # Check design variable existence
            design_var_exists = any(key == input_[1]['prom_name'] for input_ in inputs)
            self.assertTrue(design_var_exists, f"Design variable '{key}' not found in the model.")

    def test_check_initial_guesses(self):
        if not hasattr(self, 'aviary_values'):
            self.aviary_values = AviaryValues()

        initial_guesses = self.subsystem_builder.get_initial_guesses()

        mission_sys = self.subsystem_builder.build_mission(
            num_nodes=5, aviary_inputs=self.aviary_values
        )

        if mission_sys is None:
            return

        group = om.Group()
        group.add_subsystem('mission', mission_sys, promotes=['*'])
        prob = om.Problem(group)

        setup_model_options(prob, self.aviary_values)

        prob.setup()
        prob.final_setup()

        inputs = prob.model.list_inputs(out_stream=None, prom_name=True)

        for key, value in initial_guesses.items():
            # Check initial guess existence
            initial_guess_exists = any(key == input[1]['prom_name'] for input in inputs)
            self.assertTrue(initial_guess_exists, f"Initial guess '{key}' not found in the model.")


if __name__ == '__main__':
    unittest.main()
