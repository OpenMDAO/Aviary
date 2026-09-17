import unittest
from pathlib import Path

from aviary.api import AviaryProblem
from aviary.models.missions.energy_state_default import phase_info
from aviary.subsystems.test.dummy_subsystem import ExtendedMetaData, FullSubsystemBuilder


@use_tempdirs
class FullSubsystemBuilderTestSuite(unittest.TestCase):
    def setUp(self):
        self.prob = prob = AviaryProblem(verbosity=0, meta_data=ExtendedMetaData)

        prob.load_inputs(
            'models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv', phase_info
        )

        prob.load_external_subsystems([FullSubsystemBuilder()])
        prob.check_and_preprocess_inputs()
        prob.build_model()
        prob.add_driver()
        prob.add_design_variables()
        prob.add_objective()
        prob.setup()
        prob.final_setup()
        # prob.run_aviary_problem()
        # om.n2(prob, show_browser=False)
        # prob.model.list_vars(units=True, print_arrays=True)

    # needs_mission_solver also indirectly tests the SubsystemBuilder function 'build_mission'
    def test_needs_mission_solver(self):
        prob = self.prob
        self.assertTrue(
            hasattr(
                prob.model.traj.phases.climb.rhs_all.solver_sub,
                'full_suite',
            )
        )

    def test_build_pre_mission(self):
        prob = self.prob
        self.assertTrue(
            hasattr(
                prob.model.pre_mission,
                'full_suite',
            )
        )

    def test_get_states(self):
        prob = self.prob
        climb = prob.model.traj._phases['climb']
        self.assertTrue('aircraft:dummy_state_variable' in climb.indep_states.var_names)

    def test_get_controls(self):
        prob = self.prob
        climb = prob.model.traj._phases['climb']
        self.assertTrue('aircraft:dummy_control_variable' in climb.control_options)

    def test_get_parameters(self):
        prob = self.prob
        cruise = prob.model.traj._phases['cruise']
        self.assertTrue('aircraft:dummy_parameter_variable' in cruise.parameter_options)

    def test_get_constraints(self):
        prob = self.prob
        driver_vars = prob.list_driver_vars(out_stream=None)
        constraints = driver_vars.get('constraints')

        # Check that the dummy constraint variable from the FullSubsystemBuilder propagated through to phase constraints
        self.assertTrue(
            any(
                'traj.climb.aircraft:dummy_constraint_variable[final]' in tuples
                for tuples in constraints
            )
        )

    def test_get_linked_variables(self):
        prob = self.prob
        pred = prob.model._dataflow_graph.pred[
            'traj.phases.cruise.indep_states.initial_states:aircraft:dummy_state_variable'
        ]
        self.assertTrue('aircraft:dummy_state_variable' in [z for z in pred.keys()][0])

    def test_get_pre_mission_bus_variables(self):
        prob = self.prob
        pred = prob.model._dataflow_graph.pred['traj.param_comp.parameters:dummy_pre_mission_bus']
        self.assertTrue(
            'pre_mission.full_suite.dummy_pre_mission_bus' in [z for z in pred.keys()][0]
        )

    def test_build_mission(self):
        prob = self.prob
        self.assertTrue(
            hasattr(
                prob.model.traj.phases.climb.rhs_all.solver_sub,
                'full_suite',
            )
        )

    def test_mission_inputs(self):
        prob = self.prob
        climb = prob.model.traj._phases['climb']
        promoted_lists = climb.rhs_all.solver_sub.full_suite._inputs
        mission_inputs = [
            'aircraft:dummy_mission_input',
            'aircraft:dummy_parameter_variable',
            'aircraft:dummy_control_variable',
        ]
        self.assertTrue(all(s in promoted_lists for s in mission_inputs))

    def test_mission_outputs(self):
        prob = self.prob
        climb = prob.model.traj._phases['climb']
        promoted_lists = climb.rhs_all.solver_sub.full_suite._outputs
        mission_outputs = [
            'aircraft:dummy_mission_output',
            'aircraft:dummy_state_variable_rate',
            'aircraft:dummy_timeseries_variable',
            'aircraft:dummy_constraint_variable',
        ]
        self.assertTrue(all(s in promoted_lists for s in mission_outputs))

    def test_get_design_vars(self):
        prob = self.prob
        driver_vars = prob.list_driver_vars(out_stream=None)
        design_vars = driver_vars.get('design_vars')

        # Check that the dummy constraint variable from the FullSubsystemBuilder propagated through to phase constraints
        self.assertTrue(any('aircraft:dummy_design_variable' in tuples for tuples in design_vars))

    def test_get_initial_guesses(self):
        prob = self.prob
        self.assertTrue(
            prob.model.get_val('traj.phases.cruise.states:aircraft:dummy_state_variable')[0] == 10
        )

    def test_get_mass_names(self):
        prob = self.prob
        self.assertTrue(
            'aircraft_dummy_mission_output'
            in prob.model.pre_mission.external_comp_sum.get_io_metadata()
        )

    def test_preprocess_inputs(self):
        prob = self.prob
        self.assertTrue(prob.aviary_inputs.get_item('aircraft:dummy_mission_input')[0] == 5)

    def test_get_timeseries(self):
        prob = self.prob
        cruise = prob.model.traj._phases['climb']
        self.assertTrue(
            any(
                'aircraft:dummy_timeseries_variable' in tuple
                for tuple in cruise.timeseries.list_outputs()
            )
        )

    def test_get_post_mission_bus_variables(self):
        prob = self.prob
        pred = prob.model._dataflow_graph.pred[
            'traj.phases.climb.mission_bus_variables.input_values:dummy_post_mission_bus'
        ]
        self.assertTrue(
            'traj.phases.climb.rhs_all.solver_sub.full_suite.dummy_post_mission_bus'
            in [z for z in pred.keys()][0]
        )

    def test_build_post_mission(self):
        prob = self.prob
        self.assertTrue(
            hasattr(
                prob.model.post_mission,
                'full_suite',
            )
        )

    def test_report(self):
        prob = self.prob
        prob.run_aviary_problem(suppress_solver_print=True, verbosity=0)

        self.assertTrue(Path('FullSubsystemTest.md').exists())


if __name__ == '__main__':
    unittest.main()
    # test = FullSubsystemBuilderTestSuite()
    # test.setUp()
    # test.test_get_constraints()
