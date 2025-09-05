"""Benchmark test for aviary mission with an NPSS defined engine."""

import os as os
import unittest
from copy import deepcopy

from openmdao.utils.assert_utils import assert_near_equal

import aviary.api as av
from aviary.examples.external_subsystems.engine_NPSS.NPSS_engine_builder import (
    NPSSTabularEngineBuilder,
)

from aviary.examples.external_subsystems.engine_NPSS.NPSS_variable_meta_data import ExtendedMetaData


class AviaryNPSSTestCase(unittest.TestCase):
    """Test NPSS engine builder from table by building an Aviary model with NPSS engine and run."""

    @unittest.skipUnless(os.environ.get('NPSS_TOP', False), 'environment does not contain NPSS')
    def bench_test_aviary_NPSS(self):
        """Build NPSS model in Aviary."""
        phase_info = deepcopy(av.default_height_energy_phase_info)

        prob = av.AviaryProblem()

        prob.options['group_by_pre_opt_post'] = True

        # Load aircraft and options data from user
        # Allow for user overrides here
        # add engine builder
        prob.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv',
            phase_info,
            engine_builders=[NPSSTabularEngineBuilder()],
            meta_data=ExtendedMetaData,
        )

        prob.check_and_preprocess_inputs()

        prob.build_model()

        prob.add_driver('SLSQP')

        prob.add_design_variables()

        prob.add_objective()

        prob.setup()

        prob.run_aviary_problem(suppress_solver_print=True)

        rtol = 0.01

        # There are no truth values for these.
        assert_near_equal(
            prob.get_val('aircraft:engine:design_mass_flow'), 315.1648646, tolerance=rtol
        )

        assert_near_equal(
            prob.get_val('aircraft:engine:scaled_sls_thrust'), 35045.993119, tolerance=rtol
        )

        assert_near_equal(
            prob.get_val('traj.cruise.rhs_all.NPSS_prop_system.fuel_flow_rate_negative')[0],
            -1.13552634,
            tolerance=rtol,
        )
        assert_near_equal(
            prob.get_val('traj.cruise.rhs_all.NPSS_prop_system.thrust_net')[0],
            4253.95759421,
            tolerance=rtol,
        )


if __name__ == '__main__':
    test = AviaryNPSSTestCase()
    test.bench_test_aviary_NPSS()
