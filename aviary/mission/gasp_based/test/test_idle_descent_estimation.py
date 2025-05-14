import importlib
import unittest
import warnings

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary.interface.default_phase_info.two_dof_fiti import add_default_sgm_args, descent_phases
from aviary.mission.gasp_based.idle_descent_estimation import add_descent_estimation_as_submodel
from aviary.mission.gasp_based.ode.params import set_params_for_unit_tests
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.utils.process_input_decks import create_vehicle
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Dynamic, Settings


@unittest.skip(
    'Shooting method is not correctly receiving user-set options, and is currently '
    'using default values for most options'
)
@unittest.skipUnless(
    importlib.util.find_spec('pyoptsparse') is not None, 'pyoptsparse is not installed'
)
class IdleDescentTestCase(unittest.TestCase):
    """Test idle descent for 2DOF mission."""

    def setUp(self):
        input_deck = 'models/large_single_aisle_1/large_single_aisle_1_GASP.csv'
        aviary_inputs, _ = create_vehicle(input_deck)
        aviary_inputs.set_val(Settings.VERBOSITY, 0)
        aviary_inputs.set_val(Aircraft.Engine.SCALED_SLS_THRUST, val=28690, units='lbf')
        aviary_inputs.set_val(Dynamic.Vehicle.Propulsion.THROTTLE, val=0, units='unitless')
        aviary_inputs.set_val(Aircraft.Wing.FORM_FACTOR, 1.25)
        aviary_inputs.set_val(Aircraft.VerticalTail.FORM_FACTOR, 1.25)
        aviary_inputs.set_val(Aircraft.HorizontalTail.FORM_FACTOR, 1.25)
        aviary_inputs.set_val(Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR, 1.1)

        engines = [build_engine_deck(options=aviary_inputs)]
        preprocess_propulsion(aviary_inputs, engines)

        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', [build_engine_deck(aviary_inputs)]
        )

        ode_args = dict(aviary_options=aviary_inputs, core_subsystems=default_mission_subsystems)

        self.ode_args = ode_args
        self.aviary_inputs = aviary_inputs
        self.tol = 1e-5

        add_default_sgm_args(descent_phases, self.ode_args)
        self.phases = descent_phases

    def test_subproblem(self):
        prob = om.Problem()
        prob.model = om.Group()

        ivc = om.IndepVarComp()
        ivc.add_output(Aircraft.Design.OPERATING_MASS, 97500, units='lbm')
        ivc.add_output(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, 36000, units='lbm')
        ivc.add_output('parameters:interference_independent_of_shielded_area', 1.89927266)
        ivc.add_output('parameters:drag_loss_due_to_shielded_wing_area', 68.02065834)
        ivc.add_output(Aircraft.Wing.FORM_FACTOR, 1.25)
        ivc.add_output(Aircraft.VerticalTail.FORM_FACTOR, 1.25)
        ivc.add_output(Aircraft.HorizontalTail.FORM_FACTOR, 1.25)
        prob.model.add_subsystem('IVC', ivc, promotes=['*'])

        add_descent_estimation_as_submodel(
            prob,
            phases=self.phases,
            ode_args=self.ode_args,
            cruise_alt=35000,
            reserve_fuel=4500,
            all_subsystems=self.ode_args['core_subsystems'],
        )
        prob.model.promotes('idle_descent_estimation', inputs=['parameters:*'])

        setup_model_options(
            prob.model.idle_descent_estimation,
            AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')}),
        )

        prob.setup()

        set_params_for_unit_tests(prob)

        warnings.filterwarnings('ignore', category=UserWarning)
        prob.run_model()
        warnings.filterwarnings('default', category=UserWarning)

        # Values obtained by running idle_descent_estimation
        assert_near_equal(prob.get_val('descent_range', 'NM'), 98.3445738, self.tol)
        assert_near_equal(prob.get_val('descent_fuel', 'lbm'), 250.79875356, self.tol)

        # TODO: check_partials() call results in runtime error: Jacobian in 'ODE_group' is not full rank.
        # partial_data = prob.check_partials(out_stream=None, method="cs")
        # assert_check_partials(partial_data, atol=0.0005, rtol=1e-9)


if __name__ == '__main__':
    unittest.main()
