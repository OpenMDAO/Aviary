import unittest

import openmdao.api as om

from aviary.mission.flops_based.ode.landing_ode import FlareODE
from aviary.models.N3CC.N3CC_data import detailed_landing_flare, inputs, landing_subsystem_options
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import set_aviary_initial_values
from aviary.utils.preprocessors import preprocess_options
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.validation_cases.validation_tests import do_validation_test
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Dynamic


class FlareODETest(unittest.TestCase):
    """Test against data of detailed_landing_flare from models/N3CC/N3CC_data.py."""

    def test_case(self):
        prob = om.Problem()
        time, _ = detailed_landing_flare.get_item('time')
        nn = len(time)
        aviary_options = inputs

        engines = [build_engine_deck(aviary_options)]

        preprocess_options(aviary_options, engine_models=engines)

        default_mission_subsystems = get_default_mission_subsystems('FLOPS', engines)

        prob.model.add_subsystem(
            'landing_flare_ode',
            FlareODE(
                num_nodes=nn,
                subsystem_options=landing_subsystem_options,
                core_subsystems=default_mission_subsystems,
                aviary_options=aviary_options,
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1.0, units='ft**2')

        setup_model_options(prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')}))

        prob.setup(check=False, force_alloc_complex=True)

        set_aviary_initial_values(prob, aviary_options)

        do_validation_test(
            prob,
            'landing_flare_ode',
            input_validation_data=detailed_landing_flare,
            output_validation_data=detailed_landing_flare,
            input_keys=[
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                Dynamic.Mission.VELOCITY,
                Dynamic.Vehicle.MASS,
                Dynamic.Vehicle.LIFT,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.DRAG,
            ],
            output_keys=[
                Dynamic.Mission.DISTANCE_RATE,
                Dynamic.Mission.ALTITUDE_RATE,
            ],
            tol=1e-2,
            atol=5e-9,
            rtol=5e-9,
            check_values=False,
            check_partials=True,
        )


if __name__ == '__main__':
    unittest.main()
