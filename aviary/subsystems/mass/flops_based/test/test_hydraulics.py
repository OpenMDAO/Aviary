import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.hydraulics import (
    AltHydraulicsGroupMass,
    TransportHydraulicsGroupMass,
)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    Version,
    flops_validation_test,
    get_flops_case_names,
    get_flops_inputs,
    print_case,
)
from aviary.variable_info.variables import Aircraft

bwb_cases = ['BWBsimpleFLOPS', 'BWBdetailedFLOPS', 'BWB300FLOPS']


@use_tempdirs
class TransportHydraulicsGroupMassTest(unittest.TestCase):
    """Tests transport/GA hydraulics mass calculation."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        inputs = get_flops_inputs(case_name, preprocess=True)

        options = {
            Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES: inputs.get_val(
                Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES
            ),
            Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES: inputs.get_val(
                Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES
            ),
        }

        prob.model.add_subsystem(
            'hydraulics',
            TransportHydraulicsGroupMass(**options),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self,
            prob,
            case_name,
            input_keys=[
                Aircraft.Fuselage.PLANFORM_AREA,
                Aircraft.Hydraulics.SYSTEM_PRESSURE,
                Aircraft.Hydraulics.MASS_SCALER,
                Aircraft.Wing.AREA,
                Aircraft.Wing.VAR_SWEEP_MASS_PENALTY,
                Aircraft.Design.MAX_MACH,
            ],
            output_keys=Aircraft.Hydraulics.MASS,
            version=Version.TRANSPORT_and_BWB,
            tol=4.0e-4,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


@use_tempdirs
class AltHydraulicsGroupMassTest(unittest.TestCase):
    """Tests alternate hydraulics mass calculation."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'hydraulics', AltHydraulicsGroupMass(), promotes_outputs=['*'], promotes_inputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self,
            prob,
            case_name,
            input_keys=[
                Aircraft.Wing.AREA,
                Aircraft.HorizontalTail.WETTED_AREA,
                Aircraft.HorizontalTail.THICKNESS_TO_CHORD,
                Aircraft.VerticalTail.AREA,
                Aircraft.Hydraulics.MASS_SCALER,
            ],
            output_keys=Aircraft.Hydraulics.MASS,
            version=Version.ALTERNATE,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


@use_tempdirs
class BWBTransportHydraulicsGroupMassTest(unittest.TestCase):
    """Tests transport/GA hydraulics mass calculation for BWB."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(only=bwb_cases), name_func=print_case)
    def testsdg_case(self, case_name):
        prob = self.prob

        inputs = get_flops_inputs(case_name, preprocess=True)

        options = {
            Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES: inputs.get_val(
                Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES
            ),
            Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES: inputs.get_val(
                Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES
            ),
        }

        prob.model.add_subsystem(
            'hydraulics',
            TransportHydraulicsGroupMass(**options),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self,
            prob,
            case_name,
            input_keys=[
                Aircraft.Fuselage.PLANFORM_AREA,
                Aircraft.Hydraulics.SYSTEM_PRESSURE,
                Aircraft.Hydraulics.MASS_SCALER,
                Aircraft.Wing.AREA,
                Aircraft.Wing.VAR_SWEEP_MASS_PENALTY,
                Aircraft.Design.MAX_MACH,
            ],
            output_keys=Aircraft.Hydraulics.MASS,
            version=Version.BWB,
            tol=4.0e-4,
        )


if __name__ == '__main__':
    unittest.main()
