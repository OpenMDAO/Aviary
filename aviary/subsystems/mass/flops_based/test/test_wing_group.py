import unittest

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from aviary.subsystems.mass.flops_based.wing_group import WingMassGroup
from aviary.variable_info.variables import Aircraft, Mission
from aviary.validation_cases.validation_tests import flops_validation_test, get_flops_data, Version


@use_tempdirs
class BWBWingGroupTest(unittest.TestCase):
    """Tests wing mass calculation for BWB."""

    def setUp(self):
        self.prob = om.Problem()

    def test_case(self):
        case_name = 'BWBsimpleFLOPS'
        prob = self.prob

        keys = [
            Aircraft.Engine.NUM_ENGINES,
            Aircraft.Design.TYPE,
            Aircraft.Wing.DETAILED_WING,
            Aircraft.Engine.NUM_WING_ENGINES,
            Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES,
            Aircraft.Wing.INPUT_STATION_DIST,
            Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL,
            Aircraft.Wing.NUM_INTEGRATION_STATIONS,
            Aircraft.Engine.NUM_FUSELAGE_ENGINES,
            Aircraft.BWB.DETAILED_WING_PROVIDED,
        ]
        options = get_flops_data(case_name, preprocess=True, keys=keys)
        model_options = {}
        for key in keys:
            model_options[key] = options.get_item(key)[0]

        prob.model.add_subsystem(
            'wing_group',
            WingMassGroup(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        prob.model_options['*'] = model_options

        prob.model.set_input_defaults('BWB_LOAD_PATH_SWEEP_DIST', [0.0, 15.337244816], units='deg')
        prob.model.set_input_defaults(
            'BWB_THICKNESS_TO_CHORD_DIST', [0.11, 0.11, 0.11], units='unitless'
        )
        prob.model.set_input_defaults(
            'BWB_CHORD_PER_SEMISPAN_DIST', [137.5, 91.3717, 14.2848], units='unitless'
        )
        prob.model.set_input_defaults('Rear_spar_percent_chord', 0.7, units='unitless')
        prob.model.set_input_defaults('Rear_spar_percent_chord_centerline', 0.7, units='unitless')

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self,
            self.prob,
            case_name,
            input_keys=[
                # EnginePodMass
                Mission.Design.GROSS_MASS,
                Aircraft.Electrical.MASS,
                Aircraft.Fuel.FUEL_SYSTEM_MASS,
                Aircraft.Hydraulics.MASS,
                Aircraft.Instruments.MASS,
                Aircraft.Nacelle.MASS,
                Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS,
                Aircraft.Engine.MASS,
                Aircraft.Propulsion.TOTAL_STARTER_MASS,
                Aircraft.Engine.THRUST_REVERSERS_MASS,
                Aircraft.Engine.SCALED_SLS_THRUST,
                Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST,
                # BWBDetailedWingBendingFact
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Wing.ASPECT_RATIO_REFERENCE,
                Aircraft.Wing.STRUT_BRACING_FACTOR,
                Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR,
                Aircraft.Wing.THICKNESS_TO_CHORD,
                Aircraft.Wing.THICKNESS_TO_CHORD_REFERENCE,
                Aircraft.Fuselage.MAX_WIDTH,
                # WingMiscMass
                Aircraft.Wing.COMPOSITE_FRACTION,
                # Aircraft.Wing.AREA,
                Aircraft.Wing.MISC_MASS_SCALER,
                # WingShearControlMass
                Aircraft.Wing.CONTROL_SURFACE_AREA,
                Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER,
                # WingBendingMass
                Aircraft.Wing.BENDING_MATERIAL_MASS_SCALER,
                Aircraft.Wing.ENG_POD_INERTIA_FACTOR,
                Aircraft.Wing.LOAD_FRACTION,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.SWEEP,
                Aircraft.Wing.ULTIMATE_LOAD_FACTOR,
                Aircraft.Wing.VAR_SWEEP_MASS_PENALTY,
                # WingBendingMass
                # BWBAftBodyMass
                Aircraft.Fuselage.PLANFORM_AREA,
                Aircraft.Fuselage.CABIN_AREA,
                Aircraft.Fuselage.LENGTH,
                Aircraft.Wing.ROOT_CHORD,
            ],
            output_keys=[
                Aircraft.Engine.POD_MASS,
                Aircraft.Wing.BENDING_MATERIAL_FACTOR,
                Aircraft.Wing.MISC_MASS,
                Aircraft.Wing.SHEAR_CONTROL_MASS,
                Aircraft.Wing.BENDING_MATERIAL_MASS,
                Aircraft.Fuselage.AFTBODY_MASS,
                Aircraft.Wing.BWB_AFTBODY_MASS,
                Aircraft.Wing.MASS,
            ],
            version=Version.BWB,
            check_partials=False,
        )


if __name__ == '__main__':
    unittest.main()
