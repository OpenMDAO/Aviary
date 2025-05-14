import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary.subsystems.premission import CorePreMission
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.functions import set_aviary_initial_values
from aviary.utils.preprocessors import preprocess_options
from aviary.utils.test_utils.default_subsystems import get_default_premission_subsystems
from aviary.validation_cases.validation_tests import get_flops_inputs, get_flops_outputs
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Dynamic, Settings


class MissionDragTest(unittest.TestCase):
    def test_basic_large_single_aisle_1(self):
        flops_inputs = get_flops_inputs('LargeSingleAisle1FLOPS')
        flops_outputs = get_flops_outputs('LargeSingleAisle1FLOPS')

        # comparison data is unscaled drag
        flops_inputs.set_val(Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR, 1.0)
        flops_inputs.set_val(Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR, 1.0)
        flops_inputs.set_val(Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR, 1.0)
        flops_inputs.set_val(Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR, 1.0)

        key = Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST
        flops_inputs.set_val(key, *(flops_outputs.get_item(key)))
        flops_inputs.set_val(Settings.VERBOSITY, 0)

        engines = [build_engine_deck(flops_inputs)]
        preprocess_options(flops_inputs, engine_models=engines)

        # don't need mass subsystem, so we skip it
        default_premission_subsystems = get_default_premission_subsystems('FLOPS', engines)[:-1]
        # we just want aero for mission, make a copy by itself
        aero = default_premission_subsystems[-1]

        # Design conditions:
        # alt = 41000
        # mach = 0.79

        # Is this correct?
        Sref = 1370.0

        # ---------------------
        # 1D Tables over M
        # ---------------------
        mach = np.array(
            [0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.750, 0.775, 0.800, 0.825, 0.850, 0.875]
        )
        CL = np.array(
            [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
        )
        mach = np.repeat(mach, 15)
        CL = np.tile(CL, 12)

        P = 374.74437747
        T = 389.97
        nn = len(mach)

        lift = CL * Sref * 0.5 * 1.4 * P * mach**2
        mass = lift

        prob = om.Problem()
        model = prob.model

        # Upstream pre-mission analysis for aero
        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(aviary_options=flops_inputs, subsystems=default_premission_subsystems),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:*', 'mission:*'],
        )

        model.add_subsystem(
            'aero',
            aero.build_mission(num_nodes=nn, aviary_inputs=flops_inputs, **{'method': 'computed'}),
            promotes=['*'],
        )

        # Set all options
        setup_model_options(prob, flops_inputs)
        prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, np.ones(1))

        prob.setup(force_alloc_complex=True)
        prob.set_solver_print(level=2)

        # Mission params
        prob.set_val(Dynamic.Atmosphere.MACH, val=mach)
        prob.set_val(Dynamic.Atmosphere.STATIC_PRESSURE, val=P, units='lbf/ft**2')
        prob.set_val(Dynamic.Atmosphere.TEMPERATURE, val=T, units='degR')
        prob.set_val(Dynamic.Vehicle.MASS, val=mass, units='lbm')

        set_aviary_initial_values(prob, flops_inputs)

        prob.run_model()

        D = prob.get_val(Dynamic.Vehicle.DRAG, 'lbf')
        CD = D / (Sref * 0.5 * 1.4 * P * mach**2)

        # fmt: off
        data = np.array(
            [
                [
                    0.02825, 0.02849, 0.02901, 0.02981, 0.03089, 0.03223, 0.03381, 0.03549,
                    0.03752, 0.04006, 0.04270, 0.04548, 0.04861, 0.05211, 0.05600
                ],
                [
                    0.02637, 0.02662, 0.02714, 0.02794, 0.02901, 0.03035, 0.03193, 0.03361,
                    0.03564, 0.03819, 0.04082, 0.04361, 0.04674, 0.05024, 0.05412
                ],
                [
                    0.02509, 0.02533, 0.02585, 0.02665, 0.02773, 0.02907, 0.03065, 0.03233,
                    0.03436, 0.03690, 0.03954, 0.04232, 0.04545, 0.04895, 0.05284
                ],
                [
                    0.02412, 0.02434, 0.02485, 0.02565, 0.02674, 0.02808, 0.02965, 0.03133,
                    0.03336, 0.03590, 0.03854, 0.04132, 0.04445, 0.04795, 0.05184
                ],
                [
                    0.02351, 0.02370, 0.02420, 0.02500, 0.02611, 0.02745, 0.02900, 0.03068,
                    0.03272, 0.03524, 0.03788, 0.04067, 0.04380, 0.04730, 0.05118
                ],
                [
                    0.02350, 0.02362, 0.02408, 0.02489, 0.02604, 0.02741, 0.02890, 0.03057,
                    0.03261, 0.03512, 0.03776, 0.04054, 0.04368, 0.04717, 0.05106
                ],
                [
                    0.02367, 0.02373, 0.02417, 0.02498, 0.02617, 0.02754, 0.02900, 0.03066,
                    0.03270, 0.03517, 0.03781, 0.04065, 0.04393, 0.04766, 0.05186
                ],
                [
                    0.02392, 0.02394, 0.02436, 0.02518, 0.02639, 0.02777, 0.02920, 0.03084,
                    0.03288, 0.03532, 0.03804, 0.04132, 0.04533, 0.05011, 0.05570
                ],
                [
                    0.02450, 0.02443, 0.02481, 0.02563, 0.02690, 0.02833, 0.02977, 0.03140,
                    0.03349, 0.03612, 0.03970, 0.04440, 0.04951, 0.05510, 0.06116
                ],
                [
                    0.02600, 0.02567, 0.02592, 0.02677, 0.02819, 0.02984, 0.03152, 0.03357,
                    0.03637, 0.04049, 0.04542, 0.05088, 0.05801, 0.06656, 0.07652
                ],
                [
                    0.02926, 0.02811, 0.02800, 0.02891, 0.03084, 0.03348, 0.03664, 0.04043,
                    0.04577, 0.05245, 0.05885, 0.06547, 0.07224, 0.07924, 0.08650
                ],
                [
                    0.04705, 0.04397, 0.04294, 0.04394, 0.04697, 0.05224, 0.05984, 0.06647,
                    0.07918, 0.09321, 0.09807, 0.10514, 0.11100, 0.11651, 0.12197
                ],
            ]
        ).ravel()
        # fmt: on

        # Due to differences in the interp method, certain points that are extrapolated
        # (i.e., M or CL are well above design) have more than 1% error.
        assert_near_equal(CD, data, 0.05)

        assert_near_equal(CD[:134], data[:134], 0.004)

    def test_n3cc_drag(self):
        flops_inputs = get_flops_inputs('N3CC')
        flops_outputs = get_flops_outputs('N3CC')

        # comparison data is unscaled drag
        flops_inputs.set_val(Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR, 1.0)
        flops_inputs.set_val(Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR, 1.0)
        flops_inputs.set_val(Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR, 1.0)
        flops_inputs.set_val(Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR, 1.0)

        key = Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST
        flops_inputs.set_val(key, *(flops_outputs.get_item(key)))
        flops_inputs.set_val(Settings.VERBOSITY, 0)

        engines = [build_engine_deck(flops_inputs)]
        preprocess_options(flops_inputs, engine_models=engines)

        # don't need mass subsystem, so we skip it
        default_premission_subsystems = get_default_premission_subsystems('FLOPS', engines)[:-1]
        # we just want aero for mission, make a copy by itself
        aero = default_premission_subsystems[-1]

        Sref = 1220.0

        # ---------------------
        # 1D Tables over M
        # ---------------------
        mach = np.array(
            [0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.750, 0.775, 0.800, 0.825, 0.850]
        )
        CL = np.array(
            [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        )
        mach = np.repeat(mach, 15)
        CL = np.tile(CL, 11)

        P = 340.53
        T = 389.97
        nn = len(mach)

        lift = CL * Sref * 0.5 * 1.4 * P * mach**2
        mass = lift

        prob = om.Problem()
        model = prob.model

        # Upstream pre-mission analysis for aero
        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(aviary_options=flops_inputs, subsystems=default_premission_subsystems),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:*', 'mission:*'],
        )

        model.add_subsystem(
            'aero',
            aero.build_mission(num_nodes=nn, aviary_inputs=flops_inputs, **{'method': 'computed'}),
            promotes=['*'],
        )

        # Set all options
        setup_model_options(prob, flops_inputs)

        prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, np.ones(1))

        prob.setup()

        # Mission params
        prob.set_val(Dynamic.Atmosphere.MACH, val=mach)
        prob.set_val(Dynamic.Atmosphere.STATIC_PRESSURE, val=P, units='lbf/ft**2')
        prob.set_val(Dynamic.Atmosphere.TEMPERATURE, val=T, units='degR')
        prob.set_val(Dynamic.Vehicle.MASS, val=mass, units='lbm')

        set_aviary_initial_values(prob, flops_inputs)

        prob.run_model()

        D = prob.get_val(Dynamic.Vehicle.DRAG, 'lbf')
        CD = D / (Sref * 0.5 * 1.4 * P * mach**2)

        # fmt: off
        data = np.array(
            [
                [
                    0.02494, 0.02544, 0.02621, 0.02727, 0.02859, 0.03015, 0.03191, 0.03381,
                    0.03632, 0.03901, 0.04175, 0.04483, 0.04825, 0.05205, 0.05621
                ],
                [
                    0.02325, 0.02375, 0.02453, 0.02558, 0.02690, 0.02847, 0.03022, 0.03213,
                    0.03464, 0.03733, 0.04006, 0.04315, 0.04657, 0.05037, 0.05452
                ],
                [
                    0.02212, 0.02261, 0.02339, 0.02444, 0.02577, 0.02733, 0.02908, 0.03099,
                    0.03350, 0.03618, 0.03892, 0.04201, 0.04542, 0.04922, 0.05338
                ],
                [
                    0.02125, 0.02173, 0.02250, 0.02357, 0.02491, 0.02646, 0.02820, 0.03011,
                    0.03261, 0.03530, 0.03804, 0.04112, 0.04454, 0.04834, 0.05250
                ],
                [
                    0.02084, 0.02130, 0.02208, 0.02316, 0.02451, 0.02604, 0.02776, 0.02969,
                    0.03218, 0.03487, 0.03761, 0.04069, 0.04411, 0.04791, 0.05207
                ],
                [
                    0.02087, 0.02127, 0.02203, 0.02316, 0.02457, 0.02605, 0.02772, 0.02967,
                    0.03213, 0.03482, 0.03756, 0.04065, 0.04407, 0.04788, 0.05204
                ],
                [
                    0.02115, 0.02152, 0.02228, 0.02343, 0.02486, 0.02632, 0.02795, 0.02991,
                    0.03233, 0.03501, 0.03803, 0.04171, 0.04606, 0.05113, 0.05689
                ],
                [
                    0.02160, 0.02192, 0.02268, 0.02386, 0.02535, 0.02681, 0.02842, 0.03038,
                    0.03291, 0.03596, 0.04024, 0.04509, 0.05057, 0.05668, 0.06342
                ],
                [
                    0.02274, 0.02293, 0.02366, 0.02494, 0.02659, 0.02820, 0.03004, 0.03244, 
                0.03596, 0.04066, 0.04560, 0.05224, 0.06038, 0.07001, 0.08114
                ],
                [
                    0.02516, 0.02493, 0.02561, 0.02720, 0.02954, 0.03219, 0.03543, 0.03964,
                    0.04534, 0.05181, 0.05827, 0.06510, 0.07228, 0.07984, 0.08776
                ],
                [
                    0.03957, 0.03827, 0.03877, 0.04108, 0.04524, 0.05136, 0.05820, 0.06586,
                    0.08233, 0.08717, 0.09438, 0.10072, 0.10656, 0.11240, 0.11803
                ]
            ]
        ).ravel()
        # fmt: on

        # Due to differences in the interp method, certain points that are extrapolated
        # (i.e., M or CL are well above design) have more than 1% error.
        assert_near_equal(CD, data, 0.05)

        assert_near_equal(CD[:134], data[:134], 0.005)

    def test_large_single_aisle_2_drag(self):
        flops_inputs = get_flops_inputs('LargeSingleAisle2FLOPS')
        flops_outputs = get_flops_outputs('LargeSingleAisle2FLOPS')

        # comparison data is unscaled drag
        flops_inputs.set_val(Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR, 1.0)
        flops_inputs.set_val(Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR, 1.0)
        flops_inputs.set_val(Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR, 1.0)
        flops_inputs.set_val(Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR, 1.0)

        key = Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST
        flops_inputs.set_val(key, *(flops_outputs.get_item(key)))
        flops_inputs.set_val(Settings.VERBOSITY, 0)

        engines = [build_engine_deck(flops_inputs)]
        preprocess_options(flops_inputs, engine_models=engines)

        # don't need mass subsystem, so we skip it
        default_premission_subsystems = get_default_premission_subsystems('FLOPS', engines)[:-1]
        # we just want aero for mission, make a copy by itself
        aero = default_premission_subsystems[-1]

        Sref = 1341.0

        # ---------------------
        # 1D Tables over M
        # ---------------------
        mach = np.array([0.2, 0.300, 0.400, 0.500, 0.600, 0.700, 0.750, 0.775, 0.800, 0.825, 0.850])
        CL = np.array(
            [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        )
        nm = len(mach)
        nc = len(CL)
        mach = np.repeat(mach, nc)
        CL = np.tile(CL, nm)
        nn = nm * nc

        P = 374.74437747
        T = 389.97

        lift = CL * Sref * 0.5 * 1.4 * P * mach**2
        mass = lift

        prob = om.Problem()
        model = prob.model

        # Upstream pre-mission analysis for aero
        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(aviary_options=flops_inputs, subsystems=default_premission_subsystems),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:*', 'mission:*'],
        )

        model.add_subsystem(
            'aero',
            aero.build_mission(num_nodes=nn, aviary_inputs=flops_inputs, **{'method': 'computed'}),
            promotes=['*'],
        )

        # Set all options
        setup_model_options(prob, flops_inputs)

        prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, np.ones(1))

        prob.setup()

        # Mission params
        prob.set_val(Dynamic.Atmosphere.MACH, val=mach)
        prob.set_val(Dynamic.Atmosphere.STATIC_PRESSURE, val=P, units='lbf/ft**2')
        prob.set_val(Dynamic.Atmosphere.TEMPERATURE, val=T, units='degR')
        prob.set_val(Dynamic.Vehicle.MASS, val=mass, units='lbm')

        set_aviary_initial_values(prob, flops_inputs)

        prob.run_model()

        D = prob.get_val(Dynamic.Vehicle.DRAG, 'lbf')
        CD = D / (Sref * 0.5 * 1.4 * P * mach**2)

        # fmt: off
        data = np.array(
            [
                [
                    0.02748, 0.02748, 0.02777, 0.02834, 0.02919, 0.03031, 0.03167, 0.03308,
                    0.03476, 0.03721, 0.03983, 0.04278, 0.04609, 0.04996, 0.05461
                ],
                [
                    0.02562, 0.02562, 0.02591, 0.02648, 0.02733, 0.02845, 0.02982, 0.03122,
                    0.03290, 0.03536, 0.03797, 0.04093, 0.04423, 0.04810, 0.05276
                ],
                [
                    0.02435, 0.02436, 0.02465, 0.02521, 0.02606, 0.02718, 0.02855, 0.02995,
                    0.03164, 0.03409, 0.03671, 0.03966, 0.04296, 0.04683, 0.05149
                ],
                [
                    0.02338, 0.02338, 0.02366, 0.02423, 0.02508, 0.02620, 0.02757, 0.02898,
                    0.03067, 0.03310, 0.03572, 0.03867, 0.04198, 0.04585, 0.05050
                ],
                [
                    0.02276, 0.02275, 0.02303, 0.02359, 0.02446, 0.02557, 0.02693, 0.02835,
                    0.03007, 0.03246, 0.03509, 0.03804, 0.04134, 0.04521, 0.04987
                ],
                [
                
                    0.02279, 0.02269, 0.02293, 0.02350, 0.02441, 0.02555, 0.02685, 0.02829,
                    0.03002, 0.03235, 0.03499, 0.03794, 0.04124, 0.04511, 0.04977
                ],
                [
                
                    0.02297, 0.02282, 0.02303, 0.02360, 0.02454, 0.02571, 0.02699, 0.02843,
                    0.03016, 0.03241, 0.03502, 0.03803, 0.04140, 0.04537, 0.05014
                ],
                [
                
                    0.02328, 0.02306, 0.02323, 0.02380, 0.02478, 0.02598, 0.02724, 0.02868,
                    0.03042, 0.03260, 0.03512, 0.03860, 0.04232, 0.04671, 0.05204
                ],
                [
                
                    0.02399, 0.02360, 0.02370, 0.02428, 0.02533, 0.02662, 0.02789, 0.02933,
                    0.03129, 0.03403, 0.03737, 0.04141, 0.04587, 0.05069, 0.05587
                ],
                [
                
                    0.02567, 0.02496, 0.02490, 0.02548, 0.02670, 0.02821, 0.02970, 0.03157,
                    0.03426, 0.03794, 0.04203, 0.04630, 0.05183, 0.05791, 0.06470
                ],
                [
                
                    0.04212, 0.04044, 0.03992, 0.04052, 0.04226, 0.04476, 0.04771, 0.05158,
                    0.05638, 0.06166, 0.06690, 0.07238, 0.07763, 0.08296, 0.08858
                ]
            ]
        ).ravel()
        # fmt: on

        # Due to differences in the interp method, certain points that are extrapolated (i.e.,
        # M or CL are well above design) have more than 1% error.
        assert_near_equal(CD, data, 0.05)

        assert_near_equal(CD[:134], data[:134], 0.005)


if __name__ == '__main__':
    # unittest.main()
    test = MissionDragTest()
    test.test_large_single_aisle_2_drag()
