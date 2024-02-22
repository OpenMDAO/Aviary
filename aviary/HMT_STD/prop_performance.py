import openmdao.api as om
import numpy as np
from dymos.models.atmosphere import USatm1976Comp
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic
from aviary.HMT_STD.hamilton_standard import HamiltonStandard, PostHamiltonStandard, PreHamiltonStandard

def print_report(num, p):
    print(f"Case {num}")
    mach = float(p.get_val(Dynamic.Mission.MACH)[0])
    tipm = float(p.get_val('tip_mach')[0])
    advr = float(p.get_val('adv_ratio')[0])
    cpow = float(p.get_val('power_coefficient')[0])
    loss = float(p.get_val('install_loss_factor')[0])
    print(f"Inputs to PERFM: Mach = {round(mach, 3)}, Tip Mach = {round(tipm, 3)}, Advance Ratio = {round(advr, 3)}, Power Coeff = {round(cpow, 5)}, Install Loss Factor = {round(loss, 4)}")
    cthr = float(p.get_val('thrust_coefficient')[0])
    ctlf = float(p.get_val('comp_tip_loss_factor')[0])
    tccl = float(p.get_val('thrust_coefficient_comp_loss')[0])
    angb = float(p.get_val('ang_blade')[0])
    print(f"Performance: CT = {round(cthr, 5)}, XFT = {round(ctlf, 4)}, CTX = {round(tccl, 5)}, 3/4 Blade Angle = {round(angb , 2)}")
    thrt = float(p.get_val('Thrust')[0])
    peff = float(p.get_val('propeller_efficiency')[0])
    lfac = float(p.get_val('install_loss_factor')[0])
    ieff = float(p.get_val('install_efficiency')[0])
    print(f"             Thrust = {round(thrt, 1)}, Prop Eff. = {round(peff, 4)}, Instll Loss = {round(lfac, 4)}, Install Eff. = {round(ieff, 4)}")

class PropPerf(om.Group):
    def initialize(self):
        self.options.declare(
            'num_nodes', types=int,
            desc='Number of nodes to be evaluated in the RHS')
        self.options.declare(
            'num_blade', types=int,
            desc='Number of blades')
        self.options.declare(
            'compute_blockage_factor', types=bool,
            desc='Flag to compute installation factor')
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')
        self.options.declare('num_blade', default=2, desc='number of blades')

    def setup(self):
            options = self.options
            #nn = options['num_nodes']
            num_blade = options['num_blade']

            self.add_subsystem(
                name='sqa_comp',
                subsys=om.ExecComp(
                    'sqa = minimum(DiamNac_DiamProp*DiamNac_DiamProp,0.32)',
                    DiamNac_DiamProp={'units': 'unitless'},
                    sqa={'units': 'unitless'},
                    has_diag_partials=True,
                ),
                promotes_inputs=["DiamNac_DiamProp"],
                promotes_outputs=["sqa"],
            )

            self.add_subsystem(
                name='zje_comp',
                subsys=om.ExecComp(
                    'equiv_adv_ratio = minimum((1.0 - 0.254 * sqa) * 5.309 * vktas/tipspd, 5.0)',
                    vktas={'units': 'knot'},
                    tipspd={'units': 'ft/s'},
                    sqa={'units': 'unitless'},
                    equiv_adv_ratio={'units': 'unitless'},
                ),
                promotes_inputs=["sqa","vktas","tipspd"],
                promotes_outputs=["equiv_adv_ratio"],
            )

            # JK NOTE it looks like tloss can be its own component. Optionally loaded? Or install_loss_factor is an override value? Might need to talk with Ken
            if self.options['compute_blockage_factor']:
                self.blockage_factor_interp = self.add_subsystem(
                    "blockage_factor_interp",
                    om.MetaModelStructuredComp(method="scipy_slinear", extrapolate=True),
                    promotes_inputs=["sqa","equiv_adv_ratio"],
                    promotes_outputs=[
                        "blockage_factor",
                    ],
                )

                self.blockage_factor_interp.add_input(
                    "sqa",
                    0.0,
                    training_data=[0.00, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32],
                    units="unitless",
                    desc="square of DiamNac/DiamProp",
                )

                self.blockage_factor_interp.add_input(
                    "equiv_adv_ratio",
                    0.0,
                    training_data=[0., 0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
                    units="unitless",
                    desc="square of DiamNac vs DiamProp",
                )
            
                self.blockage_factor_interp.add_output(
                    "blockage_factor",
                    0.765,
                    units="unitless",
                    desc="blockage factor",
                    training_data=np.array(
                        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [0.992, 0.991, 0.988, 0.983, 0.976, 0.970, 0.963],
                        [0.986, 0.982, 0.977, 0.965, 0.953, 0.940, 0.927],
                        [0.979, 0.974, 0.967, 0.948, 0.929, 0.908, 0.887],
                        [0.972, 0.965, 0.955, 0.932, 0.905, 0.872, 0.835],
                        [0.964, 0.954, 0.943, 0.912, 0.876, 0.834, 0.786],
                        [0.955, 0.943, 0.928, 0.892, 0.848, 0.801, 0.751],
                        [0.948, 0.935, 0.917, 0.872, 0.820, 0.763, 0.706],
                        [0.940, 0.924, 0.902, 0.848, 0.790, 0.726, 0.662]]
                        ),
                )
            else:
                self.add_subsystem(
                    name='input_install',
                    subsys=om.ExecComp(
                        'blockage_factor = input_blockage_factor',
                        input_blockage_factor={'units': 'unitless'},
                        blockage_factor={'units': 'unitless'},
                        has_diag_partials=True,
                    ),
                    promotes_inputs=["input_blockage_factor"],
                    promotes_outputs=["blockage_factor"],
                )

            self.add_subsystem(
                name='FT',
                subsys=om.ExecComp(
                    'install_loss_factor = 1 - blockage_factor',
                    blockage_factor={'units': 'unitless'},
                    install_loss_factor={'units': 'unitless'},
                ),
                promotes_inputs=["blockage_factor"],
                promotes_outputs=["install_loss_factor"],
            )

            self.add_subsystem(
                name='atmosphere',
                subsys=USatm1976Comp(num_nodes=1),
                promotes_inputs=[('h', Dynamic.Mission.ALTITUDE)],
                promotes_outputs=[
                    ('sos', Dynamic.Mission.SPEED_OF_SOUND), ('rho', Dynamic.Mission.DENSITY),
                    ('temp', Dynamic.Mission.TEMPERATURE), ('pres', Dynamic.Mission.STATIC_PRESSURE)],
            )

            self.add_subsystem(
                name='pre_hamilton_standard',
                subsys=PreHamiltonStandard(),
                promotes_inputs=[
                    ("rho", Dynamic.Mission.DENSITY),
                    ("temp", Dynamic.Mission.TEMPERATURE),
                    "vktas",
                    "tipspd",
                    "diam_prop",
                    "SHP",
                ],
                promotes_outputs=[
                    Dynamic.Mission.MACH,
                    "power_coefficient",
                    "adv_ratio",
                    "tip_mach",
                    "density_ratio",
                ])

            self.add_subsystem(
                name='hamilton_standard',
                subsys=HamiltonStandard(num_blade=num_blade),
                promotes_inputs=[
                    Dynamic.Mission.MACH,
                    "power_coefficient",
                    "adv_ratio",
                    "tip_mach",
                    "act_fac",
                    "cli"],
                promotes_outputs=[
                    "thrust_coefficient",
                    "ang_blade",
                    "comp_tip_loss_factor",
                ])

            self.add_subsystem(
                name='post_hamilton_standard',
                subsys=PostHamiltonStandard(),
                promotes_inputs=[
                    "thrust_coefficient",
                    "comp_tip_loss_factor",
                    "tipspd",
                    "diam_prop",
                    "density_ratio",
                    "install_loss_factor",
                    "adv_ratio",
                    "power_coefficient",
                    ],
                promotes_outputs=[
                    "thrust_coefficient_comp_loss",
                    "Thrust",
                    "propeller_efficiency",
                    "install_efficiency",
                ])


if __name__ == "__main__":
    prob = om.Problem()
    pp = prob.model.add_subsystem(
        'pp',
        PropPerf(),
        promotes_inputs=['*'],
        promotes_outputs=["*"],
    )

    pp.set_input_defaults('tipspd', 800, units="ft/s")
    pp.set_input_defaults('vktas', 0, units="knot")
    pp.options.set(num_blade=4)
    pp.options.set(compute_blockage_factor=True)
    prob.setup()

    print()
    prob.set_val('diam_prop', 10.5, units="ft")
    prob.set_val('act_fac', 114.0, units="unitless")
    prob.set_val('cli', 0.5, units="unitless")
    prob.set_val('DiamNac_DiamProp', 0.275, units="unitless")

    #"""
    # Case 1
    prob.set_val(Dynamic.Mission.ALTITUDE, 0.0, units="ft")
    prob.set_val('vktas', 0.0, units="knot")
    prob.set_val('tipspd', 800.0, units="ft/s")
    prob.set_val('SHP', 1850.0, units="hp")

    print("Case 1")
    print(f"DPROP,BLADT,AFT,CLI: {prob.get_val('diam_prop')}, {pp.options['num_blade']}, {prob.get_val('act_fac')}, {prob.get_val('cli')}")
    print(f"Nacalle Diam to Prop Diam Ratio: {prob.get_val('DiamNac_DiamProp')}")
    print(
         f"ALT, VKTAS, VTIP, SHP & FT: {prob.get_val(Dynamic.Mission.ALTITUDE)}, {prob.get_val('vktas')}, {prob.get_val('tipspd')}, {prob.get_val('SHP')}, {0.0},")
    prob.run_model()
    print_report(1, prob)

    # Case 2
    prob.set_val(Dynamic.Mission.ALTITUDE, 0.0, units="ft")
    prob.set_val('vktas', 125.0, units="knot")
    prob.set_val('tipspd', 800.0, units="ft/s")
    prob.set_val('SHP', 1850.0, units="hp")

    print("Case 2")
    print(f"DPROP,BLADT,AFT,CLI: {prob.get_val('diam_prop')}, {pp.options['num_blade']}, {prob.get_val('act_fac')}, {prob.get_val('cli')}")
    print(f"Nacalle Diam to Prop Diam Ratio: {prob.get_val('DiamNac_DiamProp')}")
    print(
         f"ALT, VKTAS, VTIP, SHP & FT: {prob.get_val(Dynamic.Mission.ALTITUDE)}, {prob.get_val('vktas')}, {prob.get_val('tipspd')}, {prob.get_val('SHP')}, {0.0},")
    prob.run_model()
    print_report(2, prob)

    # Case 3
    prob.set_val(Dynamic.Mission.ALTITUDE, 25000.0, units="ft")
    prob.set_val('vktas', 300.0, units="knot")
    prob.set_val('tipspd', 750.0, units="ft/s")
    prob.set_val('SHP', 900.0, units="hp")

    print("Case 3")
    print(f"DPROP,BLADT,AFT,CLI: {prob.get_val('diam_prop')}, {pp.options['num_blade']}, {prob.get_val('act_fac')}, {prob.get_val('cli')}")
    print(f"Nacalle Diam to Prop Diam Ratio: {prob.get_val('DiamNac_DiamProp')}")
    print(
         f"ALT, VKTAS, VTIP, SHP & FT: {prob.get_val(Dynamic.Mission.ALTITUDE)}, {prob.get_val('vktas')}, {prob.get_val('tipspd')}, {prob.get_val('SHP')}, {0.0},")
    prob.run_model()
    print_report(3, prob)
    #"""

    # Case 4
    pp.options.set(compute_blockage_factor=False)
    prob.setup()
    prob.set_val('input_blockage_factor', 1.0, units="unitless")
    prob.set_val('diam_prop', 12.0, units="ft")
    prob.set_val('act_fac', 150.0, units="unitless")
    prob.set_val('cli', 0.5, units="unitless")
    prob.set_val('DiamNac_DiamProp', 0.2, units="unitless")
    prob.set_val(Dynamic.Mission.ALTITUDE, 10000.0, units="ft")
    prob.set_val('vktas', 200.0, units="knot")
    prob.set_val('tipspd', 750.0, units="ft/s")
    prob.set_val('SHP', 1000.0, units="hp")

    print("Case 4")
    print(f"DPROP,BLADT,AFT,CLI: {prob.get_val('diam_prop')}, {pp.options['num_blade']}, {prob.get_val('act_fac')}, {prob.get_val('cli')}")
    print(f"Nacalle Diam to Prop Diam Ratio: {prob.get_val('DiamNac_DiamProp')}")
    print(
         f"ALT, VKTAS, VTIP, SHP & FT: {prob.get_val(Dynamic.Mission.ALTITUDE)}, {prob.get_val('vktas')}, {prob.get_val('tipspd')}, {prob.get_val('SHP')}, {0.0},")
    prob.run_model()
    print_report(4, prob)

    # Case 5
    prob.set_val('input_blockage_factor', 0.95, units="unitless")
    print("Case 5")
    print(f"DPROP,BLADT,AFT,CLI: {prob.get_val('diam_prop')}, {pp.options['num_blade']}, {prob.get_val('act_fac')}, {prob.get_val('cli')}")
    print(f"Nacalle Diam to Prop Diam Ratio: {prob.get_val('DiamNac_DiamProp')}")
    print(
         f"ALT, VKTAS, VTIP, SHP & FT: {prob.get_val(Dynamic.Mission.ALTITUDE)}, {prob.get_val('vktas')}, {prob.get_val('tipspd')}, {prob.get_val('SHP')}, {0.0},")
    prob.run_model()
    print_report(5, prob)

    # Case 6
    prob.set_val(Dynamic.Mission.ALTITUDE, 0.0, units="ft")
    prob.set_val('vktas', 50.0, units="knot")
    prob.set_val('tipspd', 785.0, units="ft/s")
    prob.set_val('SHP', 1250.0, units="hp")

    print("Case 6")
    print(f"DPROP,BLADT,AFT,CLI: {prob.get_val('diam_prop')}, {pp.options['num_blade']}, {prob.get_val('act_fac')}, {prob.get_val('cli')}")
    print(f"Nacalle Diam to Prop Diam Ratio: {prob.get_val('DiamNac_DiamProp')}")
    print(
         f"ALT, VKTAS, VTIP, SHP & FT: {prob.get_val(Dynamic.Mission.ALTITUDE)}, {prob.get_val('vktas')}, {prob.get_val('tipspd')}, {prob.get_val('SHP')}, {0.0},")
    prob.run_model()
    print_report(6, prob)

    # Case 7
    pp.options.set(num_blade=3)
    prob.setup()
    prob.set_val('input_blockage_factor', 1.0, units="unitless")
    prob.set_val('diam_prop', 12.0, units="ft")
    prob.set_val('act_fac', 150.0, units="unitless")
    prob.set_val('cli', 0.5, units="unitless")
    prob.set_val('DiamNac_DiamProp', 0.2, units="unitless")
    prob.set_val(Dynamic.Mission.ALTITUDE, 10000.0, units="ft")
    prob.set_val('vktas', 200.0, units="knot")
    prob.set_val('tipspd', 750.0, units="ft/s")
    prob.set_val('SHP', 1000.0, units="hp")

    print("Case 7")
    print(f"DPROP,BLADT,AFT,CLI: {prob.get_val('diam_prop')}, {pp.options['num_blade']}, {prob.get_val('act_fac')}, {prob.get_val('cli')}")
    print(f"Nacalle Diam to Prop Diam Ratio: {prob.get_val('DiamNac_DiamProp')}")
    print(
         f"ALT, VKTAS, VTIP, SHP & FT: {prob.get_val(Dynamic.Mission.ALTITUDE)}, {prob.get_val('vktas')}, {prob.get_val('tipspd')}, {prob.get_val('SHP')}, {0.0},")
    prob.run_model()
    print_report(7, prob)

    # Case 8
    prob.set_val('input_blockage_factor', 0.95, units="unitless")
    print("Case 8")
    print(f"DPROP,BLADT,AFT,CLI: {prob.get_val('diam_prop')}, {pp.options['num_blade']}, {prob.get_val('act_fac')}, {prob.get_val('cli')}")
    print(f"Nacalle Diam to Prop Diam Ratio: {prob.get_val('DiamNac_DiamProp')}")
    print(
         f"ALT, VKTAS, VTIP, SHP & FT: {prob.get_val(Dynamic.Mission.ALTITUDE)}, {prob.get_val('vktas')}, {prob.get_val('tipspd')}, {prob.get_val('SHP')}, {0.0},")
    prob.run_model()
    print_report(8, prob)

    # Case 9
    prob.set_val(Dynamic.Mission.ALTITUDE, 0.0, units="ft")
    prob.set_val('vktas', 50.0, units="knot")
    prob.set_val('tipspd', 785.0, units="ft/s")
    prob.set_val('SHP', 1250.0, units="hp")

    print("Case 9")
    print(f"DPROP,BLADT,AFT,CLI: {prob.get_val('diam_prop')}, {pp.options['num_blade']}, {prob.get_val('act_fac')}, {prob.get_val('cli')}")
    print(f"Nacalle Diam to Prop Diam Ratio: {prob.get_val('DiamNac_DiamProp')}")
    print(
         f"ALT, VKTAS, VTIP, SHP & FT: {prob.get_val(Dynamic.Mission.ALTITUDE)}, {prob.get_val('vktas')}, {prob.get_val('tipspd')}, {prob.get_val('SHP')}, {0.0},")
    prob.run_model()
    print_report(9, prob)

    """
    print()
    print("Inputs to HamiltonStandard:")
    print(f"  power_coefficient: {prob.get_val('pp.power_coefficient')[0]}")
    print(f"  adv_ratio: {prob.get_val('pp.adv_ratio')[0]}")
    print(f"  act_fac: {prob.get_val('pp.act_fac')[0]}")
    print(f"  cli: {prob.get_val('pp.cli')[0]}")
    print(f"  mach: {prob.get_val('pp.mach')[0]}")
    print(f"  tip_mach: {prob.get_val('pp.tip_mach')[0]}")
    print(f"  num_blade: {pp.options['num_blade']}")
    print()

    # print(f"sqa = {prob.get_val('sqa')[0]}")
    # print(f"equiv_adv_ratio = {prob.get_val('equiv_adv_ratio')[0]}")
    # print(f"density = {prob.get_val(Dynamic.Mission.DENSITY)[0]}")
    # print(f"temp = {prob.get_val(Dynamic.Mission.TEMPERATURE)[0]}")
    # print(f"blockage_factor = {prob.get_val('blockage_factor')[0]}")
    # print(f"install_loss_factor = {prob.get_val('install_loss_factor')[0]}")
    # print(f"mach = {prob.get_val('mach')[0]}")
    # print(f"cp = {prob.get_val('power_coefficient')[0]}")
    # print(f"adv_ratio = {prob.get_val('adv_ratio')[0]}")
    # print(f"tip_mach = {prob.get_val('tip_mach')[0]}")
    # print(f"thrust_coefficient = {prob.get_val('thrust_coefficient')[0]}")
    # print(f"ang_blade = {prob.get_val('ang_blade')[0]}")
    # print(f"comp_tip_loss_factor = {prob.get_val('comp_tip_loss_factor')[0]}")

    print("Outputs:")
    print(f"  3/4 Blade Angle = {prob.get_val('ang_blade')[0]}")
    print(f"  CT = {prob.get_val('thrust_coefficient')[0]}")
    print(f"  XFT = {prob.get_val('comp_tip_loss_factor')[0]}")

    print(f"num_blade = {pp.options['num_blade']}")
    """
