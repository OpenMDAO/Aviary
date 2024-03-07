import openmdao.api as om
import numpy as np
from dymos.models.atmosphere import USatm1976Comp
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.variable_info.options import get_option_defaults
from aviary.subsystems.propulsion.hamilton_standard import HamiltonStandard, PostHamiltonStandard, PreHamiltonStandard
import pdb


def _print_report1(nCase, p, nBlades):
    print()
    print(f"Case {nCase}")
    diam = p.get_val(Aircraft.Engine.PROPELLER_DIAMETER)
    if diam.ndim == 1:
        diam = diam[0]
    actF = p.get_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR)
    if actF.ndim == 1:
        actF = actF[0]
    pcli = p.get_val(Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICENT)
    if pcli.ndim == 1:
        pcli = pcli[0]
    print(f"DPROP,BLADT,AFT,CLI: {diam}, {nBlades}, {actF}, {pcli}")
    try:
        # only when compute_installation_loss is true
        DiamNac = p.get_val(Aircraft.Nacelle.AVG_DIAMETER)
        DiamProp = p.get_val(Aircraft.Engine.PROPELLER_DIAMETER)
        if DiamNac.ndim == 0:
            DiamNac = float(DiamNac)
        else:
            DiamNac = float(DiamNac[0])
        if DiamProp.ndim == 0:
            DiamProp = float(DiamProp)
        else:
            DiamProp = float(DiamProp[0])
        DiamNac_DiamProp = DiamNac/DiamProp
        print(f"Nacelle Diam to Prop Diam Ratio: {round(DiamNac_DiamProp, 3)}")
    except:
        pass
    alt = p.get_val(Dynamic.Mission.ALTITUDE)
    if alt.ndim == 1:
        alt = alt[0]
    vktas = p.get_val(Dynamic.Mission.VELOCITY)
    if vktas.ndim == 1:
        vktas = vktas[0]
    tipspd = p.get_val(Dynamic.Mission.PROPELLER_TIP_SPEED)
    if tipspd.ndim == 1:
        tipspd = tipspd[0]
    shp = p.get_val(Dynamic.Mission.SHAFT_POWER)
    if shp.ndim == 1:
        shp = shp[0]
    print(
        f"ALT, VKTAS, VTIP, SHP: {alt}, {vktas}, {tipspd}, {shp}")


def _print_report2(p):
    mach = float(p.get_val(Dynamic.Mission.MACH)[0])
    tipm = float(p.get_val('tip_mach')[0])
    advr = float(p.get_val('adv_ratio')[0])
    cpow = float(p.get_val('power_coefficient')[0])
    loss = float(p.get_val(Aircraft.Engine.INSTALLATION_LOSS_FACTOR)[0])
    print(f"Inputs to PERFM: Mach = {round(mach, 3)}, Tip Mach = {round(tipm, 3)}, Advance Ratio = {round(advr, 3)}, Power Coeff = {round(cpow, 5)}, Install Loss Factor = {round(loss, 4)}")
    cthr = float(p.get_val('thrust_coefficient')[0])
    ctlf = float(p.get_val('comp_tip_loss_factor')[0])
    tccl = float(p.get_val('thrust_coefficient_comp_loss')[0])
    angb = float(p.get_val('ang_blade')[0])
    print(
        f"Performance: CT = {round(cthr, 5)}, XFT = {round(ctlf, 4)}, CTX = {round(tccl, 5)}, 3/4 Blade Angle = {round(angb , 2)}")
    thrt = float(p.get_val('Thrust')[0])
    peff = float(p.get_val('propeller_efficiency')[0])
    lfac = float(p.get_val(Aircraft.Engine.INSTALLATION_LOSS_FACTOR)[0])
    ieff = float(p.get_val('install_efficiency')[0])
    print(
        f"             Thrust = {round(thrt, 1)}, Prop Eff. = {round(peff, 4)}, Instll Loss = {round(lfac, 4)}, Install Eff. = {round(ieff, 4)}")


class InstallLoss(om.Group):
    """
    Compute installation loss
    """

    def initialize(self):
        self.options.declare(
            'num_nodes', types=int,
            desc='Number of nodes to be evaluated in the RHS')
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        self.add_subsystem(
            name='sqa_comp',
            subsys=om.ExecComp(
                'sqa = minimum(DiamNac/DiamProp*DiamNac/DiamProp, 0.32)',
                DiamNac={'units': 'ft'},
                DiamProp={'units': 'ft'},
                sqa={'units': 'unitless'},
                has_diag_partials=True,
            ),
            promotes_inputs=[("DiamNac", Aircraft.Nacelle.AVG_DIAMETER),
                             ("DiamProp", Aircraft.Engine.PROPELLER_DIAMETER)],
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
            promotes_inputs=["sqa", ("vktas", Dynamic.Mission.VELOCITY),
                             ("tipspd", Dynamic.Mission.PROPELLER_TIP_SPEED)],
            promotes_outputs=["equiv_adv_ratio"],
        )

        self.blockage_factor_interp = self.add_subsystem(
            "blockage_factor_interp",
            om.MetaModelStructuredComp(method="scipy_slinear", extrapolate=True),
            promotes_inputs=["sqa", "equiv_adv_ratio"],
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


class PropPerf(om.Group):
    """
    Computation of propeller thrust coefficient based on Hamilton Standard.
    The installation loss factor is either a user input or computed internally.
    """

    def initialize(self):
        self.options.declare(
            'num_nodes', types=int,
            desc='Number of nodes to be evaluated in the RHS')
        self.options.declare(Aircraft.Engine.NUM_BLADES, default=2,
                             desc='number of blades per propeller')
        self.options.declare(
            Aircraft.Design.COMPUTE_INSTALLATION_LOSS, types=bool,
            desc='Flag to compute installation factor')
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        options = self.options
        # nn = options['num_nodes']
        options = options['aviary_options']
        compute_installation_loss = options.get_val(
            Aircraft.Design.COMPUTE_INSTALLATION_LOSS)
        num_blades = options.get_val(Aircraft.Engine.NUM_BLADES)

        # JK NOTE it looks like tloss can be its own component. Optionally loaded? Or install_loss_factor is an override value? Might need to talk with Ken
        if compute_installation_loss:
            self.add_subsystem(
                name='loss',
                subsys=InstallLoss(),
                promotes_inputs=[
                    Aircraft.Nacelle.AVG_DIAMETER,
                    Aircraft.Engine.PROPELLER_DIAMETER,
                    Dynamic.Mission.VELOCITY,
                    Dynamic.Mission.PROPELLER_TIP_SPEED,
                ],
                promotes_outputs=["install_loss_factor"],
            )
        else:
            user_set_loss_factor = options.get_val(
                Aircraft.Engine.INSTALLATION_LOSS_FACTOR)
            comp = om.IndepVarComp()
            comp.add_output('install_loss_factor',
                            val=user_set_loss_factor, units="unitless")
            self.add_subsystem('input_install_loss', comp,
                               promotes=['install_loss_factor'])

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
                Dynamic.Mission.DENSITY,
                Dynamic.Mission.TEMPERATURE,
                Dynamic.Mission.VELOCITY,
                Dynamic.Mission.PROPELLER_TIP_SPEED,
                Aircraft.Engine.PROPELLER_DIAMETER,
                Dynamic.Mission.SHAFT_POWER,
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
            subsys=HamiltonStandard(num_blades=num_blades),
            promotes_inputs=[
                Dynamic.Mission.MACH,
                "power_coefficient",
                "adv_ratio",
                "tip_mach",
                Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR,
                Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICENT,
            ],
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
                Dynamic.Mission.PROPELLER_TIP_SPEED,
                Aircraft.Engine.PROPELLER_DIAMETER,
                "density_ratio",
                Aircraft.Engine.INSTALLATION_LOSS_FACTOR,
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
    options = get_option_defaults()
    options.set_val(Aircraft.Design.COMPUTE_INSTALLATION_LOSS,
                    val=True, units='unitless')
    options.set_val(Aircraft.Engine.NUM_BLADES,
                    val=4, units='unitless')

    prob = om.Problem()
    pp = prob.model.add_subsystem(
        'pp',
        PropPerf(aviary_options=options),
        promotes_inputs=['*'],
        promotes_outputs=["*"],
    )

    pp.set_input_defaults(Aircraft.Engine.PROPELLER_DIAMETER, 10, units="ft")
    pp.set_input_defaults(Dynamic.Mission.PROPELLER_TIP_SPEED, 800, units="ft/s")
    pp.set_input_defaults(Dynamic.Mission.VELOCITY, 0, units="knot")
    num_blades = 4
    options.set_val(Aircraft.Engine.NUM_BLADES,
                    val=num_blades, units='unitless')
    # pp.options.set(compute_installation_loss=True)
    options.set_val(Aircraft.Design.COMPUTE_INSTALLATION_LOSS,
                    val=True, units='unitless')
    prob.setup()

    print()
    prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 10.5, units="ft")
    prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 114.0, units="unitless")
    prob.set_val(Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICENT,
                 0.5, units="unitless")
    # prob.set_val('DiamNac_DiamProp', 0.275, units="unitless")
    prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, 2.8875, units='ft')

    # Case 1
    prob.set_val(Dynamic.Mission.ALTITUDE, 0.0, units="ft")
    prob.set_val(Dynamic.Mission.VELOCITY, 0.0, units="knot")
    prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED, 800.0, units="ft/s")
    prob.set_val(Dynamic.Mission.SHAFT_POWER, 1850.0, units="hp")

    _print_report1(1, prob, num_blades)
    prob.run_model()
    _print_report2(prob)

    # Case 2
    prob.set_val(Dynamic.Mission.ALTITUDE, 0.0, units="ft")
    prob.set_val(Dynamic.Mission.VELOCITY, 125.0, units="knot")
    prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED, 800.0, units="ft/s")
    prob.set_val(Dynamic.Mission.SHAFT_POWER, 1850.0, units="hp")

    _print_report1(2, prob, num_blades)
    prob.run_model()
    _print_report2(prob)

    # Case 3
    prob.set_val(Dynamic.Mission.ALTITUDE, 25000.0, units="ft")
    prob.set_val(Dynamic.Mission.VELOCITY, 300.0, units="knot")
    prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED, 750.0, units="ft/s")
    prob.set_val(Dynamic.Mission.SHAFT_POWER, 900.0, units="hp")

    _print_report1(3, prob, num_blades)
    prob.run_model()
    _print_report2(prob)

    # Case 4
    # pp.options.set(compute_installation_loss=False)
    options.set_val(Aircraft.Design.COMPUTE_INSTALLATION_LOSS,
                    val=False, units='unitless')
    options.set_val(Aircraft.Engine.INSTALLATION_LOSS_FACTOR, 0.0, units="unitless")
    prob.setup()
    prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 12.0, units="ft")
    # prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, 2.4, units='ft')
    prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 150.0, units="unitless")
    prob.set_val(Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICENT,
                 0.5, units="unitless")
    # prob.set_val('DiamNac_DiamProp', 0.2, units="unitless")
    prob.set_val(Dynamic.Mission.ALTITUDE, 10000.0, units="ft")
    prob.set_val(Dynamic.Mission.VELOCITY, 200.0, units="knot")
    prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED, 750.0, units="ft/s")
    prob.set_val(Dynamic.Mission.SHAFT_POWER, 1000.0, units="hp")

    _print_report1(4, prob, num_blades)
    prob.run_model()
    _print_report2(prob)

    # Case 5
    prob.set_val(Aircraft.Engine.INSTALLATION_LOSS_FACTOR, 0.05, units="unitless")

    _print_report1(5, prob, num_blades)
    prob.run_model()
    _print_report2(prob)

    # Case 6
    prob.set_val(Dynamic.Mission.ALTITUDE, 0.0, units="ft")
    prob.set_val(Dynamic.Mission.VELOCITY, 50.0, units="knot")
    prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED, 785.0, units="ft/s")
    prob.set_val(Dynamic.Mission.SHAFT_POWER, 1250.0, units="hp")

    _print_report1(6, prob, num_blades)
    prob.run_model()
    _print_report2(prob)

    # Case 7
    num_blades = 3
    options.set_val(Aircraft.Engine.NUM_BLADES,
                    val=num_blades, units='unitless')
    prob.setup()
    prob.set_val(Aircraft.Engine.INSTALLATION_LOSS_FACTOR, 0.0, units="unitless")
    prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 12.0, units="ft")
    # prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, 2.4, units='ft')
    prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 150.0, units="unitless")
    prob.set_val(Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICENT,
                 0.5, units="unitless")
    # prob.set_val('DiamNac_DiamProp', 0.2, units="unitless")
    prob.set_val(Dynamic.Mission.ALTITUDE, 10000.0, units="ft")
    prob.set_val(Dynamic.Mission.VELOCITY, 200.0, units="knot")
    prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED, 750.0, units="ft/s")
    prob.set_val(Dynamic.Mission.SHAFT_POWER, 1000.0, units="hp")

    _print_report1(7, prob, num_blades)
    prob.run_model()
    _print_report2(prob)

    # Case 8
    prob.set_val(Aircraft.Engine.INSTALLATION_LOSS_FACTOR, 0.05, units="unitless")
    _print_report1(8, prob, num_blades)
    prob.run_model()
    _print_report2(prob)

    # Case 9
    prob.set_val(Dynamic.Mission.ALTITUDE, 0.0, units="ft")
    prob.set_val(Dynamic.Mission.VELOCITY, 50.0, units="knot")
    prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED, 785.0, units="ft/s")
    prob.set_val(Dynamic.Mission.SHAFT_POWER, 1250.0, units="hp")

    _print_report1(9, prob, num_blades)
    prob.run_model()
    _print_report2(prob)
