import openmdao.api as om
import numpy as np

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import OutMachType
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.subsystems.propulsion.hamilton_standard import HamiltonStandard, PostHamiltonStandard, PreHamiltonStandard
from aviary.subsystems.propulsion.propeller_map import PropellerMap


class OutMachs(om.ExplicitComponent):
    """This utility sets up relations among helical Mach, free stream Mach and propeller tip Mach.
    helical_mach = sqrt(mach^2 + tip_mach^2).
    It compute the value of one from the inputs of the other two,
    """
    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare(
            "output_mach_type",
            default=OutMachType.HELICAL_MACH,
            types=OutMachType,
            desc="get one type of Mach number from the other two",
        )

    def setup(self):
        nn = self.options["num_nodes"]
        out_type = self.options["output_mach_type"]
        arange = np.arange(self.options["num_nodes"])

        if out_type is OutMachType.HELICAL_MACH:
            self.add_input(
                "mach",
                val=np.zeros(nn),
                units="unitless",
                desc="Mach number",
            )
            self.add_input(
                "tip_mach",
                val=np.zeros(nn),
                units="unitless",
                desc="tip Mach number at 3/4 of a blade",
            )
            self.add_output(
                "helical_mach",
                val=np.zeros(nn),
                units="unitless",
                desc="helical Mach number",
            )
            self.declare_partials("helical_mach", [
                                  "tip_mach", "mach"], rows=arange, cols=arange)
        elif out_type is OutMachType.MACH:
            self.add_input(
                "tip_mach",
                val=np.zeros(nn),
                units="unitless",
                desc="tip Mach number at 3/4 of a blade",
            )
            self.add_input(
                "helical_mach",
                val=np.zeros(nn),
                units="unitless",
                desc="helical Mach number",
            )
            self.add_output(
                "mach",
                val=np.zeros(nn),
                units="unitless",
                desc="Mach number",
            )
            self.declare_partials("mach", [
                                  "tip_mach", "helical_mach"], rows=arange, cols=arange)
        elif out_type is OutMachType.TIP_MACH:
            self.add_input(
                "mach",
                val=np.zeros(nn),
                units="unitless",
                desc="Mach number",
            )
            self.add_input(
                "helical_mach",
                val=np.zeros(nn),
                units="unitless",
                desc="helical Mach number",
            )
            self.add_output(
                "tip_mach",
                val=np.zeros(nn),
                units="unitless",
                desc="tip Mach number at 3/4 of a blade",
            )
            self.declare_partials("tip_mach", [
                                  "mach", "helical_mach"], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        out_type = self.options["output_mach_type"]

        if out_type is OutMachType.HELICAL_MACH:
            mach = inputs["mach"]
            tip_mach = inputs["tip_mach"]
            outputs["helical_mach"] = np.sqrt(mach * mach + tip_mach * tip_mach)
        elif out_type is OutMachType.MACH:
            tip_mach = inputs["tip_mach"]
            helical_mach = inputs["helical_mach"]
            outputs["mach"] = np.sqrt(helical_mach * helical_mach - tip_mach * tip_mach)
        elif out_type is OutMachType.TIP_MACH:
            mach = inputs["mach"]
            helical_mach = inputs["helical_mach"]
            outputs["tip_mach"] = np.sqrt(helical_mach * helical_mach - mach * mach)

    def compute_partials(self, inputs, J):
        out_type = self.options["output_mach_type"]

        if out_type is OutMachType.HELICAL_MACH:
            mach = inputs["MACH"]
            tip_mach = inputs["tip_mach"]
            J["helical_mach", "MACH"] = -mach/np.sqrt(mach * mach + tip_mach * tip_mach)
            J["helical_mach", "tip_mach"] = -tip_mach / \
                np.sqrt(mach * mach + tip_mach * tip_mach)
        elif out_type is OutMachType.MACH:
            tip_mach = inputs["tip_mach"]
            helical_mach = inputs["HELICAL_MACH"]
            J["mach", "helical_mach"] = -helical_mach / \
                np.sqrt(helical_mach * helical_mach - tip_mach * tip_mach)
            J["mach", "tip_mach"] = tip_mach / \
                np.sqrt(helical_mach * helical_mach - tip_mach * tip_mach)
        elif out_type is OutMachType.TIP_MACH:
            mach = inputs["MACH"]
            helical_mach = inputs["helical_mach"]
            J["tip_mach", "helical_mach"] = -helical_mach / \
                np.sqrt(helical_mach * helical_mach - mach * mach)
            J["tip_mach", "MACH"] = mach / \
                np.sqrt(helical_mach * helical_mach - mach * mach)


class InstallLoss(om.Group):
    """
    Compute installation loss
    """

    def initialize(self):
        self.options.declare(
            'num_nodes', types=int, default=1,
            desc='Number of nodes to be evaluated in the RHS')
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem(
            name='sqa_comp',
            subsys=om.ExecComp(
                'sqa = minimum(DiamNac**2/DiamProp**2, 0.50)',
                DiamNac={'val': 0, 'units': 'ft'},
                DiamProp={'val': 0, 'units': 'ft'},
                sqa={'units': 'unitless'},
                has_diag_partials=True,
            ),
            promotes_inputs=[("DiamNac", Aircraft.Nacelle.AVG_DIAMETER),
                             ("DiamProp", Aircraft.Engine.PROPELLER_DIAMETER)],
            promotes_outputs=["sqa"],
        )

        # We should update these minimum calls to use a smooth minimum so that the
        # gradient information is C1 continuous.
        self.add_subsystem(
            name='zje_comp',
            subsys=om.ExecComp(
                'equiv_adv_ratio = minimum((1.0 - 0.254 * sqa) * 5.309 * vktas/tipspd, 5.0)',
                vktas={'units': 'knot', 'val': np.zeros(nn)},
                tipspd={'units': 'ft/s', 'val': np.zeros(nn)},
                sqa={'units': 'unitless'},
                equiv_adv_ratio={'units': 'unitless', 'val': np.zeros(nn)},
                has_diag_partials=True,
            ),
            promotes_inputs=["sqa", ("vktas", Dynamic.Mission.VELOCITY),
                             ("tipspd", Dynamic.Mission.PROPELLER_TIP_SPEED)],
            promotes_outputs=["equiv_adv_ratio"],
        )

        self.add_subsystem(
            'convert_sqa',
            om.ExecComp(
                'sqa_array = sqa',
                sqa={'units': 'unitless'},
                sqa_array={'units': 'unitless', 'shape': (nn,)},
                has_diag_partials=True,
            ),
            promotes_inputs=["sqa"],
            promotes_outputs=["sqa_array"],
        )

        self.blockage_factor_interp = self.add_subsystem(
            "blockage_factor_interp",
            om.MetaModelStructuredComp(method="2D-slinear",
                                       extrapolate=True, vec_size=nn),
            promotes_inputs=["sqa_array", "equiv_adv_ratio"],
            promotes_outputs=[
                "blockage_factor",
            ],
        )

        self.blockage_factor_interp.add_input(
            "sqa_array",
            0.0,
            training_data=[0.00, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.50],
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
                 [0.940, 0.924, 0.902, 0.848, 0.790, 0.726, 0.662],
                 [0.904, 0.875, 0.835, 0.740, 0.655, 0.560, 0.464]]
            ),
        )

        self.add_subsystem(
            name='installation_loss_factor',
            subsys=om.ExecComp(
                'install_loss_factor = 1 - blockage_factor',
                blockage_factor={'units': 'unitless', 'val': np.zeros(nn)},
                install_loss_factor={'units': 'unitless', 'val': np.zeros(nn)},
                has_diag_partials=True,
            ),
            promotes_inputs=["blockage_factor"],
            promotes_outputs=["install_loss_factor"],
        )


class PropellerPerformance(om.Group):
    """
    Computation of propeller thrust coefficient based on Hamilton Standard or a user
    provided propeller map. Note that a propeller map allows either helical Mach number or
    free stream Mach number as input. This infomation will be detected automatically when the 
    propeller map is loaded into memory.
    The installation loss factor is either a user input or computed internally.
    """

    def initialize(self):
        self.options.declare(
            'num_nodes', types=int, default=1,
            desc='Number of nodes to be evaluated in the RHS')

        self.options.declare('aviary_options', types=AviaryValues,
                             desc='collection of Aircraft/Mission specific options')

    def setup(self):
        options = self.options
        nn = options['num_nodes']
        aviary_options = options['aviary_options']
        compute_installation_loss = aviary_options.get_val(
            Aircraft.Design.COMPUTE_INSTALLATION_LOSS)
        use_propeller_map = aviary_options.get_val(
            Aircraft.Engine.USE_PROPELLER_MAP)

        self.add_subsystem(
            'temperature_term',
            om.ExecComp(
                'theta_T = T0 * (1 + .2*mach**2)/T_amb',
                theta_T={'units': "unitless", 'shape': nn},
                T0={'units': 'degR', 'shape': nn},
                mach={'units': 'unitless', 'shape': nn},
                T_amb={'val': np.full(nn, 518.67), 'units': 'degR'},
                has_diag_partials=True,
            ),
            promotes_inputs=[
                ('T0', Dynamic.Mission.TEMPERATURE),
                ('mach', Dynamic.Mission.MACH),
            ],
            promotes_outputs=['theta_T'],
        )

        self.add_subsystem(
            'prop_tip_speed',
            om.ExecComp(
                'prop_tip_speed = minimum(pc_rotor_rpm_corrected * theta_T**.5 * max_prop_tip_spd, max_prop_tip_spd)',
                prop_tip_speed={'units': "ft/s", 'shape': nn},
                theta_T={'units': "unitless", 'shape': nn},
                pc_rotor_rpm_corrected={'units': "unitless", 'shape': nn},
                max_prop_tip_spd={'units': "ft/s", 'shape': nn},
                has_diag_partials=True,
            ),
            promotes_inputs=[
                'theta_T',
                ('pc_rotor_rpm_corrected', Dynamic.Mission.PERCENT_ROTOR_RPM_CORRECTED),
                ('max_prop_tip_spd', Aircraft.Design.MAX_PROPELLER_TIP_SPEED),
            ],
            promotes_outputs=[
                ('prop_tip_speed', Dynamic.Mission.PROPELLER_TIP_SPEED)],
        )

        if compute_installation_loss:
            self.add_subsystem(
                name='install_loss',
                subsys=InstallLoss(num_nodes=nn),
                promotes_inputs=[
                    Aircraft.Nacelle.AVG_DIAMETER,
                    Aircraft.Engine.PROPELLER_DIAMETER,
                    Dynamic.Mission.VELOCITY,
                    Dynamic.Mission.PROPELLER_TIP_SPEED,
                ],
                promotes_outputs=[
                    ("install_loss_factor", Dynamic.Mission.INSTALLATION_LOSS_FACTOR)],
            )
        else:
            self.set_input_defaults(
                Dynamic.Mission.INSTALLATION_LOSS_FACTOR, val=np.ones(nn), units="unitless")

        self.add_subsystem(
            name='pre_hamilton_standard',
            subsys=PreHamiltonStandard(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Mission.DENSITY,
                Dynamic.Mission.TEMPERATURE,
                Dynamic.Mission.VELOCITY,
                Dynamic.Mission.PROPELLER_TIP_SPEED,
                Aircraft.Engine.PROPELLER_DIAMETER,
                Dynamic.Mission.SHAFT_POWER,
            ],
            promotes_outputs=[
                "power_coefficient",
                "advance_ratio",
                "tip_mach",
                "density_ratio",
            ])

        if use_propeller_map:
            prop_model = PropellerMap('prop', aviary_options)
            prop_file_path = aviary_options.get_val(
                Aircraft.Engine.PROPELLER_DATA_FILE)
            print(f"prop_file_path: {prop_file_path}")
            mach_type = prop_model.read_and_set_mach_type(prop_file_path)
            print(f"mach type: {mach_type}")
            if mach_type == 'helical_mach':
                self.add_subsystem(
                    name='selectedMach',
                    subsys=OutMachs(
                        num_nodes=nn, output_mach_type=OutMachType.HELICAL_MACH),
                    promotes_inputs=[("mach", Dynamic.Mission.MACH), "tip_mach"],
                    promotes_outputs=[("helical_mach", "selected_mach")],
                )
            else:
                self.add_subsystem(
                    name='selectedMach',
                    subsys=om.ExecComp(
                        'selected_mach = mach',
                        mach={'units': 'unitless', 'shape': nn},
                        selected_mach={'units': 'unitless', 'shape': nn},
                        has_diag_partials=True,
                    ),
                    promotes_inputs=[("mach", Dynamic.Mission.MACH),],
                    promotes_outputs=["selected_mach"],
                )
            propeller = prop_model.build_propeller_interpolator(nn, aviary_options)
            self.add_subsystem(
                name='propeller_map',
                subsys=propeller,
                promotes_inputs=[
                    "selected_mach",
                    "power_coefficient",
                    "advance_ratio",
                ],
                promotes_outputs=[
                    "thrust_coefficient",
                ])

            # propeller map has taken compresibility into account.
            IVC = om.IndepVarComp("comp_tip_loss_factor",
                                  np.linspace(1.0, 1.0, nn),
                                  units='unitless')
            self.add_subsystem('IVC', IVC, promotes=['comp_tip_loss_factor'])
        else:
            self.add_subsystem(
                name='hamilton_standard',
                subsys=HamiltonStandard(num_nodes=nn, aviary_options=aviary_options),
                promotes_inputs=[
                    Dynamic.Mission.MACH,
                    "power_coefficient",
                    "advance_ratio",
                    "tip_mach",
                    Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR,
                    Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT,
                ],
                promotes_outputs=[
                    "thrust_coefficient",
                    "comp_tip_loss_factor",
                ])

        self.add_subsystem(
            name='post_hamilton_standard',
            subsys=PostHamiltonStandard(num_nodes=nn),
            promotes_inputs=[
                "thrust_coefficient",
                "comp_tip_loss_factor",
                Dynamic.Mission.PROPELLER_TIP_SPEED,
                Aircraft.Engine.PROPELLER_DIAMETER,
                "density_ratio",
                Dynamic.Mission.INSTALLATION_LOSS_FACTOR,
                "advance_ratio",
                "power_coefficient",
            ],
            promotes_outputs=[
                "thrust_coefficient_comp_loss",
                "propeller_thrust",
                "propeller_efficiency",
                "install_efficiency",
            ])
