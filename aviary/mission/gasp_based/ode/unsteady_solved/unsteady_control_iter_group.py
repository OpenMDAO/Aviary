import numpy as np
import openmdao.api as om

from aviary.constants import RHO_SEA_LEVEL_ENGLISH

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic

from .unsteady_solved_eom import UnsteadySolvedEOM


class UnsteadyControlIterGroup(om.Group):
    """
    This Group contains a nonlinear solver to determine alpha and thrust for a given flight condition.
    """

    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare("ground_roll", types=bool, default=False,
                             desc="True if the aircraft is confined to the ground. Removes altitude rate as an "
                                  "output and adjusts the TAS rate equation.")
        self.options.declare("clean", types=bool, default=False,
                             desc="If true then no flaps or gear are included. Useful for high-speed flight phases.")
        self.options.declare(
            'aviary_options', types=AviaryValues, default=None,
            desc='collection of Aircraft/Mission specific options'
        )

        # TODO finish description
        self.options.declare(
            'core_subsystems',
            desc='list of core subsystems'
        )

        self.options.declare(
            'subsystem_options', types=dict, default={},
            desc='dictionary of parameters to be passed to the subsystem builders'
        )

    def setup(self):
        nn = self.options["num_nodes"]
        ground_roll = self.options["ground_roll"]
        clean = self.options["clean"]
        aviary_options = self.options['aviary_options']
        core_subsystems = self.options['core_subsystems']

        kwargs = {'num_nodes': nn, 'aviary_inputs': aviary_options}

        if clean:
            kwargs['method'] = 'low_speed'
        else:
            kwargs['method'] = 'cruise'

        for subsystem in core_subsystems:
            system = subsystem.build_mission(**kwargs)
            if system is not None:
                self.add_subsystem(subsystem.name,
                                   system,
                                   promotes_inputs=subsystem.mission_inputs(**kwargs),
                                   promotes_outputs=subsystem.mission_outputs(**kwargs))

        eom_comp = UnsteadySolvedEOM(num_nodes=nn, ground_roll=ground_roll)

        self.add_subsystem("eom", subsys=eom_comp,
                           promotes_inputs=["*",
                                            (Dynamic.Mission.THRUST_TOTAL, "thrust_req")],
                           promotes_outputs=["*"])

        thrust_alpha_bal = om.BalanceComp()
        if not self.options['ground_roll']:
            thrust_alpha_bal.add_balance("alpha",
                                         units="rad",
                                         val=np.zeros(nn),
                                         lhs_name="dgam_dt_approx",
                                         rhs_name="dgam_dt",
                                         eq_units="rad/s",
                                         normalize=False)

        thrust_alpha_bal.add_balance("thrust_req",
                                     units="N",
                                     val=100*np.ones(nn),
                                     lhs_name="dTAS_dt_approx",
                                     rhs_name="dTAS_dt",
                                     eq_units="m/s**2",
                                     normalize=False)

        self.add_subsystem("thrust_alpha_bal", subsys=thrust_alpha_bal,
                           promotes_inputs=["*"],
                           promotes_outputs=["*"])

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True,
                                                atol=1.0e-10,
                                                rtol=1.0e-10)
        # self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        self.linear_solver = om.DirectSolver(assemble_jac=True)

        # Set common default values for promoted inputs
        onn = np.ones(nn)
        self.set_input_defaults(
            name="rho", val=RHO_SEA_LEVEL_ENGLISH * onn, units="slug/ft**3")
        self.set_input_defaults(
            name=Dynamic.Mission.SPEED_OF_SOUND,
            val=1116.4 * onn,
            units="ft/s")
        if not self.options['ground_roll']:
            self.set_input_defaults(name=Dynamic.Mission.FLIGHT_PATH_ANGLE,
                                    val=0.0 * onn, units="rad")
        self.set_input_defaults(name="TAS", val=250. * onn, units="kn")
