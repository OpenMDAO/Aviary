import numpy as np
import openmdao.api as om

from openmdao.core.system import System
from aviary.mission.base_ode import BaseODE as _BaseODE
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class PerformanceODE(_BaseODE):
    """The base class for all performance subsystem ODE components."""

    def initialize(self):
        super().initialize()

        self.options.declare(
            'equations_of_motion', types=System, desc='Equations of motion to be used with this ODE'
        )

        self.options.declare(
            'speed_type',
            default=SpeedType.MACH,
            desc='Type of aircraft speed that is provided as an input to the atmosphere component',
        )

    def setup(self):
        options = self.options
        nn = options['num_nodes']
        equations_of_motion = options['equations_of_motion']
        speed_type = options['speed_type']
        aviary_options = options['aviary_options']

        self.add_atmosphere(input_speed_type=speed_type)

        sub1 = self.add_subsystem('solver_sub', om.Group(), promotes=['*'])

        self.add_core_subsystems(solver_group=sub1)

        self.add_external_subsystems(solver_group=sub1)

        self.add_eom(equations_of_motion=equations_of_motion, num_nodes=nn, solver_group=sub1)

        sub1.nonlinear_solver = om.NewtonSolver(
            solve_subsystems=True,
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        sub1.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        sub1.linear_solver = om.DirectSolver(assemble_jac=True)
        sub1.nonlinear_solver.options['err_on_non_converge'] = True
        sub1.nonlinear_solver.options['iprint'] = 2

        self.options['auto_order'] = True

    def add_eom(self, equations_of_motion=None, num_nodes=1, solver_group=None):
        if solver_group is not None:
            target = solver_group
        else:
            target = self

        target.add_subsystem(
            'equations_of_motion',
            equations_of_motion(num_nodes=1),
            promotes=['*'],
        )
