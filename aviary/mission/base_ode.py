import openmdao.api as om

from aviary.subsystems.atmosphere.atmosphere import Atmosphere
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variable_meta_data import CoreMetaData


class BaseODE(om.Group):
    """The base class for all ODE components."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Turn on for all ODE systems.
        self.options['auto_order'] = True

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare(
            'subsystem_options',
            types=dict,
            default={},
            desc='dictionary of optional arguments for the subsystems in this phase',
        )
        self.options.declare(
            'user_options',
            types=dict,
            default={},
            desc='dictionary of user options for this phase',
        )
        self.options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
        )
        self.options.declare(
            'subsystems',
            desc='list of subsystem builder instances to be added to the ODE',
        )
        self.options.declare(
            'meta_data',
            default=CoreMetaData,
            desc='metadata associated with the variables to be passed into the ODE',
        )

    def add_atmosphere(self, **kwargs):
        """Adds Atmosphere component to ODE."""
        nn = self.options['num_nodes']
        self.add_subsystem(
            name='atmosphere',
            subsys=Atmosphere(num_nodes=nn, **kwargs),  # Atmosphere defaults to TAS
            promotes=['*'],
        )

    def add_subsystems_and_solver(
        self, solver_sub=None, couple_propulsion=False, couple_aero=False
    ):
        """
        Adds all specified subsystems to this ODE. Subsystems that need a solver due to coupling
        are instead added to a group called "solver_sub".

        Parameters
        ----------
        solver_sub: None or om.Group
            Pre-created subsystem for the solver.
        couple_propulsion : bool
            When True, the ODE couples with any propulsion subsystems via a throttle to commanded
            thrust balance.
        couple_aero : bool
            When True, the ODE couples with any aerodynamics subsystems via a force balance.
        Returns
        -------
        om.Group
            Target group for the ODE. This will be self unless a solver is needed, in which case it
            will be solver_sub.
        """
        nn = self.options['num_nodes']
        aviary_options = self.options['aviary_options']
        all_subsystems = self.options['subsystems']
        all_subsystem_options = self.options['subsystem_options']
        user_options = self.options['user_options']

        # Prevent circular import
        from aviary.subsystems.propulsion.propulsion_builder import PropulsionBuilder
        from aviary.subsystems.aerodynamics.aerodynamics_builder import AerodynamicsBuilder

        for subsystem in all_subsystems:
            # check if subsystem_options has entry for a subsystem of this name
            if subsystem.name in all_subsystem_options:
                subsystem_options = all_subsystem_options[subsystem.name]
            else:
                subsystem_options = {}

            subsystem_mission = subsystem.build_mission(
                num_nodes=nn,
                aviary_inputs=aviary_options,
                user_options=user_options,
                subsystem_options=subsystem_options,
            )

            if subsystem_mission is not None:
                target = self
                needs_solver = subsystem.needs_mission_solver(
                    aviary_inputs=aviary_options,
                    user_options=user_options,
                    subsystem_options=subsystem_options,
                )

                # ODE couples with propulsion.
                if couple_propulsion and isinstance(subsystem, PropulsionBuilder):
                    needs_solver = True
                elif couple_aero and isinstance(subsystem, AerodynamicsBuilder):
                    needs_solver = True

                if needs_solver:
                    if solver_sub is None:
                        solver_sub = self.add_subsystem('solver_sub', om.Group(), promotes=['*'])
                        solver_sub.options['auto_order'] = True

                        solver_sub.nonlinear_solver = om.NewtonSolver(
                            solve_subsystems=True,
                            atol=1.0e-10,
                            rtol=1.0e-10,
                        )
                        print_level = 2

                        solver_sub.nonlinear_solver.linesearch = om.BoundsEnforceLS()
                        solver_sub.linear_solver = om.DirectSolver(assemble_jac=True)
                        solver_sub.nonlinear_solver.options['err_on_non_converge'] = True
                        solver_sub.nonlinear_solver.options['iprint'] = print_level

                    target = solver_sub

                mission_in = subsystem.mission_inputs(
                    aviary_inputs=aviary_options,
                    user_options=user_options,
                    subsystem_options=subsystem_options,
                )
                mission_out = subsystem.mission_outputs(
                    aviary_inputs=aviary_options,
                    user_options=user_options,
                    subsystem_options=subsystem_options,
                )
                target.add_subsystem(
                    subsystem.name,
                    subsystem_mission,
                    promotes_inputs=mission_in,
                    promotes_outputs=mission_out,
                )

        return solver_sub if solver_sub else self
