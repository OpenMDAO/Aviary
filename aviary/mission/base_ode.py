import openmdao.api as om

from aviary.subsystems.atmosphere.atmosphere import Atmosphere
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import promote_aircraft_and_mission_vars
from aviary.variable_info.variable_meta_data import _MetaData


#
class ExternalSubsystemGroup(om.Group):
    """
    Create a lightly modified version of an OM group to add external subsystems to the
    ODE with a special configure() method that promotes all 'aircraft:*' and 'mission:*'
    variables to the ODE.
    """

    def configure(self):
        promote_aircraft_and_mission_vars(self)


class BaseODE(om.Group):
    """The base class for all ODE components."""

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
            default=_MetaData,
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

    def add_subsystems(self, solver_group=None):
        """
        Adds all specified subsystems to ODE in their own group.

        Parameters
        ----------
        solver_group : om.Group
            If not None, subsystems that require a solver (subsystem.needs_mission_solver() == True)
            are placed inside solver_group.

            If None, all subsystems are added to BaseODE regardless of if they request a solver.
            TODO add solver compatibility to all ODEs

        Returns
        -------
        use_mission_solver : bool
            Flag that communicates that one or more subsystem requests to be placed inside a solver
            (independent of the needs of an individual ODE's setup)
        """
        nn = self.options['num_nodes']
        aviary_options = self.options['aviary_options']
        all_subsystems = self.options['subsystems']
        all_subsystem_options = self.options['subsystem_options']
        user_options = self.options['user_options']
        use_mission_solver = False

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
                needs_sovler = subsystem.needs_mission_solver(
                    aviary_inputs=aviary_options,
                    subsystem_options=subsystem_options,
                )

                if needs_sovler and solver_group is not None:
                    target = solver_group
                    use_mission_solver = True

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

        return use_mission_solver
