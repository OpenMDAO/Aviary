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
            desc='dictionary of parameters to be passed to the subsystem builders',
        )
        self.options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
        )
        self.options.declare(
            'core_subsystems',
            desc='list of core subsystem builder instances to be added to the ODE',
        )
        self.options.declare(
            'external_subsystems',
            default=[],
            desc='list of external subsystem builder instances to be added to the ODE',
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

    def add_core_subsystems(self, solver_group=None):
        """
        Adds all specified external subsystems to ODE in their own group.

        Parameters
        ----------
        solver_group : om.Group
            If not None, core subsystems that require a solver
            (subsystem.needs_mission_solver() == True) are placed inside solver_group.
            If None, all core subsystems are added to BaseODE regardless of if they
            request a solver. TODO add solver compatibility to all ODEs
        """
        nn = self.options['num_nodes']
        aviary_options = self.options['aviary_options']
        core_subsystems = self.options['core_subsystems']
        subsystem_options = self.options['subsystem_options']

        for subsystem in core_subsystems:
            # check if subsystem_options has entry for a subsystem of this name
            if subsystem.name in subsystem_options:
                kwargs = subsystem_options[subsystem.name]
            else:
                kwargs = {}

            subsystem_mission = subsystem.build_mission(
                num_nodes=nn, aviary_inputs=aviary_options, **kwargs
            )

            if subsystem_mission is not None:
                if solver_group is not None:
                    target = solver_group
                else:
                    target = self

                target.add_subsystem(
                    subsystem.name,
                    subsystem_mission,
                    promotes_inputs=subsystem.mission_inputs(**kwargs),
                    promotes_outputs=subsystem.mission_outputs(**kwargs),
                )

    def add_external_subsystems(self, solver_group=None):
        """
        Adds all specified external subsystems to ODE in their own group.

        Parameters
        ----------
        solver_group : om.Group
            If not None, external subsystems that require a solver
            (subsystem.needs_mission_solver() == True) are placed inside solver_group.
            If None, all external subsystems are added to BaseODE regardless of if they
            request a solver. TODO add solver compatibility to all ODEs
        """
        nn = self.options['num_nodes']
        aviary_options = self.options['aviary_options']
        external_subsystems = self.options['external_subsystems']
        subsystem_options = self.options['subsystem_options']

        external_subsystem_group = ExternalSubsystemGroup()
        external_subsystem_group_solver = ExternalSubsystemGroup()
        add_subsystem_group = False
        add_subsystem_group_solver = False

        for subsystem in external_subsystems:
            if subsystem.name in subsystem_options:
                kwargs = subsystem_options[subsystem.name]
            else:
                kwargs = {}

            subsystem_mission = subsystem.build_mission(
                num_nodes=nn, aviary_inputs=aviary_options, **kwargs
            )

            if subsystem_mission is not None:
                target = external_subsystem_group
                if subsystem.needs_mission_solver(aviary_options) and solver_group is not None:
                    add_subsystem_group_solver = True
                    target = external_subsystem_group_solver
                else:
                    add_subsystem_group = True

                target.add_subsystem(
                    subsystem.name,
                    subsystem_mission,
                    promotes_inputs=subsystem.mission_inputs(**kwargs),
                    promotes_outputs=subsystem.mission_outputs(**kwargs),
                )

        # Only add the external subsystem group if it has at least one subsystem.
        # Without this logic there'd be an empty OM group added to the ODE.
        if add_subsystem_group:
            self.add_subsystem(
                name='external_subsystems',
                subsys=external_subsystem_group,
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )
        if add_subsystem_group_solver:
            solver_group.add_subsystem(
                name='external_subsystems',
                subsys=external_subsystem_group_solver,
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )
