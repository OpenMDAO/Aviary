import openmdao.api as om
from openmdao.utils.mpi import MPI

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import EquationsOfMotion
from aviary.variable_info.variables import Settings

HEIGHT_ENERGY = EquationsOfMotion.HEIGHT_ENERGY


class AviaryGroup(om.Group):
    """
    A standard OpenMDAO group that handles Aviary's promotions in the configure
    method. This assures that we only call set_input_defaults on variables
    that are present in the model.
    """

    def initialize(self):
        """Declare options."""
        self.options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
        )
        self.options.declare(
            'aviary_metadata', types=dict, desc='metadata dictionary of the full aviary problem.'
        )
        self.options.declare('phase_info', types=dict, desc='phase-specific settings.')
        self.builder = []

    def configure(self):
        """Configure the Aviary group."""
        aviary_options = self.options['aviary_options']
        aviary_metadata = self.options['aviary_metadata']

        # Find promoted name of every input in the model.
        all_prom_inputs = []

        # We can call list_inputs on the subsystems.
        for system in self.system_iter(recurse=False):
            var_abs = system.list_inputs(out_stream=None, val=False)
            var_prom = [v['prom_name'] for k, v in var_abs]
            all_prom_inputs.extend(var_prom)

            # Calls to promotes aren't handled until this group resolves.
            # Here, we address anything promoted with an alias in AviaryProblem.
            input_meta = system._var_promotes['input']
            var_prom = [v[0][1] for v in input_meta if isinstance(v[0], tuple)]
            all_prom_inputs.extend(var_prom)
            var_prom = [v[0] for v in input_meta if not isinstance(v[0], tuple)]
            all_prom_inputs.extend(var_prom)

        if MPI and self.comm.size > 1:
            # Under MPI, promotion info only lives on rank 0, so broadcast.
            all_prom_inputs = self.comm.bcast(all_prom_inputs, root=0)

        for key in aviary_metadata:
            if ':' not in key or key.startswith('dynamic:'):
                continue

            if aviary_metadata[key]['option']:
                continue

            # Skip anything that is not presently an input.
            if key not in all_prom_inputs:
                continue

            if key in aviary_options:
                val, units = aviary_options.get_item(key)
            else:
                val = aviary_metadata[key]['default_value']
                units = aviary_metadata[key]['units']

                if val is None:
                    # optional, but no default value
                    continue

            self.set_input_defaults(key, val=val, units=units)

        # try to get all the possible EOMs from the Enums rather than specifically calling the names here
        # This will require some modifications to the enums
        mission_method = aviary_options.get_val(Settings.EQUATIONS_OF_MOTION)

        # Temporarily add extra stuff here, probably patched soon
        if mission_method is HEIGHT_ENERGY:
            phase_info = self.options['phase_info']

            # Set a more appropriate solver for dymos when the phases are linked.
            if MPI and isinstance(self.traj.phases.linear_solver, om.PETScKrylov):
                # When any phase is connected with input_initial = True, dymos puts
                # a jacobi solver in the phases group. This is necessary in case
                # the phases are cyclic. However, this causes some problems
                # with the newton solvers in Aviary, exacerbating issues with
                # solver tolerances at multiple levels. Since Aviary's phases
                # are basically in series, the jacobi solver is a much better
                # choice and should be able to handle it in a couple of
                # iterations.
                self.traj.phases.linear_solver = om.LinearBlockJac(maxiter=5)

            # Due to recent changes in dymos, there is now a solver in any phase
            # that has connected initial states. It is not clear that this solver
            # is necessary except in certain corner cases that do not apply to the
            # Aviary trajectory. In our case, this solver merely addresses a lag
            # in the state input component. Since this solver can cause some
            # numerical problems, and can slow things down, we need to move it down
            # into the state interp component.
            # TODO: Future updates to dymos may make this unnecessary.
            for phase in self.traj.phases.system_iter(recurse=False):
                # Don't move the solvers if we are using solve segments.
                if phase_info[phase.name]['user_options'].get('distance_solve_segments'):
                    continue

                phase.nonlinear_solver = om.NonlinearRunOnce()
                phase.linear_solver = om.LinearRunOnce()
                if isinstance(phase.indep_states, om.ImplicitComponent):
                    phase.indep_states.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
                    phase.indep_states.linear_solver = om.DirectSolver(rhs_checking=True)
