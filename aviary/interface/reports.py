import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openmdao.utils.mpi import MPI
from openmdao.utils.reports_system import register_report

from aviary.interface.utils.markdown_utils import write_markdown_variable_table
from aviary.utils.named_values import NamedValues
from aviary.utils.utils import wrapped_convert_units


def register_custom_reports():
    """
    Registers Aviary reports with OpenMDAO, so they are automatically generated and
    added to the same reports folder as other default reports.
    """
    # TODO top-level aircraft report?
    # TODO add flag to skip registering reports?

    # register per-subsystem report generation
    register_report(
        name='subsystems',
        func=subsystem_report,
        desc='Generates reports for each subsystem builder in the Aviary Problem',
        class_name='AviaryProblem',
        method='run_driver',
        pre_or_post='post',
        # **kwargs
    )

    register_report(
        name='mission',
        func=mission_report,
        desc='Generates report for mission results from Aviary problem',
        class_name='AviaryProblem',
        method='run_driver',
        pre_or_post='post',
    )

    register_report(
        name='timeseries_csv',
        func=timeseries_csv,
        desc='Generates an output .csv file for variables in the timeseries of the trajectory',
        class_name='AviaryProblem',
        method='run_driver',
        pre_or_post='post',
    )

    register_report(
        name='run_status',
        func=run_status,
        desc='Generates a report on the status of the run',
        class_name='AviaryProblem',
        method='run_driver',
        pre_or_post='post',
    )

    register_report(
        name='input_checks',
        func=input_check_report,
        desc='Generates a report on the aviary inputs',
        class_name='AviaryProblem',
        method='final_setup',
        pre_or_post='post',
    )


def run_status(prob):
    """
    Creates a JSON file that contains high level overview of the run.

    Parameters
    ----------
    prob : AviaryProblem
        The AviaryProblem used to generate this report
    """
    if MPI and MPI.COMM_WORLD.rank != 0:
        return

    reports_folder = Path(prob.get_reports_dir())
    report_file = reports_folder / 'status.json'

    runtime = prob.driver.result.runtime
    runtime_ms = (runtime * 1000.0) % 1000.0
    runtime_formatted = (
        f'{time.strftime("%H hours %M minutes %S seconds", time.gmtime(runtime))} '
        f'{runtime_ms:.1f} milliseconds'
    )

    t = datetime.datetime.now()
    time_stamp = t.strftime('%Y-%m-%d %H:%M:%S %Z')

    status = {}
    status['Problem'] = prob._name
    status['Script'] = sys.argv[0]
    status['Optimizer'] = prob.driver._get_name()
    status['Number of driver iterations'] = prob.driver.result.iter_count
    status['Number of model evals'] = prob.driver.result.model_evals
    status['Number of deriv evals'] = prob.driver.result.deriv_evals
    status['Wall clock run time'] = runtime_formatted
    status['Exit status'] = prob.driver.result.exit_status
    status['Report generation date and time'] = time_stamp

    with open(report_file, 'w') as f:
        json.dump(status, f, indent=1, ensure_ascii=False)
        print(file=f)  # avoid 'no newline at end of file' message


def subsystem_report(prob, **kwargs):
    """
    Loops through all subsystem builders in the AviaryProblem calls their write_report
    method. All generated report files are placed in the "reports/subsystem_reports" folder.

    Parameters
    ----------
    prob : AviaryProblem
        The AviaryProblem used to generate this report
    """
    reports_folder = Path(prob.get_reports_dir() / 'subsystems')
    reports_folder.mkdir(exist_ok=True)

    # TODO external subsystems??
    core_subsystems = prob.core_subsystems

    for subsystem in core_subsystems.values():
        subsystem.report(prob, reports_folder, **kwargs)


def mission_report(prob, **kwargs):
    """
    Creates a basic mission summary report that is placed in the "reports" folder.

    Parameters
    ----------
    prob : AviaryProblem
        The AviaryProblem used to generate this report
    """

    def _get_phase_value(traj, phase, var_name, units, indices=None):
        try:
            vals = prob.get_val(
                f'{traj}.{phase}.timeseries.{var_name}',
                units=units,
                indices=indices,
                get_remote=True,
            )
        except KeyError:
            try:
                vals = prob.get_val(
                    f'{traj}.{phase}.{var_name}',
                    units=units,
                    indices=indices,
                    get_remote=True,
                )
            # 2DOF breguet range cruise uses time integration to track mass
            except TypeError:
                vals = prob.get_val(
                    f'{traj}.{phase}.timeseries.time',
                    units=units,
                    indices=indices,
                    get_remote=True,
                )
            except KeyError:
                vals = None

        return vals

    def _get_phase_diff(traj, phase, var_name, units, indices=[0, -1]):
        vals = _get_phase_value(traj, phase, var_name, units, indices)

        if vals is not None:
            diff = vals[-1] - vals[0]
            if isinstance(diff, np.ndarray):
                diff = diff[0]
            return diff
        else:
            return None

    reports_folder = Path(prob.get_reports_dir())
    report_file = reports_folder / 'mission_summary.md'

    # read per-phase data from trajectory
    data = {}
    for idx, phase in enumerate(prob.phase_info):
        # TODO for traj in trajectories, currently assuming single one named "traj"
        # TODO delta mass and fuel consumption need to be tracked separately
        fuel_burn = _get_phase_diff('traj', phase, 'mass', 'lbm', [-1, 0])
        time = _get_phase_diff('traj', phase, 't', 'min')
        range = _get_phase_diff('traj', phase, 'distance', 'nmi')

        # get initial values, first in traj
        if idx == 0:
            initial_mass = _get_phase_value('traj', phase, 'mass', 'lbm', 0)[0]
            initial_time = _get_phase_value('traj', phase, 't', 'min', 0)
            initial_range = _get_phase_value('traj', phase, 'distance', 'nmi', 0)[0]

        outputs = NamedValues()
        # Fuel burn is negative of delta mass
        outputs.set_val('Fuel Burn', fuel_burn, 'lbm')
        outputs.set_val('Elapsed Time', time, 'min')
        outputs.set_val('Ground Distance', range, 'nmi')
        data[phase] = outputs

        # get final values, last in traj
        final_mass = _get_phase_value('traj', phase, 'mass', 'lbm', -1)[0]
        final_time = _get_phase_value('traj', phase, 't', 'min', -1)
        final_range = _get_phase_value('traj', phase, 'distance', 'nmi', -1)[0]

    totals = NamedValues()
    totals.set_val('Total Fuel Burn', initial_mass - final_mass, 'lbm')
    totals.set_val('Total Time', final_time - initial_time, 'min')
    totals.set_val('Total Ground Distance', final_range - initial_range, 'nmi')

    if MPI and MPI.COMM_WORLD.rank != 0:
        return

    with open(report_file, mode='w') as f:
        f.write('# MISSION SUMMARY')
        write_markdown_variable_table(
            f,
            totals,
            ['Total Fuel Burn', 'Total Time', 'Total Ground Distance'],
            {
                'Total Fuel Burn': {'units': 'lbm'},
                'Total Time': {'units': 'min'},
                'Total Ground Distance': {'units': 'nmi'},
            },
        )

        f.write('\n# MISSION SEGMENTS')
        for phase in data:
            f.write(f'\n## {phase}')
            write_markdown_variable_table(
                f,
                data[phase],
                ['Fuel Burn', 'Elapsed Time', 'Ground Distance'],
                {
                    'Fuel Burn': {'units': 'lbm'},
                    'Elapsed Time': {'units': 'min'},
                    'Ground Distance': {'units': 'nmi'},
                },
            )


def input_check_report(prob, **kwargs):
    """
    Creates a basic input checking report.

    This report informs the user which aviary inputs were not specified by the user.

    Parameters
    ----------
    prob : AviaryProblem
        The AviaryProblem used to generate this report
    """
    reports_folder = Path(prob.get_reports_dir())
    report_file = reports_folder / 'input_checks.md'

    model = prob.model

    # a change in OpenMDAO 3.38.1-dev adds a resolver in place of the prom2abs/abs2prom attributes
    try:
        resolver = model._resolver

        def prom2abs(prom_name):
            return resolver.absnames(prom_name, 'input')

        def abs2prom(abs_name):
            return resolver.abs2prom(abs_name, 'input')

    except AttributeError:

        def prom2abs(prom_name):
            return model._var_allprocs_prom2abs_list['input'][prom_name]

        def abs2prom(abs_name):
            return model._var_allprocs_abs2prom['input'][abs_name]

    # Find all unconnected inputs.
    all_ivc_abs = [k for k, v in model._conn_abs_in2out.items() if 'ivc' in v]
    all_ivc_prom = [abs2prom(v) for v in all_ivc_abs]

    aviary_metadata = prob.meta_data
    aviary_inputs = prob.aviary_inputs
    bare_inputs = {v for v in all_ivc_prom if v not in aviary_inputs}
    bare_hierarchy_inputs = {
        v for v in bare_inputs if v.startswith('mission:') or v.startswith('aircraft:')
    }
    bare_local_inputs = bare_inputs - bare_hierarchy_inputs

    # There are no more collective calls, so we can exit.
    if MPI and MPI.COMM_WORLD.rank != 0:
        return

    with open(report_file, mode='w') as f:
        f.write('# Unspecified Hierarchy Variables\n')
        f.write(
            'These aviary inputs are unspecified in aviary_inputs, and may be using default values '
            'defined in the Aviary metadata.\n\n'
        )

        if bare_hierarchy_inputs:
            f.write('| Name | Value | Units | Description | Absolute Paths\n')
            f.write('| :- |  :- |  :- | :- | :- |\n')

            for var in sorted(bare_hierarchy_inputs):
                metadata = aviary_metadata.get(var)
                units = metadata['units']
                val = model.get_val(var, units=units)
                desc = metadata['desc']
                abs_paths = prom2abs(var)

                f.write(f'| **{var}** | {val} | {units} | {desc} | {abs_paths}|\n')

            f.write('\n')

        else:
            f.write('None\n')

        f.write('# Unspecified Local Variables\n')
        f.write(
            'These local subsystem inputs are unconnected, and may be using default '
            'values specified in the component.\n\n'
        )

        if bare_local_inputs:
            f.write('| Name | Value | Units | Absolute Paths\n')
            f.write('| :- |  :- |  :- | :- |\n')

            for var in sorted(bare_local_inputs):
                # Filter out dymos internals.
                if var.startswith('traj') and '.rhs_all.' not in var:
                    continue

                abs_paths = prom2abs(var)
                val = model.get_val(var)
                meta = model._var_allprocs_abs2meta['input'][abs_paths[0]]
                units = meta['units']

                f.write(f'| **{var}** | {val} | {units} | {abs_paths}|\n')

            f.write('\n\n')

        else:
            f.write('None')


def timeseries_csv(prob, **kwargs):
    """
    Generates a CSV file containing timeseries data for variables from an Aviary mission.

    This function extracts timeseries data from the provided problem object, processes the data
    to unify units across different phases of the mission, and then outputs the result to a CSV file.
    The 'time' variable is moved to the beginning of the dataset so it's always the leftmost column.
    Duplicate consecutive rows are eliminated.

    Parameters
    ----------
    prob : AviaryProblem
        The AviaryProblem used to generate this report
    kwargs : dict
        Additional keyword arguments (unused)

    The output CSV file is named 'mission_timeseries_data.csv' and is saved in the reports directory.
    The first row of the CSV file contains headers with variable names and units.
    Each subsequent row represents the mission outputs at a different time step.
    """
    timeseries_outputs = prob.model.list_outputs(
        includes='*timeseries*', out_stream=None, return_format='dict', units=True
    )
    phase_names = prob.model.traj._phases.keys()

    # There are no more collective calls, so we can exit.
    if MPI and MPI.COMM_WORLD.rank != 0:
        return

    timeseries_outputs = {value['prom_name']: value for key, value in timeseries_outputs.items()}

    timeseries_outputs = {
        key: value for key, value in timeseries_outputs.items() if not key.endswith('_phase')
    }

    unique_variable_names = set(
        [timeseries_output.split('.')[-1] for timeseries_output in timeseries_outputs]
    )

    timeseries_data = {}
    for variable_name in unique_variable_names:
        timeseries_data[variable_name] = {}
        first = True  # flag to check if this is first iteration in for loop
        units = None
        for idx_phase, phase_name in enumerate(phase_names):
            variable_str = f'traj.{phase_name}.timeseries.{variable_name}'
            time_str = f'traj.{phase_name}.timeseries.time'

            if variable_str not in timeseries_outputs:
                Warning(
                    f'Variable {variable_str} not found in timeseries_outputs for phase '
                    f'{phase_name}.'
                )
                val = np.zeros_like(timeseries_outputs[time_str]['val'])
                val[:] = np.nan
                if first:
                    val_full_traj = val
                    first = False
                else:
                    val_full_traj = np.vstack((val_full_traj, val))

            else:
                val = timeseries_outputs[variable_str]['val']

                # grab the units from the first phase that uses this variable; use these units for all others
                if units is None:
                    units = timeseries_outputs[variable_str]['units']
                    if first:
                        val_full_traj = val
                        first = False
                    else:
                        val_full_traj = np.vstack((val_full_traj, val))
                else:
                    original_units = timeseries_outputs[variable_str]['units']

                    if original_units != units:
                        val = wrapped_convert_units((val, original_units), units)

                    if first:
                        val_full_traj = val
                        first = False
                    else:
                        val_full_traj = np.vstack((val_full_traj, val))

        timeseries_data[variable_name]['val'] = val_full_traj
        timeseries_data[variable_name]['units'] = units
        timeseries_data[variable_name]['shape'] = val_full_traj.shape

    # Create a DataFrame from timeseries_data
    df_data = {
        variable_name: pd.Series(timeseries_data[variable_name]['val'].flatten())
        for variable_name in timeseries_data
    }
    df = pd.DataFrame(df_data)

    time_column = ['time']  # Isolate the 'time' column
    # Sort the rest of the columns
    other_columns = sorted([col for col in df.columns if col != 'time'])
    columns = time_column + other_columns  # Combine them, keeping 'time' first
    df = df[columns]

    # Add units to column names
    df.columns = [f'{col} ({timeseries_data[col]["units"]})' for col in df.columns]

    df.drop_duplicates()

    # The path where you want to save the CSV file
    reports_folder = Path(prob.get_reports_dir())
    report_file = reports_folder / 'mission_timeseries_data.csv'

    # Write the DataFrame to a CSV file
    df.to_csv(report_file, index=False)
