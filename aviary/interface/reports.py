from pathlib import Path
import csv

import numpy as np

from openmdao.utils.mpi import MPI
from openmdao.utils.reports_system import register_report

from aviary.interface.utils.markdown_utils import write_markdown_variable_table
from aviary.utils.named_values import NamedValues
from aviary.interface.methods_for_level2 import wrapped_convert_units


def register_custom_reports():
    """
    Registers Aviary reports with OpenMDAO, so they are automatically generated and
    added to the same reports folder as other default reports
    """
    # TODO top-level aircraft report?
    # TODO add flag to skip registering reports?

    # Note: Due to a possible bug in OpenMDAO, we need to assign Problem as the
    # class_name instead of AviaryProblem.

    # register per-subsystem report generation
    register_report(name='subsystems',
                    func=subsystem_report,
                    desc='Generates reports for each subsystem builder in the '
                         'Aviary Problem',
                    class_name='Problem',
                    method='run_driver',
                    pre_or_post='post',
                    # **kwargs
                    )

    register_report(name='mission',
                    func=mission_report,
                    desc='Generates report for mission results from Aviary problem',
                    class_name='Problem',
                    method='run_driver',
                    pre_or_post='post')

    register_report(name='timeseries_csv',
                    func=timeseries_csv,
                    desc='Generates an output .csv file for variables in the timeseries of the trajectory',
                    class_name='Problem',
                    method='run_driver',
                    pre_or_post='post')


def subsystem_report(prob, **kwargs):
    """
    Loops through all subsystem builders in the AviaryProblem calls their write_report
    method. All generated report files are placed in the "reports/subsystem_reports" folder

    Parameters
    ----------
    prob : AviaryProblem
        The AviaryProblem used to generate this report
    """

    # Note: Due to a possible bug in OpenMDAO, we need to assign Problem as the
    # class_name instead of AviaryProblem. Make sure that we don't try to write
    # aviary reports without aviary in the model.
    from aviary.interface.methods_for_level2 import AviaryProblem
    if not isinstance(prob, AviaryProblem):
        return

    reports_folder = Path(prob.get_reports_dir() / 'subsystems')
    reports_folder.mkdir(exist_ok=True)

    # TODO external subsystems??
    core_subsystems = prob.core_subsystems

    for subsystem in core_subsystems.values():
        subsystem.report(prob, reports_folder, **kwargs)


def mission_report(prob, **kwargs):
    """
    Creates a basic mission summary report that is placed in the "reports" folder

    Parameters
    ----------
    prob : AviaryProblem
        The AviaryProblem used to generate this report
    """
    def _get_phase_value(traj, phase, var_name, units, indices=None):
        try:
            vals = prob.get_val(f"{traj}.{phase}.timeseries.{var_name}",
                                units=units,
                                indices=indices,
                                get_remote=True)
        except KeyError:
            try:
                vals = prob.get_val(f"{traj}.{phase}.{var_name}",
                                    units=units,
                                    indices=indices,
                                    get_remote=True)
            # 2DOF breguet range cruise uses time integration to track mass
            except TypeError:
                vals = prob.get_val(f"{traj}.{phase}.timeseries.time",
                                    units=units,
                                    indices=indices,
                                    get_remote=True)
            except KeyError:
                vals = None

        return vals

    def _get_phase_diff(traj, phase, var_name, units, indices=[0, -1]):
        vals = _get_phase_value(traj, phase, var_name, units, indices)

        if vals is not None:
            diff = vals[-1]-vals[0]
            if isinstance(diff, np.ndarray):
                diff = diff[0]
            return diff
        else:
            return None

    # Note: Due to a possible bug in OpenMDAO, we need to assign Problem as the
    # class_name instead of AviaryProblem. Make sure that we don't try to write
    # aviary reports without aviary in the model.
    from aviary.interface.methods_for_level2 import AviaryProblem
    if not isinstance(prob, AviaryProblem):
        return

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
        write_markdown_variable_table(f, totals,
                                      ['Total Fuel Burn',
                                       'Total Time',
                                       'Total Ground Distance'],
                                      {'Total Fuel Burn': {'units': 'lbm'},
                                       'Total Time': {'units': 'min'},
                                       'Total Ground Distance': {'units': 'nmi'}})

        f.write('\n# MISSION SEGMENTS')
        for phase in data:
            f.write(f'\n## {phase}')
            write_markdown_variable_table(f, data[phase], ['Fuel Burn', 'Elapsed Time', 'Ground Distance'],
                                          {'Fuel Burn': {'units': 'lbm'},
                                           'Elapsed Time': {'units': 'min'},
                                           'Ground Distance': {'units': 'nmi'}})


def timeseries_csv(prob, **kwargs):
    timeseries_outputs = prob.model.list_outputs(
        includes='*timeseries*', out_stream=None, return_format='dict', units=True)
    phase_names = prob.model.traj._phases.keys()

    timeseries_data = {}
    for timeseries_output in timeseries_outputs:
        variable_name = timeseries_output.split('.')[-1]
        for idx_phase, phase_name in enumerate(phase_names):
            variable_str = f'traj.phases.{phase_name}.timeseries.timeseries_comp.{variable_name}'

            val = timeseries_outputs[variable_str]['val']

            # grab the units from the first phase; use these units for all others
            if idx_phase == 0:
                units = timeseries_outputs[variable_str]['units']
                val_full_traj = val
            else:
                original_units = timeseries_outputs[variable_str]['units']

                if original_units != units:
                    val = wrapped_convert_units((val, original_units), units)

                val_full_traj = np.vstack((val_full_traj, val))

        timeseries_data[variable_name] = {}
        timeseries_data[variable_name]['val'] = val_full_traj
        timeseries_data[variable_name]['units'] = units

    # move 'time' to the beginning of the dictionary
    timeseries_data = {key: timeseries_data[key] for key in [
        'time'] + [key for key in timeseries_data if key != 'time']}

    # The path where you want to save the CSV file
    reports_folder = Path(prob.get_reports_dir())
    report_file = reports_folder / 'mission_timeseries_data.csv'

    # Preparing the header with variable names and units
    header = [
        f'{variable_name} ({timeseries_data[variable_name]["units"]})' for variable_name in timeseries_data]

    # Transposing the timeseries data to match the CSV structure
    # Assuming all data are numpy arrays of the same length
    data_length = len(timeseries_data[next(iter(timeseries_data))]['val'])
    csv_data = []
    for i in range(data_length):
        row = [timeseries_data[variable_name]['val'][i][0]
               for variable_name in timeseries_data]
        # check if the row is the same as the last one in csv_data
        # only add the row if it is new
        if i > 0:
            if row != csv_data[-1]:
                csv_data.append(row)
        else:
            csv_data.append(row)

    # Writing the header and data to a CSV file
    with open(report_file, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)  # Writing the header
        writer.writerows(csv_data)  # Writing the rows of timeseries data
