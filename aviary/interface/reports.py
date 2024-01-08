from pathlib import Path
from openmdao.utils.reports_system import register_report

import numpy as np

from aviary.interface.utils.markdown_utils import write_markdown_variable_table
from aviary.utils.named_values import NamedValues


def register_custom_reports():
    """
    Registers Aviary reports with openMDAO, so they are automatically generated and
    added to the same reports folder as other default reports
    """
    # TODO top-level aircraft report?
    # TODO mission report?
    # TODO add flag to skip registering reports?

    # register per-subsystem report generation
    register_report(name='subsystems',
                    func=subsystem_report,
                    desc='Generates reports for each subsystem builder in the '
                         'Aviary Problem',
                    class_name='Problem',
                    method='run_model',
                    pre_or_post='post',
                    # **kwargs
                    )

    register_report(name='mission',
                    func=mission_report,
                    desc='Generates report for mission results from Aviary problem',
                    class_name='Problem',
                    method='run_model',
                    pre_or_post='post')


def subsystem_report(prob, **kwargs):
    """
    Loops through all subsystem builders in the AviaryProblem calls their write_report
    method. All generated report files are placed in the ./reports/subsystem_reports folder

    Parameters
    ----------
    prob : AviaryProblem
        The AviaryProblem that will be used to generate this report
    """
    reports_folder = Path(prob.get_reports_dir() / 'subsystems')
    reports_folder.mkdir(exist_ok=True)

    # TODO external subsystems??
    core_subsystems = prob.core_subsystems

    for subsystem in core_subsystems.values():
        subsystem.report(prob, reports_folder, **kwargs)


# TODO update with more detailed mission report file
def mission_report(prob, **kwargs):
    """
    Creates a basic mission summary report that is place in the reports folder

    Parameters
    ----------
    prob : AviaryProblem
        The AviaryProblem that will be used to generate this report
    """
    def _get_phase_value(traj, phase, var_name, units):
        try:
            vals = prob.get_val(
                f"{traj}.{phase}.states:{var_name}",
                units=units,
                indices=[-1, 0],
            )
        except KeyError:
            try:
                vals = prob.get_val(
                    f"{traj}.{phase}.timeseries.{var_name}",
                    units=units,
                    indices=[-1, 0],
                )
            except KeyError:
                try:
                    vals = prob.get_val(
                        f"{traj}.{phase}.{var_name}",
                        units=units,
                        indices=[-1, 0],
                    )
                except KeyError:
                    return None

        diff = vals[-1]-vals[0]
        if isinstance(diff, np.ndarray):
            diff = diff[0]

        return diff

    reports_folder = Path(prob.get_reports_dir())
    report_file = reports_folder / 'mission_summary.md'

    with open(report_file, mode='w') as f:
        f.write('# MISSION SUMMARY\n')
        for phase in prob.phase_info:
            f.write(f'## {phase}')
            # TODO for traj in trajectories, currently assuming single one named "traj"
            fuel_burn = _get_phase_value('traj', phase, 'mass', 'lbm')
            time = _get_phase_value('traj', phase, 't', 'min')
            range = _get_phase_value('traj', phase, 'range', 'nmi')
            if range is None:
                range = _get_phase_value('traj', phase, 'distance', 'nmi')
            outputs = NamedValues()
            outputs.set_val('Fuel Burn', fuel_burn, 'lbm')
            # Time, range are negative values from prob
            outputs.set_val('Elapsed Time', -time, 'min')
            outputs.set_val('Ground Distance', -range, 'nmi')
            write_markdown_variable_table(f, outputs, ['Fuel Burn', 'Elapsed Time', 'Ground Distance'],
                                          {'Fuel Burn': {'units': 'lbm'},
                                           'Elapsed Time': {'units': 'min'},
                                           'Ground Distance': {'units': 'nmi'}})
