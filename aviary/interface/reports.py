from pathlib import Path
from openmdao.utils.reports_system import register_report

from aviary.interface.utils.markdown_utils import write_markdown_variable_table


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
    reports_folder = Path(prob.get_reports_dir())
    report_file = reports_folder / 'mission_summary.md'

    with open(report_file, mode='w') as f:
        f.write('# MISSION SUMMARY')
        write_markdown_variable_table(f, prob, {})
