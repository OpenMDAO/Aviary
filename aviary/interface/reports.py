from pathlib import Path

from openmdao.utils.reports_system import register_report


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
