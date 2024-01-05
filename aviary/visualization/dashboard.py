import argparse
import glob
import json
import os
from pathlib import Path
import pathlib
import shutil
import importlib.util

import numpy as np
from bokeh.palettes import Category10
import hvplot.pandas  # noqa # need this ! Otherwise hvplot using DataFrames does not work
import pandas as pd
import panel as pn
from panel.theme import DefaultTheme

import openmdao.api as om
from openmdao.utils.general_utils import env_truthy

pn.extension(sizing_mode='stretch_width')

# Constants - # Can't get using CSS to work with frames and the raw_css for the template so going with
#    this for now
iframe_css = "width=100% height=4000vh overflow=hidden margin=0px padding=0px border=none"
aviary_variables_json_file_name = 'aviary_vars.json'


def _dashboard_setup_parser(parser):
    """
    Set up the aviary subparser for the 'aviary dashboard' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument(
        'script_name',
        type=str,
        help='Name of aviary script that was run (not including .py).'
    )

    parser.add_argument('--problem_recorder', type=str, help="Problem case recorder file name",
                        dest='problem_recorder', default='aviary_history.db')
    parser.add_argument('--driver_recorder', type=str, help="Driver case recorder file name",
                        dest='driver_recorder', default=None)
    parser.add_argument('--port', dest='port', type=int,
                        default=5000, help="dashboard server port ID (default is 5000)")

    # For future use
    parser.add_argument('-d', '--debug', action='store_true', dest='debug_output',
                        help="show debugging output")


def _dashboard_cmd(options, user_args):
    """
    Run the dashboard command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    dashboard(options.script_name, options.problem_recorder,
              options.driver_recorder, options.port)


def create_report_frame(format, text_filepath):
    """
    Create a Panel Pane that contains an embedded external file in HTML, Markdown, or text format.

    Parameters
    ----------
    format : str
        Format of the file to be embeded. Options are 'html', 'text', 'markdown'.
    text_file_name : str
        Name of the report text file.

    Returns
    -------
    pane : Panel.Pane or None
        A Panel Pane object to be displayed in the dashboard. Or None if the file 
        does not exist.
    """
    if os.path.exists(text_filepath):
        if format == 'html':
            report_pane = pn.pane.HTML(
                f'<iframe {iframe_css} src=/home/{text_filepath}></iframe>')
        elif format in ['markdown', 'text']:
            with open(text_filepath, "rb") as f:
                file_text = f.read()
                # need to deal with some encoding errors
                file_text = file_text.decode('latin-1')
            if format == 'markdown':
                report_pane = pn.pane.Markdown(file_text)
            elif format == 'text':
                report_pane = pn.pane.Markdown(f"```\n{file_text}\n```\n")
        else:
            raise RuntimeError(f"Report format of {format} is not supported.")
    else:
        report_pane = None
    return report_pane


def create_aviary_variables_table_data_nested(script_name, recorder_file):
    """
    Create a JSON file with information about Aviary variables.

    The JSON file has one level of hierarchy of the variables. The file
    is written to aviary_vars.json. That file is then read in by the 
    aviary/visualization/assets/aviary_vars/script.js script. That is inside the 
    aviary/visualization/assets/aviary_vars/index.html file that is embedded in the
    dashboard.

    The information about the variables comes from a case recorder file.

    Parameters
    ----------
    recorder_file : str
        Name of the recorder file containing the Problem cases.

    Returns
    -------
    table_data_nested
        A nested list of information about the Aviary variables.

    """
    cr = om.CaseReader(recorder_file)
    case = cr.get_case('final')
    outputs = case.list_outputs(explicit=True, implicit=True, val=True,
                                residuals=True, residuals_tol=None,
                                units=True, shape=True, bounds=True, desc=True,
                                scaling=False, hierarchical=True, print_arrays=True,
                                out_stream=None, return_format='dict')

    sorted_abs_names = sorted(outputs.keys())

    grouped = {}
    for s in sorted_abs_names:
        prefix = s.split(':')[0]
        if prefix not in grouped:
            grouped[prefix] = []
        grouped[prefix].append(s)

    sorted_group_names = sorted(grouped.keys())

    table_data_nested = []
    for group_name in sorted_group_names:
        if len(grouped[group_name]) == 1:  # a list of one var.
            var_info = grouped[group_name][0]
            table_data_nested.append(
                {
                    "abs_name": group_name,
                    "prom_name": outputs[var_info]['prom_name'],
                    "value": str(outputs[var_info]['val']),
                }
            )
        else:
            # create children
            children_list = []
            for children_name in grouped[group_name]:
                var_info = outputs[children_name]
                children_list.append(
                    {
                        "abs_name": children_name,
                        "prom_name": outputs[children_name]['prom_name'],
                        "value": str(outputs[children_name]['val']),
                    }
                )
            table_data_nested.append(  # not a real var, just a group of vars so no values
                {
                    "abs_name": group_name,
                    "prom_name": "",
                    "value": "",
                    "_children": children_list,
                }
            )

    aviary_variables_file_path = f'reports/{script_name}/aviary_vars/{aviary_variables_json_file_name}'
    with open(aviary_variables_file_path, 'w') as fp:
        json.dump(table_data_nested, fp)

    return table_data_nested


def convert_case_recorder_file_to_df(recorder_file_name):
    """
    Convert a case recorder file into a Pandas data frame.

    Parameters
    ----------
    recorder_file_name : str
        Name of the case recorder file.
    """
    cr = om.CaseReader(recorder_file_name)
    driver_cases = cr.list_cases('driver', out_stream=None)

    df = None
    for i, case in enumerate(driver_cases):
        driver_case = cr.get_case(case)

        desvars = driver_case.get_design_vars(scaled=False)
        objectives = driver_case.get_objectives(scaled=False)
        constraints = driver_case.get_constraints(scaled=False)

        if i == 0:  # Only need to get header of the data frame once
            # Need to worry about the fact that a variable can be in more than one of
            #  desvars, cons, and obj. So filter out the dupes
            initial_desvars_names = list(desvars.keys())
            initial_constraints_names = list(constraints.keys())
            objectives_names = list(objectives.keys())

            # Start with obj, then cons, then desvars
            # Give priority to having a duplicate being in the obj and cons
            #  over being in the desvars
            all_var_names = objectives_names.copy()
            constraints_names = []
            for name in initial_constraints_names:
                if name not in all_var_names:
                    constraints_names.append(name)
                    all_var_names.append(name)
            desvars_names = []
            for name in initial_desvars_names:
                if name not in all_var_names:
                    desvars_names.append(name)
                    all_var_names.append(name)
            header = ["iter_count"] + all_var_names
            df = pd.DataFrame(columns=header)

        # Now fill up a row
        row = [i,]
        # important to do in this order since that is the order added to the header
        for varname in objectives_names:
            value = objectives[varname]
            if not np.isscalar(value):
                value = np.linalg.norm(value)
            row.append(value)

        for varname in constraints_names:
            value = constraints[varname]
            if not np.isscalar(value):
                value = np.linalg.norm(value)
            row.append(value)

        for varname in desvars_names:
            value = desvars[varname]
            if not np.isscalar(value):
                value = np.linalg.norm(value)
            row.append(value)
        df.loc[i] = row

    return df


def dashboard(script_name, problem_recorder, driver_recorder, port):
    """
    Generate the dashboard app display.

    Parameters
    ----------
    script_name : str
        Name of the script file whose results will be displayed by this dashboard.
    problem_recorder : str
        Name of the recorder file containing the Problem cases.
    driver_recorder : str
        Name of the recorder file containing the Driver cases.
    """
    reports_dir = f'reports/{script_name}/'

    # TODO - use lists and functions to do this with a lot less code

    ####### Model Tab #######
    model_tabs_list = []

    # Inputs
    inputs_pane = create_report_frame('html', f'{reports_dir}/inputs.html')
    if inputs_pane:
        model_tabs_list.append(('Inputs', inputs_pane))

    #  Debug Input List
    input_list_pane = create_report_frame('text', 'input_list.txt')
    if input_list_pane:
        model_tabs_list.append(('Debug Input List', input_list_pane))

    #  Debug Output List
    output_list_pane = create_report_frame('text', 'output_list.txt')
    if output_list_pane:
        model_tabs_list.append(('Debug Output List', output_list_pane))

    # N2
    n2_pane = create_report_frame('html', f'{reports_dir}/n2.html')
    if n2_pane:
        model_tabs_list.append(('N2', n2_pane))

    # Trajectory Linkage
    traj_linkage_report_pane = create_report_frame('html',
                                                   f'{reports_dir}/traj_linkage_report.html')
    if traj_linkage_report_pane:
        model_tabs_list.append(('Trajectory Linkage Report',
                                traj_linkage_report_pane))

    ####### Optimization Tab #######
    optimization_tabs_list = []

    # Driver scaling
    driver_scaling_report_pane = create_report_frame('html',
                                                     f'{reports_dir}/driver_scaling_report.html')
    if driver_scaling_report_pane:
        optimization_tabs_list.append(('Driver Scaling Report',
                                       driver_scaling_report_pane))

    # Coloring report
    coloring_report_pane = create_report_frame('html',
                                               f'{reports_dir}/total_coloring.html')
    if coloring_report_pane:
        optimization_tabs_list.append(('Total Coloring Report',
                                       coloring_report_pane))

    # Optimization report
    opt_report_pane = create_report_frame('html',
                                          f'{reports_dir}/opt_report.html')
    if opt_report_pane:
        optimization_tabs_list.append(('Optimization Report',
                                       opt_report_pane))

    # IPOPT report
    ipopt_pane = create_report_frame('text',
                                     f'{reports_dir}/IPOPT.out')
    if ipopt_pane:
        optimization_tabs_list.append(('IPOPT Output', ipopt_pane))

    # SNOPT report
    snopt_pane = create_report_frame('text',
                                     f'{reports_dir}/SNOPT_print.out')
    if snopt_pane:
        optimization_tabs_list.append(('SNOPT Output', snopt_pane))

    # SNOPT summary
    snopt_summary_pane = create_report_frame('text',
                                             f'{reports_dir}/SNOPT_summary.out')
    if snopt_summary_pane:
        optimization_tabs_list.append(('SNOPT Summary', snopt_summary_pane))

    # PyOpt report
    pyopt_solution_pane = create_report_frame('text',
                                              f'{reports_dir}/pyopt_solution.txt')
    if pyopt_solution_pane:
        optimization_tabs_list.append(('PyOpt Solution', pyopt_solution_pane))

    # Desvars, cons, opt interactive plot
    if driver_recorder:
        if os.path.exists(driver_recorder):
            df = convert_case_recorder_file_to_df(f'{driver_recorder}')
            if df is not None:
                variables = pn.widgets.CheckBoxGroup(
                    name="Variables",
                    options=list(df.columns),
                    # just so all of them aren't plotted from the beginning. Skip the iter count
                    value=list(df.columns)[1:2]
                )
                ipipeline = df.interactive()
                ihvplot = ipipeline.hvplot(y=variables, responsive=True, min_height=400,
                                           color=list(Category10[10]),
                                           yformatter="%.0f",
                                           title="Model Optimization using OpenMDAO")
                optimization_plot_pane = pn.Column(
                    pn.Row(
                        pn.Column(
                            variables,
                            pn.VSpacer(height=30),
                            pn.VSpacer(height=30),
                            width=300
                        ),
                        ihvplot.panel(),
                    )
                )
                optimization_tabs_list.append(
                    ('Desvars, cons, opt', optimization_plot_pane))
            else:
                optimization_tabs_list.append(('Desvars, cons, opt', pn.pane.Markdown(
                    f"# Recorder file '{driver_recorder}' does not have Driver case recordings")))
        else:
            optimization_tabs_list.append(('Desvars, cons, opt', pn.pane.Markdown(
                f"# Recorder file '{driver_recorder}' not found")))

    ####### Results Tab #######
    results_tabs_list = []

    # Trajectory results
    traj_results_report_pane = create_report_frame('html',
                                                   f'{reports_dir}/traj_results_report.html')
    if traj_results_report_pane:
        results_tabs_list.append(('Trajectory Results Report',
                                  traj_results_report_pane))

    # Make the Aviary variables table pane
    if problem_recorder:
        if os.path.exists(problem_recorder):
            # Make dir reports/script_name/aviary_vars if needed
            aviary_vars_dir = pathlib.Path(f"reports/{script_name}/aviary_vars")
            aviary_vars_dir.mkdir(parents=True, exist_ok=True)

            # copy index.html file to reports/script_name/aviary_vars/index.html
            aviary_dir = pathlib.Path(importlib.util.find_spec("aviary").origin).parent

            shutil.copy(aviary_dir.joinpath(
                'visualization/assets/aviary_vars/index.html'), aviary_vars_dir.joinpath('index.html'))
            shutil.copy(aviary_dir.joinpath(
                'visualization/assets/aviary_vars/script.js'), aviary_vars_dir.joinpath('script.js'))
            # copy script.js file to reports/script_name/aviary_vars/index.html.
            # mod the script.js file to point at the json file
            # create the json file and put it in reports/script_name/aviary_vars/aviary_vars.json
            create_aviary_variables_table_data_nested(
                script_name, problem_recorder)  # create the json file
            aviary_vars_pane = create_report_frame('html',
                                                   f'{reports_dir}/aviary_vars/index.html')

            results_tabs_list.append(('Aviary Variables', aviary_vars_pane))

    ####### Subsystems Tab #######
    subsystem_tabs_list = []

    # Look through subsystems directory for markdown files
    for md_file in Path(f'{reports_dir}subsystems').glob('*.md'):
        example_subsystems_pane = create_report_frame('markdown', str(md_file))
        subsystem_tabs_list.append((md_file.stem, example_subsystems_pane))

    model_tabs = pn.Tabs(*model_tabs_list, stylesheets=['assets/aviary_styles.css'])
    optimization_tabs = pn.Tabs(*optimization_tabs_list,
                                stylesheets=['assets/aviary_styles.css'])
    results_tabs = pn.Tabs(*results_tabs_list, stylesheets=['assets/aviary_styles.css'])
    if subsystem_tabs_list:
        subsystem_tabs = pn.Tabs(*subsystem_tabs_list,
                                 stylesheets=['assets/aviary_styles.css'])

    # Add subtabs to tabs
    high_level_tabs = []
    high_level_tabs.append(('Model', model_tabs))
    high_level_tabs.append(('Optimization', optimization_tabs))
    high_level_tabs.append(('Results', results_tabs))
    if subsystem_tabs_list:
        high_level_tabs.append(('Subsystems', subsystem_tabs))
    tabs = pn.Tabs(*high_level_tabs, stylesheets=['assets/aviary_styles.css'])

    template = pn.template.FastListTemplate(
        title=f'Aviary Dashboard for {script_name}',
        logo="assets/aviary_logo.png",
        favicon="assets/aviary_logo.png",
        main=[tabs],
        accent_base_color="black",
        header_background="rgb(0, 212, 169)",
        background_color='white',
        theme=DefaultTheme,
        theme_toggle=False,
        main_layout=None,
        css_files=['assets/aviary_styles.css']
    )

    if env_truthy('TESTFLO_RUNNING'):
        show = False
        threaded = True
    else:
        show = True
        threaded = False

    assets_dir = pathlib.Path(importlib.util.find_spec(
        "aviary").origin).parent.joinpath('visualization/assets/')
    home_dir = '.'
    server = pn.serve(template, port=port, address='localhost',
                      websocket_origin=f'localhost:{port}',
                      show=show,
                      threaded=threaded,
                      static_dirs={
                          'reports': reports_dir,
                          'home': home_dir,
                          'assets': assets_dir,
                      })
    server.stop()


if __name__ == '__main__':
    # so we can get the files written to the repo top directory
    parser = argparse.ArgumentParser()
    _dashboard_setup_parser(parser)
    args = parser.parse_args()
    _dashboard_cmd(args, None)
