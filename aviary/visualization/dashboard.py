import argparse
from collections import defaultdict
import functools
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import traceback
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import openmdao.api as om
import pandas as pd
import panel as pn
from bokeh.models import (
    ColumnDataSource,
    CustomJS,
    Div,
    Legend,
    LegendItem,
    LinearAxis,
    PrintfTickFormatter,
    Range1d,
    TextInput,
)
from bokeh.palettes import Category20, d3
from bokeh.plotting import figure
from openmdao.utils.general_utils import env_truthy
from openmdao.utils.units import conversion_to_base_units

try:
    from openmdao.utils.gui_testing_utils import get_free_port
except BaseException:
    # If get_free_port is unavailable, the default port will be used
    def get_free_port():
        return 5000


from dymos.visualization.timeseries.bokeh_timeseries_report import _meta_tree_subsys_iter
from openmdao.utils.om_warnings import issue_warning

from aviary.visualization.aircraft_3d_model import Aircraft3DModel

# support getting this function from OpenMDAO post movement of the function to utils
#    but also support its old location
try:
    from openmdao.utils.array_utils import convert_ndarray_to_support_nans_in_json
except ImportError:
    from openmdao.visualization.n2_viewer.n2_viewer import (
        _convert_ndarray_to_support_nans_in_json as convert_ndarray_to_support_nans_in_json,
    )

import aviary.api as av

# Enable Panel extensions
pn.extension(sizing_mode='stretch_width')
# Initialize any custom extensions
pn.extension('tabulator')

# Constants
aviary_variables_json_file_name = 'aviary_vars.json'
documentation_text_align = 'left'

# functions for the aviary command line command


def _none_or_str(value):
    """
    Get the value of the argparse option.

    If "None", return None. Else, just return the string.

    Parameters
    ----------
    value : str
        The value used by the user on the command line for the argument.

    Returns
    -------
    option_value : str or None
        The value of the option after possibly converting from 'None' to None.
    """
    if value == 'None':
        return None
    return value


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
        nargs='*',
        help='Name of aviary script that was run (not including .py).',
    )

    parser.add_argument(
        '--problem_recorder',
        type=str,
        help='Problem case recorder file name',
        dest='problem_recorder',
        default='problem_history.db',
    )
    parser.add_argument(
        '--driver_recorder',
        type=_none_or_str,
        help='Driver case recorder file name. Set to None if file is ignored',
        dest='driver_recorder',
        default='driver_history.db',
    )
    parser.add_argument(
        '--port',
        dest='port',
        type=int,
        default=0,
        help='dashboard server port ID (default is 0, which indicates get any free port)',
    )
    parser.add_argument(
        '-b',
        '--background',
        action='store_true',
        dest='run_in_background',
        help="Run the server in the background (don't automatically open the browser)",
    )

    # For future use
    parser.add_argument(
        '-d',
        '--debug',
        action='store_true',
        dest='debug_output',
        help='show debugging output',
    )

    parser.add_argument(
        '--save',
        nargs='?',
        const=True,
        default=False,
        help='Name of zip file in which dashboard files are saved. If no argument given, use the script name to name the zip file',
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='When displaying data from a shared zip file, if the directory in the reports directory exists, overrite if this is True',
    )


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
    if options.save and not options.script_name:
        if options.save is not True:
            options.script_name = options.save
            options.save = True

    if not options.script_name:
        raise argparse.ArgumentError('script_name argument missing')

    if isinstance(options.script_name, list):
        options.script_name = options.script_name[0]

    # check to see if options.script_name is a zip file
    # if yes, then unzip into reports directory and run dashboard on it
    if zipfile.is_zipfile(options.script_name):
        report_dir_name = Path(options.script_name).stem
        report_dir_path = Path(f'{report_dir_name}_out')
        # need to check to see if that directory already exists
        if not options.force and report_dir_path.is_dir():
            raise RuntimeError(
                f'The reports directory {report_dir_path} already exists. If you wish '
                'to overrite the existing directory, use the --force option'
            )
        if (
            report_dir_path.is_dir()
        ):  # need to delete it. The unpacking will just add to what is there, not do a clean unpack
            shutil.rmtree(report_dir_path)

        shutil.unpack_archive(options.script_name, report_dir_path)
        dashboard(
            report_dir_name,
            options.problem_recorder,
            options.driver_recorder,
            options.port,
            options.run_in_background,
        )
        return

    # Save the dashboard files to a zip file that can be shared with others
    if options.save is not False:
        if options.save is True:
            save_filename_stem = options.script_name
        else:
            save_filename_stem = Path(options.save).stem
        print(f'Saving to {save_filename_stem}.zip')
        shutil.make_archive(save_filename_stem, 'zip', f'{options.script_name}_out')
        return

    dashboard(
        options.script_name,
        options.problem_recorder,
        options.driver_recorder,
        options.port,
        options.run_in_background,
    )


def _handle_pane_creation_errors():
    """
    A decorator used with the functions that creates panes for use in the dashboard.
    If the function executes without throwing an exception, the pane is added
    to the pane list that will be shown in the dashboard.
    If the function does throw an exception, it puts the error message
    and stack trace into the pane so the user can see why it failed.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(pane_title, pane_list, pane_documentation, *args, **kwargs):
            try:
                pane = func(pane_documentation, *args, **kwargs)
                pane_list.append((pane_title, pane))
                return pane
            except Exception as e:
                # Get the full stack trace as a string
                stack_trace = traceback.format_exc()
                pane = _create_message_pane(
                    pane_documentation,
                    f'Unable to create {pane_title} due to error: {e}\n\nStack trace:\n{stack_trace}',
                )
                pane_list.append((pane_title, pane))
                return None

        return wrapper

    return decorator


@_handle_pane_creation_errors()
def create_table_pane_from_json(documentation, json_filepath):
    """
    Create a Tabulator Pane with Name and Value columns using tabular data
    from a JSON file.

    Parameters
    ----------
    json_filepath : str
        Path to the JSON file containing tabular data.

    Returns
    -------
    pane : Panel.Pane or None
        A Panel Pane that is a Panel library Tabular widget.
        Or None if there was an error.
    """
    try:
        with open(json_filepath) as json_file:
            parsed_json = json.load(json_file)

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(list(parsed_json.items()), columns=['Name', 'Value'])
        table_pane = pn.widgets.Tabulator(
            df,
            show_index=False,
            selectable=False,
            sortable=False,
            disabled=True,  # disables editing of the table
            titles={
                'Name': '',
                'Value': '',
            },
        )
        table_pane_with_doc = pn.Column(
            pn.pane.HTML(
                f'<p class="pane_doc">{documentation}</p>',
                stylesheets=['assets/aviary_styles.css'],
                styles={'text-align': documentation_text_align},
            ),
            table_pane,
        )
    except Exception:
        table_pane_with_doc = pn.Column(
            pn.pane.HTML(
                f'<p class="pane_doc">{documentation}</p>',
                stylesheets=['assets/aviary_styles.css'],
                styles={'text-align': documentation_text_align},
            ),
            pn.pane.Markdown(
                f"# Table not shown because data source JSON file, '{json_filepath}', not found."
            ),
        )

    return table_pane_with_doc


# functions for creating Panel Panes given different kinds of
#    data inputs
@_handle_pane_creation_errors()
def create_csv_frame(documentation, csv_filepath):
    """
    Create a Panel Pane that contains a tabular display of the data in a CSV file.

    Parameters
    ----------
    csv_filepath : str
        Path to the input CSV file.
    documentation : str
        Explanation of what this tab is showing.

    Returns
    -------
    pane : Panel.Pane or None
        A Panel Pane object showing the tabular display of the CSV file contents.
        Or None if the CSV file does not exist.
    """
    if os.path.isfile(csv_filepath):
        df = pd.read_csv(csv_filepath)
        df_pane = pn.widgets.Tabulator(
            df,
            show_index=False,
            sortable=False,
            layout='fit_data_stretch',
            max_height=600,
            sizing_mode='scale_both',
        )
        report_pane = pn.Column(
            pn.pane.HTML(
                f'<p class="pane_doc">{documentation}</p>',
                stylesheets=['assets/aviary_styles.css'],
                styles={'text-align': documentation_text_align},
            ),
            df_pane,
        )
    else:
        report_pane = pn.Column(
            pn.pane.HTML(
                f'<p class="pane_doc">{documentation}</p>',
                stylesheets=['assets/aviary_styles.css'],
                styles={'text-align': documentation_text_align},
            ),
            pn.pane.Markdown(
                f"# Report not shown because data source CSV file, '{csv_filepath}', not found."
            ),
        )

    return report_pane


def get_run_status(status_filepath):
    """Get run status."""
    try:
        with open(status_filepath) as f:
            status_dct = json.load(f)
            if status_dct['Exit status'] == 'SUCCESS':
                return '✅ Success'
            else:
                return f'❌ {status_dct["Exit status"]}'
    except Exception:
        return 'Unknown'


def _create_message_pane(documentation, message):
    """
    A simple pane showing a text message. Mostly used for
    showing error messages.
    """
    report_pane = pn.Column(
        pn.pane.HTML(
            f'<p class="pane_doc">{documentation}</p>',
            stylesheets=['assets/aviary_styles.css'],
            styles={'text-align': 'left'},
        ),
        # pn.pane.HTML(f"<p>{message}</p>", styles={'text-align': 'left'})
        pn.pane.Str(f'<p>{message}</p>', styles={'text-align': 'left'}),
    )
    return report_pane


@_handle_pane_creation_errors()
def create_report_frame(documentation, format, text_filepath):
    """
    Create a Panel Pane that contains an embedded external file in HTML, Markdown, or text format,
    or a simple message in HTML format.

    Parameters
    ----------
    format : str
        Format of the file to be embedded. Options are 'html', 'text', 'markdown', 'simple_message'.
    text_filepath : str
        Path to the report text file or message if format is 'simple_message'.
    documentation : str
        Explanation of what this tab is showing.

    Returns
    -------
    pane : Panel.Pane or None
        A Panel Pane object to be displayed in the dashboard. Or None if the file
        does not exist.
    """
    if os.path.isfile(text_filepath):
        if format == 'html':
            iframe_css = 'width=1200px height=800px overflow-x="scroll" overflow="scroll" margin=0px padding=0px border=20px frameBorder=20px scrolling="yes"'
            report_pane = pn.Column(
                pn.pane.HTML(
                    f'<p class="pane_doc">{documentation}</p>',
                    stylesheets=['assets/aviary_styles.css'],
                    styles={'text-align': 'left'},
                ),
                pn.pane.HTML(f'<iframe {iframe_css} src=/home/{text_filepath}></iframe>'),
            )
        elif format in ['markdown', 'text']:
            with open(text_filepath, 'rb') as f:
                file_text = f.read()
                # need to deal with some encoding errors
                file_text = file_text.decode('latin-1')
            if format == 'markdown':
                report_pane = pn.pane.Markdown(file_text)
            elif format == 'text':
                report_pane = pn.pane.Str(file_text)
            report_pane = pn.Column(
                pn.pane.HTML(
                    f'<p class="pane_doc">{documentation}</p>',
                    stylesheets=['assets/aviary_styles.css'],
                    styles={'text-align': 'left'},
                ),
                report_pane,
            )
        else:
            raise RuntimeError(f'Report format of {format} is not supported.')
    else:
        report_pane = pn.Column(
            pn.pane.HTML(
                f'<p class="pane_doc">{documentation}</p>',
                stylesheets=['assets/aviary_styles.css'],
                styles={'text-align': 'left'},
            ),
            pn.pane.Markdown(
                f"# Report not shown because report file, '{text_filepath}', not found."
            ),
        )
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

    if 'final' not in cr.list_cases():
        return None

    case = cr.get_case('final')
    outputs = case.list_outputs(
        explicit=True,
        implicit=True,
        val=True,
        residuals=False,
        residuals_tol=None,
        units=True,
        shape=False,
        bounds=False,
        desc=False,
        scaling=False,
        hierarchical=False,
        print_arrays=False,
        out_stream=None,
        return_format='dict',
    )
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
            prom_name = outputs[var_info]['prom_name']
            aviary_metadata = av.CoreMetaData.get(prom_name)
            table_data_nested.append(
                {
                    'abs_name': group_name,
                    'prom_name': prom_name,
                    'value': convert_ndarray_to_support_nans_in_json(outputs[var_info]['val']),
                    'units': outputs[var_info]['units'],
                    'metadata': json.dumps(aviary_metadata),
                }
            )
        else:
            # create children
            children_list = []
            for children_name in grouped[group_name]:
                prom_name = outputs[children_name]['prom_name']
                aviary_metadata = av.CoreMetaData.get(prom_name)
                children_list.append(
                    {
                        'abs_name': children_name,
                        'prom_name': prom_name,
                        'value': convert_ndarray_to_support_nans_in_json(
                            outputs[children_name]['val']
                        ),
                        'units': outputs[children_name]['units'],
                        'metadata': json.dumps(aviary_metadata),
                    }
                )
            table_data_nested.append(  # not a real var, just a group of vars so no values
                {
                    'abs_name': group_name,
                    'prom_name': '',
                    'value': '',
                    'units': '',
                    '_children': children_list,
                }
            )

    aviary_variables_file_path = (
        f'{script_name}_out/reports/aviary_vars/{aviary_variables_json_file_name}'
    )
    with open(aviary_variables_file_path, 'w') as fp:
        json.dump(table_data_nested, fp)

    return table_data_nested


def convert_driver_case_recorder_file_to_df(recorder_file_name):
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
            header = ['iter_count'] + all_var_names
            df = pd.DataFrame(columns=header)

        # Now fill up a row
        row = [
            i,
        ]
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


def create_aircraft_3d_file(recorder_file, reports_dir, outfilepath):
    """
    Create the HTML file with the display of the aircraft design
    in 3D using the A-Frame library.

    Parameters
    ----------
    recorder_file : str
        Name of the case recorder file.
    reports_dir : str
        Path of the directory containing the reports from the run.
    outfilepath : str
        The path to the location where the file should be created.
    """
    # Get the location of the HTML template file for this HTML file
    aviary_dir = Path(importlib.util.find_spec('aviary').origin).parent
    aircraft_3d_template_filepath = aviary_dir.joinpath(
        'visualization/assets/aircraft_3d_file_template.html'
    )

    # texture for the aircraft. Need to copy it to the reports directory
    #  next to the HTML file
    shutil.copy(
        aviary_dir.joinpath('visualization/assets/aviary_airlines.png'),
        Path(reports_dir) / 'aviary_airlines.png',
    )

    aircraft_3d_model = Aircraft3DModel(recorder_file)
    aircraft_3d_model.read_variables()
    aircraft_3d_model.get_aframe_markup()
    aircraft_3d_model.get_camera_entity(aircraft_3d_model.fuselage.length)
    aircraft_3d_model.write_file(aircraft_3d_template_filepath, outfilepath)


def _get_interactive_plot_sources(data_by_varname_and_phase, x_varname, y_varname, phase):
    x = data_by_varname_and_phase[x_varname][phase]
    y = data_by_varname_and_phase[y_varname][phase]
    if len(x) > 0 and len(x) == len(y):
        return x, y
    else:
        return [], []


def create_optimization_history_plot(case_recorder, df):
    # Create a ColumnDataSource
    source = ColumnDataSource(df)

    # Create a Bokeh figure
    plotting_figure = figure(
        title='Optimization History',
        width=1000,
        height=600,
    )
    plotting_figure.title.align = 'center'
    plotting_figure.yaxis.visible = False
    plotting_figure.xaxis.axis_label = 'Iterations'
    plotting_figure.yaxis.formatter = PrintfTickFormatter(format='%5.2e')
    plotting_figure.title.text_font_size = '25px'

    # Choose a palette
    palette = Category20[20]

    # Plot each time series and keep references to the renderers
    renderers = {}
    variable_names = list(df.columns)[1:]
    for i, variable_name in enumerate(variable_names):
        color = palette[i % 20]

        renderers[variable_name] = plotting_figure.line(
            x='iter_count',
            y=variable_name,
            source=source,
            y_range_name=f'extra_y_{variable_name}',
            color=color,
            line_width=2,
            visible=False,  # hide them all initially. clicking checkboxes makes them visible
        )

        # create axes both to the right and left of the plot.
        # hide them initially
        # as the user selects/deselects variables to be plotted, they get turned on/off
        extra_y_axis = LinearAxis(
            y_range_name=f'extra_y_{variable_name}',
            axis_label=f'{variable_name}',
            axis_label_text_color=color,
        )
        plotting_figure.add_layout(extra_y_axis, 'right')
        plotting_figure.right[i].visible = False

        extra_y_axis = LinearAxis(
            y_range_name=f'extra_y_{variable_name}',
            axis_label=f'{variable_name}',
            axis_label_text_color=color,
        )
        plotting_figure.add_layout(extra_y_axis, 'left')
        plotting_figure.left[i + 1].visible = False

        # set the range
        y_min = df[variable_name].min()
        y_max = df[variable_name].max()
        # if the range is zero, the axis will not be displayed. Plus need some range to make it
        #    look good. Some other code seems to do +- 1 for the range in this case.
        if y_min == y_max:
            y_min = y_min - 1
            y_max = y_max + 1
        plotting_figure.extra_y_ranges[f'extra_y_{variable_name}'] = Range1d(y_min, y_max)

    # Make a Legend with no items in it. those will be added in JavaScript
    #    as users select variables to be plotted
    legend = Legend(items=[], location=(-50, -5), border_line_width=0)

    # make the legend items in Python. Pass them to JavaScript where they can
    # be added to the Legend
    legend_items = []
    for variable_name in variable_names:
        units = case_recorder.problem_metadata['variables'][variable_name]['units']
        legend_item = LegendItem(
            label=f'{variable_name} ({units})', renderers=[renderers[variable_name]]
        )
        legend_items.append(legend_item)

    plotting_figure.add_layout(legend, 'below')

    # make the list of variables with checkboxes
    data_source = ColumnDataSource(
        data=dict(options=variable_names, checked=[False] * len(variable_names))
    )
    # Create a Div to act as a scrollable container
    variable_scroll_box = Div(
        styles={
            'overflow-y': 'scroll',
            'height': '500px',
            'border': '1px solid #ddd',
            'padding': '10px',
        }
    )

    # make the text box used to filter variables
    filter_variables_text_box = TextInput(placeholder='Variable name filter')

    # CustomJS callback for checkbox changes
    variable_checkbox_callback = CustomJS(
        args=dict(
            data_source=data_source,
            plotting_figure=plotting_figure,
            renderers=renderers,
            legend=legend,
            legend_items=legend_items,
        ),
        code="""
    // Three things happen in this code.
    //   1. turn on/off the plot lines
    //   2. show the legend items for the items being plotted
    //   3. show the y axes for each of the lines being plotted
    // The incoming Legend is empty. The items are passed in separately

    // 1. Plots
    // turn off or on the line plot for the clicked on variable
    const checkedIndex = cb_obj.index;
    const isChecked = cb_obj.checked;
    data_source.data['checked'][checkedIndex] = isChecked;
    renderers[data_source.data['options'][checkedIndex]].visible = isChecked;

    // 2. Legend
    // empty the Legend items and then add in the ones for the variables that are checked
    legend.items = [];
    for (let i =0; i < legend_items.length; i++){
       if ( data_source.data['checked'][i] ) {
            legend.items.push(legend_items[i]);
        }
    }

    // 3. Y axes
    // first hide all of them
    for (let i =0; i < legend_items.length; i++){
       var extra_y_axis = plotting_figure.left[i + 1];
       extra_y_axis.visible = false ;

       var extra_y_axis = plotting_figure.right[i];
       extra_y_axis.visible = false ;
    }
    // alternate between making visible the axes on the left and the right to make it more even.
    // this variable keeps track of which side to add the axes to.
    let put_on_left_side = true;
    for (let i =0; i < legend_items.length; i++){
        if (data_source.data['checked'][i]){
            if (put_on_left_side){
                plotting_figure.left[i + 1].visible = true;
            } else {
                plotting_figure.right[i].visible = true;
            }
            put_on_left_side = ! put_on_left_side ;
        }
    }
    data_source.change.emit();
    """,
    )

    # CustomJS callback for the variable filtering
    filter_variables_callback = CustomJS(
        args=dict(
            data_source=data_source,
            variable_scroll_box=variable_scroll_box,
            variable_checkbox_callback=variable_checkbox_callback,
        ),
        code="""

        const filter_text = cb_obj.value.toLowerCase();
        const all_options = data_source.data['options'];
        const checked_states = data_source.data['checked'];

        // Filter options
        const filtered_options = all_options.filter(option =>
            option.toLowerCase().includes(filter_text)
        );

        // Update the scroll box content
        let checkboxes_html = '';
        filtered_options.forEach((label) => {
            const index = all_options.indexOf(label);
            checkboxes_html += `
                <label style="display:block; margin-bottom:5px;">
                    <input type="checkbox" value="${label}" ${checked_states[index] ? 'checked' : ''}
                        onchange="Bokeh.documents[0].get_model_by_id('${variable_checkbox_callback.id}').execute({index: ${index}, checked: this.checked})">
                    ${label}
                </label>
            `;
        });
        variable_scroll_box.text = checkboxes_html;
    """,
    )

    filter_variables_text_box.js_on_change('value', filter_variables_callback)

    # Initial population of the scroll box
    initial_html = ''.join(
        f"""
        <label style="display:block; margin-bottom:5px;">
            <input type="checkbox" value="{variable_name}"
                onchange="Bokeh.documents[0].get_model_by_id('{variable_checkbox_callback.id}').execute({index: {i}, checked: this.checked} )">
            {variable_name}
        </label>
    """
        for i, variable_name in enumerate(variable_names)
    )
    variable_scroll_box.text = initial_html

    # Arrange the layout using Panel
    layout = pn.Row(pn.Column(filter_variables_text_box, variable_scroll_box), plotting_figure)

    return layout


@_handle_pane_creation_errors()
def _create_interactive_xy_plot_mission_variables(documentation, problem_recorder_path):
    """
    Create a pane showing an XY plot of mission variables.
    """
    if problem_recorder_path:
        if os.path.exists(problem_recorder_path):
            cr = om.CaseReader(problem_recorder_path)

            # determine what trajectories there are
            traj_nodes = [
                n
                for n in _meta_tree_subsys_iter(
                    cr.problem_metadata['tree'], cls='dymos.trajectory.trajectory:Trajectory'
                )
            ]

            if len(traj_nodes) == 0:
                raise ValueError(
                    'No trajectories available in case recorder file for use '
                    'in generating interactive XY plot of mission variables'
                )
            traj_name = traj_nodes[0]['name']
            if len(traj_nodes) > 1:
                issue_warning(
                    'More than one trajectory found in problem case recorder file. Only using '
                    f'the first one, "{traj_name}", for the interactive XY plot of mission variables'
                )
            case = cr.get_case('final')
            outputs = case.list_outputs(out_stream=None, units=True)

            # data_by_varname_and_phase = defaultdict(dict)
            data_by_varname_and_phase = defaultdict(lambda: defaultdict(list))

            # Find the "largest" unit used for any timeseries output across all phases
            units_by_varname = {}
            phases = set()
            varnames = set()
            # pattern used to parse out the phase names and variable names
            pattern = rf'{traj_name}\.phases\.([a-zA-Z0-9_]+)\.timeseries\.([a-zA-Z0-9_]+)'
            for varname, meta in outputs:
                match = re.match(pattern, varname)
                if match:
                    phase, name = match.group(1), match.group(2)
                    phases.add(phase)
                    varnames.add(name)
                    if name not in units_by_varname:
                        units_by_varname[name] = meta['units']
                    else:
                        _, new_conv_factor = conversion_to_base_units(meta['units'])
                        _, old_conv_factor = conversion_to_base_units(units_by_varname[name])
                        if new_conv_factor < old_conv_factor:
                            units_by_varname[name] = meta['units']

            # Now get the values using those units
            for varname, meta in outputs:
                match = re.match(pattern, varname)
                if match:
                    phase, name = match.group(1), match.group(2)
                    val = case.get_val(varname, units=units_by_varname[name])
                    data_by_varname_and_phase[name][phase] = val

            # determine the initial variables used for X and Y
            varname_options = list(sorted(varnames, key=str.casefold))
            if 'distance' in varname_options:
                x_varname_default = 'distance'
            elif 'time' in varname_options:
                x_varname_default = 'time'
            else:
                x_varname_default = varname_options[0]

            if 'altitude' in varname_options:
                y_varname_default = 'altitude'
            else:
                y_varname_default = varname_options[-1]

            # need to create ColumnDataSource for each phase
            sources = {}
            for phase in phases:
                x, y = _get_interactive_plot_sources(
                    data_by_varname_and_phase, x_varname_default, y_varname_default, phase
                )
                sources[phase] = ColumnDataSource(data=dict(x=x, y=y))

            # Create the figure
            p = figure(
                width=800,
                height=400,
                tools='pan,box_zoom,xwheel_zoom,hover,undo,reset,save',
                tooltips=[
                    ('x', '@x'),
                    ('y', '@y'),
                ],
            )

            colors = d3['Category20'][20][0::2] + d3['Category20'][20][1::2]
            legend_data = []
            phases = sorted(phases, key=str.casefold)
            for i, phase in enumerate(phases):
                color = colors[i % 20]
                scatter_plot = p.scatter(
                    'x',
                    'y',
                    source=sources[phase],
                    color=color,
                    size=5,
                )
                line_plot = p.line(
                    'x',
                    'y',
                    source=sources[phase],
                    color=color,
                    line_width=1,
                )
                legend_data.append((phase, [scatter_plot, line_plot]))

            # Make the Legend
            legend = Legend(items=legend_data, location='center', label_text_font_size='8pt')
            # so users can click on the dot in the legend to turn off/on that phase in
            # the plot
            legend.click_policy = 'hide'
            p.add_layout(legend, 'right')

            # Create dropdown menus for X and Y axis selection
            x_select = pn.widgets.Select(
                name='X-Axis', value=x_varname_default, options=varname_options
            )
            y_select = pn.widgets.Select(
                name='Y-Axis', value=y_varname_default, options=varname_options
            )

            # Callback function to update the plot
            @pn.depends(x_select, y_select)
            def update_plot(x_varname, y_varname):
                for phase in phases:
                    x = data_by_varname_and_phase[x_varname][phase]
                    y = data_by_varname_and_phase[y_varname][phase]
                    x, y = _get_interactive_plot_sources(
                        data_by_varname_and_phase, x_varname, y_varname, phase
                    )
                    sources[phase].data = dict(x=x, y=y)

                p.xaxis.axis_label = f'{x_varname} ({units_by_varname[x_varname]})'
                p.yaxis.axis_label = f'{y_varname} ({units_by_varname[y_varname]})'

                p.hover.tooltips = [(x_varname, '@x'), (y_varname, '@y')]
                return p

            # Create the dashboard pane for this plot
            interactive_mission_var_plot_pane = pn.Column(
                pn.pane.Markdown(
                    f'# Interactive Mission Variable Plot for Trajectory, {traj_name}'
                ),
                pn.Row(x_select, y_select),
                pn.Row(pn.HSpacer(), update_plot, pn.HSpacer()),
            )
        else:
            interactive_mission_var_plot_pane = pn.pane.Markdown(
                f"# Recorder file '{problem_recorder_path}' not found."
            )

        interactive_mission_var_plot_pane_with_doc = pn.Column(
            pn.pane.HTML(
                f'<p class="pane_doc">{documentation}</p>',
                stylesheets=['assets/aviary_styles.css'],
                styles={'text-align': documentation_text_align},
            ),
            interactive_mission_var_plot_pane,
        )

    return interactive_mission_var_plot_pane_with_doc


# The main script that generates all the tabs in the dashboard
def dashboard(script_name, problem_recorder, driver_recorder, port, run_in_background=False):
    """
    Generate the dashboard app display.

    Parameters
    ----------
    script_name : str
        Name of the script file whose results will be displayed by this dashboard.
    problem_recorder : str
        Name of the recorder file containing the Problem cases.
    driver_recorder : str or None
        Name of the recorder file containing the Driver cases. If None, the driver tab will not be added
    port : int
        HTTP port used for the dashboard webapp. If 0, use any free port
    """
    reports_dir = f'{script_name}_out/reports/'
    out_dir = f'{script_name}_out/'

    if not Path(reports_dir).is_dir():
        raise ValueError(
            f"The script name, '{script_name}', does not have a reports folder "
            f"associated with it. The directory '{reports_dir}' does not exist."
        )

    problem_recorder_path = Path(out_dir) / problem_recorder

    if not os.path.isfile(problem_recorder_path):
        issue_warning(f'Given Problem case recorder file {problem_recorder_path} does not exist.')

    # TODO - use lists and functions to do this with a lot less code
    ####### Model Tab #######
    model_tabs_list = []

    # Input Checks
    create_report_frame(
        'Input Checks',
        model_tabs_list,
        'Detailed checks on the model inputs.',
        'markdown',
        Path(reports_dir) / 'input_checks.md',
    )

    #  Debug Input List
    create_report_frame(
        'Debug Input List',
        model_tabs_list,
        """
        A plain text display of the model inputs. Recommended for beginners. Only created if Settings.VERBOSITY is set to at least 2 in the input deck.
        The variables are listed in a tree structure. There are three columns. The left column is a list of variable names,
        the middle column is the value, and the right column is the
        promoted variable name. The hierarchy is phase, subgroups, components, and variables. An input variable can appear under
        different phases and within different components. Its values can be different because its value has
        been updated during the computation. On the top-left corner is the total number of inputs.
        That number counts the duplicates because one variable can appear in different phases.""",
        'text',
        Path(reports_dir) / 'input_list.txt',
    )

    #  Debug Output List
    create_report_frame(
        'Debug Output List',
        model_tabs_list,
        """
       A plain text display of the model outputs. Recommended for beginners. Only created if Settings.VERBOSITY is set to at least 2 in the input deck.
        The variables are listed in a tree structure. There are three columns. The left column is a list of variable names,
        the middle column is the value, and the right column is the
        promoted variable name. The hierarchy is phase, subgroups, components, and variables. An output variable can appear under
        different phases and within different components. Its values can be different because its value has
        been updated during the computation. On the top-left corner is the total number of outputs.
        That number counts the duplicates because one variable can appear in different phases.""",
        'text',
        Path(reports_dir) / 'output_list.txt',
    )

    # Inputs
    create_report_frame(
        'Inputs',
        model_tabs_list,
        'Detailed report on the model inputs.',
        'html',
        Path(reports_dir) / 'inputs.html',
    )

    # N2
    create_report_frame(
        'N2',
        model_tabs_list,
        """
        The N2 diagram, sometimes referred to as an eXtended Design Structure Matrix (XDSM), is a
        powerful tool for understanding your model in OpenMDAO. It is an N-squared diagram in the
        shape of a matrix representing functional or physical interfaces between system elements.
        It can be used to systematically identify, define, tabulate, design, and analyze functional
        and physical interfaces.""",
        'html',
        Path(reports_dir) / 'n2.html',
    )

    # Trajectory Linkage
    create_report_frame(
        'Trajectory Linkage',
        model_tabs_list,
        """
        This is a Dymos linkage report in a customized N2 diagram. It provides a report detailing how phases
        are linked together via constraint or connection. The diagram clearly shows how mission phases are linked.
        It can be used to identify errant linkages between fixed quantities.
        """,
        'html',
        Path(reports_dir) / 'traj_linkage_report.html',
    )

    # Driver scaling
    create_report_frame(
        'Driver Scaling',
        model_tabs_list,
        """
            This report is a summary of driver scaling information. After all design variables, objectives, and constraints
            are declared and the problem has been set up, this report presents all the design variables and constraints
            in all phases as well as the objectives. It also shows Jacobian information showing responses with respect to
            design variables (DV).
        """,
        'html',
        Path(reports_dir) / 'driver_scaling_report.html',
    )

    ####### Optimization Tab #######
    optimization_tabs_list = []

    # Optimization History Plot
    if driver_recorder:
        if os.path.isfile(driver_recorder):
            df = convert_driver_case_recorder_file_to_df(f'{driver_recorder}')
            cr = om.CaseReader(f'{driver_recorder}')
            opt_history_pane = create_optimization_history_plot(cr, df)
            optimization_tabs_list.append(('Optimization History', opt_history_pane))

    # IPOPT report
    if os.path.isfile(Path(reports_dir) / 'IPOPT.out'):
        ipopt_pane = create_report_frame(
            'IPOPT Output',
            optimization_tabs_list,
            """
            This report is generated by the IPOPT optimizer.
                                        """,
            'text',
            Path(reports_dir) / 'IPOPT.out',
        )

    # Optimization report
    opt_report_pane = create_report_frame(
        'Summary',
        optimization_tabs_list,
        """
        This report is an OpenMDAO optimization report. All values are in unscaled, physical units.
        On the top is a summary of the optimization, followed by the objective, design variables, constraints,
        and optimizer settings. This report is important when dissecting optimal results produced by Aviary.""",
        'html',
        Path(reports_dir) / 'opt_report.html',
    )

    # PyOpt report
    if os.path.isfile(Path(reports_dir) / 'pyopt_solution.out'):
        create_report_frame(
            'PyOpt Solution',
            optimization_tabs_list,
            """
            This report is generated by the pyOptSparse optimizer.
            """,
            'text',
            Path(reports_dir) / 'pyopt_solution.txt',
        )

    # SNOPT report
    if os.path.isfile(Path(reports_dir) / 'SNOPT_print.out'):
        create_report_frame(
            'SNOPT Output',
            optimization_tabs_list,
            """
            This report is generated by the SNOPT optimizer.
                                        """,
            'text',
            Path(reports_dir) / 'SNOPT_print.out',
        )

    # SNOPT summary
    if os.path.isfile(Path(reports_dir) / 'SNOPT_summary.out'):
        create_report_frame(
            'SNOPT Summary',
            optimization_tabs_list,
            """
            This is a report generated by the SNOPT optimizer that summarizes the optimization results.""",
            'text',
            Path(reports_dir) / 'SNOPT_summary.out',
        )

    # Coloring report
    create_report_frame(
        'Total Coloring',
        optimization_tabs_list,
        'The report shows metadata associated with the creation of the coloring.',
        'html',
        Path(reports_dir) / 'total_coloring.html',
    )

    ####### Results Tab #######
    results_tabs_list = []

    # Aircraft 3d model display
    if problem_recorder_path:
        if os.path.isfile(problem_recorder_path):
            try:
                aircraft_3d_file = Path(reports_dir) / 'aircraft_3d.html'
                create_aircraft_3d_file(problem_recorder_path, reports_dir, aircraft_3d_file)
                create_report_frame(
                    'Aircraft 3d model',
                    results_tabs_list,
                    '3D model view of designed aircraft.',
                    'html',
                    aircraft_3d_file,
                )
            except Exception as e:
                create_report_frame(
                    'Aircraft 3d model',
                    results_tabs_list,
                    '3D model view of designed aircraft.',
                    'simple_message',
                    f'Unable to create aircraft 3D model display due to error: {e}',
                )

    # Make the Aviary variables table pane
    if os.path.isfile(problem_recorder_path):
        try:
            # Make dir reports/script_name/aviary_vars if needed
            aviary_vars_dir = Path(reports_dir) / 'aviary_vars'
            aviary_vars_dir.mkdir(parents=True, exist_ok=True)

            # copy index.html file to reports/script_name/aviary_vars/index.html
            aviary_dir = Path(importlib.util.find_spec('aviary').origin).parent

            shutil.copy(
                aviary_dir.joinpath('visualization/assets/aviary_vars/index.html'),
                aviary_vars_dir.joinpath('index.html'),
            )
            shutil.copy(
                aviary_dir.joinpath('visualization/assets/aviary_vars/script.js'),
                aviary_vars_dir.joinpath('script.js'),
            )
            # copy script.js file to reports/script_name/aviary_vars/index.html.
            # mod the script.js file to point at the json file
            # create the json file and put it in
            # reports/script_name/aviary_vars/aviary_vars.json
            create_aviary_variables_table_data_nested(
                script_name, problem_recorder_path
            )  # create the json file
        except Exception as e:
            pane = _create_message_pane(
                'Table showing Aviary variables',
                f"Unable to create 'Aviary Variables' pane in dashboard due to error: {e}.",
            )
            results_tabs_list.append(('Aviary Variables', pane))

        create_report_frame(
            'Aviary Variables',
            results_tabs_list,
            'Table showing Aviary variables.',
            'html',
            Path(reports_dir) / 'aviary_vars/index.html',
        )

    # Mission Summary
    create_report_frame(
        'Mission Summary',
        results_tabs_list,
        'A report of mission results from an Aviary problem.',
        'markdown',
        Path(reports_dir) / 'mission_summary.md',
    )

    # Run status pane
    create_table_pane_from_json(
        'Run status pane',
        results_tabs_list,
        'A high level overview of the status of the run.',
        Path(reports_dir) / 'status.json',
    )
    run_status_pane_tab_number = len(results_tabs_list) - 1

    # Timeseries Mission Output Report
    create_csv_frame(
        'Timeseries Mission Output',
        results_tabs_list,
        """
        The outputs of the aircraft trajectory.
        Any value that is included in the timeseries data is included in this report.
        This data is useful for post-processing, especially those used for acoustic analysis.
        """,
        Path(reports_dir) / 'mission_timeseries_data.csv',
    )

    # Trajectory results
    traj_results_report_pane = create_report_frame(
        'Trajectory Results',
        results_tabs_list,
        """
            This is one of the most important reports produced by Aviary. It will help you visualize and
            understand the optimal trajectory produced by Aviary.
            Users should play with it and try to grasp all possible features.
            This report contains timeseries and phase parameters in different tabs.
            On the timeseries tab, users can select which phases to view.
            Other features include hovering the mouse over the solution points to see solution value and
            zooming into a particular region for details, etc.
        """,
        'html',
        Path(reports_dir) / 'traj_results_report.html',
    )

    # Interactive XY plot of mission variables
    _create_interactive_xy_plot_mission_variables(
        'Interactive Mission Variable Plot',
        results_tabs_list,
        'Plot of mission variables allowing user to select X and Y plot values.',
        problem_recorder_path,
    )

    ####### Subsystems Tab #######
    subsystem_tabs_list = []

    # Look through subsystems directory for markdown files
    # The subsystems report tab shows selected results for every major
    # subsystem in the Aviary problem

    for md_file in sorted(Path(f'{reports_dir}subsystems').glob('*.md'), key=str):
        create_report_frame(
            md_file.stem,
            subsystem_tabs_list,
            f"""

        The subsystems report tab shows selected results for every major subsystem in the Aviary problem.
        This report is for the
            {md_file.stem}
             subsystem. Reports available currently are mass, geometry, and propulsion.
            """,
            'markdown',
            str(md_file),
        )

    # Actually make the tabs from the list of Panes
    model_tabs = pn.Tabs(*model_tabs_list, stylesheets=['assets/aviary_styles.css'])
    optimization_tabs = pn.Tabs(*optimization_tabs_list, stylesheets=['assets/aviary_styles.css'])
    results_tabs = pn.Tabs(*results_tabs_list, stylesheets=['assets/aviary_styles.css'])
    if run_status_pane_tab_number:
        # make the run status tab active initially
        results_tabs.active = run_status_pane_tab_number
    if subsystem_tabs_list:
        subsystem_tabs = pn.Tabs(*subsystem_tabs_list, stylesheets=['assets/aviary_styles.css'])

    # Add subtabs to tabs
    high_level_tabs = []
    high_level_tabs.append(('Results', results_tabs))
    if subsystem_tabs_list:
        high_level_tabs.append(('Subsystems', subsystem_tabs))
    high_level_tabs.append(('Model', model_tabs))
    high_level_tabs.append(('Optimization', optimization_tabs))
    tabs = pn.Tabs(*high_level_tabs, stylesheets=['assets/aviary_styles.css'])

    save_dashboard_button = pn.widgets.Button(
        name='Save Dashboard',
        width_policy='min',
        css_classes=['save-button'],
        button_type='success',
        button_style='solid',
        stylesheets=['assets/aviary_styles.css'],
    )
    header = pn.Row(save_dashboard_button, pn.HSpacer(), pn.HSpacer(), pn.HSpacer())

    def save_dashboard(event):
        print(f'Saving dashboard files to {script_name}.zip')
        shutil.make_archive(script_name, 'zip', f'{script_name}_out')

    save_dashboard_button.on_click(save_dashboard)

    tabs.active = 0  # make the Results tab active initially

    # get status of run for display in the header of each page
    status_string_for_header = get_run_status(Path(reports_dir) / 'status.json')

    template = pn.template.FastListTemplate(
        title=f'Aviary Dashboard for {script_name}:  {status_string_for_header}',
        logo='assets/aviary_logo.png',
        favicon='assets/aviary_logo.png',
        main=[tabs],
        accent_base_color='black',
        header_background='rgb(0, 212, 169)',
        header=header,
        background_color='white',
        theme=pn.theme.DefaultTheme,
        theme_toggle=False,
        main_layout=None,
        css_files=['assets/aviary_styles.css'],
    )

    if env_truthy('TESTFLO_RUNNING'):
        show = False
        threaded = True
    else:
        show = True
        threaded = False

    # override `show` without changing `threaded`
    if run_in_background:
        show = False

    assets_dir = Path(importlib.util.find_spec('aviary').origin).parent.joinpath(
        'visualization/assets/'
    )
    home_dir = '.'
    if port == 0:
        port = get_free_port()

    server = pn.serve(
        template,
        port=port,
        address='localhost',
        websocket_origin=f'localhost:{port}',
        show=show,
        threaded=threaded,
        static_dirs={
            'reports': reports_dir,
            'home': home_dir,
            'assets': assets_dir,
        },
    )
    server.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _dashboard_setup_parser(parser)
    args = parser.parse_args()
    _dashboard_cmd(args, None)
