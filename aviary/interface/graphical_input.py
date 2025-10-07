import importlib.util  # used for opening existing phase info file
import json
import os
import platform
import pprint
import shutil
import subprocess
import tkinter as tk  # base tkinter
import tkinter.ttk as ttk  # used for combobox
from tkinter import filedialog, font, messagebox

import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# used for unit conversion of numerical data
from openmdao.utils.units import convert_units


def get_screen_geometry():
    """
    Taken from: https://stackoverflow.com/questions/3129322/how-do-i-get-monitor-resolution-in-python/56913005#56913005
    Workaround to get the size of the current screen in a multi-screen setup.
    """
    root = tk.Tk()
    root.update_idletasks()
    root.attributes('-fullscreen', True)
    root.withdraw()
    geometry = root.winfo_geometry()
    root.destroy()
    return geometry


class VerticalScrolledFrame(tk.Frame):
    """A pure Tkinter scrollable frame that actually works!
    * Use the 'interior' attribute to place widgets inside the scrollable frame.
    * Construct and pack/place/grid normally.
    * This frame only allows vertical scrolling.
    ------
    Taken from https://stackoverflow.com/questions/16188420/tkinter-scrollbar-for-frame
    """

    def __init__(self, parent, *args, **kw):
        tk.Frame.__init__(self, parent, *args, **kw)
        # Create a canvas object and a vertical scrollbar for scrolling it.
        vscrollbar = tk.Scrollbar(self, orient='vertical')
        vscrollbar.pack(fill='y', side='right', expand=False)
        # this frame will not scroll, allowing for freeze view functionality
        self.freezeframe = tk.Frame(self)
        self.freezeframe.pack(side='top', fill='x')
        # another freeze frame at bottom for warnings
        self.freezeframe_bottom = tk.Frame(self)
        self.freezeframe_bottom.pack(side='bottom', fill='x')

        canvas = tk.Canvas(self, bd=0, highlightthickness=0, yscrollcommand=vscrollbar.set)
        canvas.pack(side='left', fill='y', expand=True)
        self.vscroll_canvas = canvas
        vscrollbar.config(command=canvas.yview)

        # Reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # Create a frame inside the canvas which will be scrolled with it.
        self.interior = interior = tk.Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior, anchor='nw')

        # Track changes to the canvas and frame width and sync them,
        # also updating the scrollbar.
        def _configure_interior(event):
            # Update the scrollbars to match the size of the inner frame.
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion='0 0 %s %s' % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # Update the canvas's width to fit the inner frame.
                canvas.config(width=interior.winfo_reqwidth())

        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # Update the inner frame's width to fill the canvas.
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())

        canvas.bind('<Configure>', _configure_canvas)


class AviaryMissionEditor(tk.Tk):
    """Aviary mission editor class."""

    def __init__(self):
        screen_width, screen_height = [
            int(x) for x in get_screen_geometry().split('+')[0].split('x')
        ]

        super().__init__()
        self.title('Mission Design Utility')
        self.protocol('WM_DELETE_WINDOW', self.close_window)
        self.macOS = platform.system() == 'Darwin'
        self.default_font = font.nametofont('TkDefaultFont')
        # used by labels, buttons
        self.default_font.configure(size=12 if self.macOS else 10)
        font.nametofont('TkTextFont').configure(size=12 if self.macOS else 10)  # used by entries

        # ------
        # theme related initializations
        self.theme = 'light'
        self.pallete = {
            'light': {
                'background_primary': '#ffffff',
                'foreground_primary': '#000000',
                'foreground_secondary': '#999999',
                'crosshair': '#EE0000',
                'lines': ['#0209c6', '#aa00aa'],
                'image': 'dark_mode.png',
                'hover': '#63ebeb',
            },
            'dark': {
                'background_primary': '#1e1e1e',
                'foreground_primary': '#FEFEFE',
                'foreground_secondary': '#CCCCCC',
                'crosshair': '#EE0000',
                'lines': ['#00b6f2', '#ffff00'],
                'image': 'light_mode.png',
                'hover': '#007acc',
            },
        }
        self.style_combobox = ttk.Style()
        self.style_combobox.theme_use('alt')
        # updates image object inside pallete with PhotoImage object using absolute filepath
        self.source_directory = os.path.abspath(os.path.dirname(__file__))
        for theme_info in self.pallete.values():
            theme_info['image'] = tk.PhotoImage(
                file=os.path.join(
                    self.source_directory,
                    'static',
                    'mac_theme.png' if self.macOS else theme_info['image'],
                )
            )
        # stores/retrieves persistent settings in source directory
        self.persist_filename = os.path.join(self.source_directory, 'persist_settings.json')

        # ------
        # window geometry definition
        # tkinter size string format:  widthxheight+x+y ;  x,y are location
        # Based on subplots, matplotlib will make window 500 in height regardless of initial size
        # A width of 800 is the minimum required to see the axes labels properly
        # The minimum sized is used unless the screen is large enough in which case 50% of width/height of
        # the screen is used as the size. If user saves settings then their last saved geometry is used.
        min_win_size = (900, 500)
        self.minsize(*min_win_size)  # force a minimum size for layout to look correct

        self.store_settings = tk.BooleanVar()  # tracks if user wants to store settings or not
        if os.path.exists(self.persist_filename):
            # a file will only exist if at a previous point user wanted to store settings
            self.store_settings.set(True)
            with open(self.persist_filename, 'r') as fp:
                persist_settings = json.load(fp)
                window_geometry = persist_settings['window_geometry']
                self.theme = persist_settings['theme']
        else:
            default_win_size = (
                max(min_win_size[0], int(screen_width / 2)),
                max(min_win_size[1], int(screen_height / 2)),
            )
            default_location = (int((screen_width - default_win_size[0]) / 2), 0)
            window_geometry = (
                f'{default_win_size[0]}x{default_win_size[1]}+'
                + f'{default_location[0]}+{default_location[1]}'
            )
        self.geometry(window_geometry)

        # Set the window icon, provides 2 sizes of logos to prevent blurry icons
        self.iconphoto(
            False,
            tk.PhotoImage(file=os.path.join(self.source_directory, 'static', 'aviary_logo_16.png')),
            tk.PhotoImage(file=os.path.join(self.source_directory, 'static', 'aviary_logo_32.png')),
        )

        # ------
        # create window layout with frames for containing graph, table, and scrollbar
        self.frame_table = VerticalScrolledFrame(self)
        self.frame_table.pack(side='right', fill='y')
        self.frame_tableheaders = self.frame_table.freezeframe
        self.frame_plot_table_border = tk.Frame(self, highlightthickness=1)
        self.frame_plot_table_border.pack(side='right', fill='y')

        self.frame_plotReadouts = tk.Frame(self)
        self.frame_plotReadouts.pack(side='bottom', fill='x')
        self.frame_plots = tk.Frame(self)
        self.frame_plots.pack(side='top', expand=True, fill='both')

        # ------
        # Main definition of data which can be plotted/tabulated. Assumes single
        # independent variable followed by any number of dependent variables.
        # Plot titles inform the program of number of dependent variables.
        self.data_info = {
            'plot_titles': ['Altitude Profile', 'Mach Profile'],
            'labels': ['Time', 'Altitude', 'Mach'],
            'units': ['min', 'ft', 'unitless'],
            'limits': [400, 50e3, 1.0],
            'rounding': [0, 0, 2],
        }

        self.advanced_options = {
            'constrain_range': tk.BooleanVar(value=True),
            'distance_solve_segments': tk.BooleanVar(),
            'include_takeoff': tk.BooleanVar(),
            'include_landing': tk.BooleanVar(),
            'polynomial_order': tk.IntVar(value=1),
        }

        self.check_data_info()  # sanity checking of data_info dict
        # replace constants with stringvars which can be updated within the GUI by the user
        for key in ['units', 'limits', 'rounding']:
            self.data_info[key] = [tk.StringVar(value=item) for item in self.data_info[key]]

        # starting mach is hardcoded as 0.3 b/c Aviary models are not suitable for very low mach
        self.data = [[0], [0], [0.3]]
        self.phase_order_default = 3
        self.phase_order_list = []

        # internal variables to remember mouse state
        self.mouse_drag, self.mouse_press = False, False
        self.ptcontainer = (
            0.04  # percent of plot size, boundary around point where it can be dragged
        )

        self.popup = None
        self.show_optimize = tk.BooleanVar()  # controls display of optimize phase checkboxes
        # controls display of phase info (climb/descent rates)
        self.show_phase_slope = tk.BooleanVar()

        self.theme_button = tk.Button(
            self,
            image=self.pallete[self.theme]['image'],
            font=('Arial', 8),
            command=lambda: self.update_theme(toggle=True),
        )
        # to prevent lose of image reference from garbage collector
        self.theme_button.image = self.pallete[self.theme]['image']
        self.theme_button.bind('<Enter>', func=self.on_enter)
        self.theme_button.bind('<Leave>', func=self.on_leave)
        self.theme_button.place(anchor='sw', relx=0, rely=1.0)

        self.output_phase_info_button = tk.Button(self, text='Output Phase Info', command=self.save)
        self.output_phase_info_button.bind('<Enter>', func=self.on_enter)
        self.output_phase_info_button.bind('<Leave>', func=self.on_leave)
        self.output_phase_info_button.place(relx=0, rely=0, anchor='nw')

        self.save_option_defaults()

        self.create_plots()
        self.create_table()
        self.create_menu()
        self.update_theme()
        self.focus_force()  # focus the window

    def save_option_defaults(self):
        """Saves default values for advanced options and axes limits, these will be referenced
        if user chooses to reset advanced options or axes limits.
        """
        self.advanced_options_defaults = {}
        for key, item in self.advanced_options.items():
            self.advanced_options_defaults[key] = item.get()

        self.data_info_defaults = {}
        for key, item in self.data_info.items():
            if key == 'units' or key == 'limits' or key == 'rounding':
                self.data_info_defaults[key] = [element for element in item]

    def check_data_info(self):
        """Verifies data_info dict has consistent number of dependent variables."""
        self.num_dep_vars = len(self.data_info['plot_titles'])
        for key, item in self.data_info.items():
            if key != 'plot_titles':
                if len(item) != self.num_dep_vars + 1:
                    raise Exception(
                        f'Check data_info dictionary, expected {self.num_dep_vars + 1} elements inside {key}.'
                    )

    def update_list(self, value, index, axis):
        """Updates internal data lists based on row,col values. col corresponds
        to dependent/independent variable. row corresponds to point number.
        """
        try:
            value = float(value)
        except (ValueError, TypeError):
            self.point_warning_strvar.set('Invalid table entry!')
            return  # skip updating if value is not convertible to a float
        if value < 0:
            self.point_warning_strvar.set('Table entries must be positive values!')
            return  # skip updating negative values
        if index == len(self.data[0]):
            self.data[axis].append(value)
            if len(self.phase_order_list) < len(self.data[0]) - 1:
                # default lowest dymos phase transcription order value
                self.phase_order_list.append(self.phase_order_default)
        else:
            self.data[axis][index] = value
        self.point_warning_strvar.set('')

    def update_theme(self, toggle=False):
        """Called by theme toggle button and start of app, changes color settings for widgets
        based on current theme.
        """
        if toggle:
            self.theme = 'light' if self.theme == 'dark' else 'dark'

        # this command sets options for all the widgets
        self.tk_setPalette(
            background=self.pallete[self.theme]['background_primary'],
            foreground=self.pallete[self.theme]['foreground_primary'],
            insertBackground=self.pallete[self.theme]['foreground_primary'],
            highlightBackground=self.pallete[self.theme]['background_primary'],
            highlightColor=self.pallete[self.theme]['background_primary'],
            activeForeground=self.pallete[self.theme]['foreground_primary'],
            activeBackground=self.pallete[self.theme]['background_primary'],
            selectColor=self.pallete[self.theme]['background_primary'],
        )
        if not self.macOS:
            self.create_menu()  # recreating menu b/c tkinter menus cannot be reconfigured with new colors

        # update table header color, different from background
        self.frame_tableheaders.configure(background=self.pallete[self.theme]['hover'])
        self.frame_plot_table_border.configure(
            highlightbackground=self.pallete[self.theme]['foreground_primary']
        )
        for widget in self.table_header_widgets:
            widget.configure(background=self.pallete[self.theme]['hover'])
            widget.configure(foreground=self.pallete[self.theme]['foreground_primary'])
            if isinstance(widget, tk.Entry):
                widget.configure(readonlybackground=self.pallete[self.theme]['hover'])

        self.fig.set_facecolor(self.pallete[self.theme]['background_primary'])
        for plot in self.plots:
            plot.set_facecolor(self.pallete[self.theme]['background_primary'])
            plot.yaxis.label.set_color(self.pallete[self.theme]['foreground_primary'])
            plot.xaxis.label.set_color(self.pallete[self.theme]['foreground_primary'])
            plot.title.set_color(self.pallete[self.theme]['foreground_primary'])
            plot.grid(True, color=self.pallete[self.theme]['foreground_secondary'])
            for axis in ['x', 'y']:
                plot.tick_params(axis=axis, colors=self.pallete[self.theme]['foreground_primary'])
            for spine in ['left', 'top', 'right', 'bottom']:
                plot.spines[spine].set_color(self.pallete[self.theme]['foreground_secondary'])
        for text_list in self.plot_texts:
            for text in text_list:
                text.set(color=self.pallete[self.theme]['foreground_primary'])

        self.redraw_plot()

        # updating combobox colors
        self.option_add(
            '*TCombobox*Listbox*Background', self.pallete[self.theme]['background_primary']
        )
        self.option_add(
            '*TCombobox*Listbox*Foreground', self.pallete[self.theme]['foreground_primary']
        )
        self.option_add('*TCombobox*Listbox*selectBackground', self.pallete[self.theme]['hover'])
        self.option_add(
            '*TCombobox*Listbox*selectForeground', self.pallete[self.theme]['foreground_primary']
        )
        self.style_combobox.map(
            'TCombobox',
            fieldbackground=[('readonly', self.pallete[self.theme]['background_primary'])],
        )
        self.style_combobox.map(
            'TCombobox',
            selectbackground=[('readonly', self.pallete[self.theme]['background_primary'])],
        )
        self.style_combobox.map(
            'TCombobox',
            selectforeground=[('readonly', self.pallete[self.theme]['foreground_primary'])],
        )
        self.style_combobox.map(
            'TCombobox', background=[('readonly', self.pallete[self.theme]['hover'])]
        )
        self.style_combobox.map(
            'TCombobox', foreground=[('readonly', self.pallete[self.theme]['foreground_primary'])]
        )

        self.theme_button.configure(
            image=self.pallete[self.theme]['image'],
            bg=self.pallete[self.theme]['background_primary'],
        )
        self.theme_button.image = self.pallete[self.theme]['image']
        self.output_phase_info_button.configure(
            bg=self.pallete[self.theme]['background_primary'],
            fg=self.pallete['light' if self.macOS else self.theme]['foreground_primary'],
        )

        if self.macOS:  # macOS does not support button background color change with Tkinter, so maintain foreground color
            self.table_add_button.configure(foreground=self.pallete['light']['foreground_primary'])
            for widget in self.table_widgets:
                if isinstance(widget, tk.Button):
                    widget.configure(foreground=self.pallete['light']['foreground_primary'])

    # ----------------------
    # Plot related functions
    def create_plots(self):
        """Creates subplots according to data_info dict. Sets labels and limits.
        Ties mouse events to appropriate internal functions.
        """
        self.fig = Figure()
        self.plots = []
        self.plot_texts = [[] for _ in range(self.num_dep_vars)]
        for i in range(self.num_dep_vars):
            self.plots.append(
                self.fig.add_subplot(
                    self.num_dep_vars, 1, i + 1, title=self.data_info['plot_titles'][i]
                )
            )
        for plot in self.plots:
            self.crossX = plot.axhline(y=0)
            self.crossY = plot.axvline(x=0)
        self.crosshair = True
        self.update_axes(units=True, limits=True)

        self.fig.tight_layout(pad=2)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.figure_canvas = FigureCanvasTkAgg(self.fig, master=self.frame_plots)
        self.figure_canvas.draw()

        self.mouse_coords_str = tk.StringVar(value='Mouse Coordinates')
        self.mouse_coords = tk.Label(self.frame_plotReadouts, textvariable=self.mouse_coords_str)
        self.mouse_coords.pack()
        self.crosshair = False
        self.figure_canvas.get_tk_widget().pack(expand=True, fill='both')

    def update_axes(self, units=False, limits=False, refresh=False):
        for i, plot in enumerate(self.plots):
            if units:
                xlabel = f'{self.data_info["labels"][0]} ({self.data_info["units"][0].get()})'
                ylabel = (
                    f'{self.data_info["labels"][i + 1]} ({self.data_info["units"][i + 1].get()})'
                )
                plot.set(xlabel=xlabel, ylabel=ylabel)
            if limits:
                xlim = (0, float(self.data_info['limits'][0].get()))
                ylim = (0, float(self.data_info['limits'][i + 1].get()))
                plot.set(xlim=xlim, ylim=ylim)
        if refresh:
            self.figure_canvas.draw()

    def redraw_plot(self):
        """Redraws plot, using the new values inside data lists."""
        self.clear_plot()
        for i, plot in enumerate(self.plots):
            plot.plot(
                self.data[0],
                self.data[i + 1],
                color=self.pallete[self.theme]['lines'][i],
                marker='o',
                markersize=5,
            )

        if self.show_phase_slope.get():
            self.toggle_phase_slope(redraw=False)
        self.figure_canvas.draw()
        if len(self.data[0]) > 1:
            units = [self.data_info['units'][i].get() for i in range(len(self.data_info['units']))]
            est_range, range_unit = estimate_total_range_trapezoidal(
                times=self.data[0], mach_numbers=self.data[2], units=units
            )
            self.mouse_coords_str.set(
                self.mouse_coords_str.get().split(' | Est')[0]
                + f' | Estimated Range: {est_range} {range_unit}'
            )

    def clear_plot(self):
        """Clears all lines from plots except for crosshairs."""
        for plot in self.plots:
            for line in plot.lines:
                if line == self.crossX or line == self.crossY:
                    continue
                line.remove()

    # ----------------------
    # Mouse related functions
    def on_mouse_press(self, event):
        """Handles mouse press event, sets internal mouse state."""
        self.mouse_press = True

    def on_mouse_release(self, event):
        """Handles release of mouse button. Calls click function if mouse has not been dragged."""
        if self.mouse_press and not self.mouse_drag:  # simple click event
            self.on_mouse_click(event)
        # currently no functions operate at the end of drag
        # else: pass # drag event
        self.mouse_press, self.mouse_drag = False, False

    def on_mouse_click(self, event):
        """Called when mouse click is determined, adds new point if it is valid."""
        # this list creates default values for subplots not clicked on, half of ylim
        default_y_vals = [float(lim.get()) / 2 for lim in self.data_info['limits'][1:]]
        valid_click = False
        # if mouse click points are not None
        if event.xdata and event.ydata and event.button == MouseButton.LEFT:
            # go through each subplot first to check if click is inside a subplot
            for plot_idx, plot in enumerate(self.plots):
                # checks if mouse is inside subplot and it is the first point or next in time
                if event.inaxes == plot and (
                    len(self.data[0]) < 1 or event.xdata > max(self.data[0])
                ):
                    valid_click = True
                    break
            # once we know a subplot was clicked inside at a valid location
            if valid_click:
                self.update_list(value=event.xdata, index=len(self.data[0]), axis=0)
                for y_idx, default_val in enumerate(default_y_vals):
                    self.update_list(
                        index=len(self.data[0]),
                        axis=y_idx + 1,
                        value=event.ydata if plot_idx == y_idx else default_val,
                    )
                valid_click = False
            # update plots and tables after having changed the lists
            self.redraw_plot()
            self.update_table()

    def on_mouse_move(self, event):
        """Handles functionality related to mouse movement. Creates crosshair if mouse is inside
        a subplot and updates cursor if near a point that can be dragged. Also handles moving
        point on graph if it is being dragged.
        """
        if event.xdata and event.ydata:
            for plot_idx, plot in enumerate(self.plots):
                if event.inaxes == plot:
                    # create crosshair at current point and remove old crosshair
                    if self.crosshair:
                        self.crossX.remove()
                        self.crossY.remove()
                    self.crossX = plot.axhline(
                        y=event.ydata, color=self.pallete[self.theme]['crosshair']
                    )
                    self.crossY = plot.axvline(
                        x=event.xdata, color=self.pallete[self.theme]['crosshair']
                    )
                    self.figure_canvas.draw()
                    self.crosshair = True

                    # update mouse coordinates on screen, rounding is handled based on
                    # rounding defined in data_info
                    xvalue = self.display_rounding(event.xdata, 0)
                    yvalue = self.display_rounding(event.ydata, plot_idx + 1)
                    self.mouse_coords_str.set(
                        f'{self.data_info["labels"][0]}: {xvalue} {self.data_info["units"][0].get()} | '
                        + f'{self.data_info["labels"][plot_idx + 1]}: {yvalue} {self.data_info["units"][plot_idx + 1].get()}'
                    )

                    # check if mouse is near an existing point, use closest point for dragging
                    near = False
                    dists = []
                    if len(self.data[0]) > 0:
                        for existing_pt in zip(self.data[0], self.data[plot_idx + 1]):
                            dists.append(
                                self.get_distance((event.xdata, event.ydata), existing_pt, plot_idx)
                            )
                        min_dist = min(dists)
                        if min_dist < self.ptcontainer:
                            self.figure_canvas.set_cursor(4)
                            near = True
                            self.near_idx = dists.index(min_dist)

                    if not near:
                        self.figure_canvas.set_cursor(1)

                    # move nearby point (or if previously dragging a point)
                    if self.mouse_press and (near or self.mouse_drag):
                        self.mouse_drag = True
                        self.update_list(index=self.near_idx, axis=0, value=event.xdata)
                        self.update_list(index=self.near_idx, axis=plot_idx + 1, value=event.ydata)

            # redraw plot after looping through subplots
            self.redraw_plot()
            self.update_str_vars()

    def get_distance(self, pt1: tuple, pt2: tuple, plot_idx: int):
        """Returns a normalized distance value between 2 points. Normalization is based on the subplot's
        x and y limits, subplot specified as plot_idx.
        """
        lims = (self.plots[plot_idx].get_xlim()[1], self.plots[plot_idx].get_ylim()[1])
        return np.sqrt(sum([((pt1[i] - pt2[i]) / lims[i]) ** 2 for i in range(2)]))

    # ----------------------
    # Table related functions
    def update_str_vars(self):
        """Updates StringVar values for the table. Used when points are dragged on plot."""
        for i, vallist in enumerate(self.data):
            for j, val in enumerate(vallist):
                val = self.display_rounding(val, i)
                self.table_strvars[i][j].set(val)

    def delete_point(self, row: int):
        """When X button next to tabular point is pressed, lists are popped and plot and tables
        are updated to show the removed point.
        """
        if row < len(self.data[0]) and row > 0:
            self.phase_order_list.pop(row - 1)
            if len(self.plot_texts[0]) > 0:
                for i in range(self.num_dep_vars):
                    self.plot_texts[i][row - 1].remove()
                    self.plot_texts[i].pop(row - 1)
                self.figure_canvas.draw()

        for i in range(len(self.data)):
            self.data[i].pop(row)

        self.redraw_plot()
        self.update_table(overwrite=True)

    def update_table(self, overwrite=False, bool_list=None):
        """This function handles both adding a new entry to table and overwriting the whole table.
        Overwriting causes all table widgets to be destroyed and a new set of widgets to be created.
        This also resets the StringVars.
        """
        row = (
            len(
                # last row (assumes data lists have been updated with new point)
                self.data[0]
            )
            - 1
        )
        if overwrite and len(self.table_widgets) > 0:
            for item in self.table_widgets:
                item.destroy()
            self.table_widgets = []
            self.table_strvars = [[] for i in range(self.num_dep_vars + 1)]
            self.table_boolvars = [[] for i in range(self.num_dep_vars)]
            if len(self.data[0]) > 0:
                row = 0  # set row to 0 if overwriting entire table

        while row < len(self.data[0]) and row >= 0:
            # numerical label for each point
            rowtxt = str(row + 1)
            if row + 1 < 10:
                rowtxt = '  ' + rowtxt
            rownum_label = tk.Label(
                self.frame_table.interior,
                text=rowtxt,
                background=self.pallete[self.theme]['background_primary'],
                foreground=self.pallete[self.theme]['foreground_primary'],
            )
            rownum_label.grid(row=row * 2 + 2, column=0)
            self.table_widgets.append(rownum_label)

            if row > 0 and self.show_optimize.get():  # have at least 2 points
                optimize_label = tk.Label(
                    self.frame_table.interior,
                    text='Optimize:',
                    background=self.pallete[self.theme]['background_primary'],
                    foreground=self.pallete[self.theme]['foreground_primary'],
                )
                optimize_label.grid(row=row * 2 + 1, column=1)
                self.table_widgets.append(optimize_label)

            # entries and stringvars for each x,y value
            for col, val in enumerate([data_axis[row] for data_axis in self.data]):
                val = self.display_rounding(val, col)
                entry_text = tk.StringVar(value=val)
                self.table_strvars[col].append(entry_text)

                entry = tk.Entry(
                    self.frame_table.interior,
                    width=self.table_column_widths[col],
                    textvariable=entry_text,
                    justify='center',
                    background=self.pallete[self.theme]['background_primary'],
                    foreground=self.pallete[self.theme]['foreground_primary'],
                )
                entry.grid(row=row * 2 + 2, column=col + 1)
                # binds key release to update list function
                entry.bind(
                    '<KeyRelease>',
                    lambda e, row=row, col=col, entry_text=entry_text: [
                        self.update_list(index=row, axis=col, value=entry_text.get()),
                        self.redraw_plot(),
                    ],
                )
                self.table_widgets.append(entry)

                if (
                    col > 0 and row > 0 and self.show_optimize.get()
                ):  # have at least 2 points and for dependent var cols only
                    checkbox_label = tk.Label(
                        self.frame_table.interior,
                        text=self.data_info['labels'][col],
                        background=self.pallete[self.theme]['background_primary'],
                        foreground=self.pallete[self.theme]['foreground_primary'],
                    )
                    checkbox_label.grid(row=row * 2 + 1, column=col + 1, sticky='w')
                    self.table_widgets.append(checkbox_label)

                    optimize_variable = tk.BooleanVar()
                    self.table_boolvars[col - 1].append(optimize_variable)
                    # if bool list has already been populated (e.g. opening an existing phase info)
                    if bool_list:
                        optimize_variable.set(value=bool_list[col - 1][row - 1])
                    optimize_checkbox = tk.Checkbutton(
                        self.frame_table.interior,
                        variable=optimize_variable,
                        background=self.pallete[self.theme]['background_primary'],
                        foreground=self.pallete[self.theme]['foreground_primary'],
                        activebackground=self.pallete[self.theme]['background_primary'],
                        activeforeground=self.pallete[self.theme]['foreground_primary'],
                        selectcolor=self.pallete[self.theme]['background_primary'],
                        highlightbackground=self.pallete[self.theme]['background_primary'],
                        highlightcolor=self.pallete[self.theme]['background_primary'],
                    )
                    optimize_checkbox.grid(row=row * 2 + 1, column=col + 1, sticky='e')
                    self.table_widgets.append(optimize_checkbox)

            # delete button for each point
            delete_button = tk.Button(
                self.frame_table.interior,
                text='X',
                width=self.delete_button_width,
                font=('Arial', self.default_font.actual()['size'] - 2),
                background=self.pallete[self.theme]['background_primary'],
                foreground=self.pallete['light' if self.macOS else self.theme][
                    'foreground_primary'
                ],
            )
            delete_button.bind('<Button-1>', lambda e, row=row: self.delete_point(row))
            delete_button.bind('<Enter>', func=self.on_enter)
            delete_button.bind('<Leave>', func=self.on_leave)
            delete_button.grid(row=row * 2 + 2, column=col + 2)
            self.table_widgets.append(delete_button)

            row += 1
        # reposition add new point button based on updated table
        if len(self.data[0]) > 0:
            self.table_add_button.grid(row=row * 2 + 3, column=0, columnspan=col + 2)

    def update_header(self, new_headers):
        """Update header."""
        i = 0
        for widget in self.table_header_widgets:
            if isinstance(widget, tk.Entry):
                widget.configure(textvariable=tk.StringVar(value=new_headers[i]))
                i += 1

    def add_new_row(self):
        """Updates data lists with a generic new point and runs redraw plot and update table.
        New point is added at x = halfway between last point and x limit, y = half of y limit.
        """
        default_y_vals = [float(lim.get()) / 2 for lim in self.data_info['limits'][1:]]
        newx = 0
        if len(self.data[0]) > 0:
            newx = (float(self.data_info['limits'][0].get()) - self.data[0][-1]) / 2 + self.data[0][
                -1
            ]
        for col, item in enumerate([newx, *default_y_vals]):
            self.update_list(index=len(self.data[0]), axis=col, value=item)

        self.redraw_plot()
        self.update_table()

    def create_table(self):
        """Creates headers for table and sets column widths based on header lengths."""
        self.table_column_widths = []
        self.table_strvars = []  # list used to hold StringVars
        self.table_boolvars = []
        self.table_widgets = []  # list used to hold graphical table elements, can be used to modify them
        self.table_header_widgets = []  # list used to hold header widgets, referenced for theme changes
        header = tk.Label(self.frame_tableheaders, text='Pt')
        header.grid(row=0, column=0)
        self.table_header_widgets.append(header)
        for col, (label, unit) in enumerate(zip(self.data_info['labels'], self.data_info['units'])):
            header_str = f'{label} ({unit.get()})'
            header_text = tk.StringVar(value=header_str)
            header_width = int(len(header_str) * (0.75 if self.macOS else 1))
            header = tk.Entry(
                self.frame_tableheaders,
                textvariable=header_text,
                state='readonly',
                width=header_width,
                justify='center',
                relief='groove',
            )
            header.grid(row=0, column=col + 1)
            self.table_column_widths.append(header_width)
            self.table_strvars.append([])
            if col > 0:
                self.table_boolvars.append([])
            self.table_header_widgets.append(header)

        # this spacer prevents invisbility of delete buttons after all of them have been deleted and a new point is added
        self.delete_button_width = 1 if self.macOS else 4
        delete_button_spacer = tk.Label(self.frame_tableheaders, width=self.delete_button_width)
        delete_button_spacer.grid(row=0, column=col + 2)
        self.table_header_widgets.append(delete_button_spacer)

        # button for adding new rows to table
        self.table_add_button = tk.Button(
            self.frame_table.interior, text='Add New Point', command=self.add_new_row
        )
        self.table_add_button.bind('<Enter>', func=self.on_enter)
        self.table_add_button.bind('<Leave>', func=self.on_leave)

        self.point_warning_strvar = tk.StringVar()
        self.point_warning = tk.Label(
            self.frame_table.freezeframe_bottom, textvariable=self.point_warning_strvar
        )
        self.point_warning.pack()

        self.update_table()

    def display_rounding(self, value, col: int, extra=0):
        """Returns a rounded value based on which variable the value belongs to.
        Uses rounding amount specified in data_info.
        """
        return format(value, '.' + str(int(self.data_info['rounding'][col].get()) + extra) + 'f')

    # ----------------------
    # Popup related functions

    def close_popup(self):
        """Function to close existing popup and refocus main window."""
        self.focus_set()
        self.popup.destroy()
        self.popup = None

    def generic_popup(self, pop_title='Popup', buttons_text=[]):
        """Function to create a base window for a popup. Returns popup object to be used for adding widget
        and configuring settings. Buttons_text can be used to specify any number of buttons. These button
        objects are returned for configuring commands and location.
        """
        popup = tk.Toplevel(self)
        # Set the window icon, provides 2 sizes of logos to prevent blurry icons
        popup.iconphoto(
            False,
            tk.PhotoImage(file=os.path.join(self.source_directory, 'static', 'aviary_logo_16.png')),
            tk.PhotoImage(file=os.path.join(self.source_directory, 'static', 'aviary_logo_32.png')),
        )
        popup.resizable(False, False)
        popup.title(pop_title)
        popup.focus_set()
        popup.configure(background=self.pallete[self.theme]['background_primary'])
        self.popup = popup

        popup.protocol('WM_DELETE_WINDOW', func=self.close_popup)

        popup_content_frame = tk.Frame(
            popup, background=self.pallete[self.theme]['background_primary']
        )
        popup_content_frame.pack(side='top', fill='x')
        button_frame = tk.Frame(popup, bg=self.pallete[self.theme]['background_primary'])
        button_frame.pack(side='bottom', pady=5)

        buttons = {}
        # button width based on longest button string
        button_width = len(max(buttons_text, key=len)) + 5

        for button_txt in buttons_text:
            button = tk.Button(
                button_frame,
                text=button_txt.title(),
                width=button_width,
                background=self.pallete[self.theme]['background_primary'],
                foreground=self.pallete['light' if self.macOS else self.theme][
                    'foreground_primary'
                ],
            )
            button.bind('<Enter>', func=self.on_enter)
            button.bind('<Leave>', func=self.on_leave)
            button.pack(side='left', padx=5)
            buttons[button_txt] = button

        return popup, popup_content_frame, buttons

    def place_popup(self):
        """Generic popup lets Tkinter automatically size the popup to fit all contents.
        This function uses that size and main GUI window size/location to compute a
        location for the popup that is central to the GUI.
        """
        self.popup.update_idletasks()
        pop_wid, pop_hei = [int(x) for x in self.popup.winfo_geometry().split('+')[0].split('x')]
        win_size, win_left, win_top = self.winfo_geometry().split('+')
        win_wid, win_hei = win_size.split('x')
        win_left, win_top, win_wid, win_hei = (
            int(win_left),
            int(win_top),
            int(win_wid),
            int(win_hei),
        )
        pop_left, pop_top = (
            int(win_left + win_wid / 2 - pop_wid / 2),
            int(win_top + win_hei / 2 - pop_hei / 2),
        )
        self.popup.geometry(f'+{pop_left}+{pop_top}')

    def change_axes_popup(self):
        """Creates a popup window that allows user to edit axes limits. This function is triggered
        by the menu buttons.
        """

        def reset_options(old_list=None):
            if not old_list:
                if len(self.data[0]) > 0:  # if resetting to bring data into view
                    old_list = []
                    for val_list in self.data:
                        old_list.append(max(val_list) * 1.2)
                else:
                    old_list = [float(item.get()) for item in self.data_info_defaults['limits']]
            for i, (value, lim_str) in enumerate(zip(old_list, self.data_info['limits'])):
                lim_str.set(value=self.display_rounding(value, col=i))

        current_lims = [float(lim.get()) for lim in self.data_info['limits']]

        popup, content_frame, buttons = self.generic_popup(
            pop_title='Axes Limits', buttons_text=['apply', 'reset', 'cancel']
        )
        popup.protocol(
            'WM_DELETE_WINDOW', func=lambda: [self.close_popup(), reset_options(current_lims)]
        )
        for i in range(2):
            # allow columns to expand in frame
            content_frame.columnconfigure(i, weight=1)

        for row, (label, unit, lim_str) in enumerate(
            zip(self.data_info['labels'], self.data_info['units'], self.data_info['limits'])
        ):
            lim_str.set(value=self.display_rounding(float(lim_str.get()), col=row))
            lim_label = tk.Label(
                content_frame,
                text=f'{label} ({unit.get()})',
                justify='right',
                background=self.pallete[self.theme]['background_primary'],
                foreground=self.pallete[self.theme]['foreground_primary'],
            )
            lim_label.grid(row=row, column=0, sticky='e')
            lim_entry = tk.Entry(
                content_frame,
                textvariable=lim_str,
                width=max(6, len(lim_str.get())),
                background=self.pallete[self.theme]['background_primary'],
                foreground=self.pallete[self.theme]['foreground_primary'],
            )
            lim_entry.grid(row=row, column=1, sticky='w')

        # apply uses values in entry boxes, reset defaults to original limits, cancel uses previously set limits
        buttons['apply'].configure(
            command=lambda: [self.close_popup(), self.update_axes(limits=True, refresh=True)]
        )
        buttons['reset'].configure(
            command=lambda: [
                self.close_popup(),
                reset_options(),
                self.update_axes(limits=True, refresh=True),
            ]
        )
        buttons['cancel'].configure(
            command=lambda: [
                self.close_popup(),
                reset_options(current_lims),
                self.update_axes(limits=True, refresh=True),
            ]
        )
        self.place_popup()

    def get_phase_names(self):
        """Returns a list of phase names, these are decided based on final and starting altitudes.
        These names are only used for the dropdown menu in advanced options, and are not connected to
        phase info phase names.
        """
        names = ['Climb ', 'Cruise ', 'Descent ']
        counters = [1, 1, 1]
        phase_name_list = []
        for i in range(len(self.data[0]) - 1):
            nextpt = round(self.data[1][i + 1], int(self.data_info['rounding'][1].get()))
            nowpt = round(self.data[1][i], int(self.data_info['rounding'][1].get()))
            if nextpt > nowpt:
                j = 0
            elif nextpt < nowpt:
                j = 2
            else:
                j = 1

            phase_name_list.append(names[j] + str(counters[j]))
            counters[j] += 1
        return phase_name_list

    def advanced_options_popup(self):
        """Creates a popup window that allows user to edit advanced options for phase info.
        Options included are specified as a dict in __init__ and include solve/constrain for range,
        include landing/takeoff, polynomial order, and phase order. This function is triggered by the menu buttons.
        """

        def reset_options(self, old_dict=self.advanced_options_defaults):
            for key, value in old_dict.items():
                self.advanced_options[key].set(value=value)
            self.phase_order_list = [self.phase_order_default] * (len(self.data[0]) - 1)

        current_info = {}  # this stores option values as they are before user edits inside popup
        for key, var in self.advanced_options.items():
            current_info[key] = var.get()

        popup, content_frame, buttons = self.generic_popup(
            pop_title='Advanced Options', buttons_text=['apply', 'reset', 'cancel']
        )
        popup.protocol(
            'WM_DELETE_WINDOW', func=lambda: [self.close_popup(), reset_options(self, current_info)]
        )

        for i in range(3):
            content_frame.columnconfigure(i, weight=1)

        for row, (option_label_txt, option_var) in enumerate(self.advanced_options.items()):
            option_label = tk.Label(
                content_frame,
                text=option_label_txt.replace('_', ' ').title(),
                justify='right',
                background=self.pallete[self.theme]['background_primary'],
                foreground=self.pallete[self.theme]['foreground_primary'],
            )
            option_label.grid(row=row, column=0, sticky='e')
            if type(tk.BooleanVar()) == type(option_var):
                option_checkbox = tk.Checkbutton(
                    content_frame,
                    variable=option_var,
                    background=self.pallete[self.theme]['background_primary'],
                    foreground=self.pallete[self.theme]['foreground_primary'],
                    activebackground=self.pallete[self.theme]['background_primary'],
                    activeforeground=self.pallete[self.theme]['foreground_primary'],
                    selectcolor=self.pallete[self.theme]['background_primary'],
                    highlightbackground=self.pallete[self.theme]['background_primary'],
                    highlightcolor=self.pallete[self.theme]['background_primary'],
                )
                option_checkbox.grid(row=row, column=1, sticky='w')
            elif type(tk.IntVar()) == type(option_var):
                option_entry = tk.Entry(
                    content_frame,
                    textvariable=option_var,
                    width=3,
                    background=self.pallete[self.theme]['background_primary'],
                    foreground=self.pallete[self.theme]['foreground_primary'],
                )
                option_entry.grid(row=row, column=1, sticky='w')

        def set_var(_):
            phase_idx = order_combo.current()
            order_var.set(value=self.phase_order_list[phase_idx])

        def change_var(_):
            phase_idx = order_combo.current()
            try:
                newval = int(order_var.get())
            except ValueError:
                return
            if newval < self.phase_order_default:
                messagebox.showwarning(
                    title='Error',
                    message='Phase transcription order must be '
                    + 'at least {self.phase_order_default}!',
                )
                newval = self.phase_order_default
            self.phase_order_list[phase_idx] = newval

        if len(self.data[0]) > 1:
            order_label = tk.Label(
                content_frame,
                text='Phase Transcription Order: ',
                background=self.pallete[self.theme]['background_primary'],
                foreground=self.pallete[self.theme]['foreground_primary'],
            )
            order_label.grid(row=row + 1, column=0, sticky='e')
            order_combo = ttk.Combobox(
                content_frame, state='readonly', values=self.get_phase_names(), width=9
            )
            order_combo.bind('<<ComboboxSelected>>', set_var)
            order_combo.current(0)
            order_combo.grid(row=row + 1, column=1, sticky='w')
            order_var = tk.StringVar(value=self.phase_order_default)
            order_entry = tk.Entry(
                content_frame,
                width=3,
                textvariable=order_var,
                background=self.pallete[self.theme]['background_primary'],
                foreground=self.pallete[self.theme]['foreground_primary'],
            )
            order_entry.bind('<KeyRelease>', func=change_var)
            order_entry.grid(row=row + 1, column=2, sticky='w')

        # apply maintains user options as set by user in popup, reset reverts them to default values, cancel reverts to
        # values as they were at the start of the popup
        buttons['apply'].configure(command=lambda: self.close_popup())
        buttons['reset'].configure(command=lambda: [self.close_popup(), reset_options(self)])
        buttons['cancel'].configure(
            command=lambda: [self.close_popup(), reset_options(self, current_info)]
        )
        self.place_popup()

    # ----------------------
    # Menu related functions
    def create_menu(self):
        """Creates menu. Structure is specified as a dictionary, can add commands,
        separators, and checkbuttons.
        """
        structure = {
            'File': [
                ['command', 'Open Phase Info..', self.open_phase_info],
                ['command', 'Save Phase Info', self.save],
                ['command', 'Save Phase Info as..', self.save_as],
                ['separator'],
                ['command', 'Exit', self.close_window],
            ],
            'Edit': [
                ['command', 'Axes Limits', self.change_axes_popup],
                ['command', 'Units', self.change_units],
                ['command', 'Rounding', self.change_rounding],
                ['checkbutton', 'Store Settings?', self.remind_store_settings, self.store_settings],
            ],
            'View': [
                ['checkbutton', 'Optimize Phase', self.toggle_optimize_view, self.show_optimize],
                ['checkbutton', 'Phase Slopes', self.toggle_phase_slope, self.show_phase_slope],
                ['command', 'Advanced Options', self.advanced_options_popup],
            ],
            'Help': [['command', 'Instructions', self.show_instructions]],
        }

        menu_bar = tk.Menu(self)
        for tab_label, tab_list in structure.items():
            tab = tk.Menu(
                menu_bar,
                tearoff=False,
                background=self.pallete[self.theme]['background_primary'],
                foreground=self.pallete[self.theme]['foreground_primary'],
                activebackground=self.pallete[self.theme]['hover'],
            )
            menu_bar.add_cascade(label=tab_label, menu=tab)
            for item in tab_list:
                if item[0] == 'separator':
                    tab.add_separator()
                elif item[0] == 'command':
                    tab.add_command(label=item[1], command=item[2])
                elif item[0] == 'checkbutton':
                    tab.add_checkbutton(
                        label=item[1],
                        command=item[2],
                        variable=item[3],
                        selectcolor=self.pallete[self.theme]['foreground_primary'],
                    )
        self.config(menu=menu_bar)

    def temporary_notice(self):
        messagebox.showinfo(
            title='Under Development', message='This section is currently under development!'
        )

    def close_window(self):
        """Closes main window and saves persistent settings into a binary pickle file."""
        if self.store_settings.get():  # if user wants to store settings
            with open(self.persist_filename, 'w') as fp:
                json.dump({'window_geometry': self.winfo_geometry(), 'theme': self.theme}, fp)
        # if user doesn't want to store settings and file exists
        elif os.path.exists(self.persist_filename):
            os.remove(self.persist_filename)  # remove file
        self.destroy()

    def toggle_optimize_view(self):
        """Runs update table with overwrite on to toggle display of optimize checkboxes."""
        self.update_table(overwrite=True)

    def toggle_phase_slope(self, redraw=True):
        if len(self.data[0]) > 1 and self.show_phase_slope.get():
            y_lims = [float(item.get()) for item in self.data_info['limits'][1:]]
            for i in range(len(self.data[0]) - 1):
                for j in range(self.num_dep_vars):
                    xs = self.data[0][i : i + 2]
                    ys = self.data[j + 1][i : i + 2]
                    # offset from line by 8% of y limit
                    text_position = (np.mean(xs), np.mean(ys) + y_lims[j] * 0.1)

                    # find slope and attach units if either unit is not unitless
                    try:
                        slope = self.display_rounding(
                            (ys[1] - ys[0]) / (xs[1] - xs[0]), j + 1, extra=1
                        )
                    except ZeroDivisionError:
                        slope = 'undefined'
                    xunit, yunit = (
                        self.data_info['units'][0].get(),
                        self.data_info['units'][j + 1].get(),
                    )
                    if yunit != 'unitless' and xunit != 'unitless':
                        slope = f'{slope} {yunit}/{xunit}'

                    # matplotlib text angle is in display units, so use transform to find angle
                    scaled_pt1, scaled_pt2 = [
                        self.plots[j].transData.transform_point(pt) for pt in zip(xs, ys)
                    ]
                    line_angle = np.rad2deg(
                        np.arctan2(scaled_pt2[1] - scaled_pt1[1], scaled_pt2[0] - scaled_pt1[0])
                    )

                    if i < len(self.plot_texts[j]):
                        self.plot_texts[j][i].set(
                            text=slope, position=text_position, rotation=line_angle
                        )
                    else:
                        self.plot_texts[j].append(
                            self.plots[j].annotate(
                                slope,
                                xy=text_position,
                                rotation=line_angle,
                                verticalalignment='center',
                                horizontalalignment='center',
                                rotation_mode='anchor',
                                color=self.pallete[self.theme]['foreground_primary'],
                            )
                        )
                    self.plot_texts[j][i].set_bbox(
                        dict(
                            facecolor=self.pallete[self.theme]['background_primary'],
                            alpha=0.5,
                            linewidth=0,
                        )
                    )

        if not self.show_phase_slope.get() and len(self.plot_texts) > 0:
            for text_list in self.plot_texts:
                for text in text_list:
                    text.remove()
            self.plot_texts = [[] for _ in range(self.num_dep_vars)]

        if redraw:
            self.figure_canvas.draw()

    def show_instructions(self):
        """Shows a messagebox with instructions to use this utility."""
        message = (
            'This tool can be used to design a mission which can be used by Aviary for modelling and optimization.\n\n'
            + 'To begin, start by adding points to the Altitude Plot, Mach Plot, or the table on the right.\n\n'
            + 'Points can be edited by dragging points on the plot or editing the table values. Points can be deleted '
            + "with the 'X' button adjacent to each point on the table.\n\n"
            + "Use 'Edit'->'Axes Limits' to change the axes limits.\n\n"
            + "Use 'View'->'Optimize Phase' to add the option to optimize any mission phase.\n\n"
            + "Use 'View'->'Phase Slopes' to toggle climb/descent rate information on the plots.\n\n"
            + "Use 'View'->'Advanced Options' to edit additional options related to the mission and optimization.\n\n"
            + "If you would like to save window size, location, and theme information for subsequent runs, toggle 'Edit'->'Store Settings?'"
        )
        messagebox.showinfo(title='Mission Design Instructions', message=message)

    def remind_store_settings(self):
        status = 'be' if self.store_settings.get() else 'not be'
        messagebox.showinfo(
            title='Store Settings',
            message=f'Settings related to window location, size, and theme will {status} stored!',
        )

    def change_units(self):
        popup, content_frame, buttons = self.generic_popup(
            pop_title='Change Units', buttons_text=['apply', 'cancel']
        )
        popup.protocol('WM_DELETE_WINDOW', func=lambda: [self.close_popup()])
        for i in range(2):
            content_frame.columnconfigure(i, weight=1)

        def set_var(row):
            self.data_info['units'][row].set(unit_combos[row].get())

        def reset_units_strvar():
            for i, unit in enumerate(old_units):
                self.data_info['units'][i].set(unit)

        # this creates a copy instead of a reference
        old_units = [item.get() for item in self.data_info['units']]
        unit_combos = [None] * (self.num_dep_vars + 1)
        avail_units = [['s', 'min', 'h'], ['m', 'km', 'ft', 'mi', 'nmi']]
        for row, (var_label, var_unit) in enumerate(
            zip(self.data_info['labels'], self.data_info['units'])
        ):
            if var_unit.get() != 'unitless':
                for unit_type in avail_units:
                    if var_unit.get() in unit_type:
                        unit_list = unit_type

                tk.Label(
                    content_frame,
                    text=var_label,
                    justify='right',
                    background=self.pallete[self.theme]['background_primary'],
                    foreground=self.pallete[self.theme]['foreground_primary'],
                ).grid(row=row, column=0, sticky='e')
                unit_combos[row] = ttk.Combobox(
                    content_frame, values=unit_list, state='readonly', width=10
                )
                unit_combos[row].current(unit_list.index(var_unit.get()))
                unit_combos[row].bind('<<ComboboxSelected>>', lambda e, row=row: set_var(row))
                unit_combos[row].grid(row=row, column=1, sticky='w')

        def apply_units():
            new_headers = [
                f'{label} ({unit.get()})'
                for label, unit in zip(self.data_info['labels'], self.data_info['units'])
            ]
            self.update_header(new_headers)

            for col, (old_unit, new_unit, limit, rounding) in enumerate(
                zip(
                    old_units,
                    self.data_info['units'],
                    self.data_info['limits'],
                    self.data_info['rounding'],
                )
            ):
                new_lim = convert_units(
                    val=float(limit.get()), old_units=old_unit, new_units=new_unit.get()
                )
                limit.set(value=new_lim)
                for row, val in enumerate(self.data[col]):
                    new_val = convert_units(val=val, old_units=old_unit, new_units=new_unit.get())
                    self.update_list(index=row, axis=col, value=new_val)
                num_digs = np.floor(np.log10(new_lim)) + 1
                rounding.set(value=0 if num_digs >= 3 else 2)

            self.update_axes(limits=True, units=True)
            self.redraw_plot()
            bool_list = [[item.get() for item in axis] for axis in self.table_boolvars]
            self.update_table(overwrite=True, bool_list=bool_list)
            for i in range(2):
                self.show_phase_slope.set(not self.show_phase_slope.get())
                self.toggle_phase_slope(redraw=i == 1)

        buttons['apply'].configure(command=lambda: [self.close_popup(), apply_units()])
        buttons['cancel'].configure(command=lambda: [self.close_popup(), reset_units_strvar()])
        self.place_popup()

    def change_rounding(self):
        popup, content_frame, buttons = self.generic_popup(
            pop_title='Rounding Options', buttons_text=['apply', 'cancel']
        )
        popup.protocol('WM_DELETE_WINDOW', func=lambda: [self.close_popup()])
        for i in range(2):
            # allow columns to expand in frame
            content_frame.columnconfigure(i, weight=1)

        def apply_rounding():
            self.update_table(overwrite=True)

        def cancel_rounding():
            for changed, old in zip(self.data_info['rounding'], current_rounding):
                changed.set(old.get())

        current_rounding = [item for item in self.data_info['rounding']]

        for row, (label, unit, round_str) in enumerate(
            zip(self.data_info['labels'], self.data_info['units'], self.data_info['rounding'])
        ):
            round_label = tk.Label(
                content_frame,
                text=f'{label} ({unit.get()})',
                justify='right',
                background=self.pallete[self.theme]['background_primary'],
                foreground=self.pallete[self.theme]['foreground_primary'],
            )
            round_label.grid(row=row, column=0, sticky='e')
            round_entry = tk.Entry(
                content_frame,
                textvariable=round_str,
                width=max(4, len(round_str.get())),
                background=self.pallete[self.theme]['background_primary'],
                foreground=self.pallete[self.theme]['foreground_primary'],
            )
            round_entry.grid(row=row, column=1, sticky='w')

        buttons['apply'].configure(command=lambda: [self.close_popup(), apply_rounding()])
        buttons['cancel'].configure(command=lambda: [self.close_popup(), cancel_rounding()])
        self.place_popup()

    def open_phase_info(self):
        """Opens a dialog box to select a .py file with a phase info dict. File must contain a dict called phase_info.
        File can be placed in any directory.
        """
        file_dialog = filedialog.Open(self, filetypes=[('Python files', '*.py')])
        filename = file_dialog.show()
        if filename != '':
            # imports file similar to how a module is imported, allowing direct access to variables
            spec = importlib.util.spec_from_file_location('module_name', filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            phase_info = None
            try:
                phase_info = module.phase_info
            except AttributeError:
                raise Exception(
                    'Python File does not contain a global dictionary called phase_info!'
                )
            if phase_info:
                first_phase = True
                idx = 0
                ylabs = ['altitude', 'mach']
                self.phase_order_list = []
                units = [None] * 3
                for name, phase_dict in phase_info.items():
                    if name in ['pre_mission', 'post_mission']:
                        # Skip pre/post
                        continue

                    usr_opts = phase_dict['user_options']

                    value = usr_opts.get('distance_solve_segments', None)
                    if value is not None:
                        self.advanced_options['distance_solve_segments'].set(value=value)

                    mach_poly = usr_opts.get('mach_polynomial_order', None)
                    alt_poly = usr_opts.get('altitude_polynomial_order', None)

                    if mach_poly is not None:
                        self.advanced_options['polynomial_order'].set(value=mach_poly)
                    elif alt_poly is not None:
                        self.advanced_options['polynomial_order'].set(value=alt_poly)

                    self.phase_order_list.append(usr_opts['order'])

                    timevals, units[0] = phase_dict['initial_guesses']['time']
                    if first_phase:
                        # For first phase, initialize internal lists with correct num of elements
                        numpts = usr_opts['num_segments'] + 1
                        self.data = [[0] * numpts for _ in range(self.num_dep_vars + 1)]
                        bool_list = [[0] * (numpts - 1) for _ in range(self.num_dep_vars)]
                        self.data[0][0] = timevals[0]
                        for i in range(self.num_dep_vars):
                            self.data[i + 1][0], units[i + 1] = usr_opts[f'{ylabs[i]}_initial']
                        first_phase = False

                    self.data[0][idx + 1] = timevals[1] + timevals[0]
                    for i in range(self.num_dep_vars):
                        self.data[i + 1][idx + 1] = usr_opts[f'{ylabs[i]}_final'][0]
                        bool_list[i][idx] = usr_opts[f'{ylabs[i]}_optimize']

                    idx += 1

                self.advanced_options['constrain_range'].set(
                    value=phase_info['post_mission']['constrain_range']
                )
                self.advanced_options['include_landing'].set(
                    value=phase_info['post_mission']['include_landing']
                )
                self.advanced_options['include_takeoff'].set(
                    value=phase_info['pre_mission']['include_takeoff']
                )

                # checks if any optimize values are true, in which case checkboxes are shown
                for axis_list in bool_list:
                    for bool_var in axis_list:
                        if bool_var:
                            self.show_optimize.set(True)
                            break
                lim_margin = 1.2
                limits = [max(axis) * lim_margin for axis in self.data]
                for var, lim in zip(self.data_info['limits'], limits):
                    var.set(value=lim)
                for str_var, unit in zip(self.data_info['units'], units):
                    str_var.set(unit)
                self.update_axes(limits=True, units=True)
                self.redraw_plot()
                new_headers = [
                    f'{label} ({unit.get()})'
                    for label, unit in zip(self.data_info['labels'], self.data_info['units'])
                ]
                self.update_header(new_headers)
                self.update_table(overwrite=True, bool_list=bool_list)

    def save_as(self):
        """Creates a file dialog that saves as a phase info. User can specify filename and location."""
        filename = filedialog.asksaveasfilename(
            defaultextension='.py',
            confirmoverwrite=True,
            filetypes=[('Python files', '*.py')],
            initialfile='outputted_phase_info',
        )
        if not filename:
            return
        self.save(filename=filename)

    def save(self, filename=None):
        """Saves mission into a file as a phase info dictionary which can be used by Aviary.
        This function is also called by the save as function with a non-default filename.
        """
        for i in range(len(self.data[0]) - 1):
            if self.data[0][i] > self.data[0][i + 1]:  # going backwards in time
                messagebox.showerror(
                    title='Time Travel Error',
                    message='All mission points must go forwards in time! Edit points and try again.',
                )
                return
        low_mach = 0.25
        if min(self.data[2]) < low_mach:  # low mach value in mission
            message = (
                f'Low mach values (below {low_mach}) can cause issues with FLOPS based models.\n'
                + 'Would you like to continue saving this mission?'
            )
            continue_saving = messagebox.askyesno(title='Low Mach Values', message=message)
            if not continue_saving:
                return
        users = {
            'distance_solve_segments': self.advanced_options['distance_solve_segments'].get(),
            'constrain_range': self.advanced_options['constrain_range'].get(),
            'include_takeoff': self.advanced_options['include_takeoff'].get(),
            'include_landing': self.advanced_options['include_landing'].get(),
        }
        polyord = self.advanced_options['polynomial_order'].get()
        if len(self.table_boolvars[0]) != len(self.data[0]) - 1:
            for i in range(self.num_dep_vars):
                self.table_boolvars[i] = [tk.BooleanVar()] * (len(self.data[0]) - 1)
        if not filename:
            filename = os.path.join(os.getcwd(), 'outputted_phase_info.py')

        for j, axis in enumerate(self.data):
            for i, value in enumerate(axis):
                self.data[j][i] = float(self.display_rounding(value, col=j))

        create_phase_info(
            times=self.data[0],
            altitudes=self.data[1],
            mach_values=self.data[2],
            units=[item.get() for item in self.data_info['units']],
            polynomial_order=polyord,
            num_segments=len(self.data[0]) - 1,
            altitude_optimize_phase_vars=self.table_boolvars[0],
            mach_optimize_phase_vars=self.table_boolvars[1],
            user_choices=users,
            orders=self.phase_order_list,
            filename=filename,
        )
        self.close_window()

    # button hover color functions
    def on_enter(self, event):
        event.widget['background'] = self.pallete[self.theme]['hover']

    def on_leave(self, event):
        event.widget['background'] = self.pallete[self.theme]['background_primary']


def create_phase_info(
    times,
    altitudes,
    mach_values,
    units,
    polynomial_order,
    num_segments,
    mach_optimize_phase_vars,
    altitude_optimize_phase_vars,
    user_choices,
    orders,
    filename='outputted_phase_info.py',
):
    """
    Creates a dictionary containing the information about different flight phases
    based on input times, altitudes, and Mach values.

    The information includes details such as duration bounds, initial guesses,
    and various options for optimization and control for each phase.

    Parameters
    ----------
    times : list of float
        The times at which phase changes occur, given in minutes.
    altitudes : list of float
        The altitudes corresponding to each phase, given in feet.
    mach_values : list of float
        The Mach numbers corresponding to each phase.

    Returns
    -------
    dict
        A dictionary with all the phase information, including bounds and initial guesses.
    """
    num_phases = len(times) - 1  # Number of phases is one less than the number of points
    phase_info = {}

    # times = np.round(np.array(times)).astype(int)
    # altitudes = np.round(np.array(altitudes) / 500) * 500
    # mach_values = np.round(np.array(mach_values), 2)
    times, altitudes, mach_values = np.array(times), np.array(altitudes), np.array(mach_values)

    # Utility function to create bounds
    def create_bounds(center):
        lower_bound = max(center / 2, 0.1)  # Ensuring lower bound is not less than 0.1
        upper_bound = center * 1.5
        return (lower_bound, upper_bound)

    # Calculate duration bounds for each phase
    duration_bounds = [create_bounds(times[i + 1] - times[i]) for i in range(num_phases)]

    # Initialize the cumulative initial bounds
    cumulative_initial_bounds = [(0.0, 0.0)]  # Initial bounds for the first phase

    # Calculate the cumulative initial bounds for subsequent phases
    for i in range(1, num_phases):
        previous_duration_bounds = duration_bounds[i - 1]
        previous_initial_bounds = cumulative_initial_bounds[-1]
        new_initial_bound_min = previous_initial_bounds[0] + previous_duration_bounds[0]
        new_initial_bound_max = previous_initial_bounds[1] + previous_duration_bounds[1]
        cumulative_initial_bounds.append((new_initial_bound_min, new_initial_bound_max))

    # Add pre_mission and post_mission phases
    phase_info['pre_mission'] = {
        'include_takeoff': user_choices['include_takeoff'],
        'optimize_mass': True,
    }

    climb_count = 1
    cruise_count = 1
    descent_count = 1
    alt_margin = convert_units(500, 'ft', units[1])
    for i in range(num_phases):
        initial_altitude = altitudes[i]
        final_altitude = altitudes[i + 1]

        # Determine phase type: climb, cruise, or descent
        if final_altitude > initial_altitude:
            phase_type = 'climb'
            phase_count = climb_count
            climb_count += 1
        elif final_altitude == initial_altitude:
            phase_type = 'cruise'
            phase_count = cruise_count
            cruise_count += 1
        else:
            phase_type = 'descent'
            phase_count = descent_count
            descent_count += 1

        phase_name = f'{phase_type}_{phase_count}'

        phase_info[phase_name] = {
            'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
            'user_options': {
                'num_segments': num_segments,
                'order': orders[i],
                'mach_optimize': mach_optimize_phase_vars[i].get(),
                'mach_polynomial_order': polynomial_order,
                'mach_initial': (mach_values[i], units[2]),
                'mach_final': (mach_values[i + 1], units[2]),
                'mach_bounds': (
                    (np.min(mach_values[i : i + 2]) - 0.02, np.max(mach_values[i : i + 2]) + 0.02),
                    units[2],
                ),
                'altitude_optimize': altitude_optimize_phase_vars[i].get(),
                'altitude_polynomial_order': polynomial_order,
                'altitude_initial': (altitudes[i], units[1]),
                'altitude_final': (altitudes[i + 1], units[1]),
                'altitude_bounds': (
                    (
                        max(np.min(altitudes[i : i + 2]) - alt_margin, 0.0),
                        np.max(altitudes[i : i + 2]) + alt_margin,
                    ),
                    units[1],
                ),
                'throttle_enforcement': 'path_constraint'
                if (i == (num_phases - 1) or i == 0)
                else 'boundary_constraint',
                'time_initial_bounds': (cumulative_initial_bounds[i], units[0]),
                'time_duration_bounds': (duration_bounds[i], units[0]),
            },
            'initial_guesses': {
                'time': ([times[i], times[i + 1] - times[i]], units[0]),
            },
        }

    phase_info['post_mission'] = {
        'include_landing': user_choices['include_landing'],
        'constrain_range': True,
        'target_range': (0.0, 'nmi'),
    }

    # Apply user choices to each phase
    for phase_name, _ in phase_info.items():
        if 'pre_mission' in phase_name or 'post_mission' in phase_name:
            continue
        phase_info[phase_name]['user_options'].update(
            {
                'distance_solve_segments': user_choices.get('distance_solve_segments', False),
            }
        )

    # Apply global settings if required
    phase_info['post_mission']['constrain_range'] = user_choices.get('constrain_range', True)

    # Calculate the total range
    total_range, range_unit = estimate_total_range_trapezoidal(times, mach_values, units)
    print(f'Total range is estimated to be {total_range} {range_unit}')

    phase_info['post_mission']['target_range'] = (total_range, range_unit)

    # write a python file with the phase information
    with open(filename, 'w') as f:
        f.write('phase_info = ')
        pp = pprint.PrettyPrinter(indent=4, stream=f, sort_dicts=False)
        pp.pprint(phase_info)

    # Check for 'ruff' and format the file
    if shutil.which('ruff'):
        subprocess.run(['ruff', filename])
    else:
        if shutil.which('autopep8'):
            subprocess.run(['autopep8', '--in-place', '--aggressive', filename])
            print("File formatted using 'autopep8'")
        else:
            print(
                "'ruff' or 'autopep8' are not installed. Please consider installing one of them "
                'for better formatting.'
            )

    print(f'Phase info has been saved and formatted in {filename}')

    return phase_info


def estimate_total_range_trapezoidal(times, mach_numbers, units):
    """Source: original Aviary graphical_input.py."""
    speed_of_sound = 343  # Speed of sound in meters per second

    # convert time list into np array with units of seconds
    times_sec = np.array([convert_units(time, units[0], 's') for time in times])

    # Calculate the speeds at each Mach number
    speeds = np.array(mach_numbers) * speed_of_sound

    # Use numpy's trapz function to integrate
    total_range = np.trapz(speeds, times_sec)  # in meters
    range_unit = units[1]
    # m and ft are small units for range, change to larger ones
    if range_unit == 'm':
        range_unit = 'km'
    if range_unit == 'ft':
        range_unit = 'nmi'

    # return range in the same units as altitude units
    return round(convert_units(total_range, 'm', range_unit), 2), range_unit


def _setup_flight_profile_parser(parser):
    """
    Set up the command line options for the Flight Profile plotting tool.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser instance.
    """
    pass


def _exec_flight_profile(options, user_args):
    """
    Run the Flight Profile plotting tool.

    Parameters
    ----------
    options : argparse.Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    app = AviaryMissionEditor()
    app.mainloop()


if __name__ == '__main__':
    app = AviaryMissionEditor()
    app.mainloop()
