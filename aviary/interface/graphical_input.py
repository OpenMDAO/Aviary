from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import shutil
from tkinter import messagebox
import tkinter as tk
from tkinter import messagebox, font as tkFont
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import subprocess
import os


class DraggablePoints:
    def __init__(self, points, ax, canvas):
        self.points = points
        self.ax = ax
        self.canvas = canvas
        self.current_point = None
        self.lines, = ax.plot([p[0] for p in points], [p[1] for p in points],
                              linestyle='-', marker='o', color='b', markersize=8)

        self.cid_click = ax.figure.canvas.mpl_connect(
            'button_press_event', self.on_click)
        self.cid_release = ax.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cid_motion = ax.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_click(self, event):
        if event.inaxes != self.ax or event.button is not MouseButton.LEFT:
            return
        for i, point in enumerate(self.points):
            if abs(point[0] - event.xdata) < 5.0 and abs(point[1] - event.ydata) < 5.0:
                self.current_point = i
                return

    def on_release(self, event):
        self.current_point = None

    def on_motion(self, event):
        if self.current_point is not None and event.xdata is not None and event.ydata is not None:
            self.points[self.current_point] = (
                self.points[self.current_point][0], event.ydata)
            self.update_points(self.ax, self.points)

    def update_points(self, ax, data):
        color = 'b'
        label = 'Mach Number'
        ax.clear()
        ax.plot([p[0] for p in data], [p[1] for p in data], linestyle='-',
                marker='o', color=color, markersize=8, label=label)

        ax.set_xlim(0, 360)
        ax.set_ylim(0, 1.0)

        ax.figure.canvas.draw()


def create_phase_info(times, altitudes, mach_values,
                      polynomial_order, num_segments, optimize_mach_phase_vars, optimize_altitude_phase_vars, user_choices):
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

    times = np.round(np.array(times)).astype(int)
    altitudes = np.round(np.array(altitudes) / 500) * 500
    mach_values = np.round(np.array(mach_values), 2)

    # Utility function to create bounds
    def create_bounds(center):
        lower_bound = max(center / 2, 0.1)  # Ensuring lower bound is not less than 0.1
        upper_bound = center * 1.5
        return (lower_bound, upper_bound)

    # Calculate duration bounds for each phase
    duration_bounds = [create_bounds(times[i+1] - times[i])
                       for i in range(num_phases)]

    # Initialize the cumulative initial bounds
    cumulative_initial_bounds = [(0., 0.)]  # Initial bounds for the first phase

    # Calculate the cumulative initial bounds for subsequent phases
    for i in range(1, num_phases):
        previous_duration_bounds = duration_bounds[i-1]
        previous_initial_bounds = cumulative_initial_bounds[-1]
        new_initial_bound_min = previous_initial_bounds[0] + previous_duration_bounds[0]
        new_initial_bound_max = previous_initial_bounds[1] + previous_duration_bounds[1]
        cumulative_initial_bounds.append((new_initial_bound_min, new_initial_bound_max))

    # Add pre_mission and post_mission phases
    phase_info['pre_mission'] = {
        'include_takeoff': False,
        'optimize_mass': True,
    }

    climb_count = 1
    cruise_count = 1
    descent_count = 1

    for i in range(num_phases):
        initial_altitude = altitudes[i]
        final_altitude = altitudes[i+1]

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
            'subsystem_options': {
                'core_aerodynamics': {'method': 'computed'}
            },
            'user_options': {
                'optimize_mach': optimize_mach_phase_vars[i].get(),
                'optimize_altitude': optimize_altitude_phase_vars[i].get(),
                'polynomial_control_order': polynomial_order,
                'use_polynomial_control': True,
                'num_segments': num_segments,
                'order': 3,
                'solve_for_range': False,
                'initial_mach': (mach_values[i], 'unitless'),
                'final_mach': (mach_values[i+1], 'unitless'),
                'mach_bounds': ((np.min(mach_values[i:i+2]) - 0.02, np.max(mach_values[i:i+2]) + 0.02), 'unitless'),
                'initial_altitude': (altitudes[i], 'ft'),
                'final_altitude': (altitudes[i+1], 'ft'),
                'altitude_bounds': ((max(np.min(altitudes[i:i+2]) - 500., 0.), np.max(altitudes[i:i+2]) + 500.), 'ft'),
                'throttle_enforcement': 'path_constraint' if (i == (num_phases - 1) or i == 0) else 'boundary_constraint',
                'fix_initial': True if i == 0 else False,
                'constrain_final': True if i == (num_phases - 1) else False,
                'fix_duration': False,
                'initial_bounds': (cumulative_initial_bounds[i], 'min'),
                'duration_bounds': (duration_bounds[i], 'min'),
            },
            'initial_guesses': {
                'times': ([times[i], times[i+1]-times[i]], 'min'),
            }
        }

    phase_info['post_mission'] = {
        'include_landing': False,
        'constrain_range': True,
        'target_range': (0., 'nmi'),
    }

    # Apply user choices to each phase
    for phase_name, phase_data in phase_info.items():
        if 'pre_mission' in phase_name or 'post_mission' in phase_name:
            continue
        phase_info[phase_name]['user_options'].update({
            'solve_for_range': user_choices.get('solve_for_range', False),
        })

    # Apply global settings if required
    phase_info['post_mission']['constrain_range'] = user_choices.get(
        'constrain_range', True)

    # Calculate the total range
    total_range = estimate_total_range_trapezoidal(times, mach_values)
    print(
        f"Total range is estimated to be {total_range} nautical miles")

    phase_info['post_mission']['target_range'] = (total_range, 'nmi')

    filename = os.path.join(os.getcwd(), 'outputted_phase_info.py')

    # write a python file with the phase information
    with open(filename, 'w') as f:
        f.write(f'phase_info = {phase_info}')

    # Check for 'black' and format the file
    if shutil.which('black'):
        subprocess.run(['black', filename])
    else:
        if shutil.which('autopep8'):
            subprocess.run(['autopep8', '--in-place', '--aggressive', filename])
            print("File formatted using 'autopep8'")
        else:
            print("'black' and 'autopep8' are not installed. Please consider installing one of them for better formatting.")

    print(f"Phase info has been saved and formatted in {filename}")

    return phase_info


def estimate_total_range_trapezoidal(times, mach_numbers):
    speed_of_sound = 343  # Speed of sound in meters per second

    # Convert times to seconds from minutes
    times_sec = np.array(times) * 60

    # Calculate the speeds at each Mach number
    speeds = np.array(mach_numbers) * speed_of_sound

    # Use numpy's trapz function to integrate
    total_range = np.trapz(speeds, times_sec)

    total_range_nautical_miles = total_range / 1000 / 1.852
    return int(round(total_range_nautical_miles))


class Tooltip:
    def __init__(self, ax, tooltip_func):
        self.ax = ax
        self.tooltip_func = tooltip_func
        self.tooltip_widget = None
        self._id = None

    def __call__(self, event):
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            label = self.tooltip_func(x, y)
            if label:
                if not self.tooltip_widget:
                    self.tooltip_widget = tk.Toplevel()
                    self.tooltip_widget.wm_overrideredirect(True)
                    self.label = tk.Label(self.tooltip_widget, text="", justify='left',
                                          background='white', relief='solid', borderwidth=2,
                                          font=("tahoma", "14", "normal"))
                    self.label.pack(ipadx=1)
                self.label.config(text=label)
                self.tooltip_widget.geometry(
                    f"+{event.guiEvent.x_root+10}+{event.guiEvent.y_root+10}")
                self.tooltip_widget.deiconify()
            else:
                if self.tooltip_widget:
                    self.tooltip_widget.withdraw()
        else:
            if self.tooltip_widget:
                self.tooltip_widget.withdraw()


class IntegratedPlottingApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('Flight Profile Utility')
        self.geometry('1200x900')

        # Set default font for the application
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(size=18)  # You can adjust the size as needed
        self.option_add("*Font", default_font)

        # You might also want to adjust the font of the Matplotlib plots:
        plt.rc('font', size=12)  # Adjust the size as needed for Matplotlib plots

        # Initialize data
        self.altitude_data = [(0, 0)]
        self.mach_data = [(0, 0.2)]

        # Frame for plots
        plot_frame = tk.Frame(self)
        plot_frame.pack(side='left', fill='both', expand=True)

        # Create Altitude Plot
        self.fig_altitude, self.ax_altitude = plt.subplots()
        self.setup_plot(self.ax_altitude)
        self.canvas_altitude = FigureCanvasTkAgg(self.fig_altitude, master=plot_frame)
        self.canvas_altitude.get_tk_widget().pack(side='top', fill='both', expand=1)

        # Create Mach Plot
        self.fig_mach, self.ax_mach = plt.subplots()
        self.setup_plot(self.ax_mach)
        self.canvas_mach = FigureCanvasTkAgg(self.fig_mach, master=plot_frame)
        self.canvas_mach.get_tk_widget().pack(side='top', fill='both', expand=1)
        self.draggable_points_mach = DraggablePoints(
            self.mach_data, self.ax_mach, self.canvas_mach)

        # Frame for checkboxes
        checkbox_frame = tk.Frame(self)
        checkbox_frame.pack(side='right', fill='y', expand=False)

        # Checkboxes
        self.optimize_mach_var = tk.BooleanVar()
        self.constrain_range_var = tk.BooleanVar()
        tk.Checkbutton(checkbox_frame, text="Constrain Range",
                       variable=self.constrain_range_var).pack(anchor="w")
        self.constrain_range_var.set(True)
        self.solve_for_range_var = tk.BooleanVar()
        tk.Checkbutton(checkbox_frame, text="Solve for Range",
                       variable=self.solve_for_range_var).pack(anchor="w")

        # Textbox for Polynomial Control Order
        self.polynomial_order_var = tk.StringVar()
        tk.Label(checkbox_frame, text="Polynomial Control Order").pack(anchor="w")
        self.polynomial_order_entry = tk.Entry(
            checkbox_frame, textvariable=self.polynomial_order_var)
        self.polynomial_order_entry.pack(anchor="w")
        self.polynomial_order_entry.insert(0, "1")

        # Textbox for Number of Segments
        self.num_segments_var = tk.StringVar(value="2")  # Default value set to "2"
        tk.Label(checkbox_frame, text="Number of Segments").pack(anchor="w")
        self.num_segments_entry = tk.Entry(
            checkbox_frame, textvariable=self.num_segments_var)
        self.num_segments_entry.pack(anchor="w")

        self.cid_click_altitude = self.fig_altitude.canvas.mpl_connect(
            'button_press_event', self.on_click)
        self.cid_motion_altitude = self.fig_altitude.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.cid_release_altitude = self.fig_altitude.canvas.mpl_connect(
            'button_release_event', self.on_release)

        # Done button
        self.done_button = tk.Button(checkbox_frame, text="Done", command=self.on_done)
        self.done_button.pack(side='top', pady=10)

        # Help button
        self.help_button = tk.Button(checkbox_frame, text="Help", command=self.show_help)
        self.help_button.pack(side='top', pady=10)

        # Frame for point entries
        self.point_entry_frame = tk.Frame(checkbox_frame)
        self.point_entry_frame.pack(side='top', expand=False)
        self.altitude_entries = []
        self.mach_entries = []

        # Column labels
        tk.Label(self.point_entry_frame, text="Altitude").grid(row=0, column=1)
        tk.Label(self.point_entry_frame, text="Mach").grid(row=0, column=3)

        # Frame for phase-specific checkboxes
        self.phase_option_frame = tk.Frame(checkbox_frame)
        self.phase_option_frame.pack(side='top', expand=False)

        self.optimize_mach_phase_vars = []
        self.optimize_altitude_phase_vars = []

        # Tooltips for both plots
        self.tooltip_altitude = Tooltip(self.ax_altitude, self.tooltip_func)
        self.tooltip_mach = Tooltip(self.ax_mach, self.tooltip_func)

        self.cid_motion_altitude = self.fig_altitude.canvas.mpl_connect(
            'motion_notify_event', self.tooltip_altitude)
        self.cid_motion_mach = self.fig_mach.canvas.mpl_connect(
            'motion_notify_event', self.tooltip_mach)

        # Bind Enter key to the on_done function
        self.bind("<Return>", self.on_done)

        self.update_point_entries()

        self.current_point = None

    def tooltip_func(self, x, y):
        # Function to generate tooltip text
        for point in self.altitude_data:
            if abs(point[0] - x) < 5.0 and abs(point[1] - y) < 1000.0:
                return f"Time: {int(point[0])} min, Altitude: {int(point[1])} ft"
        for point in self.mach_data:
            if abs(point[0] - x) < 5.0 and abs(point[1] - y) < 0.1:
                return f"Time: {int(point[0])} min, Mach: {round(point[1], 2)}"
        return None

    def update_point_entries(self):
        # Clear existing entries
        for entry in self.altitude_entries + self.mach_entries:
            entry.destroy()
        self.altitude_entries.clear()
        self.mach_entries.clear()

        # Update phase-specific checkboxes
        self.update_phase_options()

        # Create new entries for each point
        for i, (alt, mach) in enumerate(zip(self.altitude_data, self.mach_data)):
            # Altitude Entry
            tk.Label(self.point_entry_frame,
                     text=f"Pt. {i+1}").grid(row=i+1, column=0, sticky="w")
            alt_var = tk.StringVar(value=str(alt[1]))
            alt_entry = tk.Entry(self.point_entry_frame,
                                 textvariable=alt_var, width=10)  # Width set to 10
            alt_entry.grid(row=i+1, column=1)
            alt_entry.bind("<KeyRelease>", lambda e, i=i,
                           var=alt_var: self.update_point_data(i, 'altitude', var.get()))

            # Mach Entry
            tk.Label(self.point_entry_frame,
                     text=f"Pt. {i+1}").grid(row=i+1, column=2, sticky="w")
            mach_var = tk.StringVar(value=str(mach[1]))
            mach_entry = tk.Entry(self.point_entry_frame,
                                  textvariable=mach_var, width=10)  # Width set to 10
            mach_entry.grid(row=i+1, column=3)
            mach_entry.bind("<KeyRelease>", lambda e, i=i,
                            var=mach_var: self.update_point_data(i, 'mach', var.get()))

            self.altitude_entries.append(alt_entry)
            self.mach_entries.append(mach_entry)

    def update_phase_options(self):
        tk.Label(self.phase_option_frame, text=f"Optimize:").grid(
            row=0, column=0, columnspan=3, sticky="s")
        # Create new checkboxes for each phase
        for i in range(len(self.altitude_data) - 1):
            if i >= len(self.optimize_mach_phase_vars):
                mach_var = tk.BooleanVar(value=False)
                alt_var = tk.BooleanVar(value=False)
                self.optimize_mach_phase_vars.append(mach_var)
                self.optimize_altitude_phase_vars.append(alt_var)

            tk.Label(self.phase_option_frame,
                     text=f"Phase {i + 1}").grid(row=i+1, column=0, sticky="w")
            tk.Checkbutton(self.phase_option_frame, text="Altitude",
                           variable=self.optimize_altitude_phase_vars[i]).grid(row=i+1, column=1)
            tk.Checkbutton(self.phase_option_frame, text="Mach",
                           variable=self.optimize_mach_phase_vars[i]).grid(row=i+1, column=2)

    def update_point_data(self, index, point_type, value):
        try:
            value = float(value)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid number.")
            return

        if point_type == 'altitude':
            self.altitude_data[index] = (self.altitude_data[index][0], value)
            self.update_plot(self.ax_altitude, self.altitude_data, 'Altitude', 'g')
        elif point_type == 'mach':
            self.mach_data[index] = (self.mach_data[index][0], value)
            self.update_plot(self.ax_mach, self.mach_data, 'Mach Number', 'b')

    def on_done(self, event=None):  # event=None to handle both button click and keypress
        times, altitudes = zip(*self.altitude_data)
        _, mach_values = zip(*self.mach_data)

        # Retrieve user choices from checkboxes
        user_choices = {
            "constrain_range": self.constrain_range_var.get(),
            "solve_for_range": self.solve_for_range_var.get()
        }

        polynomial_order = int(self.polynomial_order_var.get())
        num_segments = int(self.num_segments_var.get())

        # Call create_phase_info and close the window
        # Modify the phase_info creation to include phase-specific options
        create_phase_info(times, altitudes, mach_values, polynomial_order, num_segments,
                          self.optimize_mach_phase_vars, self.optimize_altitude_phase_vars, user_choices)

        self.destroy()

    def show_help(self):
        help_message = (
            "How to Use:\n"
            "- Click on the Altitude plot to add points.\n"
            "- Drag points in the Mach plot to adjust values vertically.\n"
            "- Use the checkboxes on the right to set options for the flight profile:\n"
            "   - Optimize Mach: If checked, the flight profile will optimize for Mach number.\n"
            "   - Optimize Altitude: If checked, the flight profile will optimize for altitude efficiency.\n"
            "   - Constrain Range: If checked, the flight profile will include constraints on the flight range.\n"
            "   - Solve for Range: If checked, the application will calculate the total flight range.\n"
            "- Press 'Done' or the Enter key to process data and close the application.\n\n"
            "Note: Dragging points in the plots will only change their vertical position"
            "while keeping the time (horizontal position) constant."
        )
        messagebox.showinfo("Help", help_message)

    def setup_plot(self, ax):
        if ax == self.ax_altitude:
            title = 'Altitude Profile'
            xlabel = 'Time (minutes)'
            ylabel = 'Altitude (feet)'
            xlim_min = 0
            xlim_max = 360
            ylim_min = 0
            ylim_max = 50000
        elif ax == self.ax_mach:
            title = 'Mach Profile'
            xlabel = 'Time (minutes)'
            ylabel = 'Mach'
            ylim_min = 0
            ylim_max = 1.0

        xlim_min = 0
        xlim_max = 360
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim_min, xlim_max)
        ax.set_ylim(ylim_min, ylim_max)

    def on_click(self, event):
        self.update_point_entries()
        if event.inaxes != self.ax_altitude:
            return  # Only interact with the altitude plot

        self.current_point = None
        # Check if the click is near an existing point
        for i, point in enumerate(self.altitude_data):
            if abs(point[0] - event.xdata) < 5.0 and abs(point[1] - event.ydata) < 1000.0:
                self.current_point = i
                return

        # Add a new point only if the click is to the right of the last point
        if event.xdata > max(self.altitude_data, key=lambda x: x[0])[0]:
            x = round(event.xdata)
            y = round(event.ydata / 500) * \
                500 if event.inaxes == self.ax_altitude else round(
                    event.ydata / 0.02) * 0.02

            self.altitude_data.append((x, y))

            # Optionally, add a corresponding point on the Mach plot at the same x
            default_mach_y = 0.72
            self.mach_data.append((x, default_mach_y))

            # Update both plots
            self.update_plot(self.ax_altitude, self.altitude_data, 'Altitude', 'g')
            self.update_plot(self.ax_mach, self.mach_data, 'Mach Number', 'b')

    def on_motion(self, event):
        if self.current_point is not None and event.xdata is not None and event.ydata is not None:
            self.altitude_data[self.current_point] = (
                self.altitude_data[self.current_point][0], event.ydata)
            self.update_plot(self.ax_altitude, self.altitude_data, 'Altitude', 'g')

    def on_release(self, event):
        self.update_point_entries()
        self.current_point = None

    def update_plot(self, ax, data, label, color):
        ax.clear()
        ax.plot([p[0] for p in data], [p[1] for p in data], linestyle='-',
                marker='o', color=color, markersize=8, label=label)

        self.setup_plot(ax)
        ax.figure.canvas.draw()


if __name__ == "__main__":
    app = IntegratedPlottingApp()
    app.mainloop()
