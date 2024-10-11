import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import tkinter as tk
from tkinter import Label, Button, StringVar, filedialog, messagebox
from tkinter.ttk import Combobox
import aviary.api as av
from aviary.utils.functions import get_path


def plot_drag_polar(input_file=None):
    """
    Plot drag polar
    """
    if input_file is None:
        input_file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not input_file:
            messagebox.showerror(
                "Error", "No file selected")
            exit()
        return

    try:
        input_path = get_path(input_file)
        polar_data = av.read_data_file(input_path, aliases={
            'altitude': 'altitude',
            'mach_number': 'mach_number',
            'alpha': 'angle_of_attack',
            'CL': 'lift_coefficient',
            'CD': 'total_drag_coefficient'
        })
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read the file: {str(e)}")
        return

    mach = polar_data.get_val('mach_number')

    mach_values = np.unique(mach)

    altitude = polar_data.get_val('altitude')

    altitude_values = np.unique(altitude)

    alpha_values = polar_data.get_val('alpha')

    CD_values = polar_data.get_val('CD')

    CL_values = polar_data.get_val('CL')

    window = tk.Tk()

    window.title('Drag Polar Plot')
    fig, ax = plt.subplots(nrows=1)

    fig.tight_layout(pad=4)
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().pack(expand=True, fill='both')

    toolbar = NavigationToolbar2Tk(canvas, window)

    def plot_polar(ax, CD, CL, color, label=None, marker='o'):
        ax.plot(CD, CL, color=color, label=label, marker='o')

    def update_plot():
        x_var = set_x_var.get()
        y_var = set_y_var.get()
        fix_variable = fix_variable_var.get()
        fix_value = float(fix_value_var.get())
        ax.clear()

        if fix_variable == 'Mach':

            indices = mach == fix_value

            fixed_values = altitude_values
            fixed_label = 'Altitude'
        else:

            index = altitude == fix_value
            fixed_values = mach_values
            fixed_label = 'Mach'

        colors = cm.viridis(np.linspace(0, 1, len(fixed_values)))
        for i, val in enumerate(fixed_values):
            if fix_variable == 'Mach':
                indices = (mach == fix_value) & (altitude == val)

                CD = np.array(CD_values[indices])

                CL = CL_values[indices]

                alpha = alpha_values[indices]

            else:
                index = (altitude == fix_value) & (mach == val)

                CD = np.array(CD_values[index])

                CL = CL_values[index]

                alpha = alpha_values[index]

            if x_var == 'CD':
                x_val = CD

            elif x_var == 'CL':
                x_val = CL

            elif x_var == 'Alpha':
                x_val = alpha

            elif x_var == 'CL/CD':
                x_val = np.array(CL)/np.array(CD)

            if y_var == 'CD':
                y_val = CD

            elif y_var == 'CL':
                y_val = CL

            elif y_var == 'Alpha':
                y_val = alpha

            elif y_var == 'CL/CD':
                y_val = np.array(CL)/np.array(CD)

            plot_polar(ax, x_val, y_val, color=colors[i], label=f'{fixed_label} {val}')

            ax.set_xlabel(f'{x_var}')
            ax.set_ylabel(f'{y_var}')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=fixed_label)
            ax.set_title(f'{y_var} vs {x_var} for fixed {fix_variable} = {fix_value}')

            canvas.draw()
            toolbar.update()

    def update_fix_value_combobox(*args):
        if fix_variable_var.get() == 'Mach':
            fix_value_combobox['values'] = [str(value) for value in mach_values]
            fix_value_var.set(str(mach_values[0]))
            fix_value_combobox.current(0)
        else:
            fix_value_combobox['values'] = [str(value) for value in altitude_values]
            fix_value_var.set(str(altitude_values[0]))

    set_x_var = StringVar(value='CD')

    set_x_label = Label(master=window, text="x-axis")
    set_x_label.pack(side='left', padx=5, pady=5)
    set_x_combobox = Combobox(master=window, textvariable=set_x_var, values=[
        'CD', 'CL', 'Alpha', 'CL/CD'])

    set_x_combobox.pack(side='left', padx=5, pady=5)
    set_y_var = StringVar(value='CL')
    set_y_label = Label(master=window, text="y-axis")
    set_y_label.pack(side='left', padx=5, pady=5)
    set_y_combobox = Combobox(master=window, textvariable=set_y_var, values=[
        'CL', 'CD', 'Alpha', 'CL/CD'])
    set_y_combobox.pack(side='left', padx=5, pady=5)
    fix_variable_var = StringVar(value='Mach')
    fix_value_var = StringVar(value=float(mach_values[0]))
    fix_variable_label = Label(master=window, text="Fix Variable:")
    fix_variable_label.pack(side='left', padx=5, pady=5)

    fix_variable_combobox = Combobox(
        master=window, textvariable=fix_variable_var, values=['Mach', 'Altitude'])
    fix_variable_combobox.pack(side='left', padx=5, pady=5)
    fix_variable_combobox.bind("<<ComboboxSelected>>", update_fix_value_combobox)
    fix_variable_combobox.current(0)
    fix_value_label = Label(master=window, text="Fix Value:")
    fix_value_label.pack(side='left', padx=5, pady=5)

    fix_value_combobox = Combobox(master=window, textvariable=fix_value_var)
    fix_value_combobox.pack(side='left', padx=5, pady=5)
    update_fix_value_combobox()

    plot_button = Button(master=window, text="Plot", command=update_plot)
    plot_button.pack(side='right', padx=5, pady=5)
    window.mainloop()


def _setup_plot_drag_polar_parser(parser):
    """
    Set up the command line options for the Model Building tool.
    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser instance.
    parser : argparse subparser
        The parser we're adding options to.
    """

    pass


def _exec_plot_drag_polar(options, user_args):
    """
    Run the Model Building tool.
    Parameters
    ----------
    options : argparse.Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    plot_drag_polar()


if __name__ == "__main__":
    plot_drag_polar()
