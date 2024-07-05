import os, shutil, subprocess, pickle
import importlib.util
from sysconfig import get_path_names
import numpy as np

import tkinter as tk # base tkinter
import tkinter.ttk as ttk # themed tkinter
from tkinter import filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseButton

# TODO: Add ability to change units, phase info specifies units and aviary can handle these unit changes.
#       Makes sense to allow unit changes in GUI based on user preference.
#       Possible unit changes: Alt -> ft, m, mi, km, nmi; Time -> min, s, hr; Mach -> none
# TODO: tooltip/another format for displaying climb/descent rates -> useful for verifying profile

# dark2 = '#252526'
# dark3 = '#2d2d30'
# dark4 = '#3e3e42'
# vsblue = '#007acc'
# fgg = '#FFFFFF'
       
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
        vscrollbar.pack(fill='y', side='left', expand=False)
        self.freezeframe = tk.Frame(self) # this frame will not scroll, allowing for freeze view functionality
        self.freezeframe.pack(side='top',fill='x')

        canvas = tk.Canvas(self, bd=0, highlightthickness=0,
                           yscrollcommand=vscrollbar.set)
        canvas.pack(side='right', fill='y', expand=True)
        self.vscroll_canvas = canvas
        vscrollbar.config(command=canvas.yview)

        # Reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # Create a frame inside the canvas which will be scrolled with it.
        self.interior = interior = tk.Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor='nw')

        # Track changes to the canvas and frame width and sync them,
        # also updating the scrollbar.
        def _configure_interior(event):
            # Update the scrollbars to match the size of the inner frame.
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
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
    def __init__(self):
        super().__init__()
        #self.app_style = tk.Style()
        self.theme_toggle = True
        self.theme = "light"
        self.pallete = {"dark":{'background_primary':'#1e1e1e',
                                'foreground_primary':'#FEFEFE',
                                'foreground_secondary':'#CCCCCC',
                                'crosshair':'#EE0000',
                                'lines':['#00b6f2','#ffff00']},
                        "light":{'background_primary':'#ffffff',
                                 'foreground_primary':'#000000',
                                 'foreground_secondary':'#999999',
                                 'crosshair':'#EE0000',
                                 'lines':['#0209c6','#ff00ff']}}
        
        self.title('Mission Design Utility')
        self.protocol("WM_DELETE_WINDOW",self.close_window)
        self.focus_set() # focus the window
        self.persist_filename = "windowlocation.pickle"

        # ---------------------------------------------------------------
        # window geometry definition, allows reuse of user's modified size/location
        # tkinter size string format:  widthxheight+x+y ;  x,y are location
        default_win_size = (900,500) # CHANGE THIS TO AFFECT DEFAULT SIZE
        window_geometry = f"{default_win_size[0]}x{default_win_size[1]}+10+10" 
        if os.path.exists(self.persist_filename):
            with open(self.persist_filename,"rb") as fp:
                persist_settings = pickle.load(fp)
                window_geometry = persist_settings['window_geometry']
                self.theme_toggle = persist_settings['theme_toggle']
        self.geometry(window_geometry)
        self.minsize(*default_win_size) # force a minimum size for layout to look correct

        # ---------------------------------------------------------------
        # create window layout with frames for containing graph, table, and scrollbar
        self.frame_table = VerticalScrolledFrame(self)
        self.frame_table.pack(side='right',fill='y')
        self.frame_tableheaders = self.frame_table.freezeframe

        self.frame_plotReadouts = tk.Frame(self)
        self.frame_plotReadouts.pack(side='bottom',fill='x')
        self.frame_plots = tk.Frame(self)
        self.frame_plots.pack(side='top',expand=True,fill='both')

        # ---------------------------------------------------------------
        # Main definition of data which can be plotted/tabulated. Assumes a singular
        # independent variable that is applied to each dependent variable. All 
        # y-attributes are in list format to support multiple dependent variables.
        self.data_info = {"xlabel":"Time","xlim":(0,400),"xunit":"min","xround":0,
                      "ylabels":["Altitude","Mach"],"ylims":[(0,50e3),(0,1.0)],
                      "yunits":["ft","unitless"],"yrounds":[0,2],
                      "plot_titles":["Altitude Profile","Mach Profile"]}
        
        # ---------------------------------------------------------------
        # sanity checking of data_info dict and creates convenient labels with units included
        # thess labels with units are used for plots, table headers, and axes limit entries
        self.check_data_info()
        self.data_info["xlabel_unit"] = self.data_info["xlabel"] + f" ({self.data_info['xunit']})"
        self.data_info["ylabels_units"] = [ self.data_info["ylabels"][i] + 
                            f" ({self.data_info['yunits'][i]})" for i in range(self.num_dep_vars)]
        self.x_list = [0]
        self.ys_list = [[0] for i in range(self.num_dep_vars)]
        self.phase_order_default = 3
        self.phase_order_list = []
        self.plot_lines = []

        # internal variables to remember mouse state
        self.mouse_drag,self.mouse_press = False, False
        self.popup = None
        self.ptcontainer = 0.04 # percent of plot size, boundary around point where it can be dragged
        self.show_optimize = tk.BooleanVar() # controls display of optimize phase checkboxes
        
        self.advanced_options_info = {"constrain_range":tk.BooleanVar(value=True),"solve_for_distance":tk.BooleanVar(),
                                      "include_takeoff":tk.BooleanVar(),"include_landing":tk.BooleanVar(),
                                      "polynomial_control_order":tk.IntVar(value=1)}

        self.save_option_defaults()
        self.create_plots()
        self.create_table()
        self.create_menu()
        self.update_theme()

    def save_option_defaults(self):
        self.advanced_options_info_defaults = {}
        for key,var in self.advanced_options_info.items():
            self.advanced_options_info_defaults[key] = var.get()
        self.axes_lim_defaults = [item[1] for item in self.data_info["ylims"]]
        self.axes_lim_defaults = [self.data_info["xlim"][1],*self.axes_lim_defaults]
        
    def check_data_info(self):
        """Verifies data_info dict has consistent number of dependent variables """
        check_keys = ["ylabels","ylims","yunits"]
        self.num_dep_vars = len(self.data_info["plot_titles"])
        for key in check_keys:
            if self.num_dep_vars != len(self.data_info[key]):
                raise Exception("Check length of lists in data_info, mismatch detected.\n"+
                                f"Expected {self.num_dep_vars} dependent variables.")

    def update_list(self,row:int,col:int,value=None):
        """Updates internal data lists based on row,col values. col corresponds 
            to dependent/independent variable. row corresponds to point number."""
        try:
            value = float(value)
        except (ValueError,TypeError):
            return
        datalists = [self.x_list,*self.ys_list]
        if row == len(self.x_list):
            datalists[col].append(value)
            if len(self.phase_order_list) < len(self.x_list) -1: 
                self.phase_order_list.append(self.phase_order_default) # default lowest dymos phase transcription order value
        else:
            datalists[col][row] = value

    def update_theme(self):
        
        self.theme = 'light' if self.theme_toggle else 'dark'
        self.theme_toggle = not self.theme_toggle
        background_primary = self.pallete[self.theme]["background_primary"]
        foreground_primary = self.pallete[self.theme]["foreground_primary"]
        self.bgpri = background_primary
        self.fgpri = foreground_primary

        self.create_menu()
        #self.theme_button.configure(bg=background_primary,image=tk.PhotoImage(file=files[self.theme=="light"]))
        #self.theme_button.image = self.imgfiles[self.theme=="light"]
        self.configure(background=background_primary)
        self.frame_plotReadouts.configure(background=background_primary)
        self.mouse_coords.configure(background=background_primary,foreground=foreground_primary)
        self.fig.set_facecolor(background_primary)

        self.frame_table.vscroll_canvas.configure(background=background_primary)
        self.frame_table.interior.configure(background=background_primary)
        self.frame_tableheaders.configure(bg=background_primary)
        for widget in self.table_widgets:
            widget.configure(background=background_primary,foreground=foreground_primary)
            if isinstance(widget,tk.Checkbutton):
                widget.configure(activebackground=background_primary,activeforeground=foreground_primary,
                                 highlightbackground=background_primary,
                                 highlightcolor=background_primary,selectcolor=background_primary)
        for widget in self.header_widgets:
            widget.configure(bg=background_primary,fg=foreground_primary)
            if isinstance(widget,tk.Entry):
                widget.configure(readonlybackground=background_primary)

        for plot in self.plots:
            plot.set_facecolor(background_primary)
            plot.yaxis.label.set_color(foreground_primary)
            plot.xaxis.label.set_color(foreground_primary)
            plot.title.set_color(foreground_primary)
            plot.grid(True,color=self.pallete[self.theme]['foreground_secondary'])
            for axis in ['x','y']: plot.tick_params(axis=axis,colors=foreground_primary)
            for spine in ['left','top','right','bottom']: plot.spines[spine].set_color(self.pallete[self.theme]['foreground_secondary'])
        self.figure_canvas.draw()
        self.redraw_plot()
        self.table_add_button.configure(background=background_primary,foreground=foreground_primary)
        
# ----------------------
# Plot related functions
    def create_plots(self):
        """Creates subplots according to data_info dict. Sets labels and limits. 
            Ties mouse events to appropriate internal functions."""
        self.fig = Figure()
        self.plots = []
        self.data_info["ylim_strvars"] = []
        for i in range(self.num_dep_vars):
            self.plots.append(self.fig.add_subplot(self.num_dep_vars,1,i+1))
            self.plots[i].set_xlabel(self.data_info["xlabel_unit"])
            self.plots[i].set_ylabel(self.data_info["ylabels_units"][i])
            self.plots[i].set_xlim(self.data_info["xlim"])
            self.plots[i].set_ylim(self.data_info["ylims"][i])
            self.plots[i].set_title(self.data_info["plot_titles"][i])
            self.data_info["ylim_strvars"].append(
                tk.StringVar(value = self.display_rounding(self.data_info["ylims"][i][1],i+1)) )
            self.crossX = self.plots[i].axhline(y=0)
            self.crossY = self.plots[i].axvline(x=0)
            self.crosshair =True

        self.data_info["xlim_strvar"] = tk.StringVar(
            value = self.display_rounding(self.data_info["xlim"][1],0))

        self.fig.tight_layout(pad=2)
        self.fig.canvas.mpl_connect('button_press_event',self.on_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event',self.on_mouse_move)
        self.fig.canvas.mpl_connect('button_release_event',self.on_mouse_release)
        self.figure_canvas = FigureCanvasTkAgg(self.fig,master=self.frame_plots)
        self.figure_canvas.draw()

        self.mouse_coords_str = tk.StringVar(value = "Mouse Coordinates")
        self.mouse_coords = tk.Label(self.frame_plotReadouts,textvariable=self.mouse_coords_str)
        self.mouse_coords.pack()
        self.crosshair = False
        self.figure_canvas.get_tk_widget().pack(expand=True,fill='both')

    def update_axes_lims(self):
        """Goes through each subplot and sets x and y limits based on the StringVar values"""
        for (plot,limstr) in zip(self.plots,self.data_info["ylim_strvars"]):
            plot.set_xlim(0,float(self.data_info["xlim_strvar"].get()))
            plot.set_ylim(0,float(limstr.get()))
        self.figure_canvas.draw()

    def redraw_plot(self):
        """Redraws plot, using the new values inside data lists"""
        self.clear_plot()
        for i,plot in enumerate(self.plots):
            self.plot_lines.append(plot.plot(self.x_list,self.ys_list[i],color=self.pallete[self.theme]['lines'][i],marker='o',markersize=5))
        self.figure_canvas.draw()

    def clear_plot(self):
        """Clears all lines from plots except for crosshairs"""
        self.plot_lines = []
        for plot in self.plots:
            for line in plot.lines:
                if line == self.crossX or line == self.crossY: continue
                line.remove()

# ----------------------
# Mouse related functions
    def on_mouse_press(self,event): 
        """Handles mouse press event, sets internal mouse state"""
        self.mouse_press = True

    def on_mouse_release(self,event):
        """Handles release of mouse button. Calls click function if mouse has not been dragged."""
        if self.mouse_press and not self.mouse_drag: # simple click event
            self.on_mouse_click(event)
        # currently no functions operate at the end of drag
        # else: pass # drag event
        self.mouse_press,self.mouse_drag = False, False

    def on_mouse_click(self,event):
        """Called when mouse click is determined, adds new point if it is valid"""
        # this list creates default values for subplots not clicked on, half of ylim
        default_y_vals = [float(self.data_info["ylim_strvars"][i].get())/2 for i in range(self.num_dep_vars)]
        valid_click = False
        # if mouse click points are not None
        if event.xdata and event.ydata and event.button == MouseButton.LEFT: 
            # go through each subplot first to check if click is inside a subplot
            for plot_idx,plot in enumerate(self.plots):
                # checks if mouse is inside subplot and it is the first point or next in time
                if event.inaxes == plot and (len(self.x_list) <1 or event.xdata > max(self.x_list)):        
                    valid_click = True
                    break
            # once we know a subplot was clicked inside at a valid location    
            if valid_click:
                self.update_list(len(self.x_list),0,event.xdata)
                for y_idx,default_val in enumerate(default_y_vals):
                    self.update_list(len(self.x_list),y_idx+1,
                        event.ydata if plot_idx == y_idx else default_val)
                valid_click = False
            # update plots and tables after having changed the lists
            self.redraw_plot()
            self.update_table()

    def on_mouse_move(self,event):
        """Handles functionality related to mouse movement. Creates crosshair if mouse is inside
            a subplot and updates cursor if near a point that can be dragged. Also handles moving
            point on graph if it is being dragged."""
        if event.xdata and event.ydata:
            for plot_idx,plot in enumerate(self.plots):
                if event.inaxes == plot:
                    # create crosshair at current point and remove old crosshair
                    if self.crosshair:
                        self.crossX.remove()
                        self.crossY.remove()
                    self.crossX = plot.axhline(y=event.ydata,color=self.pallete[self.theme]['crosshair'])
                    self.crossY = plot.axvline(x=event.xdata,color=self.pallete[self.theme]['crosshair'])
                    self.figure_canvas.draw()
                    self.crosshair = True

                    # update mouse coordinates on screen, rounding is handled based on 
                    # rounding defined in data_info
                    xvalue = self.display_rounding(event.xdata,0)
                    yvalue = self.display_rounding(event.ydata,plot_idx+1)
                    self.mouse_coords_str.set(
                        f"{self.data_info['xlabel']}: {xvalue} {self.data_info['xunit']} | "+
                        f"{self.data_info['ylabels'][plot_idx]}: {yvalue} {self.data_info['yunits'][plot_idx]}")

                    # check if mouse is near an existing point, use closest point for dragging
                    near = False
                    dists = []
                    if len(self.x_list) > 0:
                        for existing_pt in zip(self.x_list,self.ys_list[plot_idx]):
                            dists.append(self.get_distance((event.xdata,event.ydata),existing_pt,plot_idx))
                        min_dist = min(dists)
                        if min_dist < self.ptcontainer:
                            self.figure_canvas.set_cursor(4)
                            near = True
                            self.near_idx = dists.index(min_dist)
                    
                    if not near: self.figure_canvas.set_cursor(1)

                    # move nearby point (or if previously dragging a point)
                    if self.mouse_press and (near or self.mouse_drag): 
                        self.mouse_drag = True
                        self.update_list(self.near_idx,0,event.xdata)
                        self.update_list(self.near_idx,plot_idx+1,event.ydata)
            
            # redraw plot after looping through subplots
            self.redraw_plot()
            self.update_str_vars()

    def get_distance(self,pt1:tuple,pt2:tuple,plot_idx:int):
        """Returns a normalized distance value between 2 points. Normalization is based on the subplot's
        x and y limits, subplot specified as plot_idx"""
        lims = (self.plots[plot_idx].get_xlim()[1],self.plots[plot_idx].get_ylim()[1])
        return np.sqrt(sum([((pt1[i] - pt2[i])/lims[i])**2 for i in range(2)]))

# ----------------------
# Table related functions
    def update_str_vars(self):
        """Updates StringVar values for the table. Used when points are dragged on plot"""
        for i,vallist in enumerate([self.x_list,*self.ys_list]):
            for j,val in enumerate(vallist):
                val = self.display_rounding(val,i)
                self.table_strvars[i][j].set(val)

    def delete_point(self,row:int):
        """When X button next to tabular point is pressed, lists are popped and plot and tables
        are updated to show the removed point."""
        self.x_list.pop(row)
        if row < len(self.x_list)-1: self.phase_order_list.pop(row)
        for i in range(self.num_dep_vars):
            self.ys_list[i].pop(row)
        self.redraw_plot()
        self.update_table(overwrite=True)

    def update_table(self,overwrite = False,bool_list=None):
        """This function handles both adding a new entry to table and overwriting the whole table.
        Overwriting causes all table widgets to be destroyed and a new set of widgets to be created.
        This also resets the StringVars."""
        try:
            self.bgpri
        except AttributeError:
            self.bgpri = self.pallete[self.theme]["background_primary"]
        try:
            self.fgpri
        except AttributeError:
            self.fgpri = self.pallete[self.theme]["foreground_primary"]
        
        row = len(self.x_list)-1 # last row (assumes data lists have been updated with new point)
        if overwrite and len(self.table_widgets) > 0:
            for item in self.table_widgets:
                item.destroy()
            self.table_widgets = []
            self.table_strvars = [[] for i in range(self.num_dep_vars+1)]
            self.table_boolvars = [[] for i in range(self.num_dep_vars)]
            row = 0 # set row to 0 if overwriting entire table

        while row < len(self.x_list):
            # numerical label for each point
            rowtxt = str(row+1)
            if row+1 <10: rowtxt = "  "+rowtxt
            rownum_label = tk.Label(self.frame_table.interior,text = rowtxt,bg=self.bgpri,fg=self.fgpri)
            rownum_label.grid(row = row*2+2,column = 0)
            self.table_widgets.append(rownum_label)

            if row > 0 and self.show_optimize.get(): # have at least 2 points
                optimize_label = tk.Label(self.frame_table.interior,text="Optimize:",bg=self.bgpri,fg=self.fgpri)
                optimize_label.grid(row=row*2+1,column = 1)
                self.table_widgets.append(optimize_label)

            # entries and stringvars for each x,y value
            row_yvals = [self.ys_list[i][row] for i in range(self.num_dep_vars)]
            for col,val in enumerate([self.x_list[row],*row_yvals]):
                val = self.display_rounding(val,col)
                entry_text = tk.StringVar(value=val)
                self.table_strvars[col].append(entry_text)

                entry = tk.Entry(self.frame_table.interior,width=self.table_column_widths[col],
                              textvariable=entry_text,justify='center',bg=self.bgpri,fg=self.fgpri)
                entry.grid(row=row*2+2,column=col+1)
                # binds key release to update list function
                entry.bind("<KeyRelease>",lambda e,row=row,col=col,entry_text=entry_text: 
                        [self.update_list(row,col,entry_text.get()),self.redraw_plot()] )
                self.table_widgets.append(entry)

                if col > 0 and row > 0 and self.show_optimize.get(): # have at least 2 points and for dependent var cols only
                    checkbox_label = tk.Label(self.frame_table.interior,text=self.data_info["ylabels"][col-1],
                                              bg=self.bgpri,fg=self.fgpri)
                    checkbox_label.grid(row=row*2+1,column=col+1,sticky='w')
                    self.table_widgets.append(checkbox_label)

                    optimize_variable = tk.BooleanVar()
                    self.table_boolvars[col-1].append(optimize_variable)
                    if bool_list: # if bool list has already been populated (e.g. opening an existing phase info)
                        optimize_variable.set(value=bool_list[col-1][row-1])
                    optimize_checkbox = tk.Checkbutton(self.frame_table.interior,variable=optimize_variable,bg=self.bgpri,fg=self.fgpri,
                                                       activebackground=self.bgpri,activeforeground=self.fgpri,selectcolor=self.bgpri,
                                                       highlightbackground=self.bgpri,highlightcolor=self.bgpri)
                    optimize_checkbox.grid(row=row*2+1,column=col+1,sticky='e')
                    self.table_widgets.append(optimize_checkbox)

            # delete button for each point
            delete_button = tk.Button(self.frame_table.interior,text="X",width=4,bg=self.bgpri,fg=self.fgpri)
            delete_button.bind("<Button-1>",lambda e, row=row:self.delete_point(row))
            delete_button.grid(row=row*2+2,column=col+2)       
            self.table_widgets.append(delete_button)
        
            row += 1        
        # reposition add new point button based on updated table
        if len(self.x_list)>0:
            self.table_add_button.grid(row=row*2+3,column=0,columnspan=col+2)

    def add_new_row(self,_):
        """Updates data lists with a generic new point and runs redraw plot and update table.
            New point is added at x = halfway between last point and x limit, y = half of y limit"""
        default_y_vals = [float(self.data_info["ylim_strvars"][i].get())/2 for i in range(self.num_dep_vars)]
        newx = 0
        if len(self.x_list) > 0:
            newx =  (float(self.data_info["xlim_strvar"].get()) - self.x_list[-1])/2 + self.x_list[-1]
        for col,item in enumerate([newx,*default_y_vals]):
            self.update_list(row=len(self.x_list),col=col,value=item)
        self.redraw_plot()
        self.update_table()

    def create_table(self):
        """Creates headers for table and sets column widths based on header lengths."""
        self.table_column_widths = []
        self.table_strvars = [] # list used to hold StringVars 
        self.table_boolvars = []
        self.table_widgets = [] # list used to hold graphical table elements, can be used to modify them
        self.header_widgets = []
        labels_w_units = [self.data_info["xlabel_unit"],*self.data_info["ylabels_units"]]
        header = tk.Label(self.frame_tableheaders,text="Pt")
        header.grid(row = 0,column = 0)
        self.header_widgets.append(header)
        for col,label in enumerate(labels_w_units):
            header_text = tk.StringVar(value=label)
            header = tk.Entry(self.frame_tableheaders,textvariable=header_text,state='readonly',width=len(label),
                              justify='center')
            header.grid(row = 0,column = col+1)
            self.table_column_widths.append(len(label))
            self.table_strvars.append([])
            if col > 0: self.table_boolvars.append([])
            self.header_widgets.append(header)
    
        # button for adding new rows to table
        self.table_add_button = tk.Button(self.frame_table.interior,text="Add New Point")
        self.table_add_button.bind("<Button-1>",func=self.add_new_row)
        self.update_table()

    def display_rounding(self,value,col:int):
        """Returns a rounded value based on which variable the value belongs to.
        Uses rounding amount specified in data_info"""
        rounding = [self.data_info["xround"],*self.data_info["yrounds"]]
        return format(value,"."+str(rounding[col])+"f")

# ----------------------
# Menu related functions
    def create_menu(self):
        structure = {"File":[["command","Open",self.open_phase_info],
                             ["command","Save",self.save],
                             ["command","Save as",self.save_as],
                             ["separator"],
                             ["command","Exit",self.close_window]],
                    "Edit":[["command","Axes Limits",self.change_axes_popup],
                            ["command","Units",None],
                            ["command","Rounding",None]],
                    "View":[["checkbutton","Optimize Phase",self.toggle_optimize_view,self.show_optimize],
                            ["command","Advanced Options",self.advanced_options_popup]],
                    "Help":[["command","Instructions",None],
                            ["command","About",None]]}
        
        names = ["dark_mode.png","light_mode.png"]
        imgfiles = []
        dirname = os.path.abspath(os.path.dirname(__file__))
        for name in names:
            imgfiles.append(tk.PhotoImage(file=os.path.join(dirname,name)))
        menu_bar = tk.Menu(self)
        for tab_label,tab_list in structure.items():
            tab = tk.Menu(menu_bar,tearoff=False,background=self.pallete[self.theme]['background_primary'],foreground=self.pallete[self.theme]['foreground_primary'])
            menu_bar.add_cascade(label=tab_label,menu = tab)
            for item in tab_list:
                if item[0] == "separator": tab.add_separator()
                elif item[0] == "command":
                    tab.add_command(label=item[1],command=item[2])
                elif item[0] == "checkbutton":
                    tab.add_checkbutton(label=item[1],command=item[2],variable=item[3],selectcolor=self.pallete[self.theme]['foreground_primary'])
        self.config(menu=menu_bar)
        self.theme_button = tk.Button(self,command=self.update_theme,image=imgfiles[self.theme!="light"],
                                      bg=self.bgpri)
        self.theme_button.image = imgfiles[self.theme=="light"]
        self.imgfiles = imgfiles
        self.theme_button.place(anchor='nw',relx=0,rely=0)

    def close_window(self):
        """Closes main window and saves last geometry configuration into a txt """
        last_geometry = self.winfo_geometry()
        last_theme = not self.theme_toggle
        self.destroy()
        with open(self.persist_filename,"wb") as fp:
            pickle.dump({'window_geometry':last_geometry,'theme_toggle':last_theme},fp)

    def close_popup(self):
        """Function to close existing popup"""
        self.focus_set()
        self.popup.destroy()
        self.popup = None

    def generic_popup(self,pop_wid=100,pop_hei=100,pop_title="Popup",buttons_text = []):
        """Function to create a base window for a popup. Returns popup object to be used for adding widget
            and configuring settings. Also can include Apply and Cancel buttons and return these for changing
            location and command."""
        # compute middle of window location to place popup into
        win_size,win_left,win_top = self.winfo_geometry().split("+")
        win_wid,win_hei = win_size.split("x")
        win_left,win_top,win_wid,win_hei = int(win_left),int(win_top),int(win_wid),int(win_hei)
        pop_left,pop_top = int(win_left + win_wid/2 - pop_wid/2), int(win_top + win_hei/2 - pop_hei/2)

        popup = tk.Toplevel(self)
        popup.resizable(False,False)
        popup.geometry(f"{pop_wid}x{pop_hei}+{pop_left}+{pop_top}")
        popup.title(pop_title)
        popup.focus_set()
        popup.configure(background=self.bgpri)
        self.popup = popup

        popup.protocol("WM_DELETE_WINDOW",func=self.close_popup)

        popup_content_frame = tk.Frame(popup,background=self.bgpri)
        popup_content_frame.pack(side='top',fill='x')
        button_frame = tk.Frame(popup,bg=self.bgpri)
        button_frame.pack(side='bottom',pady=5)

        buttons = {}
        bwid = len(max(buttons_text,key=len))+5

        for button_txt in buttons_text:
            button = tk.Button(button_frame,text=button_txt.title(),width=bwid,
                               background=self.bgpri,foreground=self.fgpri)
            button.pack(side='left',padx=5)
            buttons[button_txt] = button
        
        return popup,popup_content_frame,buttons
    
    def change_axes_popup(self):
        """Creates a popup window that allows user to edit axes limits. This function is triggered
            by the menu buttons"""
        labels_w_units = [self.data_info["xlabel_unit"],*self.data_info["ylabels_units"]]
        lim_strs = [self.data_info["xlim_strvar"],*self.data_info["ylim_strvars"]]

        def reset_options(old_list):
            for value,lim_str in zip(old_list,lim_strs):
                lim_str.set(value=value)
        
        current_lims = [var.get() for var in lim_strs]

        popup,content_frame,buttons = self.generic_popup(pop_wid = 300, pop_hei=100, pop_title="Axes Limits",
                                           buttons_text=["apply","reset","cancel"])
        popup.protocol("WM_DELETE_WINDOW",func=lambda:[self.close_popup(),reset_options(current_lims)])
        for i in range(2): content_frame.columnconfigure(i,weight=1)

        for row,(label,lim_str) in enumerate(zip(labels_w_units,lim_strs)):
            lim_label = tk.Label(content_frame,text=label,justify='right',bg=self.bgpri,fg=self.fgpri)
            lim_label.grid(row=row,column=0,sticky='e') 
            lim_entry = tk.Entry(content_frame,textvariable=lim_str,width = max(6,len(lim_str.get())),
                                 bg=self.bgpri,fg=self.fgpri)
            lim_entry.grid(row=row,column=1,sticky='w')

        buttons["apply"].configure(command=lambda:[self.close_popup(),self.update_axes_lims()])
        buttons["reset"].configure(command=lambda:[self.close_popup(),reset_options(self.axes_lim_defaults),self.update_axes_lims()])
        buttons["cancel"].configure(command=lambda:[self.close_popup(),reset_options(current_lims),self.update_axes_lims()])

    def get_phase_names(self):
        names = ["Climb ","Cruise ","Descent "]
        counters = [1,1,1]
        phase_name_list = []
        for i in range(len(self.x_list)-1):
            nextpt = round(self.ys_list[0][i+1],self.data_info["yrounds"][0])
            nowpt = round(self.ys_list[0][i],self.data_info["yrounds"][0]) 
            if nextpt > nowpt: j = 0
            elif nextpt < nowpt: j = 2
            else: j = 1
                
            phase_name_list.append(names[j]+str(counters[j]))
            counters[j] += 1
        return phase_name_list
    
    def advanced_options_popup(self):
        """Creates a popup window that allows user to edit advanced options for phase info. 
        Options included are specified as a dict in __init__ and include solve/constrain for range,
        include landing/takeoff, polynomial order, and phase order. This function is triggered by the menu buttons"""
        def reset_options(self,old_dict = self.advanced_options_info_defaults):
            for key,value in old_dict.items():
                self.advanced_options_info[key].set(value=value)
            self.phase_order_list = [self.phase_order_default]*(len(self.x_list)-1)

        current_info = {} # this stores option values as they are before user edits inside popup
        for key,var in self.advanced_options_info.items():
            current_info[key] = var.get()

        popup,content_frame,buttons = self.generic_popup(pop_wid=300,pop_hei=175,pop_title="Advanced Options",
                                           buttons_text=["apply","reset","cancel"])
        popup.protocol("WM_DELETE_WINDOW",func=lambda:[self.close_popup(),reset_options(self,current_info)])
        
        for i in range(3): content_frame.columnconfigure(i,weight=1)

        for row,(option_label_txt,option_var) in enumerate(self.advanced_options_info.items()):
            option_label = tk.Label(content_frame,text=option_label_txt.replace("_"," ").title(),justify='right',
                                    bg=self.bgpri,fg=self.fgpri)
            option_label.grid(row=row,column=0,sticky='e')
            if type(tk.BooleanVar()) == type(option_var):
                option_checkbox = tk.Checkbutton(content_frame,variable=option_var,bg=self.bgpri,fg=self.fgpri)
                option_checkbox.grid(row=row,column=1,sticky='w')
            elif type(tk.IntVar()) == type(option_var):
                option_entry = tk.Entry(content_frame,textvariable=option_var,width = 3,bg=self.bgpri,fg=self.fgpri)
                option_entry.grid(row=row,column=1,sticky='w')

        def set_var(_):
            phase_idx = order_combo.current()
            order_var.set(value = self.phase_order_list[phase_idx])
        
        def change_var(_):
            phase_idx = order_combo.current()
            try: newval = int(order_var.get())
            except ValueError: return
            if newval < self.phase_order_default:
                messagebox.showwarning(title="Error",message=f"Phase transcription order must be at least {self.phase_order_default}!")
                newval = self.phase_order_default
            self.phase_order_list[phase_idx] = newval

        if len(self.x_list) > 1:
            order_label = tk.Label(content_frame,text="Phase Transcription Order: ",bg=self.bgpri,fg=self.fgpri)
            order_label.grid(row=row+1,column=0,sticky='e')
            order_combo = ttk.Combobox(content_frame,state='readonly',values = self.get_phase_names(),width=9)
            order_combo.bind("<<ComboboxSelected>>",set_var)
            order_combo.current(0)
            order_combo.grid(row=row+1,column=1,sticky='w')
            order_var = tk.StringVar(value = self.phase_order_default)
            order_entry = tk.Entry(content_frame,width = 3,textvariable=order_var,bg=self.bgpri,fg=self.fgpri)
            order_entry.bind("<KeyRelease>",func=change_var)
            order_entry.grid(row=row+1,column=2,sticky='w')

        # apply maintains user options as set by user in popup, reset reverts them to default values, cancel reverts to 
        # values as they were at the start of the popup
        buttons["apply"].configure(command=lambda:self.close_popup())
        buttons["reset"].configure(command=lambda:[self.close_popup(),reset_options(self)])
        buttons["cancel"].configure(command=lambda:[self.close_popup(),reset_options(self,current_info)])

    def open_phase_info(self):
        """Opens a dialog box to select a .py file with a phase info dict. File must contain a dict called phase_info. 
            File can be placed in any directory."""
        file_dialog = filedialog.Open(self,filetypes = [("Python files","*.py")])
        filename = file_dialog.show()
        if filename != "": 
            spec = importlib.util.spec_from_file_location("module_name",filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            phase_info = None
            try:
                phase_info = module.phase_info
            except AttributeError:
                raise Exception("Python File does not contain a dict called phase_info!")
            if phase_info:
                init = False
                idx = 0
                ylabs = ["altitude","mach"]
                self.phase_order_list = []
                for phase_dict in (phase_info.values()):
                    if "initial_guesses" in phase_dict: # not a pre/post mission dict
                        self.advanced_options_info["solve_for_distance"].set(
                            value = phase_dict["user_options"]["solve_for_distance"])
                        self.advanced_options_info["polynomial_control_order"].set(
                            value = phase_dict["user_options"]["polynomial_control_order"])
                        self.phase_order_list.append(phase_dict["user_options"]["order"])
                        
                        timevals = phase_dict["initial_guesses"]["time"][0]
                        if not init: # for first run initialize internal lists with correct num of elements
                            numpts = phase_dict["user_options"]["num_segments"]+1
                            self.x_list = [0]*numpts
                            self.ys_list = [[0]*numpts for _ in range(self.num_dep_vars)]
                            bool_list = [[0]*(numpts-1) for _ in range(self.num_dep_vars)]
                            self.x_list[0] = timevals[0]
                            for i in range(self.num_dep_vars):
                                self.ys_list[i][0] = phase_dict["user_options"]["initial_"+ylabs[i]][0]
                            init = True

                        self.x_list[idx+1] = timevals[1] + timevals[0]  
                        for i in range(self.num_dep_vars):
                            self.ys_list[i][idx+1] = phase_dict["user_options"]["final_"+ylabs[i]][0]
                            bool_list[i][idx] = phase_dict["user_options"]["optimize_"+ylabs[i]]
                        
                        idx +=1

                self.advanced_options_info["constrain_range"].set(value = phase_info["post_mission"]["constrain_range"])
                self.advanced_options_info["include_landing"].set(value = phase_info["post_mission"]["include_landing"])
                self.advanced_options_info["include_takeoff"].set(value = phase_info["pre_mission"]["include_takeoff"])

                # checks if any optimize values are true, in which case checkboxes are shown
                for axis_list in bool_list:
                    for bool_var in axis_list:
                        if bool_var:
                            self.show_optimize.set(True)
                            break
                lim_margin = 1.2
                lims = [max(ys)*lim_margin for ys in self.ys_list]
                lims = [max(self.x_list)*lim_margin,*lims]
                lim_strs = [self.data_info["xlim_strvar"],*self.data_info["ylim_strvars"]]
                for var,lim in zip(lim_strs,lims):
                    var.set(value = lim)

                self.redraw_plot()
                self.update_axes_lims()
                self.update_table(overwrite=True,bool_list=bool_list)

    def toggle_optimize_view(self):
        """Runs update table with overwrite on to toggle display of optimize checkboxes"""
        self.update_table(overwrite=True)

    def show_instructions(self):
        messagebox.showinfo()

    def save_as(self):
        filename = filedialog.asksaveasfilename(defaultextension='.py',confirmoverwrite=True,
                                                filetypes=[("Python files","*.py")],initialfile='outputted_phase_info')
        if not filename: return
        self.save(filename=filename)

    def save(self,filename=None):
        users = {'solve_for_distance':self.advanced_options_info["solve_for_distance"].get(),
                 'constrain_range':self.advanced_options_info["constrain_range"].get(),
                 'include_takeoff':self.advanced_options_info["include_takeoff"].get(),
                 'include_landing':self.advanced_options_info["include_landing"].get()}
        polyord = self.advanced_options_info["polynomial_control_order"].get()
        if len(self.table_boolvars[0]) != len(self.x_list)-1:
            for i in range(self.num_dep_vars):
                self.table_boolvars[i] = [tk.BooleanVar()]*(len(self.x_list)-1)
        if not filename: filename = os.path.join(os.getcwd(), 'outputted_phase_info.py')
        create_phase_info(times = self.x_list, altitudes = self.ys_list[0], mach_values = self.ys_list[1],
                          polynomial_order = polyord, num_segments = len(self.x_list)-1,
                          optimize_altitude_phase_vars = self.table_boolvars[0],
                          optimize_mach_phase_vars = self.table_boolvars[1],
                          user_choices = users, orders=self.phase_order_list,
                          filename=filename)
        self.close_window()

def create_phase_info(times, altitudes, mach_values,
                      polynomial_order, num_segments, optimize_mach_phase_vars, optimize_altitude_phase_vars, user_choices,
                      orders, filename='outputted_phase_info.py'):
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
        'include_takeoff': user_choices["include_takeoff"],
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
                'order': orders[i],
                'solve_for_distance': False,
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
                'time': ([times[i], times[i+1]-times[i]], 'min'),
            }
        }

    phase_info['post_mission'] = {
        'include_landing': user_choices["include_landing"],
        'constrain_range': True,
        'target_range': (0., 'nmi'),
    }

    # Apply user choices to each phase
    for phase_name, _ in phase_info.items():
        if 'pre_mission' in phase_name or 'post_mission' in phase_name:
            continue
        phase_info[phase_name]['user_options'].update({
            'solve_for_distance': user_choices.get('solve_for_distance', False),
        })

    # Apply global settings if required
    phase_info['post_mission']['constrain_range'] = user_choices.get(
        'constrain_range', True)

    # Calculate the total range
    total_range = estimate_total_range_trapezoidal(times, mach_values)
    print(
        f"Total range is estimated to be {total_range} nautical miles")

    phase_info['post_mission']['target_range'] = (total_range, 'nmi')

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

if __name__ == "__main__":
    app = AviaryMissionEditor()
    app.mainloop()