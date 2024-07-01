import os, importlib.util
from math import sqrt

from aviary.interface.graphical_input import create_phase_info

from tkinter import Tk, Toplevel, Canvas, Frame, BooleanVar, StringVar, IntVar, filedialog
from tkinter import Button, Checkbutton, Entry, Label, Menu, Scrollbar

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseButton

# TODO: Add ability to change units, phase info specifies units and aviary can handle these unit changes.
#       Makes sense to allow unit changes in GUI based on user preference.
#       Possible unit changes: Alt -> ft, m, mi, km, nmi; Time -> min, s, hr; Mach -> none

class VerticalScrolledFrame(Frame):
    """A pure Tkinter scrollable frame that actually works!
    * Use the 'interior' attribute to place widgets inside the scrollable frame.
    * Construct and pack/place/grid normally.
    * This frame only allows vertical scrolling.
    ------
    Taken from https://stackoverflow.com/questions/16188420/tkinter-scrollbar-for-frame
    """
    def __init__(self, parent, *args, **kw):
        Frame.__init__(self, parent, *args, **kw)

        # Create a canvas object and a vertical scrollbar for scrolling it.
        vscrollbar = Scrollbar(self, orient='vertical')
        vscrollbar.pack(fill='y', side='left', expand=False)
        self.freezeframe = Frame(self) # this frame will not scroll, allowing for freeze view functionality
        self.freezeframe.pack(side='top',fill='x')
        
        canvas = Canvas(self, bd=0, highlightthickness=0,
                           yscrollcommand=vscrollbar.set)
        canvas.pack(side='right', fill='y', expand=True)
        vscrollbar.config(command=canvas.yview)

        # Reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # Create a frame inside the canvas which will be scrolled with it.
        self.interior = interior = Frame(canvas)
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

class AviaryMissionEditor(Tk):
    def __init__(self):
        super().__init__()
        self.title('Mission Design Utility')
        self.protocol("WM_DELETE_WINDOW",self.close_window)
        self.focus_set() # focus the window

        # ---------------------------------------------------------------
        # window geometry definition, allows reuse of user's modified size/location
        # tkinter size string format:  widthxheight+x+y ;  x,y are location
        default_win_size = (900,500) # CHANGE THIS TO AFFECT DEFAULT SIZE
        window_geometry = f"{default_win_size[0]}x{default_win_size[1]}+10+10" 
        if os.path.exists("windowlocation.txt"):
            with open("windowlocation.txt","r") as fp:
                window_geometry = fp.read().split("\n")[0]
        self.geometry(window_geometry)
        self.minsize(*default_win_size) # force a minimum size for layout to look correct

        # ---------------------------------------------------------------
        # create window layout with frames for containing graph, table, and scrollbar
        self.frame_table = VerticalScrolledFrame(self)
        self.frame_table.pack(side='right',fill='y')
        self.frame_tableheaders = self.frame_table.freezeframe

        self.frame_plotReadouts = Frame(self)
        self.frame_plotReadouts.pack(side='bottom',fill='x')
        self.frame_plots = Frame(self)
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
        self.plot_lines = []

        # internal variables to remember mouse state
        self.mouse_drag,self.mouse_press = False, False
        self.popup = None
        self.ptcontainer = 0.04 # percent of plot size, boundary around point where it can be dragged
        self.show_optimize = BooleanVar() # controls display of optimize phase checkboxes
        
        self.advanced_options_info = {"constrain_range":BooleanVar(value=True),"solve_for_distance":BooleanVar(),
                                      "include_takeoff":BooleanVar(),"include_landing":BooleanVar(),
                                      "polynomial_control_order":IntVar(value=1)}

        self.save_option_defaults()
        self.create_plots()
        self.create_table()
        self.create_menu()

    def save_option_defaults(self):
        self.advanced_options_info_defaults = {}
        for key,var in self.advanced_options_info.items():
            if type(var) == type(BooleanVar()):
                self.advanced_options_info_defaults[key] = BooleanVar(value=var.get())
            elif type(var) == type(IntVar()):
                self.advanced_options_info_defaults[key] = IntVar(value=var.get())

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
        else:
            datalists[col][row] = value

# ----------------------
# Plot related functions
    def create_plots(self):
        """Creates subplots according to data_info dict. Sets labels and limits. 
            Ties mouse events to appropriate internal functions."""
        fig = Figure()
        self.plots = []
        self.data_info["ylim_strvars"] = []
        for i in range(self.num_dep_vars):
            self.plots.append(fig.add_subplot(self.num_dep_vars,1,i+1))
            self.plots[i].set_xlabel(self.data_info["xlabel_unit"])
            self.plots[i].set_ylabel(self.data_info["ylabels_units"][i])
            self.plots[i].set_xlim(self.data_info["xlim"])
            self.plots[i].set_ylim(self.data_info["ylims"][i])
            self.plots[i].set_title(self.data_info["plot_titles"][i])
            self.plots[i].grid(True)
            self.data_info["ylim_strvars"].append(
                StringVar(value = self.display_rounding(self.data_info["ylims"][i][1],i+1)) )
        
        self.data_info["xlim_strvar"] = StringVar(
            value = self.display_rounding(self.data_info["xlim"][1],0))

        fig.tight_layout(pad=2)
        fig.canvas.mpl_connect('button_press_event',self.on_mouse_press)
        fig.canvas.mpl_connect('motion_notify_event',self.on_mouse_move)
        fig.canvas.mpl_connect('button_release_event',self.on_mouse_release)
        self.figure_canvas = FigureCanvasTkAgg(fig,master=self.frame_plots)
        self.figure_canvas.draw()

        self.mouse_coords_str = StringVar(value = "Mouse Coordinates")
        self.mouse_coords = Label(self.frame_plotReadouts,textvariable=self.mouse_coords_str)
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
        colors = ['bo-','mo-']
        for i,plot in enumerate(self.plots):
            self.plot_lines.append(plot.plot(self.x_list,self.ys_list[i],colors[i],markersize=5))
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
                    self.crossX = plot.axhline(y=event.ydata,color='red')
                    self.crossY = plot.axvline(x=event.xdata,color='red')
                    self.figure_canvas.draw()
                    self.crosshair = True

                    # update mouse coordinates on screen, rounding is handled based on 
                    # rounding defined in data_info
                    xvalue = self.display_rounding(event.xdata,0)
                    yvalue = self.display_rounding(event.ydata,plot_idx+1)
                    self.mouse_coords_str.set(
                        f"{self.data_info['xlabel']}: {xvalue} {self.data_info['xunit']} | "+
                        f"{self.data_info['ylabels'][plot_idx]}: {yvalue} {self.data_info['yunits'][plot_idx]}")

                    # check if mouse is near an existing point
                    # TODO: only update point closest to mouse
                    near = False
                    for pt_idx,prevpt in enumerate(zip(self.x_list,self.ys_list[plot_idx])):
                        dist = self.get_distance((event.xdata,event.ydata),prevpt,plot_idx)
                        if dist < self.ptcontainer:
                            self.figure_canvas.set_cursor(4)
                            near = True
                            self.near_idx = pt_idx
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
        return sqrt(sum([((pt1[i] - pt2[i])/lims[i])**2 for i in range(2)]))

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
        for i in range(self.num_dep_vars):
            self.ys_list[i].pop(row)
        self.redraw_plot()
        self.update_table(overwrite=True)

    def update_table(self,overwrite = False,bool_list=None):
        """This function handles both adding a new entry to table and overwriting the whole table.
        Overwriting causes all table widgets to be destroyed and a new set of widgets to be created.
        This also resets the StringVars."""
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
            rownum_label = Label(self.frame_table.interior,text = rowtxt)
            rownum_label.grid(row = row*2+2,column = 0)
            self.table_widgets.append(rownum_label)

            if row > 0 and self.show_optimize.get(): # have at least 2 points
                optimize_label = Label(self.frame_table.interior,text="Optimize:")
                optimize_label.grid(row=row*2+1,column = 1)
                self.table_widgets.append(optimize_label)

            # entries and stringvars for each x,y value
            row_yvals = [self.ys_list[i][row] for i in range(self.num_dep_vars)]
            for col,val in enumerate([self.x_list[row],*row_yvals]):
                val = self.display_rounding(val,col)
                entry_text = StringVar(value=val)
                self.table_strvars[col].append(entry_text)

                entry = Entry(self.frame_table.interior,width=self.table_column_widths[col],
                              textvariable=entry_text,justify='center',relief='raised')
                entry.grid(row=row*2+2,column=col+1)
                # binds key release to update list function
                entry.bind("<KeyRelease>",lambda e,row=row,col=col,entry_text=entry_text: 
                        [self.update_list(row,col,entry_text.get()),self.redraw_plot()] )
                self.table_widgets.append(entry)

                if col > 0 and row > 0 and self.show_optimize.get(): # have at least 2 points and for dependent var cols only
                    checkbox_label = Label(self.frame_table.interior,text=self.data_info["ylabels"][col-1])
                    checkbox_label.grid(row=row*2+1,column=col+1,sticky='w')
                    self.table_widgets.append(checkbox_label)

                    optimize_variable = BooleanVar()
                    self.table_boolvars[col-1].append(optimize_variable)
                    if bool_list: # if bool list has already been populated (e.g. opening an existing phase info)
                        optimize_variable.set(value=bool_list[col-1][row-1])
                    optimize_checkbox = Checkbutton(self.frame_table.interior,variable=optimize_variable)
                    optimize_checkbox.grid(row=row*2+1,column=col+1,sticky='e')
                    self.table_widgets.append(optimize_checkbox)

            # delete button for each point
            delete_button = Button(self.frame_table.interior,text="X",borderwidth=2)
            delete_button.bind("<Button-1>",lambda e, row=row:self.delete_point(row))
            delete_button.grid(row=row*2+2,column=col+2)       
            self.table_widgets.append(delete_button)
        
            row += 1        
        # reposition add new point button based on updated table
        self.table_add_button.grid(row=row*2+3,column=0,columnspan=col+2)

    def add_new_row(self,_):
        """Updates data lists with a generic new point and runs redraw plot and update table.
            New point is added at x = halfway between last point and x limit, y = half of y limit"""
        default_y_vals = [float(self.data_info["ylim_strvars"][i].get())/2 for i in range(self.num_dep_vars)]
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
        labels_w_units = [self.data_info["xlabel_unit"],*self.data_info["ylabels_units"]]
        header = Label(self.frame_tableheaders,text="Pt")
        header.grid(row = 0,column = 0)
        for col,label in enumerate(labels_w_units):
            header_text = StringVar(value=label)
            header = Entry(self.frame_tableheaders,textvariable=header_text,width=len(label),
                           state='readonly',relief='solid',justify='center') # using entry maintains same col widths
            header.grid(row = 0,column = col+1)
            self.table_column_widths.append(len(label))
            self.table_strvars.append([])
            if col > 0: self.table_boolvars.append([])
    
        # button for adding new rows to table
        self.table_add_button = Button(self.frame_table.interior,text="Add New Point")
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
                             ["command","Save as",None],
                             ["separator"],
                             ["command","Exit",self.close_window]],
                    "Edit":[["command","Axes Limits",self.change_axes_popup],
                            ["command","Units",None],
                            ["command","Rounding",None]],
                    "View":[["checkbutton","Optimize Phase",self.toggle_optimize_view,self.show_optimize],
                            ["command","Advanced Options",self.advanced_options_popup]],
                    "Help":[["command","Instructions",None],
                            ["command","About",None]]}

        menu_bar = Menu(self)
        for tab_label,tab_list in structure.items():
            tab = Menu(menu_bar,tearoff=False)
            menu_bar.add_cascade(label=tab_label,menu = tab)
            for item in tab_list:
                if item[0] == "separator": tab.add_separator()
                elif item[0] == "command":
                    tab.add_command(label=item[1],command=item[2])
                elif item[0] == "checkbutton":
                    tab.add_checkbutton(label=item[1],command=item[2],variable=item[3])

        self.config(menu=menu_bar)

    def close_window(self):
        """Closes main window and saves last geometry configuration into a txt """
        last_geometry = self.winfo_geometry()
        self.destroy()
        with open("windowlocation.txt","w") as fp:
            fp.write(last_geometry)

    def close_popup(self):
        """Function to close existing popup"""
        self.focus_set()
        self.popup.destroy()
        self.popup = None

    def generic_popup(self,pop_wid=100,pop_hei=100,pop_title="Popup",apply=False,cancel=False):
        """Function to create a base window for a popup. Returns popup object to be used for adding widget
            and configuring settings. Also can include Apply and Cancel buttons and return these for changing
            location and command."""
        # compute middle of window location to place popup into
        win_size,win_left,win_top = self.winfo_geometry().split("+")
        win_wid,win_hei = win_size.split("x")
        win_left,win_top,win_wid,win_hei = int(win_left),int(win_top),int(win_wid),int(win_hei)
        pop_left,pop_top = int(win_left + win_wid/2 - pop_wid/2), int(win_top + win_hei/2 - pop_hei/2)

        popup = Toplevel(self)
        popup.resizable(False,False)
        popup.geometry(f"{pop_wid}x{pop_hei}+{pop_left}+{pop_top}")
        popup.title(pop_title)
        popup.focus_set()
        self.popup = popup

        popup.protocol("WM_DELETE_WINDOW",func=self.close_popup)

        apply_button,cancel_button = None,None
        if apply:
            apply_button = Button(popup,text = "Apply")
            apply_button.grid(column=0,sticky='e')
        if cancel:
            cancel_button = Button(popup,text = "Reset")
            cancel_button.grid(column=1,sticky='w')

        if apply and cancel: return popup, apply_button, cancel_button
        elif apply: return popup, apply_button
        elif cancel: return popup, cancel_button
        else: return popup
    
    def change_axes_popup(self):
        """Creates a popup window that allows user to edit axes limits. This function is triggered
            by the menu buttons"""
        popup,apply_button = self.generic_popup(pop_wid = 200, pop_hei=100, pop_title="Axes Limits",apply=True)

        labels_w_units = [self.data_info["xlabel_unit"],*self.data_info["ylabels_units"]]
        lim_strs = [self.data_info["xlim_strvar"],*self.data_info["ylim_strvars"]]

        for row,(label,lim_str) in enumerate(zip(labels_w_units,lim_strs)):
            lim_label = Label(popup,text=label,justify='right')
            lim_label.grid(row=row,column=0) 
            lim_entry = Entry(popup,textvariable=lim_str,width = max(6,len(lim_str.get())))
            lim_entry.grid(row=row,column=1)

        apply_button.grid(row=row+1)
        apply_button.configure(command=lambda:[self.close_popup(),self.update_axes_lims()])

    def advanced_options_popup(self):
        """Creates a popup window that allows user to edit advanced options for phase info. 
        Options included are specified as a dict in __init__ and include solve/constrain for range,
        include landing/takeoff, polynomial order, and phase order. This function is triggered by the menu buttons"""
        def reset_options(self):
            for key,var in self.advanced_options_info_defaults.items():
                self.advanced_options_info[key].set(value=var.get())
        popup,apply_button,cancel_button = self.generic_popup(pop_wid=300,pop_hei=200,pop_title="Advanced Options",apply=True,cancel=True)
        # TODO: check if this behavior is desired, closing without applying will not set user options
        # TODO: these options don't populate when opening an existing phase info, add that
        popup.protocol("WM_DELETE_WINDOW",func=lambda:[self.close_popup(),reset_options(self)])

        for row,(option_label_txt,option_var) in enumerate(self.advanced_options_info.items()):
            option_label = Label(popup,text=option_label_txt.replace("_"," ").title(),justify='right')
            option_label.grid(row=row,column=0,sticky='e')
            if type(BooleanVar()) == type(option_var):
                option_checkbox = Checkbutton(popup,variable=option_var)
                option_checkbox.grid(row=row,column=1,sticky='w')
            elif type(IntVar()) == type(option_var):
                option_entry = Entry(popup,textvariable=option_var,width = 2)
                option_entry.grid(row=row,column=1)

        # TODO: phase order (better if editable per phase)
        apply_button.grid(row=row+1,column=0)
        apply_button.configure(command=lambda:self.close_popup())
        cancel_button.grid(row=row+1,column=1)
        cancel_button.configure(command=lambda:[self.close_popup(),reset_options(self)])

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
                for phase_dict in (phase_info.values()):
                    if "initial_guesses" in phase_dict: # not a pre/post mission dict
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

                # checks if any optimize values are true, in which case checkboxes are shown
                for axis_list in bool_list:
                    for bool_var in axis_list:
                        if bool_var:
                            self.show_optimize.set(True)
                            break
                self.redraw_plot()
                self.update_table(overwrite=True,bool_list=bool_list)

    def toggle_optimize_view(self):
        """Runs update table with overwrite on to toggle display of optimize checkboxes"""
        self.update_table(overwrite=True)

    def save(self):
        # TODO: checkboxes for solve distance, constrain range, 
        #       entry for polynomial order
        # TODO: save phase info as filename with save as command
        users = {'solve_for_distance':self.advanced_options_info["solve_for_distance"].get(),
                 'constrain_range':self.advanced_options_info["constrain_range"].get()}
        polyord = self.advanced_options_info["polynomial_control_order"].get()
        if len(self.table_boolvars[0]) != len(self.x_list)-1:
            for i in range(self.num_dep_vars):
                self.table_boolvars[i] = [BooleanVar()]*(len(self.x_list)-1)
        create_phase_info(times = self.x_list, altitudes = self.ys_list[0], mach_values = self.ys_list[1],
                          polynomial_order = polyord, num_segments = len(self.x_list)-1,
                          optimize_altitude_phase_vars = self.table_boolvars[0],
                          optimize_mach_phase_vars = self.table_boolvars[1],
                          user_choices = users)
        self.close_window()

if __name__ == "__main__":
    app = AviaryMissionEditor()
    app.mainloop()