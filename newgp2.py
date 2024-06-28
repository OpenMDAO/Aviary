import os
from math import sqrt
from tkinter import Tk,Canvas,Frame,Scrollbar,Button, Entry, Label,StringVar,Menu,Toplevel
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from aviary.subsystems.aerodynamics.gasp_based import data

# TODO: Add ability to change units, phase info specifies units and aviary can handle these unit changes.
#       Makes sense to allow unit changes in GUI based on user preference.
#       Possible unit changes: Alt -> ft, m, mi, km, nmi; Time -> min, s, hr; Mach -> none
# TODO: Add button to add new rows to table for anyone only using table


class myapp(Tk):
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
        self.frame_plotReadouts = Frame(self)
        self.frame_plotReadouts.pack(side='bottom')
        self.frame_plots = Frame(self)
        self.frame_plots.pack(side='top',expand=True,fill='both')

        self.frame_table = Frame(self)
        self.frame_table.pack(side="right",fill="y")

        frame_scroll = Frame(self)
        frame_scroll.pack(side='right',fill='y')
        scroll = Scrollbar(frame_scroll)
        scroll.pack(side='right',fill='y')

        # ---------------------------------------------------------------
        # Main definition of data which can be plotted/tabulated. Assumes a singular
        # independent variable that is applied to each dependent variable. All 
        # y-attributes are in list format to support multiple dependent variables.
        self.data_info = {"xlabel":"Time","xlim":(0,400),"xunit":"min","xround":0,
                      "ylabels":["Altitude","Mach Number"],"ylims":[(0,50e3),(0,1.0)],
                      "yunits":["ft","unitless"],"yrounds":[0,2],
                      "plot_titles":["Altitude Plot","Mach Plot"]}
        
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
        self.ptcontainer = 0.04 # percent of plot size, boundary around point where it can be dragged

        self.create_plots()
        self.create_table()
        self.create_menu()

    def close_window(self):
        """Closes main window and saves last geometry configuration into a txt """
        last_geometry = self.winfo_geometry()
        self.destroy()
        with open("windowlocation.txt","w") as fp:
            fp.write(last_geometry)

    def check_data_info(self):
        """Verifies data_info dict has consistent number of dependent variables """
        check_keys = ["ylabels","ylims","yunits"]
        self.num_dep_vars = len(self.data_info["plot_titles"])
        for key in check_keys:
            if self.num_dep_vars != len(self.data_info[key]):
                raise Exception("Check length of lists in data_info, mismatch detected.\n"+
                                f"Expected {self.num_dep_vars} dependent variables.")
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
                StringVar(value = self.data_info["ylims"][i][1]) )
        
        self.data_info["xlim_strvar"] = StringVar(value = self.data_info["xlim"][1])

        fig.tight_layout(h_pad=2.0)
        fig.canvas.mpl_connect('button_press_event',self.on_mouse_press)
        fig.canvas.mpl_connect('motion_notify_event',self.on_mouse_move)
        fig.canvas.mpl_connect('button_release_event',self.on_mouse_release)
        self.figure_canvas = FigureCanvasTkAgg(fig,master=self.frame_plots)
        self.figure_canvas.draw()

        self.mouse_coords_str = StringVar(value = "Mouse Coordinates")
        self.mouse_coords = Label(self.frame_plotReadouts,textvariable=self.mouse_coords_str)
        self.mouse_coords.grid(row=0,column=0)
        self.crosshair = False
        self.figure_canvas.get_tk_widget().pack(expand=True,fill='both')

    def change_axes_popup(self):
        """Creates a popup window that allows user to edit axes limits. This function is triggered
            by the menu buttons"""
        # compute middle of window location to place popup into
        win_size,win_left,win_top = self.winfo_geometry().split("+")
        win_wid,win_hei = win_size.split("x")
        win_left,win_top,win_wid,win_hei = int(win_left),int(win_top),int(win_wid),int(win_hei)
        pop_wid,pop_hei = 200,100 # hardcoded popup size
        pop_left,pop_top = int(win_left + win_wid/2 - pop_wid/2), int(win_top + win_hei/2 - pop_hei/2)

        popup = Toplevel(self)
        popup.resizable(False,False)
        popup.geometry(f"{pop_wid}x{pop_hei}+{pop_left}+{pop_top}")
        popup.title("Axes Limits")
        popup.focus_set()

        def close_popup(self): # internal function for closing this popup and resetting focus
            self.focus_set()
            popup.destroy()
        popup.protocol("WM_DELETE_WINDOW",func=lambda:close_popup(self))

        labels_w_units = [self.data_info["xlabel_unit"],*self.data_info["ylabels_units"]]
        limstrs = [self.data_info["xlim_strvar"],*self.data_info["ylim_strvars"]]

        for row,(label,limstr) in enumerate(zip(labels_w_units,limstrs)):
            limlabel = Label(popup,text=label,justify='right')
            limlabel.grid(row=row,column=0) 
            limentry = Entry(popup,textvariable=limstr,width = max(6,len(limstr.get())))
            limentry.grid(row=row,column=1)

        change_button = Button(master=popup,command = lambda:[self.update_axes_lims(),close_popup(self)],
                               text = "Change")
        change_button.grid()

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
    def on_mouse_release(self,event):
        """Handles release of mouse button. Calls click function if mouse has not been dragged."""
        if self.mouse_press and not self.mouse_drag: # simple click event
            self.on_mouse_click(event)
        # currently no functions operate at the end of drag
        # else: pass # drag event
        self.mouse_press,self.mouse_drag = False, False

    def on_mouse_press(self,event): 
        """Handles mouse press event, sets internal mouse state"""
        self.mouse_press = True

    def on_mouse_click(self,event):
        """Called when mouse click is determined, adds new point if it is valid"""
        default_y_vals = [float(self.data_info["ylim_strvars"][i].get())/2 for i in range(self.num_dep_vars)]
        # if mouse click points are not None
        if event.xdata and event.ydata: 
            # go through each subplot
            for j,(plot,default_y_val) in enumerate(zip(self.plots,default_y_vals)):
                # checks if mouse is inside subplot and it is the first point or next in time
                if event.inaxes == plot and (len(self.x_list) <1 or event.xdata > self.x_list[-1]):  
                    # we want to update lists with either mouse data or a default value for the plot
                    # which was not clicked on        
                    for i,value in enumerate([event.xdata,*default_y_val]):
                        if i == j+1: value = event.ydata
                        self.update_list(len(self.x_list),i,value)
                    
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
                    xvalue = format(event.xdata,"."+str(self.data_info["xround"])+"f")
                    yvalue = format(event.ydata,"."+str(self.data_info["yrounds"][plot_idx])+"f")
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

    def get_distance(self,pt1,pt2,plot_idx):
        lims = (self.plots[plot_idx].get_xlim()[1],self.plots[plot_idx].get_ylim()[1])
        return sqrt(sum([((pt1[i] - pt2[i])/lims[i])**2 for i in range(2)]))

# ----------------------
# Table related functions

    def update_str_vars(self):
        """Updates StringVar values for the table. Used when points are dragged on plot"""
        rounding = [self.data_info["xround"],*self.data_info["yrounds"]]
        for i,vallist in enumerate([self.x_list,*self.ys_list]):
            for j,val in enumerate(vallist):
                val = format(val,"."+str(rounding[i])+"f")
                self.tablestrvars[i][j].set(val)

    def update_list(self,row,col,txt=None,delete=False):
        """Updates internal data lists based on row,col values. col corresponds 
            to dependent/independent variable. row corresponds to point number."""
        try:
            txt = float(txt)
        except (ValueError,TypeError):
            return
        datalists = [self.x_list,*self.ys_list]
        if row == len(self.x_list):
            datalists[col].append(txt)
        elif txt:
            datalists[col][row] = txt
        elif delete:
            if col==0: self.x_list.pop(row)
            else: self.ys_list[col].pop(row)

    def delete_point(self,row):
        self.x_list.pop(row)
        self.ys_list[0].pop(row)
        self.ys_list[1].pop(row)
        self.redraw_plot()
        self.update_table(overwrite=True)

    def update_table(self,overwrite = False):
        if overwrite and len(self.tableelems) > 0:
            for item in self.tableelems:
                item.destroy()
            self.tableelems = []
            self.tablestrvars = [[] for i in range(len(self.x_list))]

        def drawrow(row,zipobj):
            ptlabel = Label(self.frame_table,text = row+1)
            ptlabel.grid(row = row+1,column = 0)
            delsym = Button(self.frame_table,text="X")
            delsym.bind("<Button-1>",lambda e, i=row:self.delete_point(i))
            delsym.grid(row=row+1,column=4)
            self.tableelems.append(delsym)
            self.tableelems.append(ptlabel)
            for col,val in enumerate(zipobj):
                val = format(val,"."+str(rounding[col])+"f")
                etxt = StringVar(value=val)
                self.tablestrvars[col].append(etxt)
                entry = Entry(self.frame_table,width=self.colwids[col],textvariable=etxt)
                entry.grid(row=row+1,column=col+1)
                entry.bind("<KeyRelease>",lambda e,row=row,col=col,etxt=etxt: 
                        [self.update_list(row,col,etxt.get()),self.redraw_plot()] )
                self.tableelems.append(entry)

        rounding = [self.data_info["xround"],*self.data_info["yrounds"]]
        x_list,ys_list = self.x_list, self.ys_list
        if not overwrite: 
            x_list = self.x_list[-1]
            ys_list = self.ys_list[-1]
            drawrow(len(self.x_list)-1,(x_list,*ys_list))
        if overwrite: 
            for row,zipobj in enumerate(zip(x_list,*ys_list)):
                drawrow(row,zipobj)
        

    def create_table(self):
        self.colwids = []
        self.tablestrvars = [] # list used to hold StringVars 
        self.tableelems = [] # list used to hold graphical table elements, can be used to modify them
        labels_w_units = [self.data_info["xlabel_unit"],*self.data_info["ylabels_units"]]
        for col,label in enumerate(labels_w_units):
            header = Label(self.frame_table,text=label)
            header.grid(row = 0,column = col+1)
            self.colwids.append(len(label))
            self.tablestrvars.append([])

        self.update_table()

# ----------------------
# Menu related functions

    def create_menu(self):
        structure = {"File":["New File",None,"Open",None,"Save",None,None,None,"Exit",None],
                    "Edit":["Axes Limits",self.change_axes_popup,"Copy",None,"Paste",None,"Select All",None,None,None],
                    "Help":["Demo",None,"About",None]}
        menubar = Menu(self)
        for key,value in structure.items():
            tab = Menu(menubar,tearoff=False)
            menubar.add_cascade(label=key,menu = tab)
            i = 0
            while i < len(value):
                if not value[i]:
                    tab.add_separator()
                else:
                    tab.add_command(label=value[i],command = value[i+1])
                i += 2
        self.config(menu=menubar)
    

if __name__ == "__main__":
    app = myapp()
    app.mainloop()