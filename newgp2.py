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
        self.protocol("WM_DELETE_WINDOW",self.close)
        self.focus_set() # focus the window

        # if we want to reuse the user's resized/moved window
        window_geometry = "900x500+10+10" # widthxheight+x+y, x,y are location
        if os.path.exists("windowlocation.txt"):
            with open("windowlocation.txt","r") as fp:
                window_geometry = fp.read().split("\n")[0]
        self.geometry(window_geometry)
        self.minsize(900,500) # force a minimum size for layout to look correct

        self.timelist = [0]
        self.varlist = [[0],[0.2]]

        self.tablecv = Frame(self)
        self.tablecv.pack(side="right",fill="y")

        scroll_frame = Frame(self)
        scroll_frame.pack(side='right',fill='y')
        scroll = Scrollbar(scroll_frame)
        scroll.pack(side='right',fill='y')

        self.mousedrag = False
        self.mousepress = False

        self.frame_plotReadouts = Frame(self)
        self.frame_plotReadouts.pack(side='bottom')
        self.frame_plots = Frame(self)
        self.frame_plots.pack(side='top',expand=True,fill='both')
        self.crosshair = False
        self.ptcontainer = 0.04
        self.plotlines = [None,None]

        self.data_info = {"xlabel":"Time","xlim":(0,400),"xunit":"min","xround":0,
                      "ylabels":["Altitude","Mach Number"],"ylims":[(0,50e3),(0,1.0)],
                      "yunits":["ft","unitless"],"yrounds":[0,2],
                      "plot_titles":["Altitude Plot","Mach Plot"]}
        self.check_data_info()
        self.data_info["xlabel_unit"] = self.data_info["xlabel"] + f" ({self.data_info['xunit']})"
        self.data_info["ylabels_units"] = [ self.data_info["ylabels"][i] + 
                            f" ({self.data_info['yunits'][i]})" for i in range(self.num_dep_vars)]

        self.createPlots()
        self.createTable()
        self.createMenu()

    def close(self):
        last_geometry = self.winfo_geometry()
        self.destroy()
        with open("windowlocation.txt","w") as fp:
            fp.write(last_geometry)

    def check_data_info(self):
        check_keys = ["ylabels","ylims","yunits"]
        self.num_dep_vars = len(self.data_info["plot_titles"])
        for key in check_keys:
            if self.num_dep_vars != len(self.data_info[key]):
                raise Exception("Check length of lists in data_info, mismatch detected.\n"+
                                f"Expected {self.num_dep_vars} dependent variables.")


# Plot related functions

    def createPlots(self):
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
        fig.canvas.mpl_connect('button_press_event',self.onpress)
        fig.canvas.mpl_connect('motion_notify_event',self.mousemove)
        fig.canvas.mpl_connect('button_release_event',self.onrelease)
        self.figure_canvas = FigureCanvasTkAgg(fig,master=self.frame_plots)
        self.figure_canvas.draw()

        self.mouse_coords_str = StringVar(value = "Mouse Coordinates")
        self.mouse_coords = Label(self.frame_plotReadouts,textvariable=self.mouse_coords_str)
        self.mouse_coords.grid(row=0,column=0)
        self.crosshair = False
        self.figure_canvas.get_tk_widget().pack(expand=True,fill='both')

    def change_axes_popup(self):
        win_size,win_left,win_top = self.winfo_geometry().split("+")
        win_wid,win_hei = win_size.split("x")
        win_left,win_top,win_wid,win_hei = int(win_left),int(win_top),int(win_wid),int(win_hei)
        pop_wid,pop_hei = 200,100
        pop_left,pop_top = int(win_left + win_wid/2 - pop_wid/2), int(win_top + win_hei/2 - pop_hei/2)

        pop = Toplevel(self)
        pop.resizable(False,False)
        pop.geometry(f"{pop_wid}x{pop_hei}+{pop_left}+{pop_top}")
        pop.title("Axes Limits")
        pop.focus_set()
        def closePopup(self):
            self.focus_set()
            pop.destroy()
        pop.protocol("WM_DELETE_WINDOW",func=lambda:closePopup(self))

        labels_w_units = [self.data_info["xlabel_unit"],*self.data_info["ylabels_units"]]
        limstrs = [self.data_info["xlim_strvar"],*self.data_info["ylim_strvars"]]

        for row,(label,limstr) in enumerate(zip(labels_w_units,limstrs)):
            limlabel = Label(pop,text=label,justify='right')
            limlabel.grid(row=row,column=0) 
            limentry = Entry(pop,textvariable=limstr,width = max(6,len(limstr.get())))
            limentry.grid(row=row,column=1)

        change_button = Button(master=pop,command = lambda:[self.update_axes_lims(),closePopup(self)],
                               text = "Change")
        change_button.grid()

    def update_axes_lims(self):
        for (plot,limstr) in zip(self.plots,self.data_info["ylim_strvars"]):
            plot.set_xlim(0,float(self.data_info["xlim_strvar"].get()))
            plot.set_ylim(0,float(limstr.get()))
        self.figure_canvas.draw()

    def redrawPlot(self):
        self.claplot()
        colors = ['bo-','mo-']
        for i,plot in enumerate(self.plots):
            self.plotlines[i] = plot.plot(self.timelist,self.varlist[i],colors[i],markersize=5)
        self.figure_canvas.draw()

    # can be used to clear plots when deleting points
    def claplot(self):
        for plot in self.plots:
            for line in plot.lines:
                if line == self.crossX or line ==self.crossY: continue
                line.remove()

    def onrelease(self,event):
        if self.mousepress and not self.mousedrag: # simple click event
            self.mouseclick(event)
        else: pass # drag event

        self.mousepress,self.mousedrag = False, False
    def onpress(self,event): 
        self.mousepress = True

    def mouseclick(self,event):
        if event.xdata and event.ydata:
            for j,(plot,deflt) in enumerate(zip(self.plots,[0.7,30000])):
                if event.inaxes == plot and (len(self.timelist) <1 or event.xdata > self.timelist[-1]):
                    newpt = (event.xdata,event.ydata,deflt) if j==0 else (event.xdata,deflt,event.ydata)
                    for i,item in enumerate(newpt):
                        self.updateList(len(self.timelist),i,item)
                    self.redrawPlot()
                    self.updateTable()

    def mousemove(self,event):
        if event.xdata and event.ydata:
            for p,plot in enumerate(self.plots):
                if event.inaxes == plot:
                    near = False
                    if self.crosshair:
                        self.crossX.remove()
                        self.crossY.remove()
                    for i,prevpt in enumerate(zip(self.timelist,self.varlist[p])):
                        dist = self.getDist((event.xdata,event.ydata),prevpt,p)
                        if dist < self.ptcontainer:
                            self.figure_canvas.set_cursor(4)
                            near = True
                            self.neari = i
                    if not near: self.figure_canvas.set_cursor(1)
                    self.crossX = plot.axhline(y=event.ydata,color='red')
                    self.crossY = plot.axvline(x=event.xdata,color='red')
                    self.figure_canvas.draw()
                    self.crosshair = True
                    xvalue = format(event.xdata,"."+str(self.data_info["xround"])+"f")
                    yvalue = format(event.ydata,"."+str(self.data_info["yrounds"][p])+"f")
                    self.mouse_coords_str.set(
                        f"{self.data_info['xlabel']}: {xvalue} {self.data_info['xunit']} | "+
                        f"{self.data_info['ylabels'][p]}: {yvalue} {self.data_info['yunits'][p]}")

                    if self.mousepress and (near or self.mousedrag): 
                        self.mousedrag = True
                        self.updateList(self.neari,0,event.xdata)
                        self.updateList(self.neari,p+1,event.ydata)
                        self.redrawPlot()
                        self.updatestrvars()

    def getDist(self,pt1,pt2,plot_idx):
        lims = (self.plots[plot_idx].get_xlim()[1],self.plots[plot_idx].get_ylim()[1])
        return sqrt(sum([((pt1[i] - pt2[i])/lims[i])**2 for i in range(2)]))

# Table related functions

# much better for updating existing points
    def updatestrvars(self):
        rounding = [self.data_info["xround"],*self.data_info["yrounds"]]
        for i,vallist in enumerate([self.timelist,*self.varlist]):
            for j,val in enumerate(vallist):
                val = format(val,"."+str(rounding[i])+"f")
                self.tablestrvars[i][j].set(val)

    def updateList(self,row,col,txt=None,delete=False):
        try:
            txt = float(txt)
        except (ValueError,TypeError):
            return
        datalists = [self.timelist,*self.varlist]
        if row == len(self.timelist):
            datalists[col].append(txt)
        elif txt:
            datalists[col][row] = txt
        elif delete:
            if col==0: self.timelist.pop(row)
            else: self.varlist[col].pop(row)

    def deletePoint(self,row):
        self.timelist.pop(row)
        self.varlist[0].pop(row)
        self.varlist[1].pop(row)
        self.redrawPlot()
        self.updateTable(overwrite=True)

    def updateTable(self,overwrite = False):
        if overwrite and len(self.tableelems) > 0:
            for item in self.tableelems:
                item.destroy()
            self.tableelems = []
            self.tablestrvars = [[] for i in range(len(self.tablestrvars))]

        rounding = [self.data_info["xround"],*self.data_info["yrounds"]]
        timelist,varlist = self.timelist, self.varlist
        if not rounding: 
            timelist = self.timelist[-1]
            varlist = self.varlist[-1]
        for row,zipobj in enumerate(zip(timelist,*varlist)):
            ptlabel = Label(self.tablecv,text = row+1)
            ptlabel.grid(row = row+1,column = 0)
            delsym = Button(self.tablecv,text="X")
            delsym.bind("<Button-1>",lambda e, i=row:self.deletePoint(i))
            delsym.grid(row=row+1,column=4)
            self.tableelems.append(delsym)
            self.tableelems.append(ptlabel)
            for col,val in enumerate(zipobj):
                val = format(val,"."+str(rounding[col])+"f")
                etxt = StringVar(value=val)
                self.tablestrvars[col].append(etxt)
                entry = Entry(self.tablecv,width=self.colwids[col],textvariable=etxt)
                entry.grid(row=row+1,column=col+1)
                entry.bind("<KeyRelease>",lambda e,row=row,col=col,etxt=etxt: 
                        [self.updateList(row,col,etxt.get()),self.redrawPlot()] )
                self.tableelems.append(entry)

    def createTable(self):
        self.colwids = []
        self.tablestrvars = [] # list used to hold StringVars 
        self.tableelems = [] # list used to hold graphical table elements, can be used to modify them
        labels_w_units = [self.data_info["xlabel_unit"],*self.data_info["ylabels_units"]]
        for col,label in enumerate(labels_w_units):
            header = Label(self.tablecv,text=label)
            header.grid(row = 0,column = col+1)
            self.colwids.append(len(label))
            self.tablestrvars.append([])

        self.updateTable()


# Menu related functions

    def createMenu(self):
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