import os
from tkinter import Tk,Canvas,Frame,Scrollbar,Button, Entry, Label,StringVar,Menu,Toplevel
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        self.tableround = [0,0,2]
        self.tableelems = []
        self.tablestrvars = [[],[],[]]

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
        self.ptcontainerx = 0.03
        self.ptcontainery = 0.05
        self.plotlines = [None,None]
        plots_info = [{"xlabel":"Time (minutes)",
                       "ylabel":"Altitude (ft)",
                       "xlim":(0,400),"ylim":(0,50e3),
                       "title":"Altitude Plot"},

                      {"xlabel":"Time (minutes)",
                       "ylabel":"Mach Number",
                       "xlim":(0,400),"ylim":(0,1.0),
                       "title":"Mach Plot"}]
        self.createPlots(plots_info)
        self.createTable()
        self.createMenu()

    def close(self):
        print("Closing")
        last_geometry = self.winfo_geometry()
        self.destroy()
        with open("windowlocation.txt","w") as fp:
            fp.write(last_geometry)

# Plot related functions

    def createPlots(self,plots_info):
        fig = Figure()
        self.plots = []
        numplots = len(plots_info)
        for idx,plot in enumerate(plots_info):
            self.plots.append(fig.add_subplot(numplots,1,idx+1))
            self.plots[idx].set_xlabel(plot["xlabel"])
            self.plots[idx].set_ylabel(plot["ylabel"])
            self.plots[idx].set_xlim(plot["xlim"])
            self.plots[idx].set_ylim(plot["ylim"])
            self.plots[idx].set_title(plot["title"])
            self.plots[idx].grid(True)
            plot["limstrvarx"] = StringVar(value=int(plot["xlim"][1]))
            plot["limstrvary"] = StringVar(value=int(plot["ylim"][1]))
        
        self.plots_info = plots_info

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

        prevlabel = []
        for plot_idx,plot in enumerate(self.plots_info):
            for axis_idx,axis in enumerate(["x","y"]):
                labeltxt = "Max "+plot[axis+"label"]+": "
                if labeltxt in prevlabel: continue
                limlabel = Label(pop,text=labeltxt,justify='right')
                limlabel.grid(row=plot_idx*2+axis_idx,column=0) # 2 is hardcoded b/c it depends on num of widgets
                limentry = Entry(pop,textvariable=plot["limstrvar"+axis],width = 6)
                limentry.grid(row=plot_idx*2+axis_idx,column=1)
                prevlabel.append(labeltxt)

        change_button = Button(master=pop,command = lambda:[self.update_axes_lims(),closePopup(self)],
                               text = "Change")
        change_button.grid()

    def update_axes_lims(self):
        prevaxes = []
        prevlim = []
        for (plotinfo,plot) in zip(self.plots_info,self.plots):
            for (axis,func) in zip(["x","y"],[plot.set_xlim,plot.set_ylim]):
                # check if same axes exist, in which use previously set value
                # this is to account for change axes popup excluding duplicate axes entries
                if plotinfo[axis+"label"] in prevaxes: 
                    idx = prevaxes.index(plotinfo[axis+"label"])
                    newlim = prevlim[idx]
                else: newlim = float(plotinfo["limstrvar"+axis].get())
                func(0,newlim)

                prevaxes.append(plotinfo[axis+"label"])
                prevlim.append(newlim)
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
                    self.appendTable()

    def mousemove(self,event):
        if event.xdata and event.ydata:
            for p,plot in enumerate(self.plots):
                if event.inaxes == plot:
                    near = False
                    if self.crosshair:
                        self.crossX.remove()
                        self.crossY.remove()
                    for i,(x,y) in enumerate(zip(self.timelist,self.varlist[p])):
                        if self.getProx(event.xdata,x,"x",p) < self.ptcontainerx and self.getProx(event.ydata,y,"y",p)<self.ptcontainery:
                            self.figure_canvas.set_cursor(4)
                            near = True
                            self.neari = i
                    if not near: self.figure_canvas.set_cursor(1)
                    self.crossX = plot.axhline(y=event.ydata,color='red')
                    self.crossY = plot.axvline(x=event.xdata,color='red')
                    self.figure_canvas.draw()
                    self.crosshair = True
                    ys = ["Altitude","Mach"]
                    un = ["ft",""]
                    yv = int(event.ydata) if p==0 else round(event.ydata,2)
                    self.mouse_coords_str.set(f"Time: {int(event.xdata)} min, {ys[p]}: {yv} {un[p]}")

                    if self.mousepress and (near or self.mousedrag): 
                        self.mousedrag = True
                        self.updateList(self.neari,0,event.xdata)
                        self.updateList(self.neari,p+1,event.ydata)
                        self.redrawPlot()
                        self.updatestrvars()

    def getProx(self,a,b,axis,plt):
        return abs(a-b)/self.plots_info[plt][axis+"lim"][1]

# Table related functions

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
        self.redrawTable()


    def appendTable(self):
        rownum = len(self.timelist)
        ptlabel = Label(self.tablecv,text = rownum)
        ptlabel.grid(row = rownum,column = 0)
        delsym = Button(self.tablecv,text="X")
        delsym.bind("<Button-1>",lambda e, i=rownum-1:self.deletePoint(i))
        delsym.grid(row=rownum,column=4)
        self.tableelems.append(delsym)
        self.tableelems.append(ptlabel)
        for i,newval in enumerate([self.timelist[-1],self.varlist[0][-1],self.varlist[1][-1]]):
            newval = round(newval,self.tableround[i]) if self.tableround[i] != 0 else int(newval)
            etxt = StringVar(value=newval)
            self.tablestrvars[i].append(etxt)
            entry = Entry(self.tablecv,width=self.colwids[i],textvariable=etxt)
            entry.grid(row=rownum,column=i+1)
            entry.bind("<KeyRelease>",lambda e,row=rownum-1,col=i,etxt=etxt: 
                        [self.updateList(row,col,etxt.get()),self.redrawPlot()] )
            self.tableelems.append(entry)
        
    # much better for updating existing points
    def updatestrvars(self):
        for i,vallist in enumerate([self.timelist,*self.varlist]):
            for j,val in enumerate(vallist):
                val = round(val,self.tableround[i]) if self.tableround[i] != 0 else int(val)
                if j >= len(self.tablestrvars[0]):
                    self.tablestrvars[i].append(val)
                else:
                    self.tablestrvars[i][j].set(val)


    # works but delete buttons spasms while dragging points
    def redrawTable(self):
        if len(self.tableelems) > 0:
            for item in self.tableelems:
                item.destroy()
            self.tableelems = []
            self.tablestrvars = [[],[],[]]
        for row,zipobj in enumerate(zip(self.timelist,*self.varlist)):
            row +=1
            ptlabel = Label(self.tablecv,text = row)
            ptlabel.grid(row = row,column = 0)
            delsym = Button(self.tablecv,text="X")
            delsym.bind("<Button-1>",lambda e, i=row-1:self.deletePoint(i))
            delsym.grid(row=row,column=4)
            self.tableelems.append(delsym)
            self.tableelems.append(ptlabel)
            for i,val in enumerate(zipobj):
                val = round(val,self.tableround[i]) if self.tableround[i] != 0 else int(val)
                etxt = StringVar(value=val)
                self.tablestrvars[i].append(etxt)
                entry = Entry(self.tablecv,width=self.colwids[i],textvariable=etxt)
                entry.grid(row=row,column=i+1)
                entry.bind("<KeyRelease>",lambda e,row=row-1,col=i,etxt=etxt: 
                            self.updateList(row,col,etxt.get())) 
                self.tableelems.append(entry)

    def createTable(self):
        prevheaders = []
        self.colwids = []
        col = 1
        for plot in self.plots_info:
            for axis in ["x","y"]:
                headertxt = plot[axis+"label"]
                if headertxt in prevheaders: continue
                
                header = Label(self.tablecv,text=headertxt)
                header.grid(row = 0,column = col)
                col += 1
                prevheaders.append(headertxt)
                self.colwids.append(len(headertxt))
        self.appendTable()


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