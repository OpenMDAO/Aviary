import os
from tkinter import Tk,Canvas,Frame,Scrollbar,Button, Entry, Label,StringVar,Menu,Toplevel
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

class interactiveFigure():
    def __init__(self,window,parent,readout,labels,lims,titles):
        fig = Figure()
        self.window = window
        self.parent = parent
        self.altplot = fig.add_subplot(2,1,1)
        self.machplot = fig.add_subplot(2,1,2)
        self.lims = lims
        for idx,plot in enumerate([self.altplot,self.machplot]):
            plot.set_xlabel(labels[idx*2])
            plot.set_ylabel(labels[idx*2+1])
            plot.set_xlim(0,lims[0])
            plot.set_ylim(0,lims[idx+1])
            plot.set_title(titles[idx])
            plot.grid(True)
        
        fig.tight_layout(h_pad=2.0)
        fig.canvas.mpl_connect('button_press_event',self.mouseclick)
        fig.canvas.mpl_connect('motion_notify_event',self.mousemove)
        self.canvas = FigureCanvasTkAgg(fig,master=parent)
        self.canvas.draw()

        self.mouse_coords_str = StringVar(value = "Alt: T:")
        self.mouse_coords = Label(readout,textvariable=self.mouse_coords_str)
        self.mouse_coords.grid(row=0,column=0)
        self.crosshair = False
        self.canvas.get_tk_widget().pack()

    def change_axes_popup(self):
        win_size,win_left,win_top = self.window.winfo_geometry().split("+")
        win_wid,win_hei = win_size.split("x")
        win_left,win_top,win_wid,win_hei = int(win_left),int(win_top),int(win_wid),int(win_hei)
        pop_wid,pop_hei = 200,100
        pop_left,pop_top = int(win_left + win_wid/2 - pop_wid/2), int(win_top + win_hei/2 - pop_hei/2)

        pop = Toplevel(self.window)
        pop.resizable(False,False)
        pop.geometry(f"{pop_wid}x{pop_hei}+{pop_left}+{pop_top}")
        pop.title("Axes Limits")
        pop.focus_set()
        def closePopup(self):
            self.window.focus_set()
            pop.destroy()
        pop.protocol("WM_DELETE_WINDOW",func=lambda:closePopup(self))

        limlabels= ["Max Time:","Max Altitude:","Max Mach"]
        try:
            limvals = [x.get() for x in self.limstrs]
        except:
            limvals = [str(int(x)) for x in self.lims]
            self.limstrs = [None,None,None]
            
        for i in range(3):
            limlabel = Label(pop,text=limlabels[i])
            limlabel.grid(row=i,column=0)
            self.limstrs[i] = (StringVar(value=limvals[i]))
            limentry = Entry(pop,textvariable=self.limstrs[i],width = 6)
            limentry.grid(row=i,column=1)

        change_button = Button(master=pop,command = lambda:[self.update_axes_lims(),closePopup(self)],
                               text = "Change")
        change_button.grid()

    def update_axes_lims(self):
        changed = False
        funcs = [self.altplot.set_xlim,self.machplot.set_xlim,self.altplot.set_ylim,self.machplot.set_ylim]
        for i,obj in enumerate(zip(self.limstrs,self.lims)):
            lim,limstr = obj[1],obj[0]
            limval = float(limstr.get())
            if lim != limval:
                changed = True
                if i == 0:
                    funcs[0](0,limval)
                    funcs[1](0,limval)
                else:
                    funcs[i+1](0,limval)

        if changed:
            self.canvas.draw()

    def mouseclick(self,event):
        if event.xdata and event.ydata:
            if event.inaxes == self.altplot:
                self.drawPoint(event.xdata,event.ydata)

    def mousemove(self,event):
        if event.xdata and event.ydata:
            if event.inaxes == self.altplot:
                if self.crosshair:
                    self.crossX.remove()
                    self.crossY.remove()
                self.crossX = self.altplot.axhline(y=event.ydata,color='red')
                self.crossY = self.altplot.axvline(x=event.xdata,color='red')
                self.canvas.draw()
                self.crosshair = True
                self.mouse_coords_str.set(f"Altitude: {int(event.ydata)} ft, Time: {int(event.xdata)} min")
            # elif event.inaxes == self.machplot:
            #     print(f"Mach: {event.xdata,event.ydata}")
    def drawPoint(self,x,y):
        self.altplot.plot(x,y,'o',markersize=5,color='blue')
        self.machplot.plot(x,0.7,'o',markersize=5,color='blue')
        self.canvas.draw()
       
class interactiveTable:
    def __init__(self,window,table):
        self.objs = []
        self.table = table
        prefill = len(list(table.values())[0])
        numrows = max(10,prefill)+1
        self.keys = list(table.keys())

        for i in range(numrows):
            rowl = []
            
            for j,data in enumerate(table.values()):
                if i==0:
                    headertxt = data["Label"]
                    header = Label(window,text=headertxt)
                    header.grid(row = i,column = j+1)
                else:
                    val = data["Values"][i-1] if i < prefill else 0
                    etxt = StringVar(value=val)
                    entry = Entry(window,width=len(headertxt),textvariable=etxt)
                    entry.grid(row=i,column=j+1)
                    entry.bind("<KeyRelease>",lambda e,i=i,j=j,etxt=etxt: 
                               self.updatePoint(i-1,j,etxt.get()) )
                    rowl.append(entry)
            if i > 0:
                ptlabel = Label(window,text = i)
                ptlabel.grid(row = i,column = 0)
                delsym = Button(window,text="X")
                delsym.bind("<Button-1>",lambda e, i=i:self.deletePoint(i))
                delsym.grid(row=i,column=4)
                rowl.append(ptlabel)
                rowl.append(delsym)
                self.objs.append(rowl) 

    def deletePoint(self,row):
        print(row)
        for i in range(5):
            self.objs[row][i].destroy()
        
    def updatePoint(self,row,col,val):
        self.table[self.keys[col]]["Values"][row] = val

class crazypoints:
    def __init__(self,table,interactiveplots):
        self.table = table
        self.iplots = interactiveplots