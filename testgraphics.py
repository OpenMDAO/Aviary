# #from graphics import *
# #import sv_ttk
# import tkinter as tk
# from graphics import GraphWin,EasyPlot,Text,Point,Line

# w,h = 900,500
# win = GraphWin("Draw Mission", w, h,savewindowlocation=True)
# win.setBackground("white")
# #sv_ttk.set_theme("light")

# plt1 = EasyPlot(win,(0.65,0.45),(0.01,0))
# plt2 = EasyPlot(win,(0.65,0.4),(0.01,0.5))
# plt1.drawaxes(labels=["Time (minutes)","Altitude (x1000 ft)"],coords=(0,0,400,50),fontsizes=(11,10))
# plt2.drawaxes(labels=["Time (minutes)","Mach Number"],coords=(0,0,400,1.0),fontsizes=(11,10))
# # def hi():
# #     print("button press")

# # bb = tk.Button(win.root,text="Ok",command=hi)
# # bb.place(x=w/2,y=h/2)

# def newfunc(): print("new")
# def openfunc(): print("open")
# def savefunc(): print("save")
# def exitfunc(): win.close()

# structure = {"File":["New File",newfunc,"Open",openfunc,"Save",savefunc,None,None,"Exit",exitfunc],
#              "Edit":["Cut",None,"Copy",None,"Paste",None,"Select All",None,None,None],
#              "Help":["Demo",None,"About",None]}

# win.createMenubar(structure=structure)

# T = Text(Point(0.5*w,0.47*h),"")
# T.draw(win)

# while True:
#     if win.isClosed(): break
#     clk = win.checkMouseClick()
#     if clk != None:
#         print("Click!")
#         break
#     pt = win.checkMouse()
    
#     plt1.crossHair(pt)
#     #plt2.crossHair(pt)
    
#     coords1 = plt1.getCoord(pt)
#     coords2 = plt2.getCoord(pt)
        
#     if coords1:
#         T.setText(f"Time: {round(coords1[0])} minutes, Altitude: {round(coords1[1]*1000,-2)} ft")
        
# if win.isOpen(): win.close()    # Close window when done

# # current progress:
# # class for creating generic plot with ticks, labels, numbers, 
# # functions for crosshair, getting coordinate value
# #
# #

import os
from tkinter import Tk,Canvas,Frame,Scrollbar,Button, Entry, Label,StringVar
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

def mouse(e):
    print(e.xdata,e.ydata)
def move(e):
    if e.xdata and e.ydata:
        print(e.xdata,e.ydata)

def plot():
    fig = Figure(figsize = (5, 5), dpi = 100)        
    y = [i**2 for i in range(101)]
    plot1 = fig.add_subplot(111)
    plot1.plot(y)
    fig.canvas.mpl_connect('button_press_event',mouse)
    fig.canvas.mpl_connect('motion_notify_event',move)

    canvas = FigureCanvasTkAgg(fig,master = window)
    canvas.draw()
    canvas.get_tk_widget().pack()
    toolbar = NavigationToolbar2Tk(canvas,window)
    toolbar.update()
    canvas.get_tk_widget().pack()

def close():
    print("Closing")
    last_geometry = window.winfo_geometry()
    window.destroy()
    with open("windowlocation.txt","w") as fp:
        fp.write(last_geometry)

class Table():
    def __init__(self,window,table):
        self.cells = []
        numrows = max(10,len(list(table.values())[0]))
        for i in range(numrows):
            if i>0:
                ptlabel = Label(window,text=i)
                ptlabel.grid(row=i,column=0)
            if i==0:
                pth = Label(window,text="Pt")
                pth.grid(row=0,column=0)
            for j,column in enumerate(table.keys()):
                if i==0:
                    header = Label(window,text=column)
                    header.grid(row=i,column=j+1)
                else:
                    print(len(column))
                    etxt = StringVar(value="0")
                    entry = Entry(window,width=len(column),textvariable=etxt)
                    entry.grid(row=i,column=j+1)
                    entry.bind("<KeyRelease>",lambda e, i=i,
                            var=etxt: self.updatePoint(i, var.get()))
                    self.cells.append(entry)
    
    def updatePoint(self,idx,val):
        print(idx,val)

table_data = {"Time (min)":[],"Altitude (ft)":[],"Mach Number":[]}
phases = {"Mach":[],"Altitude":[]}
window = Tk()
window.title('Plotting in Tkinter')
window.protocol("WM_DELETE_WINDOW",close)

# if we want to reuse the user's resized/moved window
window_geometry = "900x500+10+10" # widthxheight+x+y, x,y are location
if os.path.exists("windowlocation.txt"):
    with open("windowlocation.txt","r") as fp:
        window_geometry = fp.read().split("\n")[0]
window.geometry(window_geometry)

tablecv = Canvas(window)
tablecv.pack(side="right",fill="y")
scv = Canvas(window)
scv.pack(side='right',fill='y')

tablecv.create_text(10,10,text="hi")
t = Table(tablecv,table_data)

scroll = Scrollbar(scv)
scroll.pack(side='left',fill='y')

cv = Canvas(window,width=100,height=100)
cv.create_text(50,50,text="Hi")
cv.pack()

plot_button = Button(master = window,command = plot, height = 2, width = 10, text = "Plot")
plot_button.pack()
window.mainloop()


#window.state('zoomed')   #zooms the screen to maxm whenever executed
