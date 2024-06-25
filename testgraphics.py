import os
from tkinter import Tk,Canvas,Frame,Scrollbar,Button, Entry, Label,StringVar,Menu,Toplevel
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.pyplot import xlabel
from graphicsDefs import interactivePlots, Table

def close():
    print("Closing")
    last_geometry = window.winfo_geometry()
    window.destroy()
    with open("windowlocation.txt","w") as fp:
        fp.write(last_geometry)

def upd():
    window.update()

def createMenu(window):
    menubar = Menu(window)
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
    window.config(menu=menubar)

table_data = {"Time (min)":[0],"Altitude (ft)":[0],"Mach Number":[0.2]}

table_data = {"x1":{"Label":"Time (min)","Values":[0]},
              "y1":{"Label":"Altitude (ft)","Values":[0]},
              "y2":{"Label":"Mach Number","Values":[0.2]}}
phases = {"Mach":[],"Altitude":[]}
window = Tk()
window.title('Mission Design Utility')
window.protocol("WM_DELETE_WINDOW",close)
window.focus_set()

# if we want to reuse the user's resized/moved window
window_geometry = "900x500+10+10" # widthxheight+x+y, x,y are location
if os.path.exists("windowlocation.txt"):
    with open("windowlocation.txt","r") as fp:
        window_geometry = fp.read().split("\n")[0]
window.geometry(window_geometry)
window.minsize(900,500)

def newfunc(): return
def openfunc(): return
def savefunc(): return
def exitfunc(): return

tablecv = Canvas(window)
tablecv.pack(side="right",fill="y")

scv = Canvas(window)
scv.pack(side='right',fill='y')

t = Table(tablecv,table_data)

scroll = Scrollbar(scv)
scroll.pack(side='right',fill='y')

canvas_plotReadouts = Canvas(window)
canvas_plotReadouts.pack(side='bottom')
canvas_plots = Canvas(window)
canvas_plots.pack(side='top')
fg = interactivePlots(window,canvas_plots,canvas_plotReadouts,
                      ["Time (minutes)","Altitude (ft)","Time (minutes)","Mach Number"],
                      [400,50e3,1.0],["Altitude Plot","Mach Plot"])

structure = {"File":["New File",newfunc,"Open",openfunc,"Save",savefunc,None,None,"Exit",exitfunc],
             "Edit":["Axes Limits",fg.change_axes_popup,"Copy",None,"Paste",None,"Select All",None,None,None],
             "Help":["Demo",None,"About",None]}
createMenu(window)

window.mainloop()

#window.state('zoomed')   #zooms the screen to maxm whenever executed
#plot_button = Button(master = window,command = plot, height = 2, width = 10, text = "Plot")
#plot_button.pack()

# toolbar = NavigationToolbar2Tk(canvas,window)
# toolbar.update()
# canvas.get_tk_widget().pack()

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