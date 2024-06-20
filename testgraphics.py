from graphics import *
w,h = 900,500
pix_per_pt = 16/12
win = GraphWin("Draw Mission", w, h,savewindowlocation=True)

def createPlot(win,size,location,labels = ["X Axis","Y Axis"],textsize = 12):
    width = size[0]*w
    height = size[1]*h
    left,top = location[0]*w, location[1]*h
    right,bottom = left+width, top+height
    pt1 = Point(left,top)
    pt2 = Point(right,bottom)
    plt = Rectangle(pt1,pt2)
    plt.setFill("red")
    plt.draw(win)

    textpix = textsize*pix_per_pt
    xlabel = Text(Point(left + width/2,bottom-textpix/2),labels[0])
    ylabel = Text(Point(left + textpix/2,top+height/2),labels[1],angle=90)
    xlabel.setSize(12)
    ylabel.setSize(12)
    xlabel.draw(win)
    ylabel.draw(win)

    ticksize = 0.03*h
    xticks = 20
    yticks = 10

    gridorigin = (left+textpix+5+ticksize/2, bottom - textpix - 5 - ticksize/2)

    tickspacingX = int((right-gridorigin[0])/xticks)
    tickspacingY = int((gridorigin[1]-top)/yticks)
 
    hticks_y = bottom - textpix - 5
    vticks_x = left + textpix + 5
    for i in range(xticks):
        xpt = i*tickspacingX + gridorigin[0]
        tick = Line(Point(xpt,hticks_y),Point(xpt,hticks_y-ticksize))
        tick.draw(win)
    for i in range(yticks):
        ypt = gridorigin[1] - i*tickspacingY
        tick = Line(Point(vticks_x+ticksize,ypt),Point(vticks_x,ypt))
        tick.draw(win)

    xaxis = Line(Point(gridorigin[0],gridorigin[1]),Point(right,gridorigin[1]))
    yaxis = Line(Point(gridorigin[0],top),Point(gridorigin[0],gridorigin[1]))
    xaxis.draw(win)
    yaxis.draw(win)

createPlot(win,(0.55,0.45),(0,0),["Time (minutes)","Altitude (ft)"])
createPlot(win,(0.55,0.45),(0,0.5),["Time (minutes)","Mach Number"])

T = Text(Point(0.5*w,0.47*h),"")
T.draw(win)

c = Circle(Point(50,50), 10)
c.draw(win)
win.checkMouseClick()
done = False
ln = None
ln2 = None
while True:
    clk = win.checkMouseClick()
    if clk != None:
        break
    pt = win.checkMouse()
    if pt:
        if ln: ln.undraw()
        if ln2: ln2.undraw()
        ln2 = Line(Point(pt.getX(),0),Point(pt.getX(),h))
        ln = Line(Point(0,pt.getY()),Point(w,pt.getY()))
        ln.draw(win)
        ln2.draw(win)
        T.setText(f"Time: {pt.getX()} minutes, Altitude: {pt.getY()} ft")
        #print(pt)

win.close()    # Close window when done