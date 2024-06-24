import time, os, math, itertools
try:  # import as appropriate for 2.x vs. 3.x
   import tkinter as tk
except:
   import Tkinter as tk


##########################################################################
# Module Exceptions

class GraphicsError(Exception):
    """Generic error class for graphics module exceptions."""
    pass

OBJ_ALREADY_DRAWN = "Object currently drawn"
UNSUPPORTED_METHOD = "Object doesn't support operation"
BAD_OPTION = "Illegal option value"

##########################################################################
# global variables and funtions

_root = tk.Tk()
_root.withdraw()

_update_lasttime = time.time()

def update(rate=None):
    global _update_lasttime
    if rate:
        now = time.time()
        pauseLength = 1/rate-(now-_update_lasttime)
        if pauseLength > 0:
            time.sleep(pauseLength)
            _update_lasttime = now + pauseLength
        else:
            _update_lasttime = now

    _root.update()

############################################################################
# Graphics classes start here
        
class GraphWin(tk.Canvas):

    """A GraphWin is a toplevel window for displaying graphics."""

    def __init__(self, title="Graphics Window",
                 width=200, height=200, windowleft = 100, windowtop = 100,autoflush=True,savewindowlocation = False,resizable = (False,False),cursor="arrow"):
        assert type(title) == type(""), "Title must be a string"
        master = tk.Toplevel(_root)
        master.grab_set()
        self.root = master
        master.protocol("WM_DELETE_WINDOW", self.close)
        tk.Canvas.__init__(self, master, width=width, height=height,
                           highlightthickness=0, bd=0,cursor=cursor)     

        # if we want to reuse the user's resized/moved window
        windowlocation = [width,height,windowleft,windowtop] # default window size/location
        if savewindowlocation and os.path.exists("windowlocation.txt"):
            with open("windowlocation.txt","r") as fp:
                temp = fp.read().split(',')[0:4]
                temp = [float(i) for i in temp]
                if len(temp) == 4:
                    windowlocation[2:] = temp[2:]
                    if resizable[0]: windowlocation[0] = temp[0] 
                    if resizable[1]: windowlocation[1] = temp[1] 
        master.geometry('%dx%d+%d+%d' % tuple(windowlocation))
        self.savewindowlocation = savewindowlocation

        self.master.title(title)
        self.pack()
        master.resizable(*resizable)
        self.foreground = "black"
        self.items = []
        self.mouseClickX = None
        self.mouseClickY = None
        self.mouseX = None
        self.mouseY = None
        self.bind("<Button-1>", self._onClick)
        self.bind('<Motion>',self._onMove)
        self.bind_all("<Key>", self._onKey)
        self.height = int(height)
        self.width = int(width)
        self.autoflush = autoflush
        self._mouseCallback = None
        self.trans = None
        self.closed = False
        master.lift()
        self.lastKey = ""
        if autoflush: _root.update()

    def __repr__(self):
        if self.isClosed():
            return "<Closed GraphWin>"
        else:
            return "GraphWin('{}', {}, {})".format(self.master.title(),
                                             self.getWidth(),
                                             self.getHeight())
    def getWinfo(self)->list[float]:
        winfo = self.root.winfo_geometry()
        return [float(i) for i in [winfo.split("x")[0],*winfo.split("x")[1].split("+")]]

    def saveloc(self)->None:
        winfo = self.getWinfo()
        with open("windowlocation.txt","w") as fp:
            for i in winfo:
                fp.write(str(i)+",")

    def __str__(self):
        return repr(self)
     
    def __checkOpen(self):
        if self.closed:
            raise GraphicsError("window is closed")

    def _onKey(self, evnt):
        self.lastKey = evnt.keysym

    def createMenubar(self,structure:dict)->None:
        menubar = tk.Menu(self.root)
        for key,value in structure.items():
            tab = tk.Menu(menubar,tearoff=False)
            menubar.add_cascade(label=key,menu = tab)
            i = 0
            while i < len(value):
                if not value[i]:
                    tab.add_separator()
                else:
                    tab.add_command(label=value[i],command = value[i+1])
                i += 2
        self.root.config(menu=menubar)

    def setBackground(self, color):
        """Set background color of the window"""
        self.__checkOpen()
        self.config(bg=color)
        self.__autoflush()
        
    def setCoords(self, x1, y1, x2, y2):
        """Set coordinates of window to run from (x1,y1) in the
        lower-left corner to (x2,y2) in the upper-right corner."""
        self.trans = Transform(self.width, self.height, x1, y1, x2, y2)
        self.redraw()

    def close(self):
        """Close the window"""

        if self.closed: return
        if self.savewindowlocation:
            self.saveloc()
        self.closed = True
        self.master.destroy()
        self.__autoflush()


    def isClosed(self):
        return self.closed


    def isOpen(self):
        return not self.closed


    def __autoflush(self):
        if self.autoflush:
            _root.update()

    
    def plot(self, x, y, color="black"):
        """Set pixel (x,y) to the given color"""
        self.__checkOpen()
        xs,ys = self.toScreen(x,y)
        self.create_line(xs,ys,xs+1,ys, fill=color)
        self.__autoflush()
        
    def plotPixel(self, x, y, color="black"):
        """Set pixel raw (independent of window coordinates) pixel
        (x,y) to color"""
        self.__checkOpen()
        self.create_line(x,y,x+1,y, fill=color)
        self.__autoflush()
      
    def flush(self):
        """Update drawing to the window"""
        self.__checkOpen()
        self.update_idletasks()

    def _onMove(self,e):
        self.mouseX = e.x
        self.mouseY = e.y  
    
    def getMouse(self):
        self.update()
        while self.mouseX == None or self.mouseY == None:
            if self.isClosed(): return None
            self.update()
            time.sleep(0.1)      
        x,y = self.toWorld(self.mouseX,self.mouseY)
        self.mouseX = None
        self.mouseY = None
        return Point(x,y)
    
    def checkMouse(self):
        if self.isClosed(): return None
        self.update()
        if self.mouseX and self.mouseY:
            x,y = self.toWorld(self.mouseX,self.mouseY)
            self.mouseX = None
            self.mouseY = None
            return Point(x,y)
        else: return None

    def getMouseClick(self):
        """Wait for mouse click and return Point object representing
        the click"""
        self.update()      # flush any prior clicks
        self.mouseClickX = None
        self.mouseClickY = None
        while self.mouseClickX == None or self.mouseClickY == None:
            self.update()
            if self.isClosed(): return None
            time.sleep(.1) # give up thread
        x,y = self.toWorld(self.mouseClickX, self.mouseClickY)
        self.mouseClickX = None
        self.mouseClickY = None
        return Point(x,y)

    def checkMouseClick(self):
        """Return last mouse click or None if mouse has
        not been clicked since last call"""
        if self.isClosed(): return None
        self.update()
        if self.mouseClickX != None and self.mouseClickY != None:
            x,y = self.toWorld(self.mouseClickX, self.mouseClickY)
            self.mouseClickX = None
            self.mouseClickY = None
            return Point(x,y)
        else:
            return None

    def getKey(self):
        """Wait for user to press a key and return it as a string."""
        self.lastKey = ""
        while self.lastKey == "":
            self.update()
            if self.isClosed(): raise GraphicsError("getKey in closed window")
            time.sleep(.1) # give up thread

        key = self.lastKey
        self.lastKey = ""
        return key

    def checkKey(self):
        """Return last key pressed or None if no key pressed since last call"""
        if self.isClosed():
            raise GraphicsError("checkKey in closed window")
        self.update()
        key = self.lastKey
        self.lastKey = ""
        return key
            
    def getHeight(self):
        """Return the height of the window"""
        return self.height
        
    def getWidth(self):
        """Return the width of the window"""
        return self.width
    
    def toScreen(self, x, y):
        trans = self.trans
        if trans:
            return self.trans.screen(x,y)
        else:
            return x,y
                      
    def toWorld(self, x, y):
        trans = self.trans
        if trans:
            return self.trans.world(x,y)
        else:
            return x,y
        
    def setMouseHandler(self, func):
        self._mouseCallback = func
        
    def _onClick(self, e):
        self.mouseClickX = e.x
        self.mouseClickY = e.y
        if self._mouseCallback:
            self._mouseCallback(Point(e.x, e.y))

    def addItem(self, item):
        self.items.append(item)

    def delItem(self, item):
        self.items.remove(item)

    def redraw(self):
        for item in self.items[:]:
            item.undraw()
            item.draw(self)
        self.update()
                          
class Transform:

    """Internal class for 2-D coordinate transformations"""
    
    def __init__(self, w, h, xlow, ylow, xhigh, yhigh):
        # w, h are width and height of window
        # (xlow,ylow) coordinates of lower-left [raw (0,h-1)]
        # (xhigh,yhigh) coordinates of upper-right [raw (w-1,0)]
        xspan = (xhigh-xlow)
        yspan = (yhigh-ylow)
        self.xbase = xlow
        self.ybase = yhigh
        self.xscale = xspan/float(w-1)
        self.yscale = yspan/float(h-1)
        
    def screen(self,x,y):
        # Returns x,y in screen (actually window) coordinates
        xs = (x-self.xbase) / self.xscale
        ys = (self.ybase-y) / self.yscale
        return int(xs+0.5),int(ys+0.5)
        
    def world(self,xs,ys):
        # Returns xs,ys in world coordinates
        x = xs*self.xscale + self.xbase
        y = self.ybase - ys*self.yscale
        return x,y

# Default values for various item configuration options. Only a subset of
#   keys may be present in the configuration dictionary for a given item
DEFAULT_CONFIG = {"fill":"",
      "outline":"black",
      "width":"1",
      "arrow":"none",
      "text":"",
      "justify":"center",
                  "font": ("helvetica", 12, "normal")}

class GraphicsObject:

    """Generic base class for all of the drawable objects"""
    # A subclass of GraphicsObject should override _draw and
    #   and _move methods.
    
    def __init__(self, options):
        # options is a list of strings indicating which options are
        # legal for this object.
        
        # When an object is drawn, canvas is set to the GraphWin(canvas)
        #    object where it is drawn and id is the TK identifier of the
        #    drawn shape.
        self.canvas = None
        self.id = None

        # config is the dictionary of configuration options for the widget.
        config = {}
        for option in options:
            config[option] = DEFAULT_CONFIG[option]
        self.config = config
        
    def setFill(self, color):
        """Set interior color to color"""
        self._reconfig("fill", color)
        
    def setOutline(self, color):
        """Set outline color to color"""
        self._reconfig("outline", color)
        
    def setWidth(self, width):
        """Set line weight to width"""
        self._reconfig("width", width)

    def draw(self, graphwin):

        """Draw the object in graphwin, which should be a GraphWin
        object.  A GraphicsObject may only be drawn into one
        window. Raises an error if attempt made to draw an object that
        is already visible."""

        if self.canvas and not self.canvas.isClosed(): raise GraphicsError(OBJ_ALREADY_DRAWN)
        if graphwin.isClosed(): raise GraphicsError("Can't draw to closed window")
        self.canvas = graphwin
        self.id = self._draw(graphwin, self.config)
        graphwin.addItem(self)
        if graphwin.autoflush:
            _root.update()
        return self

            
    def undraw(self):

        """Undraw the object (i.e. hide it). Returns silently if the
        object is not currently drawn."""
        
        if not self.canvas: return
        if not self.canvas.isClosed():
            self.canvas.delete(self.id)
            self.canvas.delItem(self)
            if self.canvas.autoflush:
                _root.update()
        self.canvas = None
        self.id = None


    def move(self, dx, dy):

        """move object dx units in x direction and dy units in y
        direction"""
        
        self._move(dx,dy)
        canvas = self.canvas
        if canvas and not canvas.isClosed():
            trans = canvas.trans
            if trans:
                x = dx/ trans.xscale 
                y = -dy / trans.yscale
            else:
                x = dx
                y = dy
            self.canvas.move(self.id, x, y)
            if canvas.autoflush:
                _root.update()
           
    def _reconfig(self, option, setting):
        # Internal method for changing configuration of the object
        # Raises an error if the option does not exist in the config
        #    dictionary for this object
        if option not in self.config:
            raise GraphicsError(UNSUPPORTED_METHOD)
        options = self.config
        options[option] = setting
        if self.canvas and not self.canvas.isClosed():
            self.canvas.itemconfig(self.id, options)
            if self.canvas.autoflush:
                _root.update()


    def _draw(self, canvas, options):
        """draws appropriate figure on canvas with options provided
        Returns Tk id of item drawn"""
        pass # must override in subclass


    def _move(self, dx, dy):
        """updates internal state of object to move it dx,dy units"""
        pass # must override in subclass
        
class Point(GraphicsObject):
    def __init__(self, x, y):
        GraphicsObject.__init__(self, ["outline", "fill"])
        self.setFill = self.setOutline
        self.x = float(x)
        self.y = float(y)

    def __repr__(self):
        return "Point({}, {})".format(self.x, self.y)
        
    def _draw(self, canvas, options):
        x,y = canvas.toScreen(self.x,self.y)
        return canvas.create_rectangle(x,y,x+1,y+1,options)
        
    def _move(self, dx, dy):
        self.x = self.x + dx
        self.y = self.y + dy
        
    def clone(self):
        other = Point(self.x,self.y)
        other.config = self.config.copy()
        return other
                
    def getX(self): return self.x
    def getY(self): return self.y

class _BBox(GraphicsObject):
    # Internal base class for objects represented by bounding box
    # (opposite corners) Line segment is a degenerate case.
    
    def __init__(self, p1, p2, options=["outline","width","fill"]):
        GraphicsObject.__init__(self, options)
        self.p1 = p1.clone()
        self.p2 = p2.clone()

    def _move(self, dx, dy):
        self.p1.x = self.p1.x + dx
        self.p1.y = self.p1.y + dy
        self.p2.x = self.p2.x + dx
        self.p2.y = self.p2.y  + dy
                
    def getP1(self): return self.p1.clone()

    def getP2(self): return self.p2.clone()
    
    def getCenter(self):
        p1 = self.p1
        p2 = self.p2
        return Point((p1.x+p2.x)/2.0, (p1.y+p2.y)/2.0)
  
class Rectangle(_BBox):
    
    def __init__(self, p1, p2):
        _BBox.__init__(self, p1, p2)

    def __repr__(self):
        return "Rectangle({}, {})".format(str(self.p1), str(self.p2))
    
    def _draw(self, canvas, options):
        p1 = self.p1
        p2 = self.p2
        x1,y1 = canvas.toScreen(p1.x,p1.y)
        x2,y2 = canvas.toScreen(p2.x,p2.y)
        return canvas.create_rectangle(x1,y1,x2,y2,options)
        
    def clone(self):
        other = Rectangle(self.p1, self.p2)
        other.config = self.config.copy()
        return other

class Oval(_BBox):
    
    def __init__(self, p1, p2):
        _BBox.__init__(self, p1, p2)

    def __repr__(self):
        return "Oval({}, {})".format(str(self.p1), str(self.p2))

        
    def clone(self):
        other = Oval(self.p1, self.p2)
        other.config = self.config.copy()
        return other
   
    def _draw(self, canvas, options):
        p1 = self.p1
        p2 = self.p2
        x1,y1 = canvas.toScreen(p1.x,p1.y)
        x2,y2 = canvas.toScreen(p2.x,p2.y)
        return canvas.create_oval(x1,y1,x2,y2,options)
    
class Circle(Oval):
    
    def __init__(self, center, radius):
        p1 = Point(center.x-radius, center.y-radius)
        p2 = Point(center.x+radius, center.y+radius)
        Oval.__init__(self, p1, p2)
        self.radius = radius

    def __repr__(self):
        return "Circle({}, {})".format(str(self.getCenter()), str(self.radius))
        
    def clone(self):
        other = Circle(self.getCenter(), self.radius)
        other.config = self.config.copy()
        return other
        
    def getRadius(self):
        return self.radius
                
class Line(_BBox):
    
    def __init__(self, p1, p2):
        _BBox.__init__(self, p1, p2, ["arrow","fill","width"])
        self.setFill(DEFAULT_CONFIG['outline'])
        self.setOutline = self.setFill

    def __repr__(self):
        return "Line({}, {})".format(str(self.p1), str(self.p2))

    def clone(self):
        other = Line(self.p1, self.p2)
        other.config = self.config.copy()
        return other
  
    def _draw(self, canvas, options):
        p1 = self.p1
        p2 = self.p2
        x1,y1 = canvas.toScreen(p1.x,p1.y)
        x2,y2 = canvas.toScreen(p2.x,p2.y)
        return canvas.create_line(x1,y1,x2,y2,options)
        
    def setArrow(self, option):
        if not option in ["first","last","both","none"]:
            raise GraphicsError(BAD_OPTION)
        self._reconfig("arrow", option)
        
class Polygon(GraphicsObject):
    
    def __init__(self, *points):
        # if points passed as a list, extract it
        if len(points) == 1 and type(points[0]) == type([]):
            points = points[0]
        self.points = list(map(Point.clone, points))
        GraphicsObject.__init__(self, ["outline", "width", "fill"])

    def __repr__(self):
        return "Polygon"+str(tuple(p for p in self.points))
        
    def clone(self):
        other = Polygon(*self.points)
        other.config = self.config.copy()
        return other

    def getPoints(self):
        return list(map(Point.clone, self.points))

    def _move(self, dx, dy):
        for p in self.points:
            p.move(dx,dy)
   
    def _draw(self, canvas, options):
        args = [canvas]
        for p in self.points:
            x,y = canvas.toScreen(p.x,p.y)
            args.append(x)
            args.append(y)
        args.append(options)
        return GraphWin.create_polygon(*args) 

class Text(GraphicsObject):
    
    def __init__(self, p, text,angle=0):
        GraphicsObject.__init__(self, ["justify","fill","text","font"])
        self.setText(text)
        self.anchor = p.clone()
        self.setFill(DEFAULT_CONFIG['outline'])
        self.setOutline = self.setFill
        self.angle = angle

    def __repr__(self):
        return "Text({}, '{}')".format(self.anchor, self.getText())
    
    def _draw(self, canvas, options):
        p = self.anchor
        x,y = canvas.toScreen(p.x,p.y)
        return canvas.create_text(x,y,options,angle=self.angle)
        
    def _move(self, dx, dy):
        self.anchor.move(dx,dy)
        
    def clone(self):
        other = Text(self.anchor, self.config['text'])
        other.config = self.config.copy()
        return other

    def setText(self,text):
        self._reconfig("text", text)
        
    def getText(self):
        return self.config["text"]
            
    def getAnchor(self):
        return self.anchor.clone()

    def setFace(self, face):
        if face in ['helvetica','arial','courier','times roman']:
            f,s,b = self.config['font']
            self._reconfig("font",(face,s,b))
        else:
            raise GraphicsError(BAD_OPTION)

    def setSize(self, size):
        if 5 <= size <= 36:
            f,s,b = self.config['font']
            self._reconfig("font", (f,size,b))
        else:
            raise GraphicsError(BAD_OPTION)

    def setStyle(self, style):
        if style in ['bold','normal','italic', 'bold italic']:
            f,s,b = self.config['font']
            self._reconfig("font", (f,s,style))
        else:
            raise GraphicsError(BAD_OPTION)

    def setTextColor(self, color):
        self.setFill(color)

class Entry(GraphicsObject):

    def __init__(self, p, width):
        GraphicsObject.__init__(self, [])
        self.anchor = p.clone()
        #print self.anchor
        self.width = width
        self.text = tk.StringVar(_root)
        self.text.set("")
        self.fill = "gray"
        self.color = "black"
        self.font = DEFAULT_CONFIG['font']
        self.entry = None

    def __repr__(self):
        return "Entry({}, {})".format(self.anchor, self.width)

    def _draw(self, canvas, options):
        p = self.anchor
        x,y = canvas.toScreen(p.x,p.y)
        frm = tk.Frame(canvas.master)
        self.entry = tk.Entry(frm,
                              width=self.width,
                              textvariable=self.text,
                              bg = self.fill,
                              fg = self.color,
                              font=self.font)
        self.entry.pack()
        #self.setFill(self.fill)
        self.entry.focus_set()
        return canvas.create_window(x,y,window=frm)

    def getText(self):
        return self.text.get()

    def _move(self, dx, dy):
        self.anchor.move(dx,dy)

    def getAnchor(self):
        return self.anchor.clone()

    def clone(self):
        other = Entry(self.anchor, self.width)
        other.config = self.config.copy()
        other.text = tk.StringVar()
        other.text.set(self.text.get())
        other.fill = self.fill
        return other

    def setText(self, t):
        self.text.set(t)

            
    def setFill(self, color):
        self.fill = color
        if self.entry:
            self.entry.config(bg=color)

            
    def _setFontComponent(self, which, value):
        font = list(self.font)
        font[which] = value
        self.font = tuple(font)
        if self.entry:
            self.entry.config(font=self.font)


    def setFace(self, face):
        if face in ['helvetica','arial','courier','times roman']:
            self._setFontComponent(0, face)
        else:
            raise GraphicsError(BAD_OPTION)

    def setSize(self, size):
        if 5 <= size <= 36:
            self._setFontComponent(1,size)
        else:
            raise GraphicsError(BAD_OPTION)

    def setStyle(self, style):
        if style in ['bold','normal','italic', 'bold italic']:
            self._setFontComponent(2,style)
        else:
            raise GraphicsError(BAD_OPTION)

    def setTextColor(self, color):
        self.color=color
        if self.entry:
            self.entry.config(fg=color)

class Image(GraphicsObject):

    idCount = 0
    imageCache = {} # tk photoimages go here to avoid GC while drawn 
    
    def __init__(self, p, *pixmap):
        GraphicsObject.__init__(self, [])
        self.anchor = p.clone()
        self.imageId = Image.idCount
        Image.idCount = Image.idCount + 1
        if len(pixmap) == 1: # file name provided
            self.img = tk.PhotoImage(file=pixmap[0], master=_root)
        else: # width and height provided
            width, height = pixmap
            self.img = tk.PhotoImage(master=_root, width=width, height=height)

    def __repr__(self):
        return "Image({}, {}, {})".format(self.anchor, self.getWidth(), self.getHeight())
                
    def _draw(self, canvas, options):
        p = self.anchor
        x,y = canvas.toScreen(p.x,p.y)
        self.imageCache[self.imageId] = self.img # save a reference  
        return canvas.create_image(x,y,image=self.img)
    
    def _move(self, dx, dy):
        self.anchor.move(dx,dy)
        
    def undraw(self):
        try:
            del self.imageCache[self.imageId]  # allow gc of tk photoimage
        except KeyError:
            pass
        GraphicsObject.undraw(self)

    def getAnchor(self):
        return self.anchor.clone()
        
    def clone(self):
        other = Image(Point(0,0), 0, 0)
        other.img = self.img.copy()
        other.anchor = self.anchor.clone()
        other.config = self.config.copy()
        return other

    def getWidth(self):
        """Returns the width of the image in pixels"""
        return self.img.width() 

    def getHeight(self):
        """Returns the height of the image in pixels"""
        return self.img.height()

    def getPixel(self, x, y):
        """Returns a list [r,g,b] with the RGB color values for pixel (x,y)
        r,g,b are in range(256)

        """
        
        value = self.img.get(x,y) 
        if type(value) ==  type(0):
            return [value, value, value]
        elif type(value) == type((0,0,0)):
            return list(value)
        else:
            return list(map(int, value.split())) 

    def setPixel(self, x, y, color):
        """Sets pixel (x,y) to the given color
        
        """
        self.img.put("{" + color +"}", (x, y))
        

    def save(self, filename):
        """Saves the pixmap image to filename.
        The format for the save image is determined from the filname extension.

        """
        
        path, name = os.path.split(filename)
        ext = name.split(".")[-1]
        self.img.write( filename, format=ext)

class EasyPlot(): # generalized class for creating plots
    def __init__(self,win,size,location=(0,0)):
        # size is given as a percent of window height/width
        # location is also a percent of window height/width for top left of plot
        self.win = win
        self.winwidth = self.win.width
        self.winheight = self.win.height
        self.width = size[0]*self.winwidth
        self.height = size[1]*self.winheight
        self.left,self.top = location[0]*self.winwidth,location[1]*self.winheight
        self.right,self.bottom = self.left + self.width, self.top+self.height
        self.bg = Rectangle(Point(self.left,self.top),Point(self.right,self.bottom))
        self.bg.setFill("white") # update to match theme settings
        self.bg.draw(self.win)
        self.crossX,self.crossY = None,None

    def drawaxes(self,coords=(0,0,100,100),labels=["X Axis","Y Axis"],fontsizes=(12,11)):
        # draws axes on the plot, with option to specify a coordinate system
        # also gives label option, along with font sizes
        # the function automatically adjusts axes location based on font sizes

        # compute pixel size for axes labels and numbers to ensure readability
        pix_per_pt = 16/12 # pixels per point (point is font size unit)
        labelsize = fontsizes[0]*pix_per_pt
        numsize = fontsizes[1]*pix_per_pt
        xnumwid = 0.7*pix_per_pt*fontsizes[1]*max(len(str(coords[0])),len(str(coords[2])))
        ynumwid = 0.7*pix_per_pt*fontsizes[1]*max(len(str(coords[1])),len(str(coords[3])))

        # getting magnitude of plot coordinates to determine rounding scheme
        # allows tick numbers to be nice even values
        # e.g., for x from 0 to 500, round each tick to nearest 10
        xexp, yexp = 0,0
        for i in range(4):
            if coords[i]!=0:
                val = find_exp(coords[i])
                if i%2==0:
                    xexp = val if abs(val) > xexp else xexp
                else:
                    yexp = val if abs(val) > yexp else yexp

        # draw X and Y axes labels
        xlabel = Text(Point(self.left+self.width/2,self.bottom-labelsize/2),labels[0])
        ylabel = Text(Point(self.left + labelsize/2,self.top+self.height/2),labels[1],angle=90)
        xlabel.setSize(fontsizes[0])
        xlabel.setFace("courier")
        ylabel.setSize(fontsizes[0])
        ylabel.setFace("courier")
        xlabel.draw(self.win)
        ylabel.draw(self.win)

        ticksize = 0.5*labelsize # tick size is a percent of label size
        margin = 2

        # determine origin location based on sizes of axes labels, numbers, ticks
        gridorigin = (self.left + labelsize + margin + ynumwid + margin + ticksize/2,
                      self.bottom - labelsize - margin - numsize - margin - ticksize/2)
        self.gridorigin = gridorigin
        
        # coordinate transformation variables (pixel coords vs. plot coords)
        self.gridwidthpix = self.right - gridorigin[0]
        self.gridheightpix = gridorigin[1] - self.top
        self.coordscalerX = (coords[2]-coords[0])/self.gridwidthpix # for x coords, there are p pixels
        self.coordscalerY = (coords[3]-coords[1])/self.gridheightpix
        self.coords = coords
        
        # this currently creates ticks to allow number labels to fit 
        # however this results in the coordinate based spacing using non even numbers
        # need to change to pick even spacings
        xticks = int(math.floor(self.gridwidthpix/xnumwid))
        yticks = int(math.floor(self.gridheightpix/numsize))
        tickspacingX = int((self.right - gridorigin[0])/xticks)
        tickspacingY = int((gridorigin[1]-self.top)/yticks)

        # x, y points for horizontal and vertical ticks
        xticks_y = gridorigin[1] + ticksize/2    
        yticks_x = gridorigin[0] - ticksize/2

        # drawing ticks and numbers 
        for combo in itertools.zip_longest(range(xticks),range(yticks)):
            xpt = (combo[0]*tickspacingX + gridorigin[0]) if combo[0] else 0
            ypt = (gridorigin[1] - combo[1]*tickspacingY) if combo[1] else 0
            xcoord,ycoord = self.pixToCoord(px=xpt,py=ypt)

            if combo[0]: # if drawing x axis ticks
                num = Text(Point(xpt,self.bottom - labelsize -margin- numsize/2),int(round(xcoord,-xexp+1)))
                num.setSize(fontsizes[1])
                num.setFace('courier')
                num.draw(self.win)

                tick = Line(Point(xpt,xticks_y),Point(xpt,xticks_y - ticksize))
                tick.draw(self.win)

            if combo[1]: # if drawing y axis ticks
                y = round(ycoord,-yexp+1)
                y = int(y) if y > 1 else y
                num = Text(Point(self.left + labelsize +margin+ ynumwid/2,ypt),y)
                num.setSize(fontsizes[1])
                num.setFace('courier')
                num.draw(self.win)

                tick = Line(Point(yticks_x,ypt),Point(yticks_x+ticksize,ypt))
                tick.draw(self.win)

        # draw main axis lines
        xaxis = Line(Point(*gridorigin),Point(self.right,gridorigin[1]))
        yaxis = Line(Point(*gridorigin),Point(gridorigin[0],self.top))
        xaxis.draw(self.win)
        yaxis.draw(self.win)

    def crossHair(self,pt):
        if pt:
            if pt.x >= self.gridorigin[0] and pt.x <= self.right and \
                pt.y >= self.top and pt.y <= self.gridorigin[1]:
                if self.crossX and self.crossY:
                    self.crossX.move(0,pt.y - self.crossoldpt.y)
                    self.crossY.move(pt.x-self.crossoldpt.x,0)
                else:
                    self.crossX = Line(Point(self.gridorigin[0],pt.y),Point(self.right,pt.y))
                    self.crossY = Line(Point(pt.x,self.top),Point(pt.x,self.gridorigin[1]))
                    self.crossX.setFill("red")
                    self.crossY.setFill("red")
                    self.crossX.draw(self.win)
                    self.crossY.draw(self.win)
                
                self.crossoldpt = pt

    def coordToPix(self,cx=0,cy=0): # check the math
        return ((cx+self.coords[0])/self.coordscalerX+self.gridorigin[0],
                self.gridorigin[1] - (self.coords[1]-cy)/self.coordscalerY)
    
    def pixToCoord(self,px=0,py=0): # convert pixel value to coordinate
        return ((px-self.gridorigin[0])*self.coordscalerX+self.coords[0],
                self.coords[1] + (self.gridorigin[1]-py)*self.coordscalerY)
    
    def getCoord(self,pt): # get coordinate value if point is inside plot
        if pt:
            if pt.x < self.gridorigin[0] or pt.x > self.right or pt.y < self.top or pt.y > self.gridorigin[1]:
                return None
            else:
                return self.pixToCoord(px=pt.x,py=pt.y)
        else: return None

def find_exp(number):
    base10 = math.log10(abs(number))
    return (math.floor(base10))

def color_rgb(r,g,b):
    """r,g,b are intensities of red, green, and blue in range(256)
    Returns color specifier string for the resulting color"""
    return "#%02x%02x%02x" % (r,g,b)

def test():
    win = GraphWin()
    win.setCoords(0,0,10,10)
    t = Text(Point(5,5), "Centered Text")
    t.draw(win)
    p = Polygon(Point(1,1), Point(5,3), Point(2,7))
    p.draw(win)
    e = Entry(Point(5,6), 10)
    e.draw(win)
    win.getMouse()
    p.setFill("red")
    p.setOutline("blue")
    p.setWidth(2)
    s = ""
    for pt in p.getPoints():
        s = s + "(%0.1f,%0.1f) " % (pt.getX(), pt.getY())
    t.setText(e.getText())
    e.setFill("green")
    e.setText("Spam!")
    e.move(2,0)
    win.getMouse()
    p.move(2,3)
    s = ""
    for pt in p.getPoints():
        s = s + "(%0.1f,%0.1f) " % (pt.getX(), pt.getY())
    t.setText(s)
    win.getMouse()
    p.undraw()
    e.undraw()
    t.setStyle("bold")
    win.getMouse()
    t.setStyle("normal")
    win.getMouse()
    t.setStyle("italic")
    win.getMouse()
    t.setStyle("bold italic")
    win.getMouse()
    t.setSize(14)
    win.getMouse()
    t.setFace("arial")
    t.setSize(20)
    win.getMouse()
    win.close()

#MacOS fix 2
#tk.Toplevel(_root).destroy()

# MacOS fix 1
update()

if __name__ == "__main__":
    test()