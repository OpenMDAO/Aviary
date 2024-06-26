import PySimpleGUI as sg
import math

# Get window resolution, then set window size to 80% of height and width, or 900 x 500, whichever is larger
# Plot width and height are also set here as percent of window's size
width,height = sg.Window.get_screen_size()
window_width, window_height = min(int(width*0.8),900),min(int(height*0.8),500)
plot_width,plot_height = int(window_width*0.7),int(window_height*0.4)

timelim = 400 # minutes, can be expanded
altlim = 50 # kft, can be expanded
machlim = 1.0

axesmargin = 0.06 # percent of plot's dimension for axes ticks, numbers, and labels
ticksize = 0.01 # percent of plot's dimension for axes ticks
axeslabelsize = 11 # pt, axes label font size
axesfontsize = 10 # pt, axes numbers font size
px_per_pt = 16./12.

restart = True
themecolors = {'Dark Mode':{'sgtheme':'DarkGray10',
                            'graphbg':'gray10',
                            'text':'white',
                            'line':'gray80'},
               'Light Mode':{'sgtheme':'SystemDefault1',
                             'graphbg':'gray90',
                             'text':'black',
                             'line':'gray9'}}

menu_def = [['&File', ['&Open', '&Save', '---', '&Properties', '&Exit']], ['&Help', ['&About...']]]

class simplePlot():
    def __init__(self,key,theme,top_right,scale=(1,1)):
        self.graph = sg.Graph(
            canvas_size=(plot_width, plot_height),
            graph_bottom_left=(-plot_width*axesmargin, -plot_height*axesmargin),
            graph_top_right=(top_right[0]*scale[0], top_right[1]*scale[1]),
            key=key,
            enable_events=True,  # mouse click events
            motion_events = True,
            background_color=theme['graphbg'],
            drag_submits=True,expand_y=False)
        self.theme = theme
        self.scale = scale

    def drawaxes(self,xlabel,ylabel):
        textcolor = self.theme['text']
        linecolor = self.theme['line']
        right,top = self.graph.TopRight
        left,bottom = self.graph.BottomLeft

        tickspacingx, tickspacingy = 20, 10
        ticksizex, ticksizey = ticksize*plot_height, ticksize*plot_width
        xticks = math.floor(right*self.scale[0]/tickspacingx)
        yticks = math.floor(top*self.scale[1]/tickspacingy)

        for i in range(xticks):
            xpt = i*tickspacingx
            if xpt >= right: break
            self.graph.draw_line((xpt,0),(xpt,-ticksizex),'gray50')
            self.graph.draw_text(round(xpt/self.scale[0]),(xpt,bottom + px_per_pt*axesfontsize*0.55),color=textcolor,font=("Arial",axesfontsize))
        for i in range(yticks):
            ypt = i*tickspacingy
            if ypt >= top: break
            self.graph.draw_line((0,ypt),(-ticksizey,ypt),'gray50')
            self.graph.draw_text(round(ypt/self.scale[1],1),(left+px_per_pt*axesfontsize*1.4,ypt),color=textcolor,font=("Arial",axesfontsize))

        self.graph.draw_lines([(0,top),(0,0),(right,0)],color = linecolor)
        self.graph.draw_text(ylabel,(left + px_per_pt*axeslabelsize*0.3,top/2),color=textcolor,angle = 90,font=("Arial",axeslabelsize))
        self.graph.draw_text(xlabel,(right/2,bottom+px_per_pt*axeslabelsize*0.2),color = textcolor,font=("Arial",axeslabelsize))
   

while restart:
    saved_theme = sg.user_settings_get_entry('theme', default='Light Mode')
    theme = themecolors[saved_theme]
    sg.theme(theme['sgtheme'])

    altplot = simplePlot(key='-ALTGRAPH-',theme=theme,top_right=(timelim,altlim))
    machplot = simplePlot(key='-MACHGRAPH-',theme=theme,top_right=(timelim,machlim),scale=(1,60))
    machplot.graph.motion_events = False
    machplot.graph.ChangeSubmits = False

    layout = [[sg.MenubarCustom(menu_def),
               sg.ButtonMenu(saved_theme,menu_def=['',list(themecolors.keys())],k='-THEME-')],
               [altplot.graph],[machplot.graph],
              [sg.Text(key='info', size=(60, 1))]]

    window = sg.Window("Mission Design GUI", layout, finalize=True,auto_save_location=True,keep_on_top=True,
                    resizable=False,size=(window_width,window_height))
    
    altplot.drawaxes("Time (minutes)","Altitude (x1000 ft)")
    machplot.drawaxes("Time (minutes)","Mach Number")
    
    # get the graph element for ease of use later
    
    altgraph = window["-ALTGRAPH-"]  # type: sg.Graph
    machgraph = window["-MACHGRAPH-"]
    info = window['info']
    prevline = False
    prevline2 = False

    #drawaxes(altgraph,theme)
    #drawaxes(machgraph,theme)
    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            restart = False
            break  # exit

        if event == "-THEME-":
            newtheme = values['-THEME-']
            if newtheme == theme:
                continue
            else:
                sg.user_settings_set_entry('theme',newtheme)
                window.close()
                break

        if "-ALTGRAPH-" in event:
            x,y = values['-ALTGRAPH-']
            if x < 0 or y < 0:
                    continue
            
            if event.endswith('+MOVE'):
                if prevline:
                    altgraph.delete_figure(l1)
                    altgraph.delete_figure(l2)
                l1 = altgraph.draw_line((0,y),(plot_width,y),'red')
                l2 = altgraph.draw_line((x,0),(x,plot_height),'red')
                info.update(value=f"Time: {x} min, Altitude: {y*1000} ft")
                prevline = True

            elif event.endswith('+UP'):  # The drawing has ended because mouse up
                altgraph.draw_point((x,y),size=8,color='red')
                machgraph.draw_point((x,0.7*60),size=8,color='red')
        elif "-MACHGRAPH-" in event:
            x,y = values['-MACHGRAPH-']
            if x < 0 or y < 0:
                    continue
            
            if event.endswith('+MOVE'):
                if prevline2:
                    machgraph.delete_figure(l1)
                    machgraph.delete_figure(l2)
                l1 = machgraph.draw_line((0,y),(plot_width,y),'red')
                l2 = machgraph.draw_line((x,0),(x,plot_height),'red')
                info.update(value=f"Time: {x} min, Mach: {y/60}")
                prevline2 = True

            elif event.endswith('+UP'):  # The drawing has ended because mouse up
                machgraph.draw_point((x,y),size=8,color='red')

        else:
            print("unhandled event", event, values)

        neww,newh = window.size
        if neww!=window_width or newh!=window_height:
            window_width,window_height = neww,newh
            gw,gh = int(window_width*0.7),int(window_height*0.5)

#################################################################################################################

# import PySimpleGUI as sg
# # import PySimpleGUIQt as sg
# # import PySimpleGUIWeb as sg

# import numpy as np
# from matplotlib.backends.backend_tkagg import FigureCanvasAgg
# import matplotlib.figure
# import matplotlib.pyplot as plt
# import io

# from matplotlib import cm
# from mpl_toolkits.mplot3d.axes3d import get_test_data
# from matplotlib.ticker import NullFormatter  # useful for `logit` scale


# """
#     Demo - Matplotlib Embedded figure in a window TEMPLATE
    
#     The reason this program is labelled as a "Template" is that it functions on 3 
#     PySimpleGUI ports by only changing the import statement. tk, Qt, Web(Remi) all
#     run this same code and produce identical results.
    
#     Copyright 2020-2023 PySimpleSoft, Inc. and/or its licensors. All rights reserved.
    
#     Redistribution, modification, or any other use of PySimpleGUI or any portion thereof is subject to the terms of the PySimpleGUI License Agreement available at https://eula.pysimplegui.com.
    
#     You may not redistribute, modify or otherwise use PySimpleGUI or its contents except pursuant to the PySimpleGUI License Agreement.
# """


# def create_axis_grid():
#     from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes

#     plt.close('all')

#     def get_demo_image():
#         # prepare image
#         delta = 0.5

#         extent = (-3, 4, -4, 3)
#         x = np.arange(-3.0, 4.001, delta)
#         y = np.arange(-4.0, 3.001, delta)
#         X, Y = np.meshgrid(x, y)
#         Z1 = np.exp(-X ** 2 - Y ** 2)
#         Z2 = np.exp(-(X - 1) ** 2 - (Y - 1) ** 2)
#         Z = (Z1 - Z2) * 2

#         return Z, extent

#     def get_rgb():
#         Z, extent = get_demo_image()

#         Z[Z < 0] = 0.
#         Z = Z / Z.max()

#         R = Z[:13, :13]
#         G = Z[2:, 2:]
#         B = Z[:13, 2:]

#         return R, G, B

#     fig = plt.figure(1)
#     ax = RGBAxes(fig, [0.1, 0.1, 0.8, 0.8])

#     r, g, b = get_rgb()
#     kwargs = dict(origin="lower", interpolation="nearest")
#     ax.imshow_rgb(r, g, b, **kwargs)

#     ax.RGB.set_xlim(0., 9.5)
#     ax.RGB.set_ylim(0.9, 10.6)

#     plt.draw()
#     return plt.gcf()



# def create_figure():
#     # ------------------------------- START OF YOUR MATPLOTLIB CODE -------------------------------
#     fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
#     t = np.arange(0, 3, .01)
#     fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))

#     return fig


# def create_subplot_3d():


#     fig = plt.figure()

#     ax = fig.add_subplot(1, 2, 1, projection='3d')
#     X = np.arange(-5, 5, 0.25)
#     Y = np.arange(-5, 5, 0.25)
#     X, Y = np.meshgrid(X, Y)
#     R = np.sqrt(X ** 2 + Y ** 2)
#     Z = np.sin(R)
#     surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
#                            linewidth=0, antialiased=False)
#     ax.set_zlim3d(-1.01, 1.01)

#     fig.colorbar(surf, shrink=0.5, aspect=5)

#     ax = fig.add_subplot(1, 2, 2, projection='3d')
#     X, Y, Z = get_test_data(0.05)
#     ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
#     return fig




# def create_pyplot_scales():

#     plt.close('all')
#     # Fixing random state for reproducibility
#     np.random.seed(19680801)

#     # make up some data in the interval ]0, 1[
#     y = np.random.normal(loc=0.5, scale=0.4, size=1000)
#     y = y[(y > 0) & (y < 1)]
#     y.sort()
#     x = np.arange(len(y))

#     # plot with various axes scales
#     plt.figure(1)

#     # linear
#     plt.subplot(221)
#     plt.plot(x, y)
#     plt.yscale('linear')
#     plt.title('linear')
#     plt.grid(True)

#     # log
#     plt.subplot(222)
#     plt.plot(x, y)
#     plt.yscale('log')
#     plt.title('log')
#     plt.grid(True)

#     # symmetric log
#     plt.subplot(223)
#     plt.plot(x, y - y.mean())
#     plt.yscale('symlog', linthreshy=0.01)
#     plt.title('symlog')
#     plt.grid(True)

#     # logit
#     plt.subplot(224)
#     plt.plot(x, y)
#     plt.yscale('logit')
#     plt.title('logit')
#     plt.grid(True)
#     # Format the minor tick labels of the y-axis into empty strings with
#     # `NullFormatter`, to avoid cumbering the axis with too many labels.
#     plt.gca().yaxis.set_minor_formatter(NullFormatter())
#     # Adjust the subplot layout, because the logit one may take more space
#     # than usual, due to y-tick labels like "1 - 10^{-3}"
#     plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
#                         wspace=0.35)
#     return plt.gcf()

# # ----------------------------- The draw figure helpful function -----------------------------

# def draw_figure(element, figure):
#     """
#     Draws the previously created "figure" in the supplied Image Element

#     :param element: an Image Element
#     :param figure: a Matplotlib figure
#     :return: The figure canvas
#     """


#     plt.close('all')        # erases previously drawn plots
#     canv = FigureCanvasAgg(figure)
#     buf = io.BytesIO()
#     canv.print_figure(buf, format='png')
#     if buf is None:
#         return None
#     buf.seek(0)
#     element.update(data=buf.read())
#     return canv


# # ----------------------------- The GUI Section -----------------------------

# def main():
#     dictionary_of_figures = {'Axis Grid': create_axis_grid,
#                              'Subplot 3D': create_subplot_3d,
#                              'Scales': create_pyplot_scales,
#                              'Basic Figure': create_figure}


#     left_col = [[sg.T('Figures to Draw')],
#                 [sg.Listbox(list(dictionary_of_figures), default_values=[list(dictionary_of_figures)[0]], size=(15, 5), key='-LB-')],
#                 [sg.T('Matplotlib Styles')],
#                 [sg.Combo(plt.style.available,  key='-STYLE-')]]

#     layout = [ [sg.T('Matplotlib Example', font='Any 20')],
#                [sg.Column(left_col), sg.Image(key='-IMAGE-')],
#                [sg.B('Draw'), sg.B('Exit')] ]

#     window = sg.Window('Matplotlib Template', layout)

#     image_element = window['-IMAGE-']       # type: sg.Image

#     while True:
#         event, values = window.read()
#         print(event, values)
#         if event == 'Exit' or event == sg.WIN_CLOSED:
#             break
#         if event == 'Draw' and values['-LB-']:
#             # Get the function to call to make figure. Done this way to get around bug in Web port (default value not working correctly for listbox)
#             func = dictionary_of_figures.get(values['-LB-'][0], list(dictionary_of_figures.values())[0])
#             if values['-STYLE-']:
#                 plt.style.use(values['-STYLE-'])
#             draw_figure(image_element, func())

#     window.close()


# if __name__ == "__main__":
#     main()

##############################################################################33
#######################################################################################333

# import PySimpleGUI as sg

# """
#     Demo - Basic window design pattern
#     * Creates window in a separate function for easy "restart"
#     * Saves theme as a user variable
#     * Puts main code into a main function so that multiprocessing works if you later convert to use
    
#     Copyright 2020-2023 PySimpleSoft, Inc. and/or its licensors. All rights reserved.
    
#     Redistribution, modification, or any other use of PySimpleGUI or any portion thereof is subject to the terms of the PySimpleGUI License Agreement available at https://eula.pysimplegui.com.
    
#     You may not redistribute, modify or otherwise use PySimpleGUI or its contents except pursuant to the PySimpleGUI License Agreement.
# """


# # ------------------- Create the window -------------------
# def make_window():
#     # Set theme based on previously saved
#     sg.theme(sg.user_settings_get_entry('theme', None))

#     # -----  Layout & Window Create  -----
#     layout = [[sg.T('This is your layout')],
#                [sg.OK(), sg.Button('Theme', key='-THEME-'), sg.Button('Exit')]]

#     return sg.Window('Pattern for theme saving', layout)


# # ------------------- Main Program and Event Loop -------------------

# def main():
#     window = make_window()

#     while True:
#         event, values = window.read()
#         if event == sg.WINDOW_CLOSED or event == 'Exit':
#             break
#         if event == '-THEME-':      # Theme button clicked, so get new theme and restart window
#             ev, vals = sg.Window('Choose Theme', [[sg.Combo(sg.theme_list(), k='-THEME LIST-'), sg.OK(), sg.Cancel()]]).read(close=True)
#             if ev == 'OK':
#                 window.close()
#                 sg.user_settings_set_entry('theme', vals['-THEME LIST-'])
#                 window = make_window()

#     window.close()


# if __name__ == '__main__':
#     main()

# import PySimpleGUI as sg

# layout = [[sg.Text('This is our persistent window')],
#           [sg.Button('1'), sg.Button('2'), sg.Button('Exit')]]
# sg.theme('SystemDefault')
# window = sg.Window('Title', layout)

# # theme options: DarkGrey10 or SystemDefault1 (dark, light)
# sg.SystemTray.notify('Notification Title', 'This is the notification message')


# while True:			# The Event Looop
#     event, values = window.read()

#     if event == sg.WIN_CLOSED or event == 'Exit':
#         break
#     if event == '1':
#         sg.popup('You clicked 1')
#     elif event == '2':
#         sg.popup('You clicked 2')

# window.close()