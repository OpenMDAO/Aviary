import numpy as np
import csv
from tkinter import *
import aviary.api as av
from copy import deepcopy
import math
import tkinter as tk
import tkinter.ttk as ttk
data = deepcopy(av.CoreMetaData)
list_values = list(data.values())
list_keys = list(data.keys())
list_len = len(list_values)
root = Tk()
wid=1200
hei=800
root.geometry("%dx%d" % (wid,hei))
root.title("Model Aircraft Input")

headers=[]
name_each_subhead = []
entries_per_subhead = []
old_subhead =''
for key in list_keys:
    if ':' in key:
        subhead = key.split(':')[1]
        header=key.split(':')[0]
        if subhead == old_subhead:
            entries_per_subhead[-1] += 1
        else:
            entries_per_subhead.append(1)
            name_each_subhead.append(subhead)
            headers.append(header)
        old_subhead = subhead
    else:
        entries_per_subhead.append(1)
        name_each_subhead.append(key)
        headers.append('dynamic')
rows_per_subhead=[]
compound_subheaders=[]
v=0
for num in entries_per_subhead:
    rows = math.ceil(num/4)
    rows_per_subhead.append(rows)
    compound_subheaders.append(v)
    v+=1
compound_data_rows=[0]
for i,nums in enumerate(rows_per_subhead): 
    val=sum(rows_per_subhead[:i+1])
    compound_data_rows.append(val)
    if len(compound_data_rows)==87:
        break
compound_data_entries=[]
for i, nums in enumerate(entries_per_subhead):
    val = sum(entries_per_subhead[:i+1])
    compound_data_entries.append(val)
index_list=[]
number = 0
mini_list=[0]
for num in compound_data_entries:
    for i in range(number,num):    
        if number < num-1:
            number +=1
            mini_list.append(number)
    index_list.append(mini_list)
    mini_list=[]

class DoubleScrolledFrame:
    """
    https://gist.github.com/novel-yet-trivial/2841b7b640bba48928200ff979204115
    """
    def __init__(self, master, **kwargs):
        width = wid
        height = hei
        self.outer = tk.Frame(master, **kwargs)

        self.vsb = ttk.Scrollbar(self.outer, orient=tk.VERTICAL)
        self.vsb.grid(row=0, column=1, sticky='ns')
        self.hsb = ttk.Scrollbar(self.outer, orient=tk.HORIZONTAL)
        self.hsb.grid(row=1, column=0, sticky='ew')
        self.canvas = tk.Canvas(self.outer, highlightthickness=0, width=width, height=height)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.outer.rowconfigure(0, weight=1)
        self.outer.columnconfigure(0, weight=1)
        self.canvas['yscrollcommand'] = self.vsb.set
        self.canvas['xscrollcommand'] = self.hsb.set
        # mouse scroll does not seem to work with just "bind"; You have
        # to use "bind_all". Therefore to use multiple windows you have
        # to bind_all in the current widget
        self.canvas.bind("<Enter>", self._bind_mouse)
        self.canvas.bind("<Leave>", self._unbind_mouse)
        self.vsb['command'] = self.canvas.yview
        self.hsb['command'] = self.canvas.xview

        self.inner = tk.Frame(self.canvas)
        # pack the inner Frame into the Canvas with the topleft corner 4 pixels offset
        self.canvas.create_window(4, 4, window=self.inner, anchor='nw')
        self.inner.bind("<Configure>", self._on_frame_configure)

        self.outer_attr = set(dir(tk.Widget))

    def __getattr__(self, item):
        if item in self.outer_attr:
            # geometry attributes etc (eg pack, destroy, tkraise) are passed on to self.outer
            return getattr(self.outer, item)
        else:
            # all other attributes (_w, children, etc) are passed to self.inner
            return getattr(self.inner, item)

    def _on_frame_configure(self, event=None):
        x1, y1, x2, y2 = self.canvas.bbox("all")
        height = self.canvas.winfo_height()
        width = self.canvas.winfo_width()
        self.canvas.config(scrollregion = (0,0, max(x2, width), max(y2, height)))

    def _bind_mouse(self, event=None):
        self.canvas.bind_all("<4>", self._on_mousewheel)
        self.canvas.bind_all("<5>", self._on_mousewheel)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mouse(self, event=None):
        self.canvas.unbind_all("<4>")
        self.canvas.unbind_all("<5>")
        self.canvas.unbind_all("<MouseWheel>")
        
    def _on_mousewheel(self, event):
        """Linux uses event.num; Windows / Mac uses event.delta"""
        func = self.canvas.xview_scroll if event.state & 1 else self.canvas.yview_scroll 
        if event.num == 4 or event.delta > 0:
            func(-1, "units" )
        elif event.num == 5 or event.delta < 0:
            func(1, "units" )
    
    def __str__(self):
        return str(self.outer)
myframe = DoubleScrolledFrame(root)
myframe.pack()

notebook=ttk.Notebook(myframe)
notebook.pack(fill='both', expand=True, anchor='center')
frame1=ttk.Frame(notebook)
frame2=ttk.Frame(notebook)
frame3=ttk.Frame(notebook)
frame4=ttk.Frame(notebook)
notebook.add(frame1,text="Aircraft")
notebook.add(frame2,text="Dynamic")
notebook.add(frame3,text="Mission")
notebook.add(frame4,text="Settings")


# list_of_inputs = []
tempdict = {}
def info():
    list_of_lists = []
    for i,(key,value) in enumerate(zip(data.keys(),data.values())):
        unit = value["units"]
        if key in tempdict.keys():
            list_of_lists.append([f'{key},{tempdict[key].get()},{unit}'])
        else:
            list_of_lists.append([f'{key},{value["default_value"]},{unit}'])
        # if list_of_inputs[i].get() != "":  
        #     list_of_lists.append([f'{key},{list_of_inputs[i].get()},{unit}'])
        # else:    
        #     list_of_lists.append([f'{key},{value["default_value"]},{unit}'])
        
    with open('Aircraft_Model.csv', 'w', newline='') as i:
        writer = csv.writer(i,delimiter=';',quoting=csv.QUOTE_MINIMAL)
        for sublist in list_of_lists:
            writer.writerow(sublist)


for i in range(len(name_each_subhead)):
    if headers[i] == 'aircraft':
        subhead = ttk.Label(frame1,justify='center', text=name_each_subhead[i],font=('TkDefaultFont',15,'bold'))  
        button = ttk.Button(frame1,text = name_each_subhead[i], width = "20", command = lambda i=i: fxn(i, frame1))
        subhead.grid(row=compound_data_rows[i]+compound_subheaders[i],columnspan=1,sticky='n',pady=10)
        button.grid(row=compound_data_rows[i]+compound_subheaders[i],column=2,sticky='n',pady=10)    
    elif headers[i] == 'dynamic':
        subhead = ttk.Label(frame2,justify='center', text=name_each_subhead[i],font=('TkDefaultFont',15,'bold'))  
        button = ttk.Button(frame2,text = name_each_subhead[i], width = "20", command = lambda i=i: fxn(i, frame2))
        subhead.grid(row=compound_data_rows[i]+compound_subheaders[i],columnspan=1,sticky='n',pady=10)
        button.grid(row=compound_data_rows[i]+compound_subheaders[i],column=2,sticky='n',pady=10)
    elif headers[i] == 'mission':
        subhead = ttk.Label(frame3,justify='center', text=name_each_subhead[i],font=('TkDefaultFont',15,'bold'))  
        button = ttk.Button(frame3,text = name_each_subhead[i], width = "20", command = lambda i=i: fxn(i, frame3))
        subhead.grid(row=compound_data_rows[i]+compound_subheaders[i],columnspan=1,sticky='n',pady=10)
        button.grid(row=compound_data_rows[i]+compound_subheaders[i],column=2,sticky='n',pady=10)
    elif headers[i] == "settings":
        subhead = ttk.Label(frame4,justify='center', text=name_each_subhead[i],font=('TkDefaultFont',15,'bold'))  
        button = ttk.Button(frame4,text = name_each_subhead[i], width = "20", command = lambda i=i: fxn(i, frame4))
        subhead.grid(row=compound_data_rows[i]+compound_subheaders[i],columnspan=1,sticky='n',pady=10)
        button.grid(row=compound_data_rows[i]+compound_subheaders[i],column=2,sticky='n',pady=10)

rows = math.ceil(list_len/4)
def fxn(x,frame):
    num_rows = rows_per_subhead[x]
    num=0
    for row in range(num_rows):
        if num < entries_per_subhead[x]:
            for col in range(3):
                print(x, num)
                i = index_list[x][num]
                user_input = StringVar(value=f'{list_values[i]["default_value"]}')  
                
                input_title = ttk.Label(frame,justify='left', text=f'{list_keys[i]}')
                input_title.grid(row = (row+compound_subheaders[x]+compound_data_rows[x]+1),column=col,pady=10,padx=1,sticky='nw')   

                user_input_entry = ttk.Entry(frame,width='10',textvariable=user_input)
                user_input_entry.grid(row = (row+compound_subheaders[x]+compound_data_rows[x]+1),column=col,pady=30,padx=1,sticky='w')   

                input_unit = ttk.Label(frame,justify='left', text=f'{list_values[i]["units"]}')
                input_unit.grid(row = (row+compound_subheaders[x]+compound_data_rows[x]+1),column=col,pady=30,padx=75,sticky='w')   

                input_desc = ttk.Label(frame,wraplength=120,justify='left', text=f'{list_values[i]["desc"]}')
                input_desc.grid(row = (row+compound_subheaders[x]+compound_data_rows[x]+1),column=col,pady=30,padx=150,sticky='w')   

                # list_of_inputs.append(user_input)
                tempdict[list_keys[i]] = user_input
                num+=1
                if num == entries_per_subhead[x]:
                    break
        else:
            break
button_main = ttk.Button(root,text = "Submit", width = "20", command = info)
button_main.pack(before=notebook,pady=10)       
root.mainloop()