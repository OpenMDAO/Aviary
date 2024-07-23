import csv
import math
from tkinter import *
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
from tkinter import filedialog
import aviary.api as av
from copy import deepcopy


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
    rows = math.ceil(num/3)
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
    
file_contents={}
file_name=StringVar(value='Aircraft_Model.csv')
file_data=[]
def Open():
    file_ = filedialog.askopenfilename(title = "Select a Model",
                                          filetypes = (('All files', '*'),))
    file_name.set(file_)
    file = open(file_)
    for line in file:
        line = line.strip()
        for i in range(len(list_keys)):
            if ',' in line:
                name = line.split(',')[0]
                numbers = line.split(',')[1:]
                if len(numbers) > 1:
                    fixed = numbers[:-1][0]
                else:
                    fixed = numbers[0]
                if "[" in fixed and "]" in fixed:
                    temp = fixed.replace("[","").replace("]","")
                    fixed = [float(num) for num in temp.split(",")]
                elif "," in fixed:
                    for num in fixed.split(","):
                        num = float(num)
                elif "FALSE" in fixed.upper() or "TRUE" in fixed.upper():
                    fixed = fixed
                else:
                    try: fixed = float(fixed)
                    except ValueError: pass

                variable = StringVar(value=name)
                if variable.get() == list_keys[i]:
                    file_contents[variable.get()]=fixed
                else: pass

    for i in range(len(list_keys)):   
        check=0
        for key in file_contents:
            if list_keys[i] == key:
                check+=1
                file_data.append(file_contents[key])
            else: pass
        if check != 1:
            file_data.append(list_values[i]["default_value"])
    return file_contents, file_name, file_data

filesaveas=StringVar(value='Aircraft_Model.csv')
checksaveas=IntVar(value=0)
def Saveas():
    files = [('CSV', '*.csv'),
             ('Text Document', '*.txt'),
             ('All Files', '*.*')]
    file__ = filedialog.asksaveasfile(filetypes = files, defaultextension = files)
    rename = str(file__).split("'")[1]
    filesaveas.set(value=rename)
    checksaveas.set(value=1)
    info(file_contents,file_name,file_data,checksaveas,filesaveas)
    return filesaveas, checksaveas

def About():
    messagebox.showinfo(title='About', message='v1.0 - 2024')


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
        self.canvas.bind("<Enter>", self._bind_mouse)
        self.canvas.bind("<Leave>", self._unbind_mouse)
        self.vsb['command'] = self.canvas.yview
        self.hsb['command'] = self.canvas.xview

        self.inner = tk.Frame(self.canvas)
        self.canvas.create_window(4, 4, window=self.inner, anchor='nw')
        self.inner.bind("<Configure>", self._on_frame_configure)

        self.outer_attr = set(dir(tk.Widget))

    def __getattr__(self, item):
        if item in self.outer_attr:
            return getattr(self.outer, item)
        else:
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

notebook=ttk.Notebook(myframe.inner)
notebook.pack(fill='both', expand=True, anchor='center')
frame1=ttk.Frame(notebook)
frame2=ttk.Frame(notebook)
frame3=ttk.Frame(notebook)
frame4=ttk.Frame(notebook)
frame5=ttk.Frame(notebook)
notebook.add(frame1,text="Aircraft")
notebook.add(frame2,text="Dynamic")
notebook.add(frame3,text="Mission")
notebook.add(frame4,text="Settings")
notebook.add(frame5,text="Search")

tempdict = {}
def info(file_contents=file_contents, file_name=file_name, file_data=file_data, checksaveas=checksaveas, filesaveas=filesaveas):
    print(checksaveas.get())
    list_of_lists = [] 
    for i,(key,value) in enumerate(zip(data.keys(),data.values())):
            unit = value["units"]
            if file_name.get() == 'Aircraft_Model.csv':  
                if key in tempdict.keys():
                    list_of_lists.append([f'{key},{tempdict[key].get()},{unit}'])
                else:
                    list_of_lists.append([f'{key},{value["default_value"]},{unit}'])
            else:
                if key in tempdict.keys():
                    list_of_lists.append([f'{key},{tempdict[key].get()},{unit}'])                
                elif key in file_contents.keys():
                    list_of_lists.append([f'{key},{file_data[i]},{unit}'])
                else:
                    list_of_lists.append([f'{key},{value["default_value"]},{unit}'])
    if file_name.get() == 'Aircraft_Model.csv' and checksaveas.get() != 1:
        with open('Aircraft_Model.csv', 'w', newline='') as i:
            writer = csv.writer(i,delimiter=';',quoting=csv.QUOTE_MINIMAL)
            for sublist in list_of_lists:
                writer.writerow(sublist)
    elif file_name.get() == 'Aircraft_Model.csv' and checksaveas.get() == 1:
        with open(filesaveas.get(), 'w', newline='') as i:
            writer = csv.writer(i,delimiter=';',quoting=csv.QUOTE_MINIMAL)
            for sublist in list_of_lists:
                writer.writerow(sublist)
    elif file_name.get() != 'Aircraft_Model.csv' and checksaveas.get() == 1:
        with open(filesaveas.get(), 'w', newline='') as i:
            writer = csv.writer(i,delimiter=';',quoting=csv.QUOTE_MINIMAL)
            for sublist in list_of_lists:
                writer.writerow(sublist)
    else:
        with open(file_name.get(), 'w', newline='') as i:
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

def fxn(x,frame,file_name=file_name,file_contents=file_contents,file_data=file_data):
    num_rows = rows_per_subhead[x]
    num=0
    for row in range(num_rows):
        if num < entries_per_subhead[x]:
            for col in range(3):
                print(x, num)
                i = index_list[x][num]
                user_input = StringVar(value=f'{list_values[i]["default_value"]}')
                if file_name.get() != 'Aircraft_Model.csv':
                    for key in list_keys:
                        if key in file_contents.keys():
                            user_input = StringVar(value=f'{file_data[i]}')

                input_title = ttk.Label(frame,justify='left',font=('TkDefaultFont',12), text=f'{list_keys[i]}')
                input_title.grid(row = (row+compound_subheaders[x]+compound_data_rows[x]+1),column=col,pady=10,padx=1,sticky='nw')   

                user_input_entry = ttk.Entry(frame,width='10',textvariable=user_input)
                user_input_entry.grid(row = (row+compound_subheaders[x]+compound_data_rows[x]+1),column=col,pady=30,padx=1,sticky='w')   

                input_unit = ttk.Label(frame,justify='left',font=('TkDefaultFont',10), text=f'{list_values[i]["units"]}')
                input_unit.grid(row = (row+compound_subheaders[x]+compound_data_rows[x]+1),column=col,pady=30,padx=75,sticky='w')   

                input_desc = ttk.Label(frame,wraplength=120,justify='left',font=('TkDefaultFont',10), text=f'{list_values[i]["desc"]}')
                input_desc.grid(row = (row+compound_subheaders[x]+compound_data_rows[x]+1),column=col,pady=30,padx=150,sticky='w')   

                tempdict[list_keys[i]] = user_input
                num+=1
                if num == entries_per_subhead[x]:
                    break
        else: break

def Searchfxn(x,y,add_row,frame,file_name=file_name,file_contents=file_contents,file_data=file_data):
    i = index_list[x][y]
    user_input = StringVar(value=f'{list_values[i]["default_value"]}')
    if file_name.get() != 'Aircraft_Model.csv':
        for key in list_keys:
            if key in file_contents.keys():
                user_input = StringVar(value=f'{file_data[i]}')

    input_title = ttk.Label(frame,justify='left',font=('TkDefaultFont',12), text=f'{list_keys[i]}')
    input_title.grid(row = 4+add_row,column=0,pady=10,padx=1,sticky='nw')   

    user_input_entry = ttk.Entry(frame,width='10',textvariable=user_input)
    user_input_entry.grid(row = 5+add_row,column=0,pady=10,padx=1,sticky='w')   

    input_unit = ttk.Label(frame,justify='left',font=('TkDefaultFont',10), text=f'{list_values[i]["units"]}')
    input_unit.grid(row = 5+add_row,column=1,pady=10,padx=20,sticky='w')   

    input_desc = ttk.Label(frame,wraplength=120,justify='left',font=('TkDefaultFont',10), text=f'{list_values[i]["desc"]}')
    input_desc.grid(row = 5+add_row,column=3,pady=10,padx=20,sticky='w')   

    tempdict[list_keys[i]] = user_input

searcheader=ttk.Frame(frame5)
searcheader.pack()
searchcontents=ttk.Frame(frame5)
searchcontents.pack(after=searcheader)

def Clear(frame=searchcontents):
    frame.destroy()

def Search(keyword):
    searchcontents=ttk.Frame(frame5)
    searchcontents.pack(after=searcheader)
    add_row=0
    for i, key in enumerate(list_keys):
        var = key.split(':')[-1]
        if var == keyword:
            add_row+=2
            count=0
            for row in index_list:
                count2=0
                for col in row:
                    if i == col:
                        Searchfxn(count,count2,add_row,searchcontents)
                    count2+=1
                count+=1
    return searchcontents

Search_input = StringVar(value='Enter Variable Name')
Search_label = ttk.Label(searcheader,justify='center', text='Search variable name',font=('TkDefaultFont',15,'bold'))  
Search_label.grid(row=2,column=0)
Search_entry = ttk.Entry(searcheader,width='30',textvariable=Search_input)
Search_entry.grid(row=3, column=0)
Search_button = ttk.Button(searcheader, text = 'Search', width = "10", command = lambda i=i:Search(Search_input.get()))
Search_button.grid(row=3, column=1)
Clear_button = ttk.Button(searcheader, text = 'Clear', width = "10", command = lambda:Clear(searchcontents))
Clear_button.grid(row=3, column=2)


menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label='Open', command=Open)
filemenu.add_command(label='Save', command=info)
filemenu.add_command(label='Save as...', command=Saveas)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)
helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="About...", command=About)
menubar.add_cascade(label="Help", menu=helpmenu)
root.config(menu=menubar)



button_main = ttk.Button(root,text = "Submit", width = "20", command = info)
button_main.pack(before=notebook,pady=10)
root.mainloop()