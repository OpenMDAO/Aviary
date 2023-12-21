import glob
import os
import subprocess
from importlib import import_module
from pathlib import Path


os.chdir(Path(__file__).parents[0])

for file_name in glob.iglob("*xdsm.py"):
    xdsm = import_module(file_name[:-3])
    # remove the grayed out text decoration in spec files
    spec_name = file_name[:-7] + "specs"
    for item in xdsm.x.systems:
        spec = item.spec_name
        try:
            str_line = ""
            with open(spec_name + "/" + spec + ".json", "r") as fr:
                for line in fr:
                    if "textcolor{gray}" in line:
                        str_line += line.replace(
                            "\\\\textcolor{gray}{", "").replace("}", "")
                    else:
                        str_line += line
            with open(spec_name + "/" + spec + ".json", "w") as fw:
                fw.write(str_line)
        except:
            pass

# go through all *xdsm.py files again and replace "_" by "\_"
print("post-processing")
for file_name in glob.iglob("*xdsm.py"):
    file_name = file_name[:-3]
    print(f"post-processing {file_name}.tikz and {file_name}.tex")
    # backup .tikz files, assuming Python 3.3+
    os.replace(f"{file_name}.tikz", f"{file_name}_save.tikz")
    str_line = ""
    # replace "_" by "\_" in .tikz files
    with open(file_name+"_save.tikz", "r") as fr:
        for line in fr:
            if "begin{array}" in line:
                index_begin = line.index("begin{array}")
                str_line += line[:index_begin] + line[index_begin:].replace("_", "\\_")
            else:
                str_line += line
    # save updated .tikz files
    with open(file_name+".tikz", "w") as fw:
        fw.write(str_line)

    # generate .pdf file again
    cmd = f"pdflatex {file_name}.tex"
    try:
        output = subprocess.check_output(cmd.split())
    except subprocess.CalledProcessError as err:
        print("Command '{}' failed.  Return code: {}".format(cmd, err.returncode))
    os.remove(file_name+".log")
    os.remove(file_name+".aux")
    os.remove(file_name+"_save.tikz")
