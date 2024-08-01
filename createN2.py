from openmdao.api import n2
from os.path import basename, dirname, join, abspath


def createN2(fileref, prob):
    n2folder = join(dirname(abspath(__file__)), "N2s")
    n2(prob, outfile=join(n2folder,
       f"n2_{basename(fileref).split('.')[0]}.html"))
