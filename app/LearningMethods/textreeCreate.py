# from openpyxl import Workbook, load_workbook
import re
import math
import pandas
import networkx as nx
import pickle


"""
every bacteria is an object to easily store it's information
"""
class Bacteria:
    def __init__(self, string, val):
        string = string.replace(" ", "")
        lst = re.split("; |__|;", string)
        self.val = val
        # removing letters and blank spaces
        for i in range(0, len(lst)):
            if len(lst[i]) < 2:
                lst[i] = 0
        lst = [value for value in lst if value != 0]
        # Default fall value
        if len(lst) == 0:
            lst = ["Bacteria"]
        self.lst = lst


def create_tax_tree(series, zeroflag=True):
    tempGraph = nx.Graph()
    """workbook = load_workbook(filename="random_Otus.xlsx")
    sheet = workbook.active"""
    valdict = {("Bacteria",): 0, ("Archaea",): 0}
    bac = []
    for i, (tax, val) in enumerate(series.items()):
        # adding the bacteria in every column
        bac.append(Bacteria(str(tax), val))
        if len(bac[i].lst) == 1 and bac[i].lst[0] == "Bacteria":
            valdict[("Bacteria",)] += bac[i].val
        if len(bac[i].lst) == 1 and bac[i].lst[0] == "Archaea":
            valdict[("Archaea",)] += bac[i].val
        # connecting to the root of the tempGraph
        tempGraph.add_edge(("anaerobe",), (bac[i].lst[0],))
        # connecting all levels of the taxonomy
        for j in range(0, len(bac[i].lst) - 1):
            updateval(tempGraph, bac[i], valdict, j, True)
        # adding the value of the last node in the chain
        updateval(tempGraph, bac[i], valdict, len(bac[i].lst) - 1, False)
    valdict[("anaerobe",)] = valdict[("Bacteria",)] + valdict[("Archaea",)]
    return create_final_graph(tempGraph, valdict, zeroflag)


def updateval(graph, bac, vald, num, adde):
    if adde:
        graph.add_edge(tuple(bac.lst[:num + 1]), tuple(bac.lst[:num + 2]))
        # adding the value of the nodes
    if tuple(bac.lst[:num + 1]) in vald:
        vald[tuple(bac.lst[:num + 1])] += bac.val
    else:
        vald[tuple(bac.lst[:num + 1])] = bac.val


def create_final_graph(tempGraph, valdict, zeroflag):
    graph = nx.Graph()
    for e in tempGraph.edges():
        node1_name = e[0]
        node1_val = valdict[e[0]]
        node2_name = e[1]
        node2_val = valdict[e[1]]
        graph.add_node(node1_name, val=node1_val)
        graph.add_node(node2_name, val=node2_val)
        if not zeroflag or node1_val * node2_val != 0:
            graph.add_edge(node1_name, node2_name)
    return graph

if __name__ == "__main__":
    create_tax_tree(pickle.load(open("series.p", "rb")))
