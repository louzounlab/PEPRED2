import networkx as nx
import pandas as pd
from LearningMethods.textreeCreate import create_tax_tree
import ete3
import svgutils.compose as sc
from IPython.display import SVG # /!\ note the 'SVG' function also in svgutils.compose
import numpy as np
import matplotlib.pyplot as plt
from LearningMethods import CorrelationFramework
import re
def name_to_newick(tup):
    return str(tup).replace(", ", "|").replace("(", "<") \
        .replace(")", ">").replace("'", "").replace(",", "")


def newick_to_name(string):
    return tuple(string.strip("<>").split("|"))


def tree_to_newick_recursion(g, root=("anaerobe",)):
    if root is None:
        roots = list(filter(lambda p: p[1] == 0, g.in_degree()))
        assert 1 == len(roots)
        root = roots[0][0]
    subgs = []
    x = g[root]
    for child in g[root]:
        if len(child) > len(root) or root == ("anaerobe",):
            if len(g[child]) > 1:
                subgs.append(tree_to_newick_recursion(g, root=child))
            else:
                subgs.append(name_to_newick(child))
    return "(" + ','.join(subgs) + ")" + name_to_newick(root)


def tree_to_newick(s):
    graph = create_tax_tree(s)
    newick = tree_to_newick_recursion(graph) + ";"
    return newick, graph


def get_tree_shape(newick, graph, lower_threshold, higher_threshold, dict):
    t = ete3.Tree(newick, format=8)
    not_yellows = []
    for n in t.traverse():
        nstyle = ete3.NodeStyle()
        nstyle["fgcolor"] = dict["netural"]
        name = newick_to_name(n.name)
        if name != '' and name in graph.nodes:
            if graph.nodes[name]["val"] > higher_threshold:
                nstyle["fgcolor"] = dict["positive"]
                not_yellows.append(n)
            elif graph.nodes[name]["val"] < lower_threshold:
                nstyle["fgcolor"] = dict["negative"]
                not_yellows.append(n)
        nstyle["size"] = 5
        n.set_style(nstyle)
    d = 1
    print("successfully got nstlyes")
    while (d > 0):
        d = 0
        for n in t.traverse():
            if n.is_leaf():
                flag = 1
                if nstyle["fgcolor"] != "yellow":
                    continue
                parent = n.up
                if parent == n.get_tree_root():
                    n.delete()
                    continue
                if parent.up == n.get_tree_root():
                    n.delete()
                    continue
                if parent.up.up == n.get_tree_root():
                    n.delete()
                    continue
                while not parent.up.up.is_root():
                    if parent in not_yellows:
                        flag = 0
                        break
                    parent = parent.up
                if flag:
                    d += 1
                    n.delete()
    ts = ete3.TreeStyle()
    ts.show_leaf_name = True
    ts.min_leaf_separation = 0.5
    ts.mode = "c"
    ts.root_opening_factor = 0.75
    ts.show_branch_length = False
    return t, ts


def draw_tree(ax: plt.Axes ,series, dict, folder):
    if type(dict["treshold"]) == tuple:
        lower_threshold, higher_threshold = dict["treshold"]
    else:
        lower_threshold, higher_threshold = -dict["treshold"], dict["treshold"]
    newick, graph = tree_to_newick(series)
    try:
        t, ts = get_tree_shape(newick, graph, lower_threshold, higher_threshold, dict)
    except:
        print("not enough bacterias to create a tree")
        return None
    for n in t.traverse():
        while re.match(r'_+\d', n.name.split("|")[-1]):
            n.name = "<" + '|'.join(n.name.split('|')[:-1]) + ">"
        c = n.name.count('|') + 1
        n.name = ";".join(n.name.strip("<>").split("|")[-2:]).replace("[", "").replace("]", "")
        # n.name = delete_suffix(str(n.name))

    t.render("phylotree.svg", tree_style=ts)
    t.render("phylotree.png", tree_style=ts)
    tree = plt.imread('./phylotree.png')
    im = ax.imshow(tree)
    return 1


def delete_suffix(i):
    m = re.search(r'_+\d+$', i)
    if m is not None:
        i = i[:-(m.end()-m.start())]
    return i