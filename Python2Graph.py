import ast
import sys
import time
import ShowGraphs
import pandas as pd
import networkx as nx
#from Map2Graphs import *
from draw_graph import *
from toCSV import *
from getPythonFiles import *
from vf2 import *
startTime = time.time()

#getresults()
"""Settings"""
methods_only = True  # if you want to find the individual methods in a file
draw = False  # if you want to draw the full graph of a whole python file
convert_to_csv = True  # convert the full file into one csv file
draw_methods = True  # if you want to draw the individual methods
convert_method_to_csv = True  # convert only mehtods to csv's
print_tree = False  # if you want to print the tree to the console
global current_file

"""where the python code is located"""
basePath = os.path.dirname(os.path.realpath(__file__,)).replace('\\', '/') + '/'
# basePath = "C:/Users/dodonoghue/Documents/Python-Me/Cre8blend/"
""" where you want the csv's to be saved"""
csvLocation = basePath + "output-graphs/"  # where you want the csv's to be saved
pyLocation = basePath + "python-corpus/"  # where the corpus of python files is located
# newLocation = basePath + "py-Corpus/"  # where you want the corpus of python file to be copied to

filenum = 1
filename = ''
methodnum = 0
arguments = []
num = 0


""" for getting attributes e.g(car.colour())"""
def getattributes(node, attr):
    attribute_name = str(node)
    if (attribute_name[0:9] == "<ast.Name"):
        attr = node.id + "." + attr
    if (attribute_name[0:14] == "<ast.Attribute"):
        attr = str(node.attr) + "." + attr
        try:
            x = node.value
            return(getattributes(x, attr))
        except:
            pass
    return(attr[:-1] + "()")


"""getting the relationship between the nodes and adding label to the edge"""
def generate_edge_labels(index, preindex, labels, edge_labels, arg, attr, dep, con):
    valid = False
    while not valid:
        if preindex in labels:
            valid = True
        else:
            preindex -= 1
        x = labels[preindex].split('.')
        striped = x[-1]

    if x[-1].strip() == "Return":
        edge_labels[index, preindex] = "Return"
    elif striped[1:10] == "CompareOp":
        edge_labels[index, preindex] = "Compare"
    elif striped[1:5] == "Call":
        edge_labels[index, preindex] = "Parameter"
    elif con == True:
        edge_labels[index, preindex] = "Condition"
    elif dep == True:
        edge_labels[index, preindex] = "Depends"
    elif arg == True:
        edge_labels[index, preindex] = "Parameter"
    elif attr == True:
        edge_labels[index, preindex] = "Attribute"
    else:
        edge_labels[index, preindex] = "Contains"


""" The method that creates the graphs in networkx"""
def generate_graph(node, nodes, preindex, labels, edge_labels, nm, preNode, g):
    global convert_method_to_csv
    global draw_methods
    global num
    global methods_only
    global filenames

    node_names = ["Module", "Assign", "Name", "Constant", "UnaryOp", "For", "While",
                  "Expr", "If", "For", "Try", "And", "Or", "UAdd", "USub", "Not",
                  "Tuple", "Set", "Slice", "Return", "ExceptHandler", "ImportFrom", "Pass", "Break",
                  "Continue","Compare"]

    comparators = ["Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn"]

    mathoperators = ["Add", "Mult", "Div", "FloorDiv", "Mod", "Pow", "LShift", "RShift", "BitOr", "BitXor", "BitAnd",
                     "MatMult"]
    name = str(type(node).__name__)
    name1 = str(type(node).__name__)
    global arguments
    index = len(g.nodes)

    if name == "FunctionDef":
        if methods_only == True:
            if index > 85:
                print(" dud", index, end=" ")
            global methodnum
            global filename
            methodnum += 1
            newIndex = 0
            nm3 = {}
            lb2 = {}
            edge_lb2 = {}
            gm = nx.Graph()
            gm.add_node(newIndex)
            lb2[newIndex] = str(newIndex) + ". Module"
            preindex = 0

            newIndex = len(gm.nodes)
            gm.add_node(newIndex)
            gm.add_edge(preindex, newIndex)
            lb2[newIndex] = str(newIndex) + ". Block"
            generate_edge_labels(newIndex, preindex, lb2, edge_lb2, False, False, False, False)
            preindex = 1

            newIndex = len(gm.nodes)
            gm.add_node(newIndex)
            gm.add_edge(preindex, newIndex)
            lb2[newIndex] = str(newIndex) + ". " + name + " : " + node.name
            generate_edge_labels(newIndex, preindex, lb2, edge_lb2, False, False, False, False)
            tempIndex = newIndex

            for arg in node.args.args:
                newIndex = len(gm.nodes)
                nodes.append(newIndex)
                gm.add_node(newIndex)
                preindex = tempIndex
                lb2[newIndex] = str(newIndex) + ". " + arg.arg
                if arg.arg not in nm3:
                    nm3[arg.arg] = newIndex
                if newIndex != preindex:
                    gm.add_edge(tempIndex, newIndex)
                    generate_edge_labels(newIndex, tempIndex, lb2, edge_lb2, True, False, False, False)

            for f in ast.iter_fields(node):
                if f[0] == "body":
                    newIndex = len(gm.nodes)
                    nodes.append(newIndex)
                    gm.add_node(newIndex)
                    preindex = newIndex
                    lb2[newIndex] = str(newIndex) + ". Block"
                    gm.add_edge(tempIndex, newIndex)
                    generate_edge_labels(newIndex, tempIndex, lb2, edge_lb2, False, False, False, False)

            for n in ast.iter_child_nodes(node):
                generate_graph(n, [], newIndex, lb2, edge_lb2, nm3, node, gm)
            if convert_method_to_csv == True:
                method_to_csv(edge_lb2, lb2, methodnum, node.name, filename, csvLocation)
            if methodnum == 10000:
                sys.exit()
            if draw_methods == True:
                # draw_graph(gm, lb2, edge_lb2)
                new_graph = nx.MultiDiGraph()
                new_graph.clear()
                for x in edge_lb2:
                    rel = edge_lb2[(x[0], x[1])]
                    new_graph.add_node(lb2[x[0]], label=lb2[x[0]])
                    new_graph.add_node(lb2[x[1]], label=lb2[x[1]])
                    new_graph.add_edge(lb2[x[0]], lb2[x[1]], label=rel )
                preds = returnEdgesAsList(new_graph)
                ShowGraphs.show_blended_space_big_nodes(new_graph, preds, [], [],
                                                        output_filename=current_file[:-3])
        else:
            nm2 = {}
            index = len(g.nodes)
            tempIndex = index
            nodes.append(index)
            g.add_node(index)
            labels[index] = str(index) + ". " + name + " : " + node.name

            if index != preindex:
                g.add_edge(preindex, index)
                generate_edge_labels(index, preindex, labels, edge_labels, False, False, False, False)

            for arg in node.args.args:
                index = len(g.nodes)
                nodes.append(index)
                g.add_node(index)
                preindex = tempIndex
                labels[index] = str(index) + ". " + arg.arg
                if arg.arg not in nm2:
                    nm2[arg.arg] = index
                if index != preindex:
                    g.add_edge(preindex, index)
                    generate_edge_labels(index, preindex, labels, edge_labels, True, False, False, False)

            for f in ast.iter_fields(node):
                if f[0] == "body":
                    index = len(g.nodes)
                    nodes.append(index)
                    g.add_node(index)
                    preindex = tempIndex
                    labels[index] = str(index) + ". Block"
                    if index != preindex:
                        g.add_edge(preindex, index)
                        generate_edge_labels(index, preindex, labels, edge_labels, False, False, False, False)

            for n in ast.iter_child_nodes(node):
                generate_graph(n, nodes, index, labels, edge_labels, nm2, node, g)

    elif name == "ImportFrom":
        tempindex = len(g.nodes)
        nodes.append(index)
        labels[tempindex] = str(tempindex) + ". ImportFrom : " + node.module
        if index != preindex:
            g.add_edge(preindex, tempindex)
            generate_edge_labels(tempindex, preindex, labels, edge_labels, False, False, True, False)

        for x in node.names:
            index = len(g.nodes)
            nodes.append(index)
            labels[index] = str(index) + ". Alias: " + str(x.name)
            if index != preindex:
                g.add_edge(tempindex, index)
                generate_edge_labels(index, tempindex, labels, edge_labels, False, False, True, False)

    elif name == "Import":
        tempindex = len(g.nodes)
        nodes.append(index)
        labels[tempindex] = str(tempindex) + ". Import"
        if index != preindex:
            g.add_edge(preindex, tempindex)
            generate_edge_labels(tempindex, preindex, labels, edge_labels, False, False, True, False)

        for x in node.names:
            index = len(g.nodes)
            nodes.append(index)
            labels[index] = str(index) + ". Alias: " + str(x.name)
            if index != preindex:
                g.add_edge(tempindex, index)
                generate_edge_labels(index, tempindex, labels, edge_labels, False, False, True, False)

    elif name == "BoolOp":
        index = len(g.nodes)
        nodes.append(index)
        g.add_node(index)
        preindex = index - 2
        name = str(node.op)
        if name[0:8] == "<ast.And":
            name = "And"
        else:
            name = "Or"
        labels[index] = str(index) + ". BoolOp : " + name
        if index != preindex:
            g.add_edge(preindex, index)
            generate_edge_labels(index, preindex, labels, edge_labels, False, False, False, False)
        for n in node.values:
            generate_graph(n, nodes, index + 1, labels, edge_labels, nm, node, g)

    elif name == "Call":
        tempindex = len(g.nodes)
        nodes.append(index)
        g.add_node(tempindex)
        attribute_names = ""
        try:
            labels[tempindex] = str(tempindex) + ". Call: "  + str(node.func.id) +"()"
        except:
            attr = getattributes(node.func, "")
            labels[tempindex] = str(tempindex) + ". Call: "  + getattributes(node.func, "")

        if index != preindex:
            g.add_edge(preindex, tempindex)
            generate_edge_labels(tempindex, preindex, labels, edge_labels, False, False, False, False)
        for x in node.args:
            generate_graph(x, nodes, index, labels, edge_labels, nm, node, g)

        for keyword in node.keywords:
            generate_graph(keyword.value, nodes, index, labels, edge_labels, nm, node, g)

    elif name == "While":
        index = len(g.nodes)
        tempIndex = index
        nodes.append(index)
        g.add_node(index)
        g.add_edge(preindex, index)
        labels[index] = str(index) + ". WhileLoop"
        generate_edge_labels(index, preindex, labels, edge_labels, False, False, False, False)
        try:
            name2 = str(node.test)
            if name2[0:13] == "<ast.Constant":
                index = len(g.nodes)
                nodes.append(index)
                g.add_node(index)
                g.add_edge(index - 1, index)
                labels[index] = str(index) + ". " + str(node.test.value)
                generate_edge_labels(index, index - 1, labels, edge_labels, False, False, False, True)
            if name2[0:9] == "<ast.Name":
                if node.test.id in nm:
                    index = len(g.nodes)
                    nodes.append(node.test.id)
                    labels[nm[node.test.id]] = str(node.test.id)
                    g.add_edge(index - 1, nm[node.test.id])
                    generate_edge_labels(nm[node.test.id], index - 1, labels, edge_labels, False, False, False, True)
                else:
                    index = len(g.nodes)
                    nodes.append(index)
                    g.add_node(index)
                    g.add_edge(index - 1, index)
                    labels[index] = str(index) + ". " + str(node.test.id)
                    generate_edge_labels(index, index - 1, labels, edge_labels, False, False, False, True)

        except:
            pass

        for f in ast.iter_fields(node):
            if f[0] == "body":
                index = len(g.nodes)
                nodes.append(index)
                g.add_node(index)
                preindex = tempIndex
                labels[index] = str(index) + ". Block"
                if index != preindex:
                    g.add_edge(preindex, index)
                    generate_edge_labels(index, preindex, labels, edge_labels, False, False, False, False)
        for n in ast.iter_child_nodes(node):
            if str(n)[0:9] == "<ast.Name" or str(n)[0:13] == "<ast.Constant":
                continue

            generate_graph(n, nodes, index, labels, edge_labels, nm, node, g)
    elif name == "If":
        index = len(g.nodes)
        tempIndex = index
        nodes.append(index)
        g.add_node(index)
        g.add_edge(preindex, index)
        labels[index] = str(index) + ". If"
        generate_edge_labels(index, preindex, labels, edge_labels, False, False, False, False)
        try:
            name2 = str(node.test)
            if name2[0:13] == "<ast.Constant":
                index = len(g.nodes)
                nodes.append(index)
                g.add_node(index)
                g.add_edge(index-1, index)
                labels[index] = str(index) + ". " + str(node.test.value)
                generate_edge_labels(index, index-1, labels, edge_labels, False, False, False, True)
            if name2[0:9] == "<ast.Name":
                if node.test.id in nm:

                    index = len(g.nodes)
                    nodes.append(node.test.id)

                    labels[nm[node.test.id]] = str(node.test.id)
                    g.add_edge(index-1, nm[node.test.id])
                    generate_edge_labels(nm[node.test.id], index-1, labels, edge_labels, False, False, False, True)
                else:
                    index = len(g.nodes)
                    nodes.append(index)
                    g.add_node(index)
                    g.add_edge(index-1, index)
                    labels[index] = str(index) + ". " + str(node.test.id)
                    generate_edge_labels(index, index-1, labels, edge_labels, False, False, False, True)
        except:
            pass

        for f in ast.iter_fields(node):
            if f[0] == "body":
                index = len(g.nodes)
                nodes.append(index)
                g.add_node(index)
                preindex = tempIndex
                labels[index] = str(index) + ". Block"
                if index != preindex:
                    g.add_edge(preindex, index)
                    generate_edge_labels(index, preindex, labels, edge_labels, False, False, False, False)
        for n in ast.iter_child_nodes(node):
            if str(n)[0:9] == "<ast.Name" or str(n)[0:13] == "<ast.Constant":
                continue
            generate_graph(n, nodes, index, labels, edge_labels, nm, node, g)

    elif name == "For":
        index = len(g.nodes)
        nodes.append(index)
        g.add_node(index)
        tempIndex = index
        labels[index] = str(index) + ". ForLoop"
        g.add_edge(preindex, index)
        generate_edge_labels(index, preindex, labels, edge_labels, False, False, False, False)

        try:
            name2 = str(node.target)
            if name2[0:13] == "<ast.Constant":

                index = len(g.nodes)
                nodes.append(index)
                g.add_node(index)
                g.add_edge(index-1, index)
                labels[index] = str(index) + ". Constant: " + str(node.target.value)
                generate_edge_labels(index, tempIndex, labels, edge_labels, False, False, False, True)
            elif name2[0:9] == "<ast.Name":
                if node.target.id in nm:
                    index = len(g.nodes)
                    nodes.append(index)
                    g.add_edge(index-1, nm[node.target.id])
                    generate_edge_labels(nm[node.target.id], index-1, labels, edge_labels, False, False, False, True)
                else:
                    index = len(g.nodes)
                    nodes.append(index)
                    g.add_node(index)
                    g.add_edge(index-1, index)
                    labels[index] = str(index) + ". Name: " + str(node.target.id)
                    generate_edge_labels(index, tempIndex, labels, edge_labels, False, False, False, True)
            else:
                generate_graph(node.target, nodes, tempIndex, labels, edge_labels, nm, node, g)

        except:
            pass

        try:
            name2 = str(node.iter)
            if name2[0:13] == "<ast.Constant":

                index = len(g.nodes)
                nodes.append(index)
                g.add_node(index)
                g.add_edge(index-1, index)
                labels[index] = str(index) + ". Constant: " + str(node.iter.value)
                generate_edge_labels(index, tempIndex, labels, edge_labels, False, False, False, True)
            elif name2[0:9] == "<ast.Name":
                if node.target.id in nm:
                    index = len(g.nodes)
                    nodes.append(index)
                    g.add_edge(tempIndex, nm[node.iter.id])
                    generate_edge_labels(nm[node.iter.id], tempIndex, labels, edge_labels, False, False, False, True)
                else:
                    index = len(g.nodes)
                    nodes.append(index)
                    g.add_node(index)
                    g.add_edge(tempIndex, index)
                    labels[index] = str(index) + ". Name: " + str(node.iter.id)
                    generate_edge_labels(index, tempIndex, labels, edge_labels, False, False, False, True)
            else:
                generate_graph(node.iter, nodes, tempIndex, labels, edge_labels, nm, node, g)
        except:
            pass

        for f in ast.iter_fields(node):
            if f[0] == "body":
                index = len(g.nodes)
                nodes.append(index)
                g.add_node(index)
                preindex = tempIndex
                labels[index] = str(index) + ". Block"
                if index != preindex:
                    g.add_edge(preindex, index)
                    generate_edge_labels(index, preindex, labels, edge_labels, False, False, False, False)

        for n in ast.iter_child_nodes(node):
            if str(n)[0:9] == "<ast.Name" or str(n)[0:13] == "<ast.Constant" or str(n)[0:9] == "<ast.List" or str(n)[0:9] == "<ast.Call" or str(n)[0:10] == "<ast.Tuple":
                continue
            generate_graph(n, nodes, index, labels, edge_labels, nm, node, g)


    elif name == "Dict":
        index = len(g.nodes)
        nodes.append(index)
        g.add_node(index)
        labels[index] = str(index) + ". Dictionary"
        g.add_edge(preindex, index)
        generate_edge_labels(index, preindex, labels, edge_labels, False, False, False, False)
        for f in ast.iter_fields(node):
            for t in f:
                for i in t:
                    if f[0] == "keys":
                        Name2 = "Key :"
                    else:
                        Name2 = "value :"
                    try:
                        print(i.value)
                        index = len(g.nodes)
                        nodes.append(index)
                        g.add_node(index)

                        labels[index] = str(index) + ". " + Name2 + " " + str(i.value)
                        if index != preindex:
                            g.add_edge(preindex + 2, index)
                            generate_edge_labels(index, preindex + 2, labels, edge_labels, False, False, False, False)
                    except:
                        try:
                            print(i.id)
                            index = len(g.nodes)
                            nodes.append(index)
                            g.add_node(index)
                            labels[index] = str(index) + ". " + Name2 + " " + str(i.id)
                            if index != preindex:
                                g.add_edge(preindex + 2, index)
                                generate_edge_labels(index, preindex + 2, labels, edge_labels, False, False, False, False)
                        except:
                            pass


    elif name == "Compare":
        index = len(g.nodes)
        nodes.append(index)
        g.add_node(index)
        g.add_edge(preindex - 1, index)

        for i in ast.iter_child_nodes(node):
            name2 = str(type(i).__name__)
            op = ""
            if name2 in comparators:
                if name2 == "Lt":
                    op += "<"
                elif name2 == "Gt":
                    op += ">"
                elif name2 == "LtE":
                    op += "<="
                elif name2 == "GtE":
                    op += ">="
                elif name2 == "Eq":
                    op += "=="
                elif name2 == "NotEq":
                    op += "!="
                else:
                    op += name2
                name = "CompareOp: " + op

        labels[index] = str(index) + ". " + name
        if index != preindex:
            generate_edge_labels(index, preindex - 1, labels, edge_labels, False, False, False, True)

        for n in ast.iter_child_nodes(node):
            generate_graph(n, nodes, index, labels, edge_labels, nm, node, g)

    elif name == "BinOp":
        index = len(g.nodes)
        nodes.append(index)
        g.add_node(index)
        g.add_edge(preindex, index)

        for i in ast.iter_child_nodes(node):
            name2 = str(type(i).__name__)
            op = ""
            if name2 in mathoperators:
                if name2 == "Add":
                    op += "+"
                elif name2 == "Sub":
                    op += "-"
                elif name2 == "Mult":
                    op += "*"
                elif name2 == "Div":
                    op += "/"
                else:
                    op += name2
                name = "mathOp: " + op

        labels[index] = str(index) + ". " + name
        if index != preindex:
            generate_edge_labels(index, preindex, labels, edge_labels, False, False, False, False)

        for n in ast.iter_child_nodes(node):
            generate_graph(n, nodes, index, labels, edge_labels, nm, node, g)

    elif name == "AugAssign":
        index = len(g.nodes)
        nodes.append(index)
        g.add_node(index)
        g.add_edge(preindex, index)

        for i in ast.iter_child_nodes(node):
            name2 = str(type(i).__name__)
            if name2 in mathoperators:
                name = "AugAssign: " + name2

        labels[index] = str(index) + ". " + name

        if index != preindex:
            generate_edge_labels(index, preindex, labels, edge_labels, False, False, False, False)

        for n in ast.iter_child_nodes(node):
            generate_graph(n, nodes, index, labels, edge_labels, nm, node, g)

    elif name == "List":
        index = len(g.nodes)
        nodes.append(index)
        g.add_node(index)
        labels[index] = str(index) + ". List"
        g.add_edge(preindex, index)
        generate_edge_labels(index, preindex, labels, edge_labels, False, False, False, False)
        for n in node.elts:
            try:
                generate_graph(n, nodes, index, labels, edge_labels, nm, node, g)
            except:
                pass

    elif name == "ClassDef":
        fnm = {}
        index = len(g.nodes)
        tempIndex = index
        nodes.append(index)
        g.add_node(index)
        labels[index] = str(index) + ". " + name + " : " + node.name
        if index != preindex:
            g.add_edge(preindex, index)
            generate_edge_labels(index, preindex, labels, edge_labels, False, False, False, False)

        for x in node.bases:
            generate_graph(x, nodes, index, labels, edge_labels, nm, node, g)

        for f in ast.iter_fields(node):
            if f[0] == "body":
                index = len(g.nodes)
                nodes.append(index)
                g.add_node(index)
                preindex = tempIndex
                labels[index] = str(index) + ". Block"
                if index != preindex:
                    g.add_edge(preindex, index)
                    generate_edge_labels(index, preindex, labels, edge_labels, False, False, False, False)
                for x in f:
                    for n in x:
                        try:
                            generate_graph(n, nodes, index, labels, edge_labels, fnm, node, g)
                        except:
                            pass

        for n in ast.iter_child_nodes(node):

            generate_graph(n, nodes, index, labels, edge_labels, fnm, node,g)

    elif name in node_names:
        # if node is variable get the variable name
        if name == "Name":
            name = "Name : " + node.id
            if node.id not in nm:
                nm[node.id] = index
        if name == "Constant":
            name = "Constant : " + str(node.value)
        if name == "For":
            name = "Loop: For"
        if name == "Import":
            name = "Import :"
            for x in node.names:
                name += x.name + ", "
                if x.asname is not None:
                    name += " as " + x.asname

        # adding node to graph
        if name1 == "Name" and node.id in nm:
            index = len(g.nodes)
            nodes.append(nm[node.id])
            g.add_node(nm[node.id])

            if str(type(preNode).__name__) == "Attribute":
                g.add_edge(preindex, nm[node.id])
                labels[nm[node.id]] = str(nm[node.id]) + ". " + name + "." + preNode.attr
                if index != preindex:
                    generate_edge_labels(nm[node.id], preindex, labels, edge_labels, False, True, False, False)
            else:
                labels[nm[node.id]] = str(nm[node.id]) + ". " + name
                g.add_edge(preindex, nm[node.id])
                generate_edge_labels(nm[node.id], preindex, labels, edge_labels, False, False, False, False)

        elif name == "Return":
            index = len(g.nodes)
            nodes.append(index)
            g.add_node(index)
            labels[index] = str(index) + ". " + name
            if index != preindex:
                g.add_edge(preindex, index)
                generate_edge_labels(index, preindex, labels, edge_labels, False, False, False, False)
        else:
            index = len(g.nodes)
            nodes.append(index)
            g.add_node(index)
            g.add_edge(preindex, index)
            if str(type(preNode).__name__) == "Attribute":

                labels[index] = str(index) + ". " + name + "." + preNode.attr
                if index != preindex:
                    generate_edge_labels(index, preindex, labels, edge_labels, False, True, False, False)
            else:

                labels[index] = str(index) + ". " + name
                if index != preindex:
                    g.add_edge(preindex, index)
                    generate_edge_labels(index, preindex, labels, edge_labels, False, False, False, False)

        # to add block/body node to graph
        for f in ast.iter_fields(node):
            if f[0] == "body":
                index = len(g.nodes)
                nodes.append(index)
                g.add_node(index)
                preindex = index - 1
                labels[index] = str(index) + ". Block"
                if index != preindex:
                    g.add_edge(preindex, index)
                    generate_edge_labels(index, preindex, labels, edge_labels, False, False, False, False)

        for n in ast.iter_child_nodes(node):
            generate_graph(n, nodes, index, labels, edge_labels, nm, node, g)
    else:
        for n in ast.iter_child_nodes(node):
            generate_graph(n, nodes, index - 1, labels, edge_labels, nm, node, g)

# randomSelection()
# countFiles()


def returnEdgesAsList(G):  # returnEdgesAsList(sourceGraph)
    """ returns a list of lists, each composed of triples"""
    res = []
    for (u, v, reln) in G.edges.data('label'):
        res.append([u, reln, v])
    return res
# returnEdgesAsList(targetGraph)


def generate_full_graph():
    global filename
    global basePath
    global filenum
    count = 0

    for filenames in os.listdir(basePath):
        try:
            if filenames.endswith(".py"):
                filename = filenames[:-3]
                f = open(basePath + filenames, "r", encoding='utf8')
                while True:
                    try:
                        line = f.readline()
                    except UnicodeDecodeError:
                        f.close()
                        encodeing = 'windows-1252'
                        break
                    if not line:
                        f.close()
                        encoding = 'utf8'
                        break
                with open(basePath + filenames, "r", encoding=encoding) as source:
                    tree = ast.parse(source.read())
                file = open("logs.txt", "w+")

                print(ast.dump(tree, indent=4))
                g = nx.Graph()
                lb = {}
                edge_lb = {}
                nm = {}
                # print(filename)
                """Generating the graph"""
                generate_graph(tree, [], 0, lb, edge_lb, nm, g.nodes, g)
                # print(nm)
                # print(list(g.nodes))
                # print(g.edges.data())
                # print(lb)
                # print(edge_lb)
                if draw == True:
                    draw_graph(g,lb,edge_lb)
                if convert_to_csv == True:
                    to_csv(edge_lb, lb, filenum, filename)
                print(count)
                count += 1
                filenum += 1

        except Exception as er:

            """log what files have errors"""
            print(er)
            try:
                file = open("logs.txt", "a")
                file.write("\nError in " + filename + ": " + str(er))
                file.close
                filenum += 1
            except Exception as er:
                # file = open("logs.txt", "a")
                file.write("\nError in " + filename + ": " + str(er))
                file.close
                filenum += 1


"""function to kickstart the graph generation"""
def start():
    global filename
    global basePath
    global filenum
    global draw
    global convert_to_csv
    global current_file
    count = 0

    """if methods_only == False:
        generate_full_graph()
        sys.exit()"""
    file_list = os.listdir(pyLocation)  # os.listdir(basePath)
    for filenames in file_list:
        current_file =  filenames
        print("\n\n###Filename:",  filenames)
        try:
            if filenames.endswith(".py"):
                filename = filenames[:-3]
                f = open(pyLocation + filenames, "r", encoding='utf8')
                while True:
                    try:
                        line = f.readline()  # FIXME Detect comments here
                    except UnicodeDecodeError:
                        f.close()
                        encoding = 'windows-1252'
                        break
                    if not line:
                        f.close()
                        encoding = 'utf8'
                        break
                with open(pyLocation + filenames, "r", encoding=encoding) as source:
                    tree = ast.parse(source.read())
                file = open("logs.txt", "w+")
                if print_tree == True:
                    print(ast.dump(tree, indent=4))  # indent = 4 has issues working in linux best to take it out

                g = nx.Graph()
                lb, edge_lb, nm = {}, {}, {}
                generate_graph(tree, [], 0, lb, edge_lb, nm, g.nodes, g)  # Generating the graph
                print(edge_lb)
                print(g.nodes)
                print(lb)
                if convert_to_csv == True and methods_only == False:
                    to_csv(edge_lb, lb, filenum, filename, csvLocation)
                if draw == True and methods_only == False:
                     draw_graph(g, lb, edge_lb)

                print(count)  # counting the number of files that are done
                count += 1
                filenum += 1

        except Exception as er:
            """log what files have errors"""
            print(er)
            try:
                file = open("logs.txt", "a")
                file.write("\nError in " + filename + ": " + str(er))
                file.close
                filenum += 1
            except Exception as er:
                print(er)
        print(filenames, " done")


def get_results():
    df = pd.read_csv("F:/dfs3/ResultsOutput/summary.csv")
    df1 = pd.DataFrame([], columns=['TARGET', 'SOURCE', '#T Conc', '#T Rels', '#S Cons', '#S rels',
                                    '#Map Preds', '%Mapp', 'AvLin Con', 'AvWu Con', '#Infs',
                                    '#MapCncpts', 'AvgRelSim', 'LrgCmpnt', '0',
                                    '#WekConnCpnt, Score', 'DFS'])
    print(df)
    try:
        for index, row in df.iterrows():
            #print(row['#Map Preds'], row['#T Rels'])

            sim = (float(row['#Map Preds'])/ float(row['#T Rels']))

            print(str(row))

            if(float(row['#Map Preds'])/ float(row['#T Rels']) > 0.9):
                file3 = open("succesful3.txt", "a")
                df1 = df1.append(row)
                print(sim)
                file3.write(str(row.values) + "/n")
                file3.close
    except:
        pass
    print('test-this')
    df1.to_csv('success5.csv')

start()

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))


"""Some utility functions"""
#get_results()
#getPyFiles(pyLocation, newLocation) # to bring all python files in a directory to the top and then move them to a new location
#randomSelection("F:/Testing/", "F:/dfs3/", 10000)  # for getting a random selction of 10000 python files from a directory
#delrandomSelection("F:/matchingC2/", 5000)  # do delete a random selection from a directory
#test_csv("F:/dfs3/")  # some csv's have the wrong encoding and will throw an error in the graph matching and need to be removed
#countFiles(csvLocation)  # count all the files in a directory
#add_columns()
#add_columns()
#match_vf2()
