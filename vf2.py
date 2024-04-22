from networkx.algorithms import isomorphism
import networkx as nx
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
#from Map2Graphs import build_graph_from_csv
from draw_graph import draw_graph
def match_vf2():
    location1 = "F:/Test3/"
    location2 = "F:/Test4/"
    count = 0

    for root, dirs, files in os.walk(location1):
        for name in files:
            with open(location1 + name) as file:
                edge_labels = {}
                labels = {}
                reader = csv.reader(file)
                try:
                    df = pd.read_csv(location1 + name)

                    """for index, row in df.iterrows():
                        edge_labels[row['index1'], row['index2']] = row['relationship']
                        labels[row['index1']] = row['source']
                        if row['index2'] not in labels:
                            labels[row['index2']] = row['target']"""
                    #G = nx.from_pandas_edgelist(df, 'index1', 'index2', edge_attr='relationship', create_using=nx.Graph())
                    # draw_graph(g, labels, edge_labels)
                except:
                    pass

                G1 = nx.from_pandas_edgelist(df, 'source', 'target', edge_attr='relationship', create_using=nx.Graph())

                for root2, dirs2, files2 in os.walk(location2):
                    for name2 in files2:
                        with open(location2 + name2) as file2:
                            edge_labels = {}
                            labels = {}
                            reader = csv.reader(file2)
                            try:
                                df = pd.read_csv(location2 + name2)

                                G2 = nx.from_pandas_edgelist(df, 'source', 'target', edge_attr='relationship',create_using=nx.Graph())
                                GM = isomorphism.GraphMatcher(G1, G2)
                                if GM.is_isomorphic() == True:
                                    count += 1
                                    file3 = open("succesful2.txt", "a")
                                    file3.write("\nNo." + str(count) +" Match in : " + name + " -> " + name2)
                                    file.close
                            except:
                                pass


def add_columns():
    for root, dirs, files in os.walk("F:/Test3/"):

            for name in files:
                try:
                    with open("F:/Test3/" + name) as file:
                        reader = csv.reader(file)

                        df1 = pd.read_csv("F:/Test3/" + name)
                        columns = ['filename','source','relationship', 'target', 'index1', 'index2']
                        df1.columns = columns

                        with open("F:/Test3/" + name, 'w', newline=''):
                            pass
                        df1.to_csv("F:/Test3/" + name, index=False)
                except:
                    pass