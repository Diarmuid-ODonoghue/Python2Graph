import csv
import os
import random
from itertools import islice
import pandas as pd

"""convert full graphs to csv"""
def to_csv(G, lb, filenum, filename, location):
    with open(location + 'graph%s.csv' % filename[:-3], 'w', newline='') as file:

        w = csv.writer(file, delimiter=',')
        if filename.endswith('.py'):
            mod_filename = filename[:-3]

        for x in G:
            w.writerow([mod_filename, lb[x[1]], G[x], lb[x[0]], x[1], x[0]])


"""Convert method graphs to csv"""
def method_to_csv(G, lb, filenum, name, filename, location):
    with open(location + str(filenum) + "_" + filename + "_" + name + '.csv', 'w+', newline='') as file:
        w = csv.writer(file, delimiter=',')
        for x in G:
            w.writerow([filename, lb[x[1]], G[x], lb[x[0]], x[1], x[0]])
    print(filename, "CSV generated")


"""testing to see if all the csv are readable"""
def test_csv(location):
    for root, dirs, files in os.walk(location):
            for name in files:
                try:
                    with open(location + name) as file:
                        reader = csv.reader(file)
                        for row in reader:
                           pass
                except:
                    print("ERROR in" + name)
                    os.remove(location + name)
                    pass

def getresults():

    with open('F:/DFSMatching/ResultsOutput/summary.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        with open('F:/dfsSummary/summary2.csv', 'w', newline='') as file:
            w = csv.writer(file, delimiter=',')
            reader1 = csv.reader(csvfile)
            choices = ()
            #lines = [line for line in csvfile]

            #random_choice = random.sample(lines, 10)
            choices = ["9566_demo_data_main.csv,21206_key_point_value_test_test_high.csv",
                       "5926_celery_node_monitor_test_tearDown.csv", "8029_CourseSelection_populate.csv",
                       "57276_workflowscene_items.csv", "8724_dashboard_page_is_previous_button_enabled.csv",
                       "7988_cost_functions___init__.csv,28388_PairAlign___len__.csv", "58767_znc_rehashconf.csv",
                       "6236_CimiRunner_getVmTemplatesParameters.csv", "57684_XSDataBioSaxsv1_0_marshal.csv",
                       "9716_DiagramObject_getDiagramObjectPoints.csv"]

            #for x in random_choice:
                #print(x)

            for row in reader:  # first 10 only
                if row['TARGET'] in choices:
                    w.writerow(
                        [row['TARGET'], row['SOURCE'], row['#T Conc'], row['#T Rels'], row['#S Cons'], row['#S rels'],
                         row['#Map Preds'], row['%Mapp'], row['AvLin Con'], row['AvWu Con'], row['#Infs'],
                         row['#MapCncpts'], row['AvgRelSim'], row['LrgCmpnt'], row['0'],
                         row['#WekConnCpnt, Score'], row['DFS']])



