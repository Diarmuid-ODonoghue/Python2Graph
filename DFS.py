# DFS is a heuristic algorithm that presumes that edges with greater total significance
# should be given a greater priority than edges with a smaller total significance
# It explores the edge-space of graph-subgraph near isomorphism.
# loosely inspired by Nawaz, Enscore and Ham (NEH) algorithm
# local optimisation, near the global optimum.
# Edges are described by a 4-tuple of in/out degrees from a di-graph. 2 edges are compared by Wasserstein metric.
# I believe it's an admissible heuristic! A narrow search space is explored heuristically.
#
# import networkx as nx
import sys
import math
# import Map2Graphs.mode as mode
# from Map2Graphs import mode
# import numpy as np
from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet as wn
# import pdb

wnl = WordNetLemmatizer()

global mode
global term_separator
global max_topology_distance
global semantic_factors

max_topology_distance = 50  # in terms of a node's in/out degree
mode = "English"
# mode = 'code'
if mode == "English":
    term_separator = "_"  # Map2Graphs.term_separator
else:
    term_separator = ":"
semantic_factors = True

from sense2vec import Sense2Vec

# s2v = Sense2Vec().from_disk("C:/Users/user/Documents/Python-Me/Sense2Vec/s2v_reddit_2019_lg/")
s2v = Sense2Vec().from_disk("F:\s2v\s2v_reddit_2019_lg\s2v_reddit_2019_lg")
query = "drive|VERB"
assert query in s2v
s2v.similarity(['run' + '|VERB'], ['eat' + '|VERB'])
global s2v_cache
s2v_cache = dict()
# query = "can|VERB"
# assert query in s2v
vector = s2v[query]
freq = s2v.get_freq(query)


# most_similar = s2v.most_similar(query, n=3)
# print(most_similar)
def find_nearest(vec):
    for key, vec in s2v.items():
        print(key, vec)


beam_size = 4  # beam breadth for beam search
epsilon = 100
current_best_mapping = []
bestEverMapping = []


#def __init__(G1, G2):
#    self.G1 = G1
#    self.G2 = G2
#    self.core_1 = {}
#    self.core_2 = {}
#    self.mapping = self.core_1.copy()


def sim_to_dist(n):
    return 1 - 1 / (0.000001 + n)


def MultiDiGraphMatcher(target_graph, souce_graph):
    generate_and_explore_mapping_space(target_graph, souce_graph)


def is_isomorphic():
    print(" DFS.is_isomorphic() ")


# @staticmethod
def generate_and_explore_mapping_space(target_graph, source_graph):
    global current_best_mapping
    global bestEverMapping
    global semantic_factors
    current_best_mapping = []
    bestEverMapping = []
    if target_graph.number_of_nodes() == 0 or source_graph.number_of_nodes() == 0:
        return [], 0
    global beam_size
    global epsilon
    source_preds = return_sorted_predicates(source_graph)
    target_preds = return_sorted_predicates(target_graph)
    candidate_sources = []
    if target_graph.graph['Graphid'] == "2685984": # debuugery
        dud = 0
    for t_subj_in, t_subj_out, t_obj_in, t_obj_out, t_subj, t_reln, t_obj in target_preds:
        best_distance, composite_distance, best_subj, best_reln, best_obj \
            = sys.maxsize, sys.maxsize, "nil", "nil", "nil"
        beam = 0
        alternate_candidates = alternates_confirmed = []
        ## print("\nTT:", t_subj, t_reln, t_obj, end="  SS:   \t")
        numeric_influencing_factors = []
        for s_subj_in, s_subj_out, s_obj_in, s_obj_out, s_subj, s_reln, s_obj in source_preds:
            beam += 1
            topology_dist = math.sqrt( euclidean_distance(t_subj_in, t_subj_out, t_obj_in, t_obj_out,
                                               s_subj_in, s_subj_out, s_obj_in, s_obj_out) )
            if topology_dist > max_topology_distance:
                continue
            elif (t_subj == t_obj) and (s_subj != s_obj):  # what if one is a self-map && the other not ...
                continue
            #topology_dist = euc_to_unit(topology_dist)
            if semantic_factors:
                reln_dist = max(0.001, relational_distance(t_reln, s_reln))
                subj_dist = conceptual_distance(t_subj, s_subj)
                obj_dist = conceptual_distance(t_obj, s_obj)
            else:
                reln_dist = subj_dist = obj_dist = 1
            h_prime = math.sqrt(scoot_ahead(t_subj, s_subj, t_reln, s_reln, t_obj, s_obj, source_graph, target_graph))
            composite_sum = (reln_dist*4 + subj_dist + obj_dist) + topology_dist + h_prime
            composite_distance = (reln_dist*4 + subj_dist + obj_dist) * topology_dist * h_prime
            if composite_distance < best_distance:         # minimize distance
                best_distance = composite_distance
            alternate_candidates.append([s_subj_in, s_subj_out, s_obj_in, s_obj_out, s_subj, s_reln, s_obj,
                                         composite_distance, reln_dist, subj_dist, obj_dist, topology_dist, h_prime])
            numeric_influencing_factors.append([s_subj, s_reln, s_obj, (reln_dist + subj_dist + obj_dist), topology_dist,
                                                h_prime, composite_distance, composite_sum])
        alternate_candidates.sort(key=lambda x: x[7])
        numeric_influencing_factors.sort(key=lambda x: x[6])
        if False and numeric_influencing_factors != []:    # display
            for item in numeric_influencing_factors: #[:3]: ## only interested in first few results
                print(item[0], item[1], item[2],  '{:.2f}'.format(item[3]),  '{:.2f}'.format(item[4]),
                      '{:.2f}'.format(item[5]),  '({:.2f})'.format(item[6]), '[{:.2f}] '.format(item[7]), end= "")
        if len(alternate_candidates) > 0:
            alternates_confirmed = []
            for x in alternate_candidates:
                if abs(x[7] - best_distance) < epsilon: # and best_distance < 250.00
                    alternates_confirmed.append(x)  # flat list of sublists
        alternates_confirmed = alternates_confirmed[:beam_size]  # consider only best beam_size options
        candidate_sources.append(alternates_confirmed)  # ...(alternates_confirmed[0])
    #print("")
    #candidate_sources.sort(key=lambda x: x[0][7], reverse=True)
    reslt = explore_mapping_space(target_preds, candidate_sources, [])
    # zz = evaluateMapping(bestEverMapping[0])
    print("RESULT ", len(bestEverMapping), end="  ")
    return bestEverMapping, len(bestEverMapping)


#################################################################################################################
######################################### Scoot Ahead ###########################################################
#################################################################################################################


def return_best_in_combo(targetGraph, sourceGraph, t_subj, s_subj, t_preds, s_preds):  # {'lake': {0: {'label': 'fed_by'}}, 'they': {0: {'label': 'conveyed'}}})
    result_list = []
    t_in_deg, h_prime, rel_dist = 0, 0, 0
    for in_t_nbr, foovalue in t_preds:  # find MOST similar S & T pair
        in_t_rel = foovalue[0]['label']
        t_in_deg = targetGraph.in_degree[t_subj]
        for in_s_nbr, foovalue2 in s_preds:
            in_s_rel = foovalue2[0]['label']
            rel_dist = max(relational_distance(in_t_rel, in_s_rel), 0.01)
            s_in_deg = max(0.1, sourceGraph.in_degree[s_subj])
            scor = dist_to_sim(rel_dist) * min(max(1, t_in_deg), s_in_deg)
            result_list.append([scor, in_t_rel, in_s_rel, in_t_nbr, in_s_nbr])
    result_list.sort(key=lambda x: x[0], reverse=True)  # sum the first out_degree numbers
    h_prime = sum(i[0] for i in result_list[:t_in_deg])
    return h_prime, dist_to_sim(rel_dist)


def return_best_out_combo(targetGraph, sourceGraph, t_obj, s_obj, t_preds,
                          s_preds):  # {'lake': {0: {'label': 'fed_by'}}, 'they': {0: {'label': 'conveyed'}}})
    result_list = []
    t_out_deg, h_prime, rel_dist = 0, 0, 0
    for out_t_nbr, foovalue in t_preds:
        in_t_rel = foovalue[0]['label']
        t_out_deg = targetGraph.out_degree[t_obj]
        for out_s_nbr, foovalue2 in s_preds:
            in_s_rel = foovalue2[0]['label']
            rel_dist = relational_distance(in_t_rel, in_s_rel)
            s_out_deg = max(0.1, sourceGraph.out_degree[s_obj])
            scor = dist_to_sim(rel_dist) * min(max(1, targetGraph.out_degree[t_obj]), s_out_deg)
            result_list.append([scor, in_t_rel, in_s_rel, out_t_nbr, out_s_nbr])
    result_list.sort(key=lambda x: x[0], reverse=True)  # sum the first out_degree numbers
    h_prime = sum(i[0] for i in result_list[:t_out_deg])
    return h_prime, dist_to_sim(rel_dist)


def scoot_ahead(t_subj, s_subj, t_reln, s_reln, t_obj, s_obj, sourceGraph, targetGraph, reslt=[], level=1):
    if level == 0:
        return 0  # apply sum iterator over reslt
    reln_sim = dist_to_sim(relational_distance(t_reln, s_reln))
    subj_dist = conceptual_distance(t_subj, s_subj)
    obj_dist = conceptual_distance(t_obj, s_obj)
    best_in_links, in_rel_sim = return_best_in_combo(targetGraph, sourceGraph, t_subj, s_subj,
                                                     targetGraph.pred[t_subj].items(), sourceGraph.pred[s_subj].items())
    best_out_links, out_rel_sim = return_best_out_combo(targetGraph, sourceGraph, t_obj, s_obj,
                                                        targetGraph.succ[t_obj].items(),
                                                        sourceGraph.succ[s_obj].items())
    reln_val = reln_sim + subj_dist + obj_dist + in_rel_sim + out_rel_sim
    reln_val = max(0.01, reln_val)
    return reln_val


def sim_to_dist(sim):
    return 1 - sim


def dist_to_sim(dis):
    return (dis - 1) * -1


def euc_dist_to_sim(sim):
    return 1 / (1 + (sim ** 2))


def euc_to_unit(dist):
    return 1 - (dist - 1) * -1

# #####################################################################################################################
# #####################################################################################################################
# #####################################################################################################################


def explore_mapping_space(t_preds_list, s_preds_list, globl_mapped_predicates):
    """ Map the next target pred, by finding a mapping from the sources"""
    global max_topology_distance
    global bestEverMapping
    if len(globl_mapped_predicates) > len(bestEverMapping):  # compare scores, not lengths?
        if len(bestEverMapping) >0:
            print("¬", end="")
        # print("NB:", bestEverMapping)
        bestEverMapping = globl_mapped_predicates
    if t_preds_list == [] or s_preds_list == []:
        return globl_mapped_predicates
    elif t_preds_list != [] and s_preds_list[0] == []:
        explore_mapping_space(t_preds_list[1:], s_preds_list[1:], globl_mapped_predicates)  # skip t_pred
        explore_mapping_space(t_preds_list, s_preds_list[1:], globl_mapped_predicates)
        # bestEverMapping = globl_mapped_predicates  # works for Greedy solution
        # return globl_mapped_predicates
    elif t_preds_list == [] or s_preds_list == []:
        if len(globl_mapped_predicates) > len(bestEverMapping):
            bestEverMapping = globl_mapped_predicates
        return globl_mapped_predicates
        # recursive_search(t_preds_list[1:], s_preds_list[1:], globl_mapped_predicates)
    t_subj_in, t_subj_out, t_obj_in, t_obj_out, t_subj, t_reln, t_obj = t_preds_list[0]
    if type(s_preds_list[0]) is int:  # Error
        sys.exit(22)
    if type(s_preds_list[0]) is list:
        if s_preds_list[0] == []:
            if len(s_preds_list) > 0:
                if s_preds_list[0] == []:
                    current_options = []
                else:
                    current_options = s_preds_list[1:][0]
            else:
                current_options = []
        elif type(s_preds_list[0][0]) is list:  # alternates list
            current_options = s_preds_list[0]
        else:
            current_options = [s_preds_list[0]]  # wrap the single pred within a list
    else:
        sys.exit("DFS.py Error - s_preds_list malformed :-(")
    candidates = []
    for singlePred in current_options:  # from s_preds_list
        s_subj_in, s_subj_out, s_obj_in, s_obj_out, s_subj, s_reln, s_obj, composite_distance,\
            reln_dist, subj_dist, obj_dist, topology_dist, h_prime= singlePred
        mapped_subjects = check_if_already_mapped(t_subj, s_subj, globl_mapped_predicates) # together
        mapped_objects = check_if_already_mapped(t_obj, s_obj, globl_mapped_predicates)
        unmapped_subjects = check_if_both_unmapped(t_subj, s_subj, globl_mapped_predicates) # unmapped target
        unmapped_objects = check_if_both_unmapped(t_obj, s_obj, globl_mapped_predicates)
        already_mapped_source = check_if_source_already_mapped([s_subj, s_reln, s_obj], globl_mapped_predicates) # check if source already mapped to something else
        if already_mapped_source: # unavailable for mapping 22 Feb. 24
            continue
        elif mapped_subjects:  # Bonus for intersecting with the current mapping
            composite_distance = composite_distance / 2
        elif not unmapped_subjects:
            composite_distance = composite_distance # max_topology_distance
            continue
        if mapped_objects:
            composite_distance = composite_distance / 2
        elif not unmapped_objects:
            continue
        if s_subj == s_obj or t_subj == t_obj:    # Reflexive relations? detect and duplicate
            if (s_subj == s_obj and t_subj != t_obj) or (s_subj != s_obj and t_subj == t_obj): #
                composite_distance = max_topology_distance   #composite_product = (reln_dist*4 + subj_dist + obj_dist) * topology_dist * h_prime
            else:
                candidates = candidates + [[composite_distance, s_subj, s_reln, s_obj]] # add mapping option
        else:
            candidates = candidates + [[composite_distance, s_subj, s_reln, s_obj]]
    candidates.sort(key=lambda x: x[0])
    for dist, s_subj, s_reln, s_obj in candidates:  # assign best
        candidate_pair = [[t_subj, t_reln, t_obj], [s_subj, s_reln, s_obj], dist]
        if (check_if_already_mapped(t_subj, s_subj, globl_mapped_predicates) or    # add candidate_pair to mapping
                check_if_both_unmapped(t_subj, s_subj, globl_mapped_predicates)):
            if check_if_already_mapped(t_obj, s_obj, globl_mapped_predicates) or \
                    check_if_both_unmapped(t_obj, s_obj, globl_mapped_predicates):
                return explore_mapping_space(t_preds_list[1:], s_preds_list[1:],
                                             globl_mapped_predicates + [candidate_pair])
        #else:  # candidate_pair incompatible with current mapping
        dud = 0
    return explore_mapping_space(t_preds_list[1:], s_preds_list[1:], globl_mapped_predicates) # skip candidate_pair
    return explore_mapping_space(t_preds_list[1:], s_preds_list, globl_mapped_predicates) # 22 Feb 28
    return explore_mapping_space(t_preds_list, s_preds_list[1:], globl_mapped_predicates) # 22 Feb 28


def check_if_source_already_mapped(s_pred, globl_mapped_predicates):
    if s_pred == [] or globl_mapped_predicates == []:
        return False
    for mapped in globl_mapped_predicates:
        if s_pred[0] == mapped[1][0] and s_pred[1] == mapped[1][1] and s_pred[2] == mapped[1][2]:
            return True
    return False


# assert( checkIfUnmapped(1, 11, [ [1,2,3], [11,12,13] ])  ) == False
# checkIfUnmapped(1, 11, [[1,11]])


def check_if_already_mapped(t_subj, s_subj, globl_mapped_predicates):
    if globl_mapped_predicates == []:
        return False
    for x in globl_mapped_predicates:  # second_head()
        t_s, t_v, t_o = x[0]
        s_s, s_v, s_o = x[1]
        if t_subj == t_s and s_subj == s_s:
            return True
        elif t_subj == t_o and s_subj == s_o:
            return True  # and if its a commutitative relation...
        elif t_subj == s_s and s_subj == t_o:
            return True
        elif t_subj == s_o and s_subj == t_s:
            return True
    return False


def check_if_both_unmapped(t_subj, s_subj, globl_mapped_predicates):
    """ check if both are unmapped and are free to form any new mmapping """
    if globl_mapped_predicates == []:
        return True
    for x in globl_mapped_predicates:
        t_s, t_v, t_o = x[0]  # target
        s_s, s_v, s_o = x[1]
        if t_subj == t_s or s_subj == s_s:
            return False
        elif t_subj == t_o or s_subj == s_o:
            return False
        #else: # hide
        #    return True
    return True


def check_if_unmapped_DEPRECATED(t_subj, s_subj, globl_mapped_predicates):
    if t_subj == 'shot':
        x = 0
    if globl_mapped_predicates == []:
        return True
    for x in globl_mapped_predicates:
        if t_subj in x[0] or t_subj in x[1]:
            return False
        elif s_subj in x[0] or s_subj in x[1]:
            return False
    return True


def evaluateMapping(globl_mapped_predicates):
    """[[['hawk_Karla_she', 'saw', 'hunter'], ['hawk_Karla_she', 'know', 'hunter'], 0.715677797794342], [['h"""
    mapping = dict()
    relatio_structural_dist = 0
    for t_pred, s_pred, val in globl_mapped_predicates:  # t_pred,s are full predicates
        relatio_structural_dist += val
        if t_pred == [] or s_pred == []:
            continue
        # print("{: >20} {: >10} {: >20}".format(s, v, o, "    ==    "))
        if t_pred[0] not in mapping.keys() and s_pred[0] not in mapping.values():
            mapping[t_pred[0]] = s_pred[0]
            mapping[t_pred[2]] = s_pred[2]
        elif mapping[t_pred[0]] != s_pred[0]:
            print("\n-- Mis-Mapping 1 ", s_pred, t_pred, "         ")
            # sys.exit(" Mis-Mapping 1 in DFS ")
        elif t_pred[2] not in mapping.keys() and s_pred[2] not in mapping.values():
            print("-- Mis-Mapping 2 ", t_pred[2], s_pred[2], "       ", mapping)  # , end=" ")
            # print(mapping[t_pred[2]], end="   ")
            mapping[t_pred[2]] = s_pred[2]
        elif t_pred[2] in mapping.keys() and mapping[t_pred[2]] != s_pred[2]:  # and
            print("\n*** Mis-Mapping 2", s_pred[2], t_pred[2], "*** Possible Reflexive relation ***")
            # sys.exit("Mis-Mapping 2.2 in DFS ")
        else:
            print("-- Mis-Mappping 3 ", t_pred[2], s_pred[2], "       ", mapping)
    print("     MAPPING", len(mapping), relatio_structural_dist, "  ", mapping, end="     \t")
    return relatio_structural_dist


def relational_distance(t_in, s_in):  # using s2v sense2vec
    global term_separator
    global s2v_cache
    t_reln = t_in.split(term_separator)[0]
    s_reln = s_in.split(term_separator)[0]
    if t_in == s_in:
        return 0.001
    elif second_head(t_reln) == second_head(s_reln):
        return 0.01
    #elif t_reln + "-" + s_reln in s2v_cache:
    #    return 1 - s2v_cache[t_reln + "-" + s_reln]
    #elif s_reln + "-" + t_reln in s2v_cache:
    #    return 1 - s2v_cache[s_reln + "-" + t_reln]
    else:
        if s2v.get_freq(t_reln + '|VERB') is None or s2v.get_freq(s_reln + '|VERB') is None:
            t_root = wnl.lemmatize(t_reln)
            s_root = wnl.lemmatize(s_reln)
            if s2v.get_freq(t_root + '|VERB') is None or s2v.get_freq(s_root + '|VERB') is None:
                return 1.0
            else:
                return 1 - s2v.similarity([t_root + '|VERB'], [s_root + '|VERB'])
        else:
            sims_core = s2v.similarity([t_reln + '|VERB'], [s_reln + '|VERB'])
            if sims_core >= 0.001:
                sims_core = 1 - sims_core
            else:
                sims_core = 1.0
            s2v_cache[t_reln + "-" + s_reln] = sims_core
            return sims_core


def second_head(node):
    global term_separator
    if not (isinstance(node, str)):
        return ""
    else:
        lis = node.split(term_separator)
        if len(lis) >= 2:
            wrd = lis[1].strip()
        else:
            wrd = lis[0].strip()
        return wrd


def linear_distance_string(strg):  # based on graph properties in/out degrees
    t_subj_in, t_subj_out, t_obj_in, t_obj_out, t_subj, t_reln, t_obj, \
    s_subj_in, s_subj_out, s_obj_in, s_obj_out, s_subj, s_reln, s_obj = strg
    z = math.sqrt((t_subj_in - s_subj_in) ** 2 + (t_subj_out - s_subj_out) ** 2 +
                  (t_obj_in - s_obj_in) ** 2 + (t_obj_out - s_obj_out) ** 2)
    return z + 0.1


def euclidean_distance(t_subj_in, t_subj_out, t_obj_in, t_obj_out,
                       s_subj_in, s_subj_out, s_obj_in, s_obj_out):
    z = math.sqrt((t_subj_in - s_subj_in) ** 2 + (t_subj_out - s_subj_out) ** 2 +
                  (t_obj_in - s_obj_in) ** 2 + (t_obj_out - s_obj_out) ** 2)
    return z + 0.001


def conceptual_distance(str1, str2):
    """for simple conceptual similarity"""
    global term_separator
    if str1 == str2:
        return 0.01
    str1 = str1.split(term_separator)
    str2 = str2.split(term_separator)
    if mode == "Code" and str1[0] == str2[0]:  # intersection over difference
        inter_sectn = list(set(str1) & set(str2))
        if len(inter_sectn) > 0:
            return min(0.1, 0.2 ** len(inter_sectn))
        else:
            return 0.25
    elif mode == "English":
        # print(str1, str2, end=" ")
        if s2v.get_freq(str1[0] + '|NOUN') is None or \
                s2v.get_freq(str2[0] + '|NOUN') is None:
            return 2
        else:
            sim_score = s2v.similarity([str1[0] + '|NOUN'], [str2[0] + '|NOUN'])
            #if sim_score ==
            return 1 - sim_score
    else:
        return 2


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def return_sorted_predicates(grf):
    edge_list = grf.edges.data("label")
    zz = [(n, d) for n, d in grf.degree()]
    yy = sorted(zz, key=lambda x: x[1], reverse=True)
    pred_list = []
    for (s, o, v) in edge_list:
        pred_list.append([grf.in_degree(s), grf.out_degree(s), grf.in_degree(o), grf.out_degree(o), s, v, o])
    z = sorted(pred_list, key=lambda x: x[1], reverse=True)
    return sorted(z, key=lambda x: x[0] + x[1] + x[2] + x[3], reverse=True)


def findArity(nod, tupl_list):
    for n, x in tupl_list:
        if n == nod:
            return x


# pred_comparison_vector = subj_in, subj_out, reln, obj_in, obj_out   (reln1 to reln2)

def quick_Wasserstein(a, b):  # inspired by Wasserstein distance (quick Wasserstein approximation)
    a_prime = sorted(list(set(a) - set(b)))
    b_prime = sorted(list(set(b) - set(a)))
    if len(a_prime) < len(b_prime):  # longer list is a_prime
        temp = b_prime.copy()
        b_prime = a_prime.copy()
        a_prime = temp.copy()
    sum1 = sum(abs(i - j) for i, j in zip(a_prime, b_prime))
    b_len = len(b_prime)
    sum2 = sum(a_prime[b_len:])
    return sum1 + sum2


k3_words = ['a', 'abandon', 'ability', 'able', 'abortion', 'about', 'above', 'abroad', 'absence', 'absolute',
            'absolutely', 'absorb', 'abuse', 'academic', 'accept', 'access',
            'accident', 'accompany', 'accomplish', 'according', 'account', 'accurate', 'accuse', 'achieve',
            'achievement', 'acid', 'acknowledge', 'acquire', 'across', 'act',
            'action', 'active', 'activist', 'activity', 'actor', 'actress', 'actual', 'actually', 'ad', 'adapt', 'add',
            'addition', 'additional', 'address', 'adequate', 'adjust',
            'adjustment', 'administration', 'administrator', 'admire', 'admission', 'admit', 'adolescent', 'adopt',
            'adult', 'advance', 'advanced', 'advantage', 'adventure',
            'advertising', 'advice', 'advise', 'adviser', 'advocate', 'affair', 'affect', 'afford', 'afraid', 'African',
            'African-American', 'after', 'afternoon', 'again', 'against',
            'age', 'agency', 'agenda', 'agent', 'aggressive', 'ago', 'agree', 'agreement', 'agricultural', 'ah',
            'ahead', 'aid', 'aide', 'AIDS', 'aim', 'air', 'aircraft', 'airline',
            'airport', 'album', 'alcohol', 'alive', 'all', 'alliance', 'allow', 'ally', 'almost', 'alone', 'along',
            'already', 'also', 'alter', 'alternative', 'although', 'always', 'AM',
            'amazing', 'American', 'among', 'amount', 'analysis', 'analyst', 'analyze', 'ancient', 'and', 'anger',
            'angle', 'angry', 'animal', 'anniversary', 'announce', 'annual',
            'another', 'answer', 'anticipate', 'anxiety', 'any', 'anybody', 'anymore', 'anyone', 'anything', 'anyway',
            'anywhere', 'apart', 'apartment', 'apparent',
            'apparently', 'appeal', 'appear', 'appearance', 'apple', 'application', 'apply', 'appoint', 'appointment',
            'appreciate', 'approach', 'appropriate', 'approval',
            'approve', 'approximately', 'Arab', 'architect', 'area', 'argue', 'argument', 'arise', 'arm', 'armed',
            'army', 'around', 'arrange', 'arrangement', 'arrest', 'arrival',
            'arrive', 'art', 'article', 'artist', 'artistic', 'as', 'Asian', 'aside', 'ask', 'asleep', 'aspect',
            'assault',
            'assert', 'assess', 'assessment', 'asset', 'assign', 'assignment', 'assist', 'assistance', 'assistant',
            'associate', 'association', 'assume', 'assumption', 'assure', 'at',
            'athlete', 'athletic', 'atmosphere', 'attach', 'attack', 'attempt', 'attend', 'attention', 'attitude',
            'attorney', 'attract', 'attractive', 'attribute', 'audience', 'author',
            'authority', 'auto', 'available', 'average', 'avoid', 'award', 'aware', 'awareness', 'away', 'awful',
            'baby', 'back', 'background', 'bad', 'badly', 'bag', 'bake', 'balance',
            'ball', 'ban', 'band', 'bank', 'bar', 'barely', 'barrel', 'barrier', 'base', 'baseball', 'basic',
            'basically', 'basis', 'basket', 'basketball', 'bathroom', 'battery', 'battle',
            'be', 'beach', 'bean', 'bear', 'beat', 'beautiful', 'beauty', 'because', 'become', 'bed', 'bedroom', 'beer',
            'before', 'begin', 'beginning', 'behavior', 'behind', 'being',
            'belief', 'believe', 'bell', 'belong', 'below', 'belt', 'bench', 'bend', 'beneath', 'benefit', 'beside',
            'besides', 'best', 'bet', 'better', 'between', 'beyond', 'Bible', 'big',
            'bike', 'bill', 'billion', 'bind', 'biological', 'bird', 'birth', 'birthday', 'bit', 'bite', 'black',
            'blade', 'blame', 'blanket', 'blind', 'block', 'blood', 'blow', 'blue',
            'board', 'boat', 'body', 'bomb', 'bombing', 'bond', 'bone', 'book', 'boom', 'boot', 'border', 'born',
            'borrow', 'boss', 'both', 'bother', 'bottle', 'bottom', 'boundary', 'bowl',
            'box', 'boy', 'boyfriend', 'brain', 'branch', 'brand', 'bread', 'break', 'breakfast', 'breast', 'breath',
            'breathe', 'brick', 'bridge', 'brief', 'briefly', 'bright', 'brilliant',
            'bring', 'British', 'broad', 'broken', 'brother', 'brown', 'brush', 'buck', 'budget', 'build', 'building',
            'bullet', 'bunch', 'burden', 'burn', 'bury', 'bus', 'business', 'busy',
            'but', 'butter', 'button', 'buy', 'buyer', 'by', 'cabin', 'cabinet', 'cable', 'cake', 'calculate', 'call',
            'camera', 'camp', 'campaign', 'campus', 'can', 'Canadian', 'cancer',
            'candidate', 'cap', 'capability', 'capable', 'capacity', 'capital', 'captain', 'capture', 'car', 'carbon',
            'card', 'care', 'career', 'careful', 'carefully', 'carrier', 'carry',
            'case', 'cash', 'cast', 'cat', 'catch', 'category', 'Catholic', 'cause', 'ceiling', 'celebrate',
            'celebration', 'celebrity', 'cell', 'center', 'central', 'century', 'CEO',
            'ceremony', 'certain', 'certainly', 'chain', 'chair', 'chairman', 'challenge', 'chamber', 'champion',
            'championship', 'chance', 'change', 'changing', 'channel', 'chapter',
            'character', 'characteristic', 'characterize', 'charge', 'charity', 'chart', 'chase', 'cheap', 'check',
            'cheek', 'cheese', 'chef', 'chemical', 'chest', 'chicken', 'chief',
            'child', 'childhood', 'Chinese', 'chip', 'chocolate', 'choice', 'cholesterol', 'choose', 'Christian',
            'Christmas', 'church', 'cigarette', 'circle', 'circumstance', 'cite',
            'citizen', 'city', 'civil', 'civilian', 'claim', 'class', 'classic', 'classroom', 'clean', 'clear',
            'clearly', 'client', 'climate', 'climb', 'clinic', 'clinical', 'clock',
            'close', 'closely', 'closer', 'clothes', 'clothing', 'cloud', 'club', 'clue', 'cluster', 'coach', 'coal',
            'coalition', 'coast', 'coat', 'code', 'coffee', 'cognitive', 'cold',
            'collapse', 'colleague', 'collect', 'collection', 'collective', 'college', 'colonial', 'color', 'column',
            'combination', 'combine', 'come', 'comedy', 'comfort', 'comfortable',
            'command', 'commander', 'comment', 'commercial', 'commission', 'commit', 'commitment', 'committee',
            'common', 'communicate', 'communication', 'community', 'company', 'compare',
            'comparison', 'compete', 'competition', 'competitive', 'competitor', 'complain', 'complaint', 'complete',
            'completely', 'complex', 'complicated', 'component', 'compose', 'composition',
            'comprehensive', 'computer', 'concentrate', 'concentration', 'concept', 'concern', 'concerned', 'concert',
            'conclude', 'conclusion', 'concrete', 'condition', 'conduct',
            'conference', 'confidence', 'confident', 'confirm', 'conflict', 'confront', 'confusion', 'Congress',
            'congressional', 'connect', 'connection', 'consciousness', 'consensus',
            'consequence', 'conservative', 'consider', 'considerable', 'consideration', 'consist', 'consistent',
            'constant', 'constantly', 'constitute', 'constitutional', 'construct',
            'construction', 'consultant', 'consume', 'consumer', 'consumption', 'contact', 'contain', 'container',
            'contemporary', 'content', 'contest', 'context', 'continue', 'continued',
            'contract', 'contrast', 'contribute', 'contribution', 'control', 'controversial', 'controversy',
            'convention', 'conventional', 'conversation', 'convert', 'conviction', 'convince',
            'cook', 'cookie', 'cooking', 'cool', 'cooperation', 'cop', 'cope', 'copy', 'core', 'corn', 'corner',
            'corporate', 'corporation', 'correct', 'correspondent', 'cost', 'cotton',
            'couch', 'could', 'council', 'counselor', 'count', 'counter', 'country', 'county', 'couple', 'courage',
            'course', 'court', 'cousin', 'cover', 'coverage', 'cow', 'crack', 'craft',
            'crash', 'crazy', 'cream', 'create', 'creation', 'creative', 'creature', 'credit', 'crew', 'crime',
            'criminal', 'crisis', 'criteria', 'critic', 'critical', 'criticism', 'criticize',
            'crop', 'cross', 'crowd', 'crucial', 'cry', 'cultural', 'culture', 'cup', 'curious', 'current', 'currently',
            'curriculum', 'custom', 'customer', 'cut', 'cycle', 'dad', 'daily',
            'damage', 'dance', 'danger', 'dangerous', 'dare', 'dark', 'darkness', 'data', 'date', 'daughter', 'day',
            'dead', 'deal', 'dealer', 'dear', 'death', 'debate', 'debt', 'decade',
            'decide', 'decision', 'deck', 'declare', 'decline', 'decrease', 'deep', 'deeply', 'deer', 'defeat',
            'defend', 'defendant', 'defense', 'defensive', 'deficit', 'define', 'definitely',
            'definition', 'degree', 'delay', 'deliver', 'delivery', 'demand', 'democracy', 'Democrat', 'democratic',
            'demonstrate', 'demonstration', 'deny', 'department', 'depend', 'dependent',
            'depending', 'depict', 'depression', 'depth', 'deputy', 'derive', 'describe', 'description', 'desert',
            'deserve', 'design', 'designer', 'desire', 'desk', 'desperate', 'despite',
            'destroy', 'destruction', 'detail', 'detailed', 'detect', 'determine', 'develop', 'developing',
            'development', 'device', 'devote', 'dialogue', 'die', 'diet', 'differ', 'difference',
            'different', 'differently', 'difficult', 'difficulty', 'dig', 'digital', 'dimension', 'dining', 'dinner',
            'direct', 'direction', 'directly', 'director', 'dirt', 'dirty', 'disability',
            'disagree', 'disappear', 'disaster', 'discipline', 'discourse', 'discover', 'discovery', 'discrimination',
            'discuss', 'discussion', 'disease', 'dish', 'dismiss', 'disorder', 'display',
            'dispute', 'distance', 'distant', 'distinct', 'distinction', 'distinguish', 'distribute', 'distribution',
            'district', 'diverse', 'diversity', 'divide', 'division', 'divorce', 'DNA',
            'do', 'doctor', 'document', 'dog', 'domestic', 'dominant', 'dominate', 'door', 'double', 'doubt', 'down',
            'downtown', 'dozen', 'draft', 'drag', 'drama', 'dramatic', 'dramatically',
            'draw', 'drawing', 'dream', 'dress', 'drink', 'drive', 'driver', 'drop', 'drug', 'dry', 'due', 'during',
            'dust', 'duty', 'each', 'eager', 'ear', 'early', 'earn', 'earnings', 'earth',
            'ease', 'easily', 'east', 'eastern', 'easy', 'eat', 'economic', 'economics', 'economist', 'economy', 'edge',
            'edition', 'editor', 'educate', 'education', 'educational', 'educator',
            'effect', 'effective', 'effectively', 'efficiency', 'efficient', 'effort', 'egg', 'eight', 'either',
            'elderly', 'elect', 'election', 'electric', 'electricity', 'electronic', 'element',
            'elementary', 'eliminate', 'elite', 'else', 'elsewhere', 'e-mail', 'embrace', 'emerge', 'emergency',
            'emission', 'emotion', 'emotional', 'emphasis', 'emphasize', 'employ', 'employee',
            'employer', 'employment', 'empty', 'enable', 'encounter', 'encourage', 'end', 'enemy', 'energy',
            'enforcement', 'engage', 'engine', 'engineer', 'engineering', 'English', 'enhance', 'enjoy',
            'enormous', 'enough', 'ensure', 'enter', 'enterprise', 'entertainment', 'entire', 'entirely', 'entrance',
            'entry', 'environment', 'environmental', 'episode', 'equal', 'equally', 'equipment',
            'era', 'error', 'escape', 'especially', 'essay', 'essential', 'essentially', 'establish', 'establishment',
            'estate', 'estimate', 'etc', 'ethics', 'ethnic', 'European', 'evaluate', 'evaluation',
            'even', 'evening', 'event', 'eventually', 'ever', 'every', 'everybody', 'everyday', 'everyone',
            'everything', 'everywhere', 'evidence', 'evolution', 'evolve', 'exact', 'exactly', 'examination',
            'examine', 'example', 'exceed', 'excellent', 'except', 'exception', 'exchange', 'exciting', 'executive',
            'exercise', 'exhibit', 'exhibition', 'exist', 'existence', 'existing', 'expand',
            'expansion', 'expect', 'expectation', 'expense', 'expensive', 'experience', 'experiment', 'expert',
            'explain', 'explanation', 'explode', 'explore', 'explosion', 'expose', 'exposure',
            'express', 'expression', 'extend', 'extension', 'extensive', 'extent', 'external', 'extra', 'extraordinary',
            'extreme', 'extremely', 'eye', 'fabric', 'face', 'facility', 'fact', 'factor',
            'factory', 'faculty', 'fade', 'fail', 'failure', 'fair', 'fairly', 'faith', 'fall', 'false', 'familiar',
            'family', 'famous', 'fan', 'fantasy', 'far', 'farm', 'farmer', 'fashion', 'fast',
            'fat', 'fate', 'father', 'fault', 'favor', 'favorite', 'fear', 'feature', 'federal', 'fee', 'feed', 'feel',
            'feeling', 'fellow', 'female', 'fence', 'few', 'fewer', 'fiber', 'fiction', 'field',
            'fifteen', 'fifth', 'fifty', 'fight', 'fighter', 'fighting', 'figure', 'file', 'fill', 'film', 'final',
            'finally', 'finance', 'financial', 'find', 'finding', 'fine', 'finger', 'finish', 'fire',
            'firm', 'first', 'fish', 'fishing', 'fit', 'fitness', 'five', 'fix', 'flag', 'flame', 'flat', 'flavor',
            'flee', 'flesh', 'flight', 'float', 'floor', 'flow', 'flower', 'fly', 'focus', 'folk',
            'follow', 'following', 'food', 'foot', 'football', 'for', 'force', 'foreign', 'forest', 'forever', 'forget',
            'form', 'formal', 'formation', 'former', 'formula', 'forth', 'fortune', 'forward',
            'found', 'foundation', 'founder', 'four', 'fourth', 'frame', 'framework', 'free', 'freedom', 'freeze',
            'French', 'frequency', 'frequent', 'frequently', 'fresh', 'friend', 'friendly',
            'friendship', 'from', 'front', 'fruit', 'frustration', 'fuel', 'full', 'fully', 'fun', 'function', 'fund',
            'fundamental', 'funding', 'funeral', 'funny', 'furniture', 'furthermore', 'future',
            'gain', 'galaxy', 'gallery', 'game', 'gang', 'gap', 'garage', 'garden', 'garlic', 'gas', 'gate', 'gather',
            'gay', 'gaze', 'gear', 'gender', 'gene', 'general', 'generally', 'generate',
            'generation', 'genetic', 'gentleman', 'gently', 'German', 'gesture', 'get', 'ghost', 'giant', 'gift',
            'gifted', 'girl', 'girlfriend', 'give', 'given', 'glad', 'glance', 'glass', 'global',
            'glove', 'go', 'goal', 'God', 'gold', 'golden', 'golf', 'good', 'government', 'governor', 'grab', 'grade',
            'gradually', 'graduate', 'grain', 'grand', 'grandfather', 'grandmother', 'grant',
            'grass', 'grave', 'gray', 'great', 'greatest', 'green', 'grocery', 'ground', 'group', 'grow', 'growing',
            'growth', 'guarantee', 'guard', 'guess', 'guest', 'guide', 'guideline', 'guilty',
            'gun', 'guy', 'habit', 'habitat', 'hair', 'half', 'hall', 'hand', 'handful', 'handle', 'hang', 'happen',
            'happy', 'hard', 'hardly', 'hat', 'hate', 'have', 'he', 'head', 'headline', 'headquarters',
            'health', 'healthy', 'hear', 'hearing', 'heart', 'heat', 'heaven', 'heavily', 'heavy', 'heel', 'height',
            'helicopter', 'hell', 'hello', 'help', 'helpful', 'her', 'here', 'heritage', 'hero',
            'herself', 'hey', 'hi', 'hide', 'high', 'highlight', 'highly', 'highway', 'hill', 'him', 'himself', 'hip',
            'hire', 'his', 'historian', 'historic', 'historical', 'history', 'hit', 'hold', 'hole',
            'holiday', 'holy', 'home', 'homeless', 'honest', 'honey', 'honor', 'hope', 'horizon', 'horror', 'horse',
            'hospital', 'host', 'hot', 'hotel', 'hour', 'house', 'household', 'housing', 'how',
            'however', 'huge', 'human', 'humor', 'hundred', 'hungry', 'hunter', 'hunting', 'hurt', 'husband',
            'hypothesis', 'I', 'ice', 'idea', 'ideal', 'identification', 'identify', 'identity', 'ie',
            'if', 'ignore', 'ill', 'illegal', 'illness', 'illustrate', 'image', 'imagination', 'imagine', 'immediate',
            'immediately', 'immigrant', 'immigration', 'impact', 'implement', 'implication',
            'imply', 'importance', 'important', 'impose', 'impossible', 'impress', 'impression', 'impressive',
            'improve', 'improvement', 'in', 'incentive', 'incident', 'include', 'including', 'income',
            'incorporate', 'increase', 'increased', 'increasing', 'increasingly', 'incredible', 'indeed',
            'independence', 'independent', 'index', 'Indian', 'indicate', 'indication', 'individual',
            'industrial', 'industry', 'infant', 'infection', 'inflation', 'influence', 'inform', 'information',
            'ingredient', 'initial', 'initially', 'initiative', 'injury', 'inner', 'innocent',
            'inquiry', 'inside', 'insight', 'insist', 'inspire', 'install', 'instance', 'instead', 'institution',
            'institutional', 'instruction', 'instructor', 'instrument', 'insurance', 'intellectual',
            'intelligence', 'intend', 'intense', 'intensity', 'intention', 'interaction', 'interest', 'interested',
            'interesting', 'internal', 'international', 'Internet', 'interpret', 'interpretation',
            'intervention', 'interview', 'into', 'introduce', 'introduction', 'invasion', 'invest', 'investigate',
            'investigation', 'investigator', 'investment', 'investor', 'invite', 'involve', 'involved',
            'involvement', 'Iraqi', 'Irish', 'iron', 'Islamic', 'island', 'Israeli', 'issue', 'it', 'Italian', 'item',
            'its', 'itself', 'jacket', 'jail', 'Japanese', 'jet', 'Jew', 'Jewish', 'job', 'join', 'joint',
            'joke', 'journal', 'journalist', 'journey', 'joy', 'judge', 'judgment', 'juice', 'jump', 'junior', 'jury',
            'just', 'justice', 'justify', 'keep', 'key', 'kick', 'kid', 'kill', 'killer', 'killing',
            'kind', 'king', 'kiss', 'kitchen', 'knee', 'knife', 'knock', 'know', 'knowledge', 'lab', 'label', 'labor',
            'laboratory', 'lack', 'lady', 'lake', 'land', 'landscape', 'language', 'lap', 'large', 'largely',
            'last', 'late', 'later', 'Latin', 'latter', 'laugh', 'launch', 'law', 'lawn', 'lawsuit', 'lawyer', 'lay',
            'layer', 'lead', 'leader', 'leadership', 'leading', 'leaf', 'league', 'lean', 'learn', 'learning',
            'least', 'leather', 'leave', 'left', 'leg', 'legacy', 'legal', 'legend', 'legislation', 'legitimate',
            'lemon', 'length', 'less', 'lesson', 'let', 'letter', 'level', 'liberal', 'library', 'license', 'lie',
            'life', 'lifestyle', 'lifetime', 'lift', 'light', 'like', 'likely', 'limit', 'limitation', 'limited',
            'line', 'link', 'lip', 'list', 'listen', 'literally', 'literary', 'literature', 'little', 'live',
            'living', 'load', 'loan', 'local', 'locate', 'location', 'lock', 'long', 'long-term', 'look', 'loose',
            'lose', 'loss', 'lost', 'lot', 'lots', 'loud', 'love', 'lovely', 'lover', 'low', 'lower', 'luck',
            'lucky', 'lunch', 'lung', 'machine', 'mad', 'magazine', 'mail', 'main', 'mainly', 'maintain', 'maintenance',
            'major', 'majority', 'make', 'maker', 'makeup', 'male', 'mall', 'man', 'manage', 'management',
            'manager', 'manner', 'manufacturer', 'manufacturing', 'many', 'map', 'margin', 'mark', 'market',
            'marketing', 'marriage', 'married', 'marry', 'mask', 'mass', 'massive', 'master', 'match', 'material',
            'math',
            'matter', 'may', 'maybe', 'mayor', 'me', 'meal', 'mean', 'meaning', 'meanwhile', 'measure', 'measurement',
            'meat', 'mechanism', 'media', 'medical', 'medication', 'medicine', 'medium', 'meet', 'meeting',
            'member', 'membership', 'memory', 'mental', 'mention', 'menu', 'mere', 'merely', 'mess', 'message', 'metal',
            'meter', 'method', 'Mexican', 'middle', 'might', 'military', 'milk', 'million', 'mind', 'mine',
            'minister', 'minor', 'minority', 'minute', 'miracle', 'mirror', 'miss', 'missile', 'mission', 'mistake',
            'mix', 'mixture', 'mm-hmm', 'mode', 'model', 'moderate', 'modern', 'modest', 'mom', 'moment', 'money',
            'monitor', 'month', 'mood', 'moon', 'moral', 'more', 'moreover', 'morning', 'mortgage', 'most', 'mostly',
            'mother', 'motion', 'motivation', 'motor', 'mount', 'mountain', 'mouse', 'mouth', 'move', 'movement',
            'movie', 'Mr', 'Mrs', 'Ms', 'much', 'multiple', 'murder', 'muscle', 'museum', 'music', 'musical',
            'musician', 'Muslim', 'must', 'mutual', 'my', 'myself', 'mystery', 'myth', 'naked', 'name', 'narrative',
            'narrow', 'nation', 'national', 'native', 'natural', 'naturally', 'nature', 'near', 'nearby', 'nearly',
            'necessarily', 'necessary', 'neck', 'need', 'negative', 'negotiate', 'negotiation', 'neighbor',
            'neighborhood', 'neither', 'nerve', 'nervous', 'net', 'network', 'never', 'nevertheless', 'new', 'newly',
            'news', 'newspaper', 'next', 'nice', 'night', 'nine', 'no', 'nobody', 'nod', 'noise', 'nomination',
            'none', 'nonetheless', 'nor', 'normal', 'normally', 'north', 'northern', 'nose', 'not', 'note', 'nothing',
            'notice', 'notion', 'novel', 'now', 'nowhere', 'not', 'nuclear', 'number', 'numerous', 'nurse', 'nut',
            'object', 'objective', 'obligation', 'observation', 'observe', 'observer', 'obtain', 'obvious', 'obviously',
            'occasion', 'occasionally', 'occupation', 'occupy', 'occur', 'ocean', 'odd', 'odds', 'of', 'off',
            'offense', 'offensive', 'offer', 'office', 'officer', 'official', 'often', 'oh', 'oil', 'ok', 'okay', 'old',
            'Olympic', 'on', 'once', 'one', 'ongoing', 'onion', 'online', 'only', 'onto', 'open', 'opening',
            'operate', 'operating', 'operation', 'operator', 'opinion', 'opponent', 'opportunity', 'oppose', 'opposite',
            'opposition', 'option', 'or', 'orange', 'order', 'ordinary', 'organic', 'organization', 'organize',
            'orientation', 'origin', 'original', 'originally', 'other', 'others', 'otherwise', 'ought', 'our',
            'ourselves', 'out', 'outcome', 'outside', 'oven', 'over', 'overall', 'overcome', 'overlook', 'owe', 'own',
            'owner', 'pace', 'pack', 'package', 'page', 'pain', 'painful', 'paint', 'painter', 'painting', 'pair',
            'pale', 'Palestinian', 'palm', 'pan', 'panel', 'pant', 'paper', 'parent', 'park', 'parking', 'part',
            'participant', 'participate', 'participation', 'particular', 'particularly', 'partly', 'partner',
            'partnership', 'party', 'pass', 'passage', 'passenger', 'passion', 'past', 'patch', 'path', 'patient',
            'pattern', 'pause', 'pay', 'payment', 'PC', 'peace', 'peak', 'peer', 'penalty', 'people', 'pepper', 'per',
            'perceive', 'percentage', 'perception', 'perfect', 'perfectly', 'perform', 'performance',
            'perhaps', 'period', 'permanent', 'permission', 'permit', 'person', 'personal', 'personality', 'personally',
            'personnel', 'perspective', 'persuade', 'pet', 'phase', 'phenomenon', 'philosophy', 'phone',
            'photo', 'photograph', 'photographer', 'phrase', 'physical', 'physically', 'physician', 'piano', 'pick',
            'picture', 'pie', 'piece', 'pile', 'pilot', 'pine', 'pink', 'pipe', 'pitch', 'place', 'plan',
            'plane', 'planet', 'planning', 'plant', 'plastic', 'plate', 'platform', 'play', 'player', 'please',
            'pleasure', 'plenty', 'plot', 'plus', 'PM', 'pocket', 'poem', 'poet', 'poetry', 'point', 'pole',
            'police', 'policy', 'political', 'politically', 'politician', 'politics', 'poll', 'pollution', 'pool',
            'poor', 'pop', 'popular', 'population', 'porch', 'port', 'portion', 'portrait', 'portray',
            'pose', 'position', 'positive', 'possess', 'possibility', 'possible', 'possibly', 'post', 'pot', 'potato',
            'potential', 'potentially', 'pound', 'pour', 'poverty', 'powder', 'power', 'powerful',
            'practical', 'practice', 'pray', 'prayer', 'precisely', 'predict', 'prefer', 'preference', 'pregnancy',
            'pregnant', 'preparation', 'prepare', 'prescription', 'presence', 'present', 'presentation',
            'preserve', 'president', 'presidential', 'press', 'pressure', 'pretend', 'pretty', 'prevent', 'previous',
            'previously', 'price', 'pride', 'priest', 'primarily', 'primary', 'prime', 'principal', 'principle',
            'print', 'prior', 'priority', 'prison', 'prisoner', 'privacy', 'private', 'probably', 'problem',
            'procedure', 'proceed', 'process', 'produce', 'producer', 'product', 'production', 'profession',
            'professional', 'professor', 'profile', 'profit', 'program', 'progress', 'project', 'prominent', 'promise',
            'promote', 'prompt', 'proof', 'proper', 'properly', 'property', 'proportion', 'proposal',
            'propose', 'proposed', 'prosecutor', 'prospect', 'protect', 'protection', 'protein', 'protest', 'proud',
            'prove', 'provide', 'provider', 'province', 'provision', 'psychological', 'psychologist',
            'psychology', 'public', 'publication', 'publicly', 'publish', 'publisher', 'pull', 'punishment', 'purchase',
            'pure', 'purpose', 'pursue', 'push', 'put', 'qualify', 'quality', 'quarter', 'quarterback',
            'question', 'quick', 'quickly', 'quiet', 'quietly', 'quit', 'quite', 'quote', 'race', 'racial', 'radical',
            'radio', 'rail', 'rain', 'raise', 'range', 'rank', 'rapid', 'rapidly', 'rare', 'rarely',
            'rate', 'rather', 'rating', 'ratio', 'raw', 'reach', 'react', 'reaction', 'read', 'reader', 'reading',
            'ready', 'real', 'reality', 'realize', 'really', 'reason', 'reasonable', 'recall', 'receive',
            'recent', 'recently', 'recipe', 'recognition', 'recognize', 'recommend', 'recommendation', 'record',
            'recording', 'recover', 'recovery', 'recruit', 'red', 'reduce', 'reduction', 'refer', 'reference',
            'reflect', 'reflection', 'reform', 'refugee', 'refuse', 'regard', 'regarding', 'regardless', 'regime',
            'region', 'regional', 'register', 'regular', 'regularly', 'regulate', 'regulation', 'reinforce',
            'reject', 'relate', 'relation', 'relationship', 'relative', 'relatively', 'relax', 'release', 'relevant',
            'relief', 'religion', 'religious', 'rely', 'remain', 'remaining', 'remarkable', 'remember',
            'remind', 'remote', 'remove', 'repeat', 'repeatedly', 'replace', 'reply', 'report', 'reporter', 'represent',
            'representation', 'representative', 'Republican', 'reputation', 'request', 'require',
            'requirement', 'research', 'researcher', 'resemble', 'reservation', 'resident', 'resist', 'resistance',
            'resolution', 'resolve', 'resort', 'resource', 'respect', 'respond', 'respondent', 'response',
            'responsibility', 'responsible', 'rest', 'restaurant', 'restore', 'restriction', 'result', 'retain',
            'retire', 'retirement', 'return', 'reveal', 'revenue', 'review', 'revolution', 'rhythm', 'rice',
            'rich', 'rid', 'ride', 'rifle', 'right', 'ring', 'rise', 'risk', 'river', 'road', 'rock', 'role', 'roll',
            'romantic', 'roof', 'room', 'root', 'rope', 'rose', 'rough', 'roughly', 'round', 'route',
            'routine', 'row', 'rub', 'rule', 'run', 'running', 'rural', 'rush', 'Russian', 'sacred', 'sad', 'safe',
            'safety', 'sake', 'salad', 'salary', 'sale', 'sales', 'salt', 'same', 'sample', 'sanction',
            'sand', 'satellite', 'satisfaction', 'satisfy', 'sauce', 'save', 'saving', 'say', 'scale', 'scandal',
            'scared', 'scenario', 'scene', 'schedule', 'scheme', 'scholar', 'scholarship', 'school',
            'science', 'scientific', 'scientist', 'scope', 'score', 'scream', 'screen', 'script', 'sea', 'search',
            'season', 'seat', 'second', 'secret', 'secretary', 'section', 'sector', 'secure', 'security',
            'see', 'seed', 'seek', 'seem', 'segment', 'seize', 'select', 'selection', 'self', 'sell', 'Senate',
            'senator', 'send', 'senior', 'sense', 'sensitive', 'sentence', 'separate', 'sequence', 'series',
            'serious', 'seriously', 'serve', 'service', 'session', 'set', 'setting', 'settle', 'settlement', 'seven',
            'several', 'severe', 'sex', 'sexual', 'shade', 'shadow', 'shake', 'shall', 'shape', 'share',
            'sharp', 'she', 'sheet', 'shelf', 'shell', 'shelter', 'shift', 'shine', 'ship', 'shirt', 'shit', 'shock',
            'shoe', 'shoot', 'shooting', 'shop', 'shopping', 'shore', 'short', 'shortly', 'shot',
            'should', 'shoulder', 'shout', 'show', 'shower', 'shrug', 'shut', 'sick', 'side', 'sigh', 'sight', 'sign',
            'signal', 'significance', 'significant', 'significantly', 'silence', 'silent', 'silver',
            'similar', 'similarly', 'simple', 'simply', 'sin', 'since', 'sing', 'singer', 'single', 'sink', 'sir',
            'sister', 'sit', 'site', 'situation', 'six', 'size', 'ski', 'skill', 'skin', 'sky', 'slave',
            'sleep', 'slice', 'slide', 'slight', 'slightly', 'slip', 'slow', 'slowly', 'small', 'smart', 'smell',
            'smile', 'smoke', 'smooth', 'snap', 'snow', 'so', 'so-called', 'soccer', 'social', 'society',
            'soft', 'software', 'soil', 'solar', 'soldier', 'solid', 'solution', 'solve', 'some', 'somebody', 'somehow',
            'someone', 'something', 'sometimes', 'somewhat', 'somewhere', 'son', 'song', 'soon',
            'sophisticated', 'sorry', 'sort', 'soul', 'sound', 'soup', 'source', 'south', 'southern', 'Soviet', 'space',
            'Spanish', 'speak', 'speaker', 'special', 'specialist', 'species', 'specific',
            'specifically', 'speech', 'speed', 'spend', 'spending', 'spin', 'spirit', 'spiritual', 'split', 'spokesman',
            'sport', 'spot', 'spread', 'spring', 'square', 'squeeze', 'stability', 'stable',
            'staff', 'stage', 'stair', 'stake', 'stand', 'standard', 'standing', 'star', 'stare', 'start', 'state',
            'statement', 'station', 'statistics', 'status', 'stay', 'steady', 'steal', 'steel',
            'step', 'stick', 'still', 'stir', 'stock', 'stomach', 'stone', 'stop', 'storage', 'store', 'storm', 'story',
            'straight', 'strange', 'stranger', 'strategic', 'strategy', 'stream', 'street',
            'strength', 'strengthen', 'stress', 'stretch', 'strike', 'string', 'strip', 'stroke', 'strong', 'strongly',
            'structure', 'struggle', 'student', 'studio', 'study', 'stuff', 'stupid', 'style',
            'subject', 'submit', 'subsequent', 'substance', 'substantial', 'succeed', 'success', 'successful',
            'successfully', 'such', 'sudden', 'suddenly', 'sue', 'suffer', 'sufficient', 'sugar', 'suggest',
            'suggestion', 'suicide', 'suit', 'summer', 'summit', 'sun', 'super', 'supply', 'support', 'supporter',
            'suppose', 'supposed', 'Supreme', 'sure', 'surely', 'surface', 'surgery', 'surprise', 'surprised',
            'surprising', 'surprisingly', 'surround', 'survey', 'survival', 'survive', 'survivor', 'suspect', 'sustain',
            'swear', 'sweep', 'sweet', 'swim', 'swing', 'switch', 'symbol', 'symptom', 'system',
            'table', 'tablespoon', 'tactic', 'tail', 'take', 'tale', 'talent', 'talk', 'tall', 'tank', 'tap', 'tape',
            'target', 'task', 'taste', 'tax', 'taxpayer', 'tea', 'teach', 'teacher', 'teaching',
            'team', 'tear', 'teaspoon', 'technical', 'technique', 'technology', 'teen', 'teenager', 'telephone',
            'telescope', 'television', 'tell', 'temperature', 'temporary', 'ten',
            'tend', 'tendency', 'tennis', 'tension', 'tent', 'term', 'terms', 'terrible', 'territory', 'terror',
            'terrorism', 'terrorist', 'test', 'testify', 'testimony', 'testing', 'text', 'than', 'thank',
            'thanks', 'that', 'the', 'theater', 'their', 'them', 'theme', 'themselves', 'then', 'theory', 'therapy',
            'there', 'therefore', 'these', 'they', 'thick', 'thin', 'thing', 'think', 'thinking',
            'third', 'thirty', 'this', 'those', 'though', 'thought', 'thousand', 'threat', 'threaten',
            'three', 'throat', 'through', 'throughout', 'throw', 'thus', 'ticket', 'tie', 'tight', 'time', 'tiny',
            'tip', 'tire', 'tired', 'tissue', 'title', 'to', 'tobacco', 'today', 'toe', 'together',
            'tomato', 'tomorrow', 'tone', 'tongue', 'tonight', 'too', 'tool', 'tooth', 'top', 'topic', 'toss', 'total',
            'totally', 'touch', 'tough', 'tour', 'tourist', 'tournament', 'toward', 'towards',
            'tower', 'town', 'toy', 'trace', 'track', 'trade', 'tradition',
            'traditional', 'traffic', 'tragedy', 'trail', 'train', 'training', 'transfer', 'transform',
            'transformation', 'transition', 'translate', 'transportation', 'travel', 'treat', 'treatment',
            'treaty', 'tree', 'tremendous', 'trend', 'trial', 'tribe', 'trick', 'trip', 'troop', 'trouble', 'truck',
            'true', 'truly', 'trust', 'truth', 'try', 'tube', 'tunnel', 'turn', 'TV',
            'twelve', 'twenty', 'twice', 'twin', 'two', 'type', 'typical', 'typically', 'ugly', 'ultimate',
            'ultimately', 'unable', 'uncle', 'under', 'undergo', 'understand',
            'understanding', 'unfortunately', 'uniform', 'union', 'unique', 'unit', 'United', 'universal', 'universe',
            'university', 'unknown', 'unless', 'unlike', 'unlikely',
            'until', 'unusual', 'up', 'upon', 'upper', 'urban', 'urge', 'us', 'use', 'used', 'useful', 'user', 'usual',
            'usually', 'utility', 'vacation', 'valley', 'valuable',
            'value', 'variable', 'variation', 'variety', 'various', 'vary', 'vast', 'vegetable', 'vehicle', 'venture',
            'version', 'versus', 'very', 'vessel', 'veteran', 'via', 'victim', 'victory', 'video', 'view', 'viewer',
            'village', 'violate', 'violation', 'violence', 'violent', 'virtually', 'virtue', 'virus', 'visible',
            'vision', 'visit', 'visitor', 'visual', 'vital', 'voice', 'volume', 'volunteer', 'vote', 'voter', 'vs',
            'vulnerable',
            'wage', 'wait', 'wake', 'walk', 'wall', 'wander', 'want', 'war', 'warm', 'warn', 'warning', 'wash', 'waste',
            'watch', 'water', 'wave', 'way', 'we', 'weak', 'wealth', 'wealthy', 'weapon', 'wear', 'weather',
            'wedding', 'week', 'weekend', 'weekly', 'weigh', 'weight', 'welcome', 'welfare', 'well', 'west', 'western',
            'wet', 'what', 'whatever', 'wheel', 'when', 'whenever', 'where', 'whereas', 'whether', 'which', 'while',
            'whisper', 'white', 'who', 'whole', 'whom', 'whose', 'why', 'wide', 'widely', 'widespread', 'wife', 'wild',
            'will', 'willing', 'win', 'wind', 'window', 'wine', 'wing', 'winner', 'winter', 'wipe', 'wire', 'wisdom',
            'wise', 'wish', 'with', 'withdraw', 'within', 'without', 'witness', 'woman', 'wonder', 'wonderful', 'wood',
            'wooden', 'word', 'work', 'worker', 'working', 'works', 'workshop', 'world', 'worried', 'worry', 'worth',
            'would', 'wound', 'wrap', 'write', 'writer', 'writing', 'wrong', 'yard', 'yeah', 'year', 'yell', 'yellow',
            'yes', 'yesterday', 'yet', 'yield', 'you', 'young', 'your', 'yours', 'yourself', 'youth', 'zone']


def return_best_candidate(w1, target, PoS_tag):
    candidates1 = s2v.most_similar(w1 + "|" + PoS_tag, n=10)
    # 3000 most common english words
    candidates1_list = []
    for (x, y) in candidates1:
        candidates1_list.append([s2v.similarity([x], [target + "|" + PoS_tag]), x])
    candidates1_list
    res = sorted(candidates1_list, key=lambda x: x[0])[::-1]
    return res


def midpoint_between_words(w1, w2, PoS_tag):
    print("MID ", w1, w2, PoS_tag, end=" :: ")
    query1 = w1
    query2 = w2
    assert query1 + "|" + PoS_tag in s2v
    assert query2 + "|" + PoS_tag in s2v
    count = 0
    score1 = []
    score2 = []
    previous_best = sys.maxsize
    while count < 2:
        simlr1_to_target = return_best_candidate(query1, w2, PoS_tag)
        simlr1_to_home = return_best_candidate(query1, w1, PoS_tag)
        simlr2_to_target = return_best_candidate(query2, w1, PoS_tag)
        simlr2_to_home = return_best_candidate(query2, w2, PoS_tag)
        for (a, b) in zip(simlr1_to_target, simlr1_to_home):  # a score, b word
            if b[1].split("|")[0] != w1 and b[1].split("|")[0] != w2 and b[1].split("|")[1] == PoS_tag:
                if a[1].split("|")[0] != w1 and a[1].split("|")[0] != w2 and a[1].split("|")[1] == PoS_tag:
                    score1.append([b[0] + a[0] + abs(a[0] - b[0]), a[1]])
                # score1.append([b[0] + a[0], a[1]] )
                # score1.append( [ (abs(b[0] - a[0])), a[1]] )
        for (a, b) in zip(simlr2_to_target, simlr2_to_home):
            if b[1].split("|")[0] != w1 and b[1].split("|")[0] != w2 and b[1].split("|")[1] == PoS_tag:
                if a[1].split("|")[0] != w1 and a[1].split("|")[0] != w2 and a[1].split("|")[1] == PoS_tag:
                    score2.append([b[0] + a[0] + abs(a[0] - b[0]), a[1]])
                # score2.append([b[0] + a[0], a[1]] )
                # score2.append( [ (abs(b[0] - a[0])), a[1] ] )
        midpoint1_estimate = sorted(score1, key=lambda x: abs(x[0]))  # [::-1]
        midpoint2_estimate = sorted(score2, key=lambda x: abs(x[0]))  # [::-1]
        big_list = sorted(midpoint1_estimate + midpoint2_estimate, key=lambda x: abs(x[0]))
        if big_list[0][0] < previous_best:
            previous_best = big_list[0][0]
            best_answer_so_far = big_list[0][1]
        query1 = midpoint1_estimate[0][1].split("|")[0]
        query2 = midpoint2_estimate[0][1].split("|")[0]
        count += 1
        print("LOOP: ", best_answer_so_far, previous_best, end="     ")
    rslt1 = sorted(big_list, key=lambda x: abs(x[0]))
    print()
    print("ClearnResult ", end="")
    for (a, b) in rslt1:
        if not "_" in b:
            print("{: <20} {:.3f} ".format(b, a), end="    ")
    return rslt1


def generalise_word(word, pos):
    from nltk.corpus import wordnet as wn  # names, stopwords, words
    # words.words('en-basic')  # 850 words
    if pos == "NOUN":
        root = wn.morphy(word, wn.NOUN)  # + "|" + pos + '|01'
    elif pos == "VERB":
        root = wn.morphy(word, wn.VERB)  # + "|" + pos + '|01'
    if wn.synsets(root):
        my_synset = wn.synsets(root)[0]
        parents = my_synset.hypernyms()
        res = [root, my_synset] + recurse_over_hypernymns(my_synset)  # [:-1]
        # all_hypernyms = list(set([w for s in my_synset.closure(lambda s: s.hyponyms()) for w in s.lemma_names()]))
        print(res, "\n \n")


global sy_set_res
sy_set_res = []
def recurse_over_hypernymns(sy_set):
    if sy_set.lemmas()[0].name() == "entity" or sy_set.lemmas()[0].name() == "placental":
        return  # []
    else:
        parent1 = [sy_set.hypernyms()[0]]
        # print(parent1, end="  ")
        sy_set_res.extend([parent1])
        return recurse_over_hypernymns(parent1[0])

# z = generalise_word('cats', "NOUN")
# print("\n \nRES:", z, end=":: end")

# print("dog pet ", s2v.similarity("dog"+"|"+"NOUN", "pet"+"|"+"NOUN") )
# print("cat pet ", s2v.similarity("cat"+"|"+"NOUN", "pet"+"|"+"NOUN") )

# print(midpoint_between_words("dog", "cat", "NOUN"))
# print(midpoint_between_words("tumor", "fortress", "NOUN"))
# print(midpoint_between_words("drive", "fly", "VERB"))

# print( s2v.similarity("man"+"|"+"NOUN", "drive"+"|"+"VERB") )

# print( s2v.similarity("tumour"+"|"+"NOUN", "fortress"+"|"+"NOUN") )
