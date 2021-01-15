# -*- coding: utf-8 -*-
# Requirements

import sys
sys.stdout = open('pigs.txt', 'a')
# !pip install numpy
# !pip install xlrd
# !pip install matplotlib
# !pip install pandas
# !pip install networkx
# !pip install sklearn
# !pip install -U -q PyDrive
# !pip install -q xlrd
# !pip install matlab
# !pip install scipy

"""#Main codes

#PartialCorreTest
"""

#PartialCorreTest
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import solve
import math
from scipy.stats import norm

def PartialCorreTest(x_main, y_main, Z_main, alpha):
  # alpha = 0.01


  # print('alpha', alpha)

  x = copy.deepcopy(x_main)
  y = copy.deepcopy(y_main)
  Z = copy.deepcopy(Z_main)

  # print('Z.shape',Z.shape)


  if Z.shape[1]==0:
    x =np.squeeze(np.asarray(x))
    y =np.squeeze(np.asarray(y))
    n = x.shape[0]
    ncit = 0
    pcc= numpy.corrcoef(x,y)
    # print('pccx', pcc)
    pcc = pcc[0][1]

  else:
    n,ncit = Z.shape
    # print('shape', n, ncit)
    Z = np.concatenate((np.ones((n,1)), Z),axis=1)
    # print('ZZZZZ',Z)

    wx = np.linalg.lstsq(Z, x)  #  A*wx = x
    wx= wx[0]
    wx = wx.reshape(len(wx),1)
    rx = x - np.matmul(Z, wx)


    wy = np.linalg.lstsq(Z, y)
    wy = wy[0]
    wy = wy.reshape(len(wy),1)
    ry = y - np.matmul(Z, wy)

    # print('rx',rx.shape)
    # print('ry',ry.shape)

    # n = samle size (int), rx,ry = diff (nx1) ,
    num = (n* np.matmul(np.transpose(rx), ry)  - np.sum(rx)* np.sum(ry))
    # denx = math.sqrt((n* np.matmul(np.transpose(rx), rx) - np.sum(np.square(rx))))
    denx = math.sqrt((n* np.matmul(np.transpose(rx), rx) - np.sum(rx)**2 ))
    deny = math.sqrt((n* np.matmul(np.transpose(ry), ry) - np.sum(ry)**2 ))

    pcc = num/(denx*deny)


  zpcc = 0.5 * np.log((1+pcc)/(1-pcc))
  A = math.sqrt(n-ncit-3)* abs(zpcc)
  B = norm.ppf(1-alpha/2)  #inverse of the CDF of the standard normal distribution with mean 0 , std 1
  sig = (B-A)/(A+B)

  if math.sqrt(n-ncit-3) * abs(zpcc) >  norm.ppf(1-alpha/2):
    cit = False
  else:
    cit = True

  return cit, sig

#PartialCorreTest
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import solve
import math
from scipy.stats import norm

def PartialCorreTest_dir(x_main, y_main, Z_main, alpha):
  # alpha = 0.01


  # print('alpha', alpha)

  x = copy.deepcopy(x_main)
  y = copy.deepcopy(y_main)
  Z = copy.deepcopy(Z_main)

  # print('Z.shape',Z.shape)


  if Z.shape[1]==0:
    x =np.squeeze(np.asarray(x))
    y =np.squeeze(np.asarray(y))
    n = x.shape[0]
    ncit = 0
    # print(x)
    # print(y)
    pcc= numpy.corrcoef(x,y)
    pcc = pcc[0][1]
    # print('pccxxx', pcc)

  else:
    n,ncit = Z.shape
    # print('shape', n, ncit)
    Z = np.concatenate((np.ones((n,1)), Z),axis=1)
    # print('ZZZZZ',Z)

    wx = np.linalg.lstsq(Z, x)  #  A*wx = x
    wx= wx[0]
    wx = wx.reshape(len(wx),1)
    rx = x - np.matmul(Z, wx)


    wy = np.linalg.lstsq(Z, y)
    wy = wy[0]
    wy = wy.reshape(len(wy),1)
    ry = y - np.matmul(Z, wy)

    # print('rx',rx.shape)
    # print('ry',ry.shape)

    # n = samle size (int), rx,ry = diff (nx1) ,
    num = (n* np.matmul(np.transpose(rx), ry)  - np.sum(rx)* np.sum(ry))
    # denx = math.sqrt((n* np.matmul(np.transpose(rx), rx) - np.sum(np.square(rx))))
    denx = math.sqrt((n* np.matmul(np.transpose(rx), rx) - np.sum(rx)**2 ))
    deny = math.sqrt((n* np.matmul(np.transpose(ry), ry) - np.sum(ry)**2 ))

    pcc = num/(denx*deny)


  zpcc = 0.5 * np.log((1+pcc)/(1-pcc))
  A = math.sqrt(n-ncit-3)* abs(zpcc)
  B = norm.ppf(1-alpha/2)  #inverse of the CDF of the standard normal distribution with mean 0 , std 1
  sig = (B-A)/(A+B)

  if math.sqrt(n-ncit-3) * abs(zpcc) >  norm.ppf(1-alpha/2):
    cit = False
  else:
    cit = True

  return cit, sig

"""# Check independency

##Current code
"""

import random
import numpy as np
import copy
import itertools





def is_independent(vi_name, vj_name, Z_name, graph):


  global number_of_citest
  number_of_citest+= 1


  vi = graph.get_data([vi_name])
  vj = graph.get_data([vj_name])
  Z = graph.get_data(Z_name)

  # alpha = 0.01
  cit, sig = PartialCorreTest(vi,vj, Z , graph.ind_alpha)

  # print('cit',cit, 'sig', sig)
  # if cit:
  #   vid = graph.vertices.index(vi_name)
  #   vjd = graph.vertices.index(vj_name)
  #   graph.M[vid][vjd] = 1
  #   graph.M[vjd][vid] = 1
    # 000

    # if Z.shape[1] > 0:
    #   ind2,ind3 = get_direction_KCIT(vi, vj, Z)

    #   if ind2:
    #     graph.dirCM[vid][vjd] = 1

    #   if ind3:
    #     graph.dirDM[vid][vjd] = 1
    #   # print(vid, vjd, 'ind2', ind2, 'ind3', ind3)

  # print('---->',vi_name,vj_name,'is independent?', cit, sig )
  return cit, sig



# make sure Z is minimized
# vi and vj are independent given k size subset of V_main. graph is the current undirected situation  in this partition
def is_korder_conditional_independent(vi,vj, V_main, k_order , graph):

    vid = graph.vertices.index(vi)
    vjd = graph.vertices.index(vj)

    V_main = list(set(V_main))

    global graph_sep_set

    if (graph.M[vid][vjd] == 1 or graph.M[vjd][vid] ==1):   #the variables that make these pair ind must be in the same partition

      prev_sepset = set(graph_sep_set[(vi,vj)])
      curr_vertices= set(V_main)
      if prev_sepset.issubset(curr_vertices):
        # print('Yes matched sepset found %s  of %s'%(prev_sepset,V_main))
        return True, graph_sep_set[(vi,vj)] , graph.adj_sigvalue[vid][vjd]


    if vi==vj:
      return False,[],0

    # implement from ci paper-mk
    # sent here just to check not modify
    V=copy.deepcopy(V_main)

    if vi in V:
        V.remove(vi)
    if vj in V:
        V.remove(vj)

    # find subset of V that make vi,vj independent with k order
    # print('V==',V,'k=',k_order)
    k_var_list = list(itertools.combinations(V, k_order))

    # print('k_var_list', k_var_list)



    Z = []; sig_value = 0
    cit_test= False
    for k_var in k_var_list:
      k_var= list(k_var)
      # print('kvars', k_var)

      global exp_param, cp_param, capa_param, sada_param

      if exp_param.algo_state ==  'refining_causal_graphs_EXP':
        exp_param.actual_refined_citest+= 1
      elif cp_param.algo_state ==  'refining_causal_graphs_CP':
        cp_param.actual_refined_citest+= 1
      elif capa_param.algo_state == 'CAPA_Partition3':
        capa_param.actual_refined_citest+= 1
      elif sada_param.algo_state == 'refining_causal_graphs_SADA':
        sada_param.actual_refined_citest+= 1

      cit,sig = is_independent(vi, vj, k_var, graph)

      graph.adj_sigvalue[vid][vjd] = sig
      graph.adj_sigvalue[vjd][vid] = sig

      if cit == True:
        # print(vi,' is ind ',vj,'|', k_var )



        graph_sep_set[(vi,vj)] = k_var
        graph_sep_set[(vj,vi)] = k_var


        cit_test, Z, sig_value = (cit, k_var, sig)
        break



    return cit_test, Z, sig_value



def is_conditional_independent_for_any_k(vi, vj, V_main , graph):

  for k_order in range(graph.k_thresh):
    res = is_korder_conditional_independent(vi,vj, V_main, k_order , graph)
    if res[0]== True:
      return res[0],res[1],res[2], k_order

  return False,None,None,None



# w independent for all in V given C, with k_order
def independent_forall(w, V, C, graph):

    flagV1 = True
    for vi in V:

        cit_test,Z, p_value = is_korder_conditional_independent(w, vi, C, graph.k_order, graph)
        if cit_test== False:
            # print('%s dependent %s ' %(w,vi))
            flagV1 = False
            break
        else:
            pass
            # print('%s ind %s | %s' %(w,vi,Z))


    return flagV1

"""## exp->independent for all new"""

# w independent for all in V given C, with k_order
def independent_forall_new(w, V, C, graph):


    wi = graph.vertices.index(w)
    dep_set = numpy.where(graph.M[:,wi] == 0)[0]
    dep_set = [ graph.vertices[i] for i in dep_set]
    if w in dep_set:
        dep_set.remove(w)


    flagV1 = True
    for vi in V:
        # print(' for var %s  ignore this indiv set=%s' %(w,graph.individual_ciset[w]))
        # updated_C = [var for var in C if var not in graph.individual_ciset[w]]

        updated_C_again = [var for var in dep_set if var not in graph.individual_ciset[w]]


        # print('Now, is updated, %s ind from %s | %s'%(w,vi,updated_C))
        # print('4. depsets are ',dep_set)
        if graph.case == 0:
          updated_C = C
          # print('the normal C')
        else:
          # print('using the updated one')
          updated_C= updated_C_again

        # print('C= ', C)
        # print('updated_C_again= ', updated_C_again)
        # print('Going in',updated_C)

        cit_test,Z, p_value = is_korder_conditional_independent(w, vi, updated_C, graph.k_order, graph)
        if cit_test== False:
            # print('%s dependent %s ' %(w,vi))
            flagV1 = False
            break
        else:
            # print('%s ind %s | %s' %(w,vi,Z))
            graph.individual_ciset[w].append(vi)

    return flagV1

"""#Get_direction

## PC direction
"""

#Get_direction



def dfs(graph ,par_idx, curr_idx, end_idx):
  if curr_idx == end_idx:
    return True
  for j in range(0,graph.V):
    if graph.directed_adj[curr_idx][j] == 1 and j!=par_idx:
      if dfs(graph, par_idx, j, end_idx) == True:
        return True;

  return False;

def directed_path_exists(graph, A, B):

  return dfs(graph, -1, A, B)



def get_direction_pc(graph):
    for X in graph.vertices:
        for Y in graph.vertices:
            for Z in graph.vertices:
                if graph.adj[X][Y]==1 and graph.adj[Y, Z]==1 and graph.adj[X][Z]==0 and Y not in graph.sep_set[(X, Z)]:
                    graph.directed_adj[X][Y] = 1
                    graph.directed_adj[Z][Y] = 1

    atleast_one_directed = True

    while atleast_one_directed:

        atleast_one_directed = False

        for i in range(0, graph.V):  # A
            for j in range(0, graph.V):  # B
                if graph.directed_adj[i][j] == 1:  # A->B

                    for k in range(0, graph.V):  # C
                        if graph.adj[j][k] == 1 and graph.adj[i][k] == 0 and graph.directed_adj[:j].astype(
                                numpy.float).sum() == 0:
                            graph.directed_adj[j][k] = 1
                            atleast_one_directed = True



                if directed_path_exists(graph, i, j) and graph.adj[i][j] == 1:
                    graph.directed_adj[i][j] = 1
                    atleast_one_directed = True


    return graph

"""## ReCITdirection"""

# Depends on CausalGraph class. Objects: graph, new_graph

# returns ReCIT directed graph new_graph for input graph
def ReCITdirection(graph):

    # print('recitdirection starting')
    global graph_sep_set

    new_graph = CausalGraph(graph.vertices)
    for vi in range(graph.V):
        for vj in range(graph.V):
            if vi != vj and graph.M[vi][vj] == 1:

                # adding necessary edges to the new graph
                vi_name = graph.vertices[vi]
                vj_name = graph.vertices[vj]
                if graph.dirCM[vi][vj] == 1:

                  # print('creating1 %s -> %s' %(graph_sep_set[(vi_name, vj_name)], vi_name))
                  for vk_name in graph_sep_set[(vi_name, vj_name)]:
                      vk = graph.vertices.index(vk_name)
                      if graph.directed_adj[vi][vk] == 1 or graph.directed_adj[vk][vi] == 1:
                          new_graph.directed_adj[vk][vi] = 1
                          new_graph.directed_adj[vi][vk] = 0

                if graph.dirDM[vi][vj] == 1:
                  # print('creating2 %s -> %s' %(graph_sep_set[(vi_name, vj_name)], vj_name))
                  for vk_name in graph_sep_set[(vi_name, vj_name)]:
                    vk = graph.vertices.index(vk_name)
                    if graph.directed_adj[vj][vk] == 1 or graph.directed_adj[vk][vj] == 1:
                        new_graph.directed_adj[vk][vj] = 1
                        new_graph.directed_adj[vj][vk] = 0
    return new_graph

"""## VSdirection"""

def VSdirection(graph):
    new_graph = CausalGraph(graph.vertices, graph)
    for vi in range(graph.V):
        for vj in range(graph.V):
            if vi != vj and graph.M[vi][vj] == 1:

                # adding necessary edges to the new graph
                chx = np.nonzero(graph.directed_adj[vi])
                pax = np.nonzero(graph.directed_adj[:,vi])
                pcx = np.union1d(pax,chx)
                chy = np.nonzero(graph.directed_adj[vj])
                pay = np.nonzero(graph.directed_adj[:,vj])
                pcy = np.union1d(pay,chy)
                interPc = np.intersect1d(pcx,pcy)
                if interPc.size != 0:
                    if len(graph_sep_set[(vi, vj)]) == 0:
                        for vk in interPc:
                            new_graph.directed_adj[vi][vk] = 1
                            new_graph.directed_adj[vj][vk] = 1
                            new_graph.directed_adj[vk][vi] = 0
                            new_graph.directed_adj[vk][vj] = 0
                    if len(graph_sep_set[(vi, vj)]) > 0:
                        diffPc = np.setdiff1d(interPc,graph_sep_set[(vi, vj)])
                        if diffPc.size == 0:
                            for vk in diffPc:
                                new_graph.directed_adj[vi][vk] = 1
                                new_graph.directed_adj[vj][vk] = 1
                                new_graph.directed_adj[vk][vi] = 0
                                new_graph.directed_adj[vk][vj] = 0
    return new_graph

"""##True direction"""

def get_true_direction(graph):
  # print('here for getting true direction')
  # print(graph.M)
  # graph.draw_graph()

  for vi in range(graph.V):
    for vj in range(graph.V):
      # print('here I am')
      if vi != vj and graph.directed_adj[vi][vj] == 1:

        newi = true_graph.vertices.index(graph.vertices[vi])
        newj = true_graph.vertices.index(graph.vertices[vj])
        # print('inside true direction', newi,newj)
        if true_graph.directed_adj[newi][newj] == 1:

          # print('edge %s - > %s'  %(graph.vertices[vi], graph.vertices[vj]))
          graph.directed_adj[vi][vj] = 1
          graph.directed_adj[vj][vi] = 0
        elif true_graph.directed_adj[newj][newi] == 1:  #redundant

          # print('edge %s - > %s'  %(graph.vertices[vj], graph.vertices[vi]))
          graph.directed_adj[vi][vj] = 0
          graph.directed_adj[vj][vi] = 1


  # print('after')
  # graph.draw_graph()
  return graph

"""#Basic PC algorithm

##Exp run_PC_new
"""

# experiment includes run_PC_new

def get_adjacents(vi, vj, graph):


  vi =graph.vertices.index(vi)
  vj =graph.vertices.index(vj)

  adjacents = []
  for nbrId in range(graph.V):
    if nbrId != vj and graph.directed_adj[vi][nbrId]==1:
      adjacents.append(nbrId)

  adjacents = [ graph.vertices[id] for id in adjacents ]
  return adjacents



def run_PC_new(graph):

  global graph_sep_set
  vertices_list = graph.vertices
  V = len(vertices_list)
  graph = CausalGraph(vertices_list, graph)
  graph.create_complete()

  # print('Running the PC algorithm for', vertices_list)

  # print('->>>',graph.directed_adj)
# we will run the algo with vertices name but in graph class it will work with indices

  for i in range(V-1):
    for j in range(i+1,V):

      # matlab
      # if  Cskeleton(idx(i),idx(j))==0
      #     continue;
      # end

      # print(V-1, V, 'id->',i,j)
      vi = graph.vertices[i]
      vj = graph.vertices[j]
      # print('edge between ',vi,vj)

      A = get_adjacents(vi, vj, graph)
      # print('adjacents of ',vi,A)
      B = get_adjacents(vj, vi, graph)
      # print('adjacents of ',vj,B)
      conSepSet = list(set(A).union(set(B)))
      # conSepSet = list(set(A) & set(B))


      # print('graph vertices', graph.vertices)
      # print('conSepSet', conSepSet)
      max_korder = min(graph.k_thresh, len(conSepSet)+1)
      # print('max_koder',max_korder)
      for k in range(max_korder):   #go till the max order including it

        # cond, lst = selectable(n, u, v, graph)

        # if len(conSepSet)<k:
        #   break  # didnt found any. no need to iterate anymore

        # print(vi,vj,'independent checking |', conSepSet, ' korder=',k)

        # print('conSepSet',conSepSet)
        cit_test, Z, sig_value = is_korder_conditional_independent(vi, vj, conSepSet, k, graph)
        # print(cit_test, Z)

        if cit_test == True:
          # print(vi,vj,'independent found |', Z)
          graph.remove_edge(graph.vertices.index(vi), graph.vertices.index(vj))
          graph.remove_edge(graph.vertices.index(vj), graph.vertices.index(vi))
          graph_sep_set[vi, vj] = Z
          graph_sep_set[vj, vi] = Z
          break  # found independent . no need to iterate anymore

        # print(vi,vj,'not ind |%s with k=%s' %(conSepSet,k))


  graph = get_true_direction(graph)
  # graph.print_graph()
  # print('got true direction in running pc and now graph showing')
  # graph.draw_graph()
  # print('true direction')
  # true_graph.draw_graph()
  return graph

# PC Algo old

def selectable(n, x, y, graph):
  extra = deepcopy(graph.get_adj_undirected(x))
  if extra.count(y) <= 0:
    return 0, []

  if y in extra:
      extra.remove(y)

  for i in range(len(extra) + 1):
        for j in range(i + 1, len(extra) + 1):
            sub = extra[i:j]
            if len(sub) == n and len(extra) >= n:
              return 1, sub
  return 0, []


def run_PC_old(graph):

  vertices_list = graph.vertices
  V = len(vertices_list)
  graph = CausalGraph(vertices_list)
  graph.create_complete()

  n = 0

  while True:

    for u in range(V):
      for v in range(V):
        if u != v:



          cond, lst = selectable(n, u, v, graph)

          if cond == 1:
            graph.remove_edge(u, v)
            graph.sep_set[u, v] = lst
            graph.sep_set[v, u] = lst


    n = n + 1
    if n >= len(vertices_list):
      break

  graph.print_graph()

# run_PC([2, 3, 4])

"""Pasting here

# Discover_Causal_Relations
"""

# Discover_Causal_Relations
import random, numpy
import numpy as np
from random import seed
from random import gauss
import matplotlib.pyplot as plt
import math



def discover_causal_relation(graph):

  result_graph = run_PC_new(graph)
  return result_graph

"""# Merge_Partial_Results


1. In CP_code they didnt kept any merge function just passed the skeleton to different partitions and updated it

## merge main codes
"""

# Merge_Partial_Results

import copy


# this is actually a dfs function



def merge_directed_adjacency(new_graph, graph, visited):


  for i in range(0,graph.V):

        ii = graph.vertices[i]
        new_i = new_graph.vertices.index(ii)   #getting index of 6 and 7 of the new graph vertices
        for j in range(0,graph.V):

          # vertices = [5,6,7]

          jj = graph.vertices[j]   #7
          new_j = new_graph.vertices.index(jj)

          global capa_param
          if visited[new_i][new_j] == 1:
            if new_graph.directed_adj[new_i][new_j]==1 and  graph.directed_adj[i][j]==0:
              new_graph.directed_adj[new_i][new_j] = 0


              # print('Alertt edge wants delete existing %s->%s, so disconnected them ' %(new_graph.vertices[new_i],new_graph.vertices[new_j]))

            elif new_graph.directed_adj[new_i][new_j]==0 and  graph.directed_adj[i][j]==1:
              new_graph.directed_adj[new_i][new_j] =0


              # print('Alertt new edge wants to connect for %s->%s, but didnt connect them' %(new_graph.vertices[new_i],new_graph.vertices[new_j]))


          else:
            new_graph.directed_adj[new_i][new_j] = graph.directed_adj[i][j]
            new_graph.adj_sigvalue[new_i][new_j] = graph.adj_sigvalue[i][j]
            visited[new_i][new_j] = 1   # for conflict tracking


        got_data = np.array(graph.get_data([ii]))
        # print('data ffrom ',ii)
        # print(got_data)

        new_graph.dataset[:,new_i] = copy.deepcopy(got_data[:,0])  #check if its okay
        # print('copied',new_graph.dataset[:,new_i])


  return new_graph


def merge_graph_structure(*graphs):


      verticesList = []
      numberofsamples= 0
      for G in graphs:
        verticesList = verticesList + G.vertices
        numberofsamples = G.dataset.shape[0]

      verticesList = list(set(verticesList))
      new_graph = CausalGraph(verticesList)
      D= numpy.zeros((numberofsamples , new_graph.V))  #may paroblem
      new_graph.load_data(D)

      # print('merging starting:')
      # new_graph.draw_graph()

      visited = numpy.zeros((new_graph.V , new_graph.V))
      for G in graphs:
        new_graph = merge_directed_adjacency(new_graph, G, visited)   # check if problem occurs pass by reference

        # print('vertices', G.vertices)
        # print('dataset', G.dataset)
        # print('graph merged')
        # new_graph.draw_graph()


      # print('merged vertices', new_graph.vertices)
      # print('merged dataset', new_graph.dataset)
      # remove_conf(new_graph)
      # remove_redun(new_graph)
      return new_graph







def merge_SADA(*graphs):

    G = merge_graph_structure(*graphs)

    # G.run_floydwarshall()
    # G = remove_conflicts(G)
    # G = remove_redundancy(G)

    return G


def merge_partial_results(*graphs):
    G = merge_SADA(*graphs)
    return G

"""# Find_Causal_Partition

## Ocaam Razor new
Runtime independency calculating may need fixing
"""

# from new CAPA with Occam Razor
def divide_non_overlapping_parts_occam_new(graph):

  Ds = copy.deepcopy(graph.M)

  # Ds= np.random.rand(3,3)
  non_adjacents = np.sum(Ds, axis=0)
  # print('non_adjacent', non_adjacents)
  # print('max of them', np.argmax(non_adjacents))

  # var_list = []
  # for idx in range(len(graph.V)):
  #     var_list.append((graph.vertices[idx], non_adjacents[idx]))

  # var_list.sort(key=lambda x: x[1], reverse=True)
  # V_main = [var[0] for var in var_list]



  # find V1,V2,C  implement from sada

  maxphiLength=0
  maxPhi=None
  V= copy.deepcopy(graph.vertices)

  V1=[]
  V2=[]
  C=[]

  not_assigned = [i for i in range(graph.V)]  #ids of the variables

  for iter in range(len(V)):

    # calculating for indices not for values

    sum_array = np.sum(Ds, axis=0)
    max_val = np.max(sum_array)
    # print('max val', max_val)

    if  max_val !=0:
      wi = np.argmax(sum_array)
    else:
      wi = not_assigned[0]



    tempV1 = [graph.vertices[i] for i in V1]
    tempV2 = [graph.vertices[i] for i in V2]
    tempC = [graph.vertices[i] for i in C]
    # print('V1=%s  V2=%s w=%s with k=%s'%(tempV1,tempV2, graph.vertices[wi] ,graph.k_order))


    w_diff = graph.vertices[wi]
    V1_diff = tempV1
    V2_diff = tempV2

    if len(V1) > len(V2):

      # if np.sum(graph.M[V1,wi]) == len(V1):

      if independent_forall_new(w_diff,V1_diff, graph.vertices, graph)==True:
          V2.append(wi)
      elif independent_forall_new(w_diff,V2_diff,graph.vertices, graph)==True:
          V1.append(wi)
      else:
          C.append(wi)

    else:
      if independent_forall_new(w_diff,V2_diff,graph.vertices, graph)==True:
          V1.append(wi)
      elif independent_forall_new(w_diff,V1_diff, graph.vertices, graph)==True:
          V2.append(wi)
      else:
          C.append(wi)

    Ds[:,wi]= 0
    if wi in not_assigned:
        not_assigned.remove(wi)



  # print('current C',C)
  iter_C= copy.deepcopy(C)
  # in CAPA its not conditioned on C rather on whole V
  for s in iter_C:
    _C = copy.deepcopy(C)
    if s in _C:
        _C.remove(s)

    if np.sum(graph.M[V1,s]) == len(V1):
        if s in C:
            C.remove(s)
        V2.append(s)
    elif np.sum(graph.M[V2,s]) == len(V2):
        if s in C:
            C.remove(s)
        V1.append(s)

  # print('After removing C',C)


  V1 = [graph.vertices[i] for i in V1]
  V2 = [graph.vertices[i] for i in V2]
  C = [graph.vertices[i] for i in C]

  phi= (V1,C,V2)

  # phiLen= min( len(V1), len(V2))
  # if phiLen > maxphiLength:
  #   maxphiLength= phiLen
  #   maxPhi=phi

  maxPhi = phi

  return maxPhi

  # return V1,V2,C
  # these are non overlapping

"""##SADA find causal cut"""

# Find_Causal_Partition





def pick_random_two(graph):

    # print('id->', id(graph))

    # print('start')
    v1 = None
    v2 = None
    resZ = None
    while graph.k_order < graph.k_thresh:

      # alternative brute force approach
      Z, p_value = ([],0)
      two_var_list = list(itertools.combinations(graph.vertices, 2))
      random.shuffle(two_var_list)   #very nice idea :v
      # print('two_var_list',two_var_list)
      for (vi,vj) in two_var_list:
        cit_test,Z, p_value=is_korder_conditional_independent(vi, vj, graph.vertices, graph.k_order , graph)
        # print('random two pick', vi, vj, cit_test)
        if cit_test == True:   # found the suitable Z set
          # print('Found')
          v1= vi
          v2= vj
          resZ = Z
          break

        # print(graph.k_order, vi ,vj, Z)


      if cit_test == True:   # found the suitable Z set
        break
      else:
        graph.k_order +=1

    # print(" found for" ,graph.k_order,",", v1, v2,'->', resZ)
    # print('End')
    return v1,v2,resZ








# from sada paper algo 2
def divide_non_overlapping_parts_sada(graph):

    V_main = graph.vertices
    init_pairs_k=5

    # find V1,V2,C  implement from sada

    maxphiLength=0
    maxPhi=None
    for j in range(1, init_pairs_k):
        V = copy.deepcopy(graph.vertices)

        # print(init_pairs_k,' ->main vertice set', V)
        # graph and k_order
        vi,vj,Z = pick_random_two(graph)
        # print('random picked -> vi, vj, Z', vi, vj, Z)
        if vi==None and vj==None:
          maxPhi = (V,[],[])
          break

        V1 = [vi]
        V2 = [vj]
        C = Z
        if vi in V:
            V.remove(vi)
        if vj in V:
            V.remove(vj)
        for _c in C:
            if _c in V:
                V.remove(_c)


        for w in V:
            # print('consider about ',w)
            if independent_forall(w,V1,C, graph)==True:
                # print(w,'is independent for all in ',V1, ' given ',C)
                # print('so adding in ',V2)
                V2.append(w)
            elif independent_forall(w,V2,C, graph)==True:
                # print(w,'is independent for all in ',V2, ' given ',C)
                # print('so adding in ',V1)
                V1.append(w)
            else:
                C.append(w)


            tempV1 = [v1 for v1 in V1]
            tempV2 = [v2 for v2 in V2]
            tempC = [c for c in C]
            # print('V1=%s  V2=%s w=%s with k=%s'%(tempV1,tempV2, w ,graph.k_order))



        iter_C= copy.deepcopy(C)
        for s in iter_C:
            _C = copy.deepcopy(C)
            if s in _C:
                _C.remove(s)

            if independent_forall(s, V1, _C, graph):
                if s in C:
                    C.remove(s)
                V2.append(s)
            elif independent_forall(s,V2, _C, graph):
                if s in C:
                    C.remove(s)
                V1.append(s)


        phi= (V1,C,V2)

        phiLen= min( len(V1), len(V2))
        if phiLen > maxphiLength:
            maxphiLength= phiLen
            maxPhi=phi


    return maxPhi

    # return V1,V2,C
    # these are non overlapping

"""##SADA partition method"""

def form_three_probable_partitions_sada(graph):
    # line 4

    # optimize  C
    V =  copy.deepcopy(graph.vertices)

    # A, C, B = divide_non_overlapping_parts_occam_new(graph)
    # A, C, B = divide_non_overlapping_parts_occam(graph)
    A, C, B = divide_non_overlapping_parts_sada(graph)

    # print('divided_non_overlapping_parts', A,C,B)




    return A, C, B




def find_causal_partition_sada(graph):
  # print("Finding partition of graph.vertices",graph.vertices)

  row, col = graph.M.shape
  # print("diemnsion of the matrix:" , (row, col))
  V = graph.vertices

  global graph_sep_set

  while True:
    for r in range(0, graph.V-1):   #need optimization
        for c in range(r+1, graph.V):
            # if is_korder_conditional_independent(graph.vertices[r], graph.vertices[c], V, graph.k_order, graph)[0]:

            cit, Z,sig = is_korder_conditional_independent(graph.vertices[r], graph.vertices[c], V, graph.k_order, graph)

            if cit :
                graph.M[r][c] = 1
                graph.M[c][r] = 1

                # graph_sep_set[(graph.vertices[r],graph.vertices[c])] = Z
                # graph_sep_set[(graph.vertices[c],graph.vertices[r])] = Z
                graph.adj_sigvalue[r][c] = sig
                graph.adj_sigvalue[c][r] = sig


    # print('graph.M completed with', graph.k_order)
    # print(graph.M)

    A, C, B = form_three_probable_partitions_sada(graph)
    # A, C, B = form_three_probable_partitions_testing(graph)


    # print(graph.k_order,' found partition size of %s %s %s from %s->' %(len(V1), len(V3), len(V2), len(V)))

    # if max( max(len(V1), len(V2)), len(V3) ) == len(V)  and  graph.k_order+1 < graph.k_thresh:
    if len(C) >= len(A)+len(B)  and  graph.k_order+1 < graph.k_thresh:
      graph.k_order= graph.k_order+ 1
      # print('increasing graph.k_order')
      continue
    else:
      # print('returning from find_causal_partition')


      V1 = sorted(A + C)
      V2 = sorted(B + C)
      return V1,V2

"""## CAPA partitioning method"""

def form_three_probable_partitions_capa(graph):
    # line 4

    # optimize  C
    V =  copy.deepcopy(graph.vertices)

    A, C, B = divide_non_overlapping_parts_occam_new(graph)
    # A, C, B = divide_non_overlapping_parts_occam(graph)
    # A, C, B = divide_non_overlapping_parts_sada(graph)

    # print('divided_non_overlapping_parts', A,C,B)

    # line 5,6
    # V3 and C is not same
    A_B= sorted(A+B)   # have to write sorted function
    for vi in A_B:
        if independent_forall(vi, C , V, graph):  # vi independent for all in C given V  # btw can there remain some elements which goes to no set
            if vi in V:
                V.remove(vi)

    # line 8
    V1 = sorted(A + C)
    V2 = sorted(B + C)
    V3 = copy.deepcopy(V)


    return V1,V3,V2


def find_causal_partition_capa(graph):
  # print("Finding partition of graph.vertices",graph.vertices)

  row, col = graph.M.shape
  # print("diemnsion of the matrix:" , (row, col))
  V = graph.vertices

  global graph_sep_set

  while True:
    for r in range(0, graph.V-1):   #need optimization
        for c in range(r+1, graph.V):

            cit, Z,sig = is_korder_conditional_independent(graph.vertices[r], graph.vertices[c], V, graph.k_order, graph)

            if cit :
                graph.M[r][c] = 1
                graph.M[c][r] = 1

                # graph_sep_set[(graph.vertices[r],graph.vertices[c])] = Z
                # graph_sep_set[(graph.vertices[c],graph.vertices[r])] = Z
                graph.adj_sigvalue[r][c] = sig
                graph.adj_sigvalue[c][r] = sig


    # print('graph.M completed with', graph.k_order)
    # print(graph.M)

    V1, V3, V2 = form_three_probable_partitions_capa(graph)

    # print(graph.k_order,' found partition size of %s %s %s from %s->' %(len(V1), len(V3), len(V2), len(V)))

    if max( max(len(V1), len(V2)), len(V3) ) == len(V)  and  graph.k_order+1 < graph.k_thresh:
      graph.k_order= graph.k_order+ 1
      # print('increasing graph.k_order')
      continue
    else:
      # print('returning from find_causal_partition')
      return V1, V3, V2

"""##CP partition method"""

def form_three_probable_partitions_cp(graph):
    # line 4

    # optimize  C
    V =  copy.deepcopy(graph.vertices)

    A, C, B = divide_non_overlapping_parts_occam_new(graph)
    # A, C, B = divide_non_overlapping_parts_occam(graph)
    # A, C, B = divide_non_overlapping_parts_sada(graph)

    # print('divided_non_overlapping_parts', A,C,B)



    return A, C, B




def find_causal_partition_cp(graph):
  # print("Finding partition of graph.vertices",graph.vertices)

  row, col = graph.M.shape
  # print("diemnsion of the matrix:" , (row, col))
  V = graph.vertices

  global graph_sep_set

  while True:
    for r in range(0, graph.V-1):   #need optimization
        for c in range(r+1, graph.V):
            # if is_korder_conditional_independent(graph.vertices[r], graph.vertices[c], V, graph.k_order, graph)[0]:

            cit, Z,sig = is_korder_conditional_independent(graph.vertices[r], graph.vertices[c], V, graph.k_order, graph)

            if cit :
                graph.M[r][c] = 1
                graph.M[c][r] = 1

                # graph_sep_set[(graph.vertices[r],graph.vertices[c])] = Z
                # graph_sep_set[(graph.vertices[c],graph.vertices[r])] = Z
                graph.adj_sigvalue[r][c] = sig
                graph.adj_sigvalue[c][r] = sig


    # print('graph.M completed with', graph.k_order)
    # print(graph.M)

    A, C, B = form_three_probable_partitions_cp(graph)
    # A, C, B = form_three_probable_partitions_testing(graph)


    # print(graph.k_order,' found partition size of %s %s %s from %s->' %(len(V1), len(V3), len(V2), len(V)))

    # if max( max(len(V1), len(V2)), len(V3) ) == len(V)  and  graph.k_order+1 < graph.k_thresh:
    if len(C) >= len(A)+len(B)  and  graph.k_order+1 < graph.k_thresh:
      graph.k_order= graph.k_order+ 1
      # print('increasing graph.k_order')
      continue
    else:
      # print('returning from find_causal_partition')


      V1 = sorted(A + C)
      V2 = sorted(B + C)
      return V1,V2

"""#CausalGraph class"""

# CausalGraph class
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class CausalGraph:
    V = 0
    vertices = []
    names = {}

    adj = None
    directed_adj = None    #This mat just provides the directed connectivity between vars
    M = None          # graph.M gives the independency between variables for current k_order, some vars may be ind later for higher k_order
    dirCM = None
    dirDM = None
    adj_sigvalue = None
    dataset= None

    # sep_set = {}
    individual_ciset ={}

    size_thres= 3
    k_order =0
    k_thresh = 4  #  CP used 6
    ind_alpha = 0.05


    case= None

    def __init__(self, vertices_list, *args):
        vertices_list.sort()
        self.V = len(vertices_list)
        self.vertices = vertices_list
        self.k_thresh= 4   #   k_order <4  ->  0<=k_order<=3
        self.size_thres =3
        self.ind_alpha = 0.05
        self.case = None

        # initializing them
        self.adj = np.zeros((self.V, self.V))
        self.directed_adj = np.zeros((self.V, self.V))
        self.adj_sigvalue = np.zeros((self.V, self.V))
        self.M = np.zeros((self.V, self.V))
        self.dirCM = np.zeros((self.V, self.V))
        self.dirDM = np.zeros((self.V, self.V))
        self.k_order =0

        # for x in self.vertices:
        #   for y in self.vertices:
        #     self.sep_set[(x,y)] = []

        for x in self.vertices:
          self.individual_ciset[x] = []
          self.names[x]= str(x)


        # getting data from previous structure
        if len(args) !=0:
          for old_graph in args:
            # old_graph = copy.deepcopy(o_graph)
            # print('insider constructor', old_graph.sep_set)
            self.get_from_old_structure(old_graph)

        else:  # only declared for the first time
          self.size_thres = max(math.floor(self.V/10), 3)







    def create_complete(self):  # for the directed mat
      for i in range(self.V):
        for j in range(self.V):
          if i != j:
            self.directed_adj[i][j] = 1


    # def add_edge(self, s, d):
    #     self.adj[s][d] = 1
    #     self.adj[d][s] = 1


    # def get_adj_undirected(self, x):
    #   ret = []
    #   for i in range(len(self.adj[x])):
    #     if self.adj[x][i] == 1:
    #       ret.append(i)
    #   return ret


    def get_edges(self):
      ret = []
      for i in range(self.V):
        for j in range(self.V):
          if self.directed_adj[i][j] == 1 :
            ret.append((i, j))
      return ret

    def get_parents():
      pass


    def print_graph(self):
        for i in range(self.V):
            # print(i, ": ")
            for j in range(self.V):
              if self.directed_adj[i][j] == 1:
                  pass
                # print(j, " ")
        # print('\n')

    def remove_edge(self, s, d):   # for the directed mat considering only indices
        # s= self.vertices.index(s)
        # d= self.vertices.index(d)
        self.directed_adj[s][d] = 0
        # self.directed_adj[d][s] = 0

    def add_edge(self, s,d):
        self.directed_adj[s][d] = 1
        # self.directed_adj[d][s] = 1


    def merge_graph(self, graph):
      newV= self.V+ graph.V
      #new_adj=


    def load_data(self, data):
      self.dataset = data
      return


    def get_data(self, var_set):

      # print('getting data, my vertices', self.vertices)
      # print('requested', var_set)

      var_idset = [self.vertices.index(var) for var in var_set]
      # print('varidset', var_idset)
      return self.dataset[:, var_idset]


    def get_from_old_structure(self, old_graph):

      self.k_order = old_graph.k_order
      self.load_data(old_graph.get_data(self.vertices))  #may problem

      # print('inside old struction', old_graph.sep_set)

      for i in range(self.V):
        for j in range(self.V):

          vidx = old_graph.vertices.index( self.vertices[i]  )
          vjdx = old_graph.vertices.index( self.vertices[j]  )
          # print('indexes', vidx,vjdx)
          # print(old_graph.adj)
          # print(old_graph.vertices)
          self.adj[i][j] = old_graph.adj[vidx][vjdx]
          self.directed_adj[i][j] = old_graph.directed_adj[vidx][vjdx]
          self.M[i][j] = old_graph.M[vidx][vjdx]
          self.dirCM[i][j] = old_graph.dirCM[vidx][vjdx]
          self.dirDM[i][j] = old_graph.dirDM[vidx][vjdx]
          self.adj_sigvalue[i][j] = old_graph.adj_sigvalue[vidx][vjdx]

          vi = self.vertices[i]
          vj = self.vertices[j]
          # self.sep_set[(vi,vj)] =old_graph.sep_set[(vi,vj)]
          # print('copying = %s %s -> %s' %(vi,vj, old_graph.sep_set[(vi,vj)]))


        vi = self.vertices[i]
        self.individual_ciset[vi]= old_graph.individual_ciset[vi]




      return




    def draw_graph(self):
      adj_mat = self.directed_adj
      rows, cols = np.where(adj_mat == 1)

      rows = [self.vertices[r] for r in rows]
      cols = [self.vertices[c] for c in cols]

      edges = zip(rows, cols)
      G = nx.DiGraph()

      G.add_nodes_from(self.vertices)
      G.add_edges_from(edges)
      # nx.draw(gr, node_size=500, labels=self.vertices, with_labels=True)
      # pos=nx.spring_layout(G,scale=3)
      pos=nx.spring_layout(G, k=0.20,iterations=20)
      nx.draw(G, pos, with_labels=True, font_weight='bold')
      plt.show()
      return


    #can be optimized
    def get_colliders(self):
      ret = np.sum(self.directed_adj, axis=0)
      coll_ids=np.where(ret>1)
      # print(coll_ids)
      coll_ids = coll_ids[0]
      colliders = [ self.vertices[i] for i in coll_ids ]
      # print('colliders:',colliders)


      # new_refined_edges = {}
      # new_refined= 0
      # for col in coll_ids:
      #   # print(col)
      #   nbrs = np.nonzero(true_graph.directed_adj[:,col])[0]
      #   new_refined_edges[col] = [ (col,nbr) for nbr in nbrs ]
      #   new_refined +=  len(new_refined_edges[col])

      # print(new_refined,'  new refined edges', new_refined_edges)

      return colliders

"""# Graph parameters class"""

# graph parameter class
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



class GraphParameters:


  def __init__(self, algo_name):
    self.algo_name = algo_name
    self.R=0
    self.P=0
    self.F1=0
    self.refined_edges = {}
    self.number_of_citest =0
    self.refined_time =0
    self.executed_time = 0
    self.actual_refined_citest = 0
    self.algo_state = ""
    self.result_graph = None


  def print_stat(self):
    print('%s R = %s, P = %s, F1 = %s' %(self.algo_name, self.R, self.P, self.F1))


  def print_test_time(self):
    print(self.algo_name,' executed time =', self.executed_time)
    print(self.algo_name,' actual refined citest =', self.actual_refined_citest)
    print(self.algo_name,' refined executed time =', self.refined_time)


  def print_edges(self):
    print(self.algo_name,' refine edges =',len(self.refined_edges))
    capa_miss = get_missing_edges(true_graph, self.result_graph)
    capa_extra = get_extra_edges(true_graph, self.result_graph)
    print(self.algo_name,' missing edges =',len(capa_miss))
    print(self.algo_name,' extra edges =',len(capa_extra))

  def compare(self, *args):

    # print('\npercentage improvement: with respect to', self.algo_name)
    for param in args:
      # print('comparison with ', param.algo_name)
      # print('actual_refined_citest = %s%%' %( 100 - ( self.actual_refined_citest/ param.actual_refined_citest )*100 ))
      # print('executed_time = %s%%' %( 100 - ( self.executed_time/ param.executed_time )*100 ))
      # print('refined_time = %s%%' %( 100 - ( self.refined_time/ param.refined_time )*100 ))
      print('')



  def get_miss_with_sep(self, exp_graph_sep_set):

    miss_with_sep = {}
    for miss in exp_miss:
      if miss in exp_graph_sep_set:
        miss_with_sep[miss]= exp_graph_sep_set[miss]
      else:
        miss_with_sep[miss]= None

    return miss_with_sep

"""# Data generation process

## Read Net type 2
"""

# Depends on CausalGraph class. Object: true_graph

from google.colab import files
import re

# _file can be either a file path as string or the a file object
def read_net(_file):

    try:
      fp = open(_file)
    except:
      files.upload()
      fp = open(_file)

    # print(fp)
    index = 0
    nodes = {}
    edges = []
    for line in fp:
        line = re.split('[ ()|\n\']',line)
        while(None in line):
            line.remove(None)
        while('' in line):
            line.remove('')
        if len(line) != 0:
            if line[0] == 'var':
                nodes[line[1]] = index
                index += 1
            elif line[0] == 'parents':
                line.remove('parents')
                for str in line:
                    if str != line[0] and str in nodes:
                        edges.append((nodes[str], nodes[line[0]]))

    true_graph = CausalGraph([vi for vi in range(len(nodes))])
    for vi in nodes:
        true_graph.names[nodes[vi]] = vi
    for ei in edges:
        true_graph.directed_adj[ei[0]][ei[1]] = 1
    # true_graph.draw_graph()
    return true_graph


# true_graph = read_net('pigs.net')
# true_graph = read_net('graph.net')
# draw_temp_graph(true_graph)

"""## Write Net type 2"""

# Depends on CausalGraph class. Object: true_graph

from google.colab import files
from os import path
import re
from google.colab import files

def write_net(true_graph, _file, dataset, _data):
    fp = open(_file, 'w+')
    for vi in range(true_graph.V):
        fp.write('var '+true_graph.names[vi]+'\n')
    for vj in range(true_graph.V):
        fp.write('parents '+true_graph.names[vj])
        for vi in range(true_graph.V):
            if true_graph.directed_adj[vi][vj]:
                fp.write(' '+true_graph.names[vi])
        fp.write('\n')
    fp.close()
    fp = open(_data, 'w+')
    for ri in dataset:
        for ci in ri:
            fp.write(str(ci)+' ')
        fp.write('\n')
    fp.close()

    # print('file downloading')
    # files.download(_file)
    # files.download(_data)

"""## Read Net / Hugin"""

# Depends on CausalGraph class. Object: true_graph

from google.colab import files
from os import path
import re

def read_net_hugin(_file):
    if path.exists(_file) == False:
        files.upload()
    fp = open(_file)
    index = 0
    nodes = {}
    edges = []
    for line in fp:
        line = re.split('[ ()|\n]',line)
        while(None in line):
            line.remove(None)
        while('' in line):
            line.remove('')
        if len(line) != 0:
            if line[0] == 'node':
                nodes[line[1]] = index
                index += 1
            elif line[0] == 'potential':
                line.remove('potential')
                for str in line:
                    if str != line[0]:
                        edges.append((nodes[str], nodes[line[0]]))

    fp.close()
    true_graph = CausalGraph([vi for vi in range(len(nodes))])
    for vi in nodes:
        true_graph.names[nodes[vi]] = vi
    for ei in edges:
        true_graph.directed_adj[ei[0]][ei[1]] = 1
    # true_graph.draw_graph()
    return true_graph

import os
# os.remove('asia.net')
# read_net_hugin('asia.net')

"""## Write Net / Hugin"""

# Depends on CausalGraph class. Object: true_graph

from google.colab import files
from os import path
import re

def write_net_hugin(true_graph, _file, dataset, _data):
    fp = open(_file, 'w+')
    for vi in range(true_graph.V):
        fp.write('node '+true_graph.names[vi]+'\n')
    for vj in range(true_graph.V):
        fp.write('potential '+true_graph.names[vj])
        for vi in range(true_graph.V):
            if true_graph.directed_adj[vi][vj]:
                fp.write(' '+true_graph.names[vi])
        fp.write('\n')
    fp.close()
    fp = open(_data, 'w+')
    for ri in dataset:
        for ci in ri:
            fp.write(str(ci)+' ')
        fp.write('\n')
    fp.close()

"""##Random dag with networkx"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

def generate_random_DAG(num_of_nodes, edge_prob):
  num_nodes = num_of_nodes
  G=nx.gnp_random_graph(num_nodes, edge_prob ,directed=True)

  DAG = nx.DiGraph([(u,v,{'weight':random.randint(1,1)}) for (u,v) in G.edges() if u<v])
  # pos=nx.spring_layout(DAG,scale=2)
  # nx.draw(DAG, pos, with_labels=True, font_weight='bold')
  # plt.show()

  nodelist= [ nd for nd in range(num_nodes) ]
  adj= nx.to_numpy_matrix(DAG, nodelist= nodelist)

  true_graph = CausalGraph(nodelist)


  for r in range(adj.shape[0]):

    for c in range (adj.shape[1]):
      if adj[r,c]==False:
        true_graph.directed_adj[r][c]= False
      else:
        true_graph.directed_adj[r][c]= True


  # true_graph.draw_graph()

  # print(adj==true_graph.directed_adj)
  return true_graph

"""## Generate true causal structure"""

# Graph generation

from random import seed
from random import gauss,randint
import matplotlib.pyplot as plt
import math


def generate_true_causal_structure(num_of_nodes, in_degree):
    # from sada appendix

    # create n*n adjacency matrix,
    # then get di value from parents by lineare relation
    #take from capa experiemental evaluation

    true_graph = CausalGraph([nd for nd in range(0, num_of_nodes)])

    seed(1)
    x=[]
    y=[]
    for i in range(0,num_of_nodes):

      cur_degree = gauss(in_degree, 1)  # mean= in_degree ,  std = 1
      y.append(cur_degree)
      upper =math.ceil(in_degree)
      lower = math.floor(in_degree)
      if abs(cur_degree-upper) < abs(cur_degree-lower):     #get the closest value
        cur_degree= upper
      else:
        cur_degree = lower


      # print('cur_degree->',cur_degree)



      cur_degree = min(cur_degree, i)
      j=0
      while j<cur_degree:
        # k = randint(1,i)   # in pseudocode, maybe bcz its 1 based
        k = randint(0,i)     # I am making it 0 based
        if i==k:
          continue
        if true_graph.directed_adj[i][k]== False:
          true_graph.directed_adj[i][k] = True
          j+= 1


    indegress= sum(true_graph.directed_adj.sum(axis=0))
    # print('indegress',indegress, 'avg', indegress/true_graph.V)
    # true_graph.draw_graph()
    return true_graph

"""##Generate Linear non-gaussian data"""

# Data generation

# Linear non-gaussian data generation


import random, numpy
from random import seed
from random import gauss
import matplotlib.pyplot as plt
import math
from scipy import stats
import seaborn as sns


# with variance 1

def zero_mean_unit_variance(X):
  Z = (X - X.mean())/ X.std()
  return Z



# this one is correct bcz we draw data uniformly and this is not normal data. supported by CAPA code
def get_non_gaussian_uniform(m):

  # the range is given in sada
  noise= np.random.uniform(size=(m))
  noise= zero_mean_unit_variance(noise)

  return noise


def topologicalSortUtil(true_graph, v, visited, stack):
    visited[v] = True

    # print('xxx',true_graph.directed_adj[v][0].shape)

    for i in range(0, true_graph.V):
        # print('->>>',true_graph.directed_adj[v][i])
        if true_graph.directed_adj[v][i] == 1 and visited[i] == False:
            topologicalSortUtil(true_graph, i,visited,stack)

    stack.insert(0,v)

def topologicalSort(true_graph):
    visited = [False]*true_graph.V
    stack =[]

    for i in range(0, true_graph.V):
        if visited[i] == False:
          topologicalSortUtil(true_graph,i,visited,stack)

    # print('the stack')
    # print(stack)
    return stack


# fix sample size = 2|number of nodes|  [MeCIT]
def generate_LiNGAM_data(true_graph, noiseRatio, num_of_nodes, num_of_samples):

  D= numpy.zeros((num_of_samples, num_of_nodes))

  vertices = topologicalSort(true_graph)

  # topological sort then calculate
  for var in vertices:
    # print('calculating data for ',var)

    parent_ids = list(np.nonzero(true_graph.directed_adj[:,var])[0])

    # noise = get_non_gaussian_uniform(num_of_samples)
    noise = zero_mean_unit_variance(np.random.uniform(size=(num_of_samples)))

    if len(parent_ids)==0:
      # print('no parent')
      D[:,var]= noise

    else:
      # print('parents', parent_ids)
      W = [ 1/len(parent_ids) for id in parent_ids]
      D[:,var] = noise*noiseRatio
      for j in parent_ids:  # true_id vs index
        D[:,var] = D[:,var] + W[parent_ids.index(j)] * D[:,j]


      D[:,var] = zero_mean_unit_variance(D[:,var])

    # normalized to zero mean and unit variance
    # D[:,var] = (D[:,var] - D[:,var].mean())/ D[:,var].std()
    # sns.distplot(D[:,var])

  return D

"""## Read Dataset"""

from google.colab import files
from os import path
import re

def read_dataset(_file):
    if path.exists(_file) == False:
        files.upload()
    fp = open(_file)
    dataset = []
    for line in fp:
        line = re.split('[ ,\n]',line)
        while(None in line):
            line.remove(None)
        while('' in line):
            line.remove('')
        row = []
        for vi in line:
            row.append(float(vi))
        dataset.append(row)
    fp.close()
    return np.array(dataset)

# read_dataset('1. cd3cd28.csv')

"""#Performance measure

## Get directed score
"""

# Depends on CausalGraph class. Objects: original_graph, result_graph

# returns recall score, precision score and F1-score
def get_directed_score(original_graph, result_graph):
    result_edge_count = 0
    original_edge_count = 0
    intersection_edge_count = 0

    # assuming both graph has same vertices, counting the necessary edges
    for vi in range(original_graph.V):
        for vj in range(original_graph.V):
            if vi != vj:
                if result_graph.directed_adj[vi][vj] == 1:
                    result_edge_count += 1
                if original_graph.directed_adj[vi][vj] == 1:
                    original_edge_count += 1
                    if result_graph.directed_adj[vi][vj] == 1:
                        intersection_edge_count += 1

    # calculating and returning scores
    recall_score = intersection_edge_count / original_edge_count
    precision_score = intersection_edge_count / result_edge_count
    f1_score = 2 * recall_score * precision_score / (recall_score + precision_score)
    return recall_score, precision_score, f1_score

"""## Get undirected score"""

# Depends on CausalGraph class. Objects: original_graph, result_graph

# returns recall score, precision score and F1-score
def get_undirected_score(original_graph, result_graph):
    result_edge_count = 0
    original_edge_count = 0
    intersection_edge_count = 0

    # assuming both graph has same vertices, counting the necessary edges
    for vi in range(original_graph.V-1):
        for vj in range(vi+1, original_graph.V):
            if vi != vj:
                if result_graph.directed_adj[vi][vj] == 1 or result_graph.directed_adj[vj][vi]==1:
                    result_edge_count += 1
                if original_graph.directed_adj[vi][vj] == 1 or original_graph.directed_adj[vj][vi] == 1:
                    original_edge_count += 1
                    if result_graph.directed_adj[vi][vj] == 1 or result_graph.directed_adj[vj][vi]==1:
                        intersection_edge_count += 1

    # calculating and returning scores
    recall_score = intersection_edge_count / original_edge_count
    precision_score = intersection_edge_count / result_edge_count
    f1_score = (2 * recall_score * precision_score) / (recall_score + precision_score)
    return recall_score, precision_score, f1_score

"""## Missing or extra edges"""

def get_missing_edges(true_graph, result_graph):
  missing_edges = []
  for i in range(true_graph.V):
    for j in range(true_graph.V):
      if true_graph.directed_adj[i][j]==1 and result_graph.directed_adj[i][j]==0:
        missing_edges.append((i,j))

  # print('EXP missing edges', missing_edges)
  return missing_edges

def get_extra_edges(true_graph, result_graph):
  extra_edges = []
  for i in range(true_graph.V-1):
    for j in range(i+1,true_graph.V):
      # if (true_graph.directed_adj[i][j]==0 and true_graph.directed_adj[j][i]==0) and (result_graph.directed_adj[i][j]==1 and result_graph.directed_adj[j][i]==1):
      #   if (j,i) not in extra_edges:
        if result_graph.directed_adj[i][j]==1 and result_graph.directed_adj[j][i]==1:
          extra_edges.append((i,j))

  # print('EXP extra edges', extra_edges)

  return extra_edges

"""#CAPA

##Redundancy & conflict removing
for SADA
"""

def get_names(graph,idlist):
  return [graph.vertices[id] for id in idlist]


def get_neighbors_sada(varid, graph):

  par_idset1 = np.nonzero(graph.directed_adj[:, varid])[0]
  # print('par_idset1', par_idset1)
  par_idset2 = np.nonzero(graph.directed_adj[varid, :])[0]
  # print('par_idset2', par_idset2)
  par_idset = list(set(par_idset1).union(par_idset2))

  return par_idset



def is_colliders_sada(nodeid, graph):

  # print('is colliders %s????' %(graph.vertices[nodeid]) )

  var = graph.vertices[nodeid]

  par_idset = get_neighbors_sada(nodeid, graph)
  parent_set = [graph.vertices[i] for i in par_idset]

  collider_parents = []
  is_collider = False

  for parvi in parent_set:
    for parvj in parent_set:
      if parvi == parvj:
        continue

      conseptset= []
      if (parvi, parvj) in graph_sep_set:
        conseptset = graph_sep_set[(parvi, parvj)]

      # print('parent, %s ind %s | consepset= %s' %(parvi, parvj, conseptset))

      if var not in conseptset:
        is_collider = True
        collider_parents.append(parvi)
        collider_parents.append(parvj)


  collider_parents = list(set(collider_parents))
  collider_parents_idset = [graph.vertices.index(par) for par in collider_parents ]

  # if is_collider == True:
  #   # print('collider %s, its parents =%s' %(var,collider_parents ))

  return is_collider, collider_parents_idset




def is_reachable(graph, src, dest, visited, path_nodes):


  if src == dest:
    # print('finally found ',dest)
    return True,path_nodes


  visited.append(src)

  # nbrset = get_neighbors_sada(src, graph)
  nbrset = np.nonzero(graph.directed_adj[src, :])[0]
  # print(src,' -> total nbrs', get_names(graph,nbrset))
  # is_col,collider_parents =  is_colliders_sada(src, graph)
  # if is_col == True:
  #   nbrset = [nbr for nbr in nbrset if nbr not in collider_parents]
  #   # print('%s is collider so nbrset=%s' %(graph.vertices[src], get_names(graph,nbrset)))

  flag = False
  for nbr in nbrset:

      if nbr not in visited:
          # print('going %s -> %s' %(graph.vertices[src], graph.vertices[nbr]))

          # parent[nbr]=src

          ret = is_reachable(graph, nbr, dest, visited, path_nodes)
          if ret[0]==True:
            path_nodes = ret[1]
            path_nodes.append(nbr)
            flag= True

  path_nodes = list( set(path_nodes))
  return flag, path_nodes



def get_path(src, dest, parent):
    path=[]
    while dest!=src:
        path.append(dest)
        dest=parent[dest]

    path.append(src)
    return path



# after getting all the directions
def remove_conf(new_graph):    # calculated considering the indices of vi,vj not true name

  # print('entering into remove conf')
  # new_graph.draw_graph()

  ls = []
  edges = new_graph.get_edges()
  for tmp in edges:
    score = new_graph.adj_sigvalue[tmp[0]][tmp[1]]
    ls.append([score, tmp[0], tmp[1]])

  ls.sort(key=lambda x: int(x[0]))



  reachable_set=[]

  for edge in ls:
    v1 = edge[1]
    v2 = edge[2]

    # print('considering edge %s -> %s with sigval=%s' %(new_graph.vertices[v1], new_graph.vertices[v2], new_graph.adj_sigvalue[v1][v2]))
    # print('is reachable from %s to %s!!!' %(new_graph.vertices[v2], new_graph.vertices[v1]))
    new_graph.remove_edge(v1,v2)

    flag= False
    if new_graph.directed_adj[v2][v1] ==1:
      new_graph.remove_edge(v2,v1)
      flag= True

    # if (v2,v1) in reachable_set or is_reachable(new_graph, v2, v1, [], [] )[0] == True:
    ret = is_reachable(new_graph, v2, v1, [], [] )
    if ret[0] == True:
      # print('yes reachable.removing %s -> %s' %(new_graph.vertices[v1], new_graph.vertices[v2]))
      # print('path_nodes', get_names(new_graph, ret[1]))
      continue

    else:
      # print('not reachable')
      new_graph.add_edge(v1,v2)
      # reachable_set.append((v1,v2))
      if flag== True:
        new_graph.add_edge(v2,v1)


  # print('final graph after conflict removal')
  # new_graph.draw_graph()


  return new_graph


def remove_redun(new_graph):
  # print('remove redun');

  edges = new_graph.get_edges()
  for tmp in edges:
    v1id = tmp[0]
    v1 = new_graph.vertices[v1id]
    v2id = tmp[1]
    v2 = new_graph.vertices[v2id]

    path_nodes = is_reachable(new_graph, v1id , v2id, [], [] )[1]
    # print('getting path between ', v1,v2)

    path = [new_graph.vertices[vid] for vid in path_nodes]
    # print('path_nodes', path)

    if len(path) > 0 and is_conditional_independent_for_any_k(v1, v2, path, new_graph)[0]:
      # print('yes redundant removed %s -> %s' %(v1,v2))
      new_graph.remove_edge(v1id, v2id)
      new_graph.remove_edge(v2id, v1id)

  return new_graph

# true_graph = generate_true_causal_structure(10, 2)
# is_reachable(true_graph, 0, 9, [], {})

"""##SADA refinement"""

# refines graph with original variables of graph.vertices_list and threashold = graph.k_order
def refining_causal_graphs_SADA(graph):

  global sada_param
  sada_param.algo_state = 'refining_causal_graphs_SADA'

  tic = time.time()

  graph = remove_conf(graph)
  graph = remove_redun(graph)

  toc = time.time()
  sada_param.refined_time += (toc-tic)



  sada_param.algo_state = 'sada_end'


  return graph

"""##CP refinement"""

# Depends on is_korder_conditional_independent() function
# Depends on CausalGraph class. Object: graph

# refines graph with original variables of graph.vertices_list and threashold = graph.k_order
def refining_causal_graphs_CP(graph):

  global cp_param
  cp_param.algo_state = 'refining_causal_graphs_CP'

  tic = time.time()

  # print('graph refining for', graph.vertices)
  for vi in range(graph.V):
    for vj in range(graph.V):
      if graph.directed_adj[vi][vj] == 1:



        # finding all the neighbors of vi and vj
        set_of_neighbors = []
        for vk in range(graph.V):
          if vi != vk and vj != vk and (graph.directed_adj[vi][vk] == 1 or graph.directed_adj[vk][vi] == 1 or graph.directed_adj[vj][vk] == 1 or graph.directed_adj[vk][vj] == 1):
              set_of_neighbors.append(vk)

        set_of_neighbors = [graph.vertices[nbr] for nbr in set_of_neighbors]






        # removing the edge between vi and vj if they are independent when conditioned on small subsets of the set of neighbors
        for k_order in range(graph.k_thresh):


          # cit, zvl, sig = is_korder_conditional_independent(graph.vertices[vi], graph.vertices[vj], set_of_neighbors, k_order + 1, graph)
          cit, zvl, sig = is_korder_conditional_independent(graph.vertices[vi], graph.vertices[vj], set_of_neighbors, k_order, graph)
          if cit == True:
              # print('edge removed',graph.vertices[vi], graph.vertices[vj])
              cp_param.refined_edges[(graph.vertices[vi], graph.vertices[vj])]= zvl
              # print(cp_param.refined_edges)
              graph.directed_adj[vi][vj] = 0
              graph.directed_adj[vj][vi] = 0
          if graph.directed_adj[vi][vj] == 0:
              break

  toc = time.time()
  cp_param.refined_time += (toc-tic)


  cp_param.algo_state = 'cp_end'


  return graph

"""##EXP refinement"""

def get_neighbors(varid, graph):

  par_idset1 = np.nonzero(graph.directed_adj[:, varid])[0]
  # print('par_idset1', par_idset1)
  par_idset2 = np.nonzero(graph.directed_adj[varid, :])[0]
  # print('par_idset2', par_idset2)
  par_idset = list(set(par_idset1).union(par_idset2))

  return par_idset

def get_colliders(merged_graph, graph1, graph2):

  # print('Getting colliders')

  colliders = []

  for varid in range(merged_graph.V):
    var = merged_graph.vertices[varid]
    if (var not in graph1.vertices) or (var not in graph2.vertices):
      continue


    par_idset = get_neighbors(varid, merged_graph)
    # print('par_idset', par_idset)
    # print('all vertices',graph.vertices)
    # par_idset= par_idset1
    # print(par_idset)
    # print(merged_graph.vertices[par])
    parent_set = [merged_graph.vertices[i] for i in par_idset]

    is_collider = False
    for parvi in parent_set:
      for parvj in parent_set:
        if parvi == parvj:
          continue

        conseptset= None
        if (parvi, parvj) in graph_sep_set:
          conseptset = graph_sep_set[(parvi, parvj)]


        if conseptset!= None and var not in conseptset:
          colliders.append(varid)
          # print('!!!%s is collider for %s ind %s cz | %s' %(var, parvi,parvj,conseptset))
          is_collider = True
          break

      if is_collider:
        break



  colliders = list(set(colliders))
  return colliders

# Depends on is_korder_conditional_independent() function
# Depends on CausalGraph class. Object: graph

# refines graph with original variables of graph.vertices_list and threashold = graph.k_order
def refining_causal_graphs_EXP(merged_graph, graph1, graph2):
  global exp_param
  exp_param.algo_state =  'refining_causal_graphs_EXP'

  tic = time.time()



  # print('graph refining for', merged_graph.vertices)



  # ret = np.sum(merged_graph.directed_adj, axis=0)
  # coll_ids=np.where(ret>2)   #atleast 3, parent, spurious and from other partition
  # print(coll_ids)
  # coll_idset = coll_ids[0]
  # print('colliders_without_direction',colliders_without_direction)

  colliders_without_direction = get_colliders(merged_graph, graph1, graph2)
  coll_idset= colliders_without_direction
  collider_set = [merged_graph.vertices[i] for i in coll_idset]
  # print('colliders:',collider_set)

  for collid in coll_idset:

      collider = merged_graph.vertices[collid]
      if (collider not in graph1.vertices) or (collider not in graph2.vertices):
        continue

      # par_idset = np.nonzero(merged_graph.directed_adj[:, collid])[0]
      par_idset = get_neighbors(collid, merged_graph)

      parent_set = [merged_graph.vertices[i] for i in par_idset]

      # print('collider %s parents= %s' %(collider,parent_set))


      # this is each collider must have more than two neighbors
      if len(parent_set)<=2:
        # print('but for less than equal to two parents skipped')
        continue


      for parid in par_idset:
        parent = merged_graph.vertices[parid]
        # if (parent in graph1.vertices and parent not in graph2.vertices) or  can we optimize it?
        #     (parent in graph1.vertices and parent not in graph2.vertices):  #will not work

        #  this is a parent of collider




        ## these are neighbors of the parent of the collider
        ## finding all the neighbors of parent and collider

        # par_par_idset = get_neighbors(parid, merged_graph)
        # par_parent_set = [merged_graph.vertices[i] for i in par_par_idset]
        # print('parentset', parent_set)
        # print('par_parent_set',par_parent_set)
        # set_of_neighbors = copy.deepcopy(list(set(sorted(parent_set+par_parent_set))))   #may occur problem


        ## these are the neighbors of the collider
        # print('parentset', parent_set)
        set_of_neighbors = copy.deepcopy(parent_set)
        # print('conditioning on only parents of the collider')

        # print('set_of_neighbors',set_of_neighbors)
        if parent in set_of_neighbors:
            set_of_neighbors.remove(parent)



        # print('checking condition between par=%s child=%s',(parent,collider))

        # removing the edge between vi and vj if they are independent when conditioned on small subsets of the set of neighbors
        for k_order in range(merged_graph.k_thresh):


          # cit, zvl, sig = is_korder_conditional_independent(parent, collider, set_of_neighbors, k_order + 1, graph)
          # print('is ind? %s <-> %s| %s  where k=%s' %(parent, collider, set_of_neighbors,k_order))
          cit, zvl, sig = is_korder_conditional_independent(parent, collider, set_of_neighbors, k_order, merged_graph)
          if cit == True:
              # print('edge removed',parent, collider)
              exp_param.refined_edges[(parent, collider)]= zvl
              # print(refined_removed_edge)
              merged_graph.directed_adj[collid][parid] = 0
              merged_graph.directed_adj[parid][collid] = 0
          if merged_graph.directed_adj[parid][collid] == 0:
              break


  toc = time.time()
  exp_param.refined_time += (toc-tic)

  exp_param.algo_state =  'exp_end'

  return merged_graph

# Depends on is_korder_conditional_independent() function
# Depends on CausalGraph class. Object: graph

# refines graph with original variables of graph.vertices_list and threashold = graph.k_order
def refining_causal_graphs_EXP_backup(merged_graph, graph1, graph2):

  # print('graph refining for', merged_graph.vertices)


  # colliders_without_direction = get_colliders(merged_graph, graph1, graph2)

  ret = np.sum(merged_graph.directed_adj, axis=0)
  coll_ids=np.where(ret>2)   #atleast 3, parent, spurious and from other partition
  # print(coll_ids)
  coll_idset = coll_ids[0]
  collider_set = [merged_graph.vertices[i] for i in coll_idset]
  # print('colliders:',collider_set)
  # print('colliders_without_direction',colliders_without_direction)

  # collider_set= colliders_without_direction

  for collid in coll_idset:

      collider = merged_graph.vertices[collid]
      if (collider not in graph1.vertices) or (collider not in graph2.vertices):
        continue

      par_idset = np.nonzero(merged_graph.directed_adj[:, collid])[0]
      parent_set = [merged_graph.vertices[i] for i in par_idset]

      # print('collider %s parents= %s' %(collider,parent_set))



      for parid in par_idset:
        parent = merged_graph.vertices[parid]
        # if (parent in graph1.vertices and parent not in graph2.vertices) or  can we optimize it?
        #     (parent in graph1.vertices and parent not in graph2.vertices):  #will not work




        par_par_idset = np.nonzero(merged_graph.directed_adj[:, parid])[0]
        par_parent_set = [merged_graph.vertices[i] for i in par_par_idset]



        # finding all the neighbors of vi and vj
        set_of_neighbors = copy.deepcopy(list(set(sorted(parent_set+par_parent_set))))
        if parent in set_of_neighbors:
            set_of_neighbors.remove(parent)

        global exp_param

        # print('checking condition between par=%s child=%s',(parent,collider))

        # removing the edge between vi and vj if they are independent when conditioned on small subsets of the set of neighbors
        for k_order in range(merged_graph.k_thresh):


          # cit, zvl, sig = is_korder_conditional_independent(parent, collider, set_of_neighbors, k_order + 1, graph)
          # print('is ind? %s <-> %s| %s  where k=%s' %(parent, collider, set_of_neighbors,k_order))
          cit, zvl, sig = is_korder_conditional_independent(parent, collider, set_of_neighbors, k_order, merged_graph)
          if cit == True:
              # print('edge removed',parent, collider)
              exp_param.refined_edges[(parent, collider)]= zvl
              # print(refined_removed_edge)
              merged_graph.directed_adj[collid][parid] = 0
              merged_graph.directed_adj[parid][collid] = 0
          if merged_graph.directed_adj[parid][collid] == 0:
              break

  return merged_graph

"""##SADA main"""

# SADA

def SADA(graph):

    V = graph.vertices

    if len(V) <= graph.size_thres:
      # print('basic solver needed bcz of size less than threshold')
      result_graph= discover_causal_relation(graph)
      return result_graph


    # graph will go as parameter as read-only
    # V1,V3,V2 = find_causal_partition(graph)
    V1,V2 = find_causal_partition_sada(graph)
    # print('final causal partitions',V1,V2)
    if max(len(V1), len(V2))== len(V):
        # print('basic solver needed  since can not partition anymore')
        result_graph= discover_causal_relation(graph)
        return result_graph

    else:
        graph1 = SADA(CausalGraph(V1, graph))
        graph2 = SADA(CausalGraph(V2, graph))
        # graph3 = CAPA(CausalGraph(V3, graph))

        # here graphs will be read-write
        # print('merging graphs')
        # print(graph1.vertices, graph2.vertices)
        merged_graph= merge_partial_results(graph1,graph2)
        # print('hello')
        merged_graph= refining_causal_graphs_SADA(merged_graph)
        return merged_graph

"""##CP main"""

# CP

def CP(graph):

    V = graph.vertices

    if len(V) <= graph.size_thres:
      # print('basic solver needed bcz of size less than threshold')
      result_graph= discover_causal_relation(graph)
      return result_graph


    # graph will go as parameter as read-only
    # V1,V3,V2 = find_causal_partition(graph)
    V1,V2 = find_causal_partition_cp(graph)
    # print('final causal partitions',V1,V2)
    if max(len(V1), len(V2))== len(V):
        # print('basic solver needed  since can not partition anymore')
        result_graph= discover_causal_relation(graph)
        return result_graph

    else:
        graph1 = CP(CausalGraph(V1, graph))
        graph2 = CP(CausalGraph(V2, graph))


        # graph3 = CAPA(CausalGraph(V3, graph))

        # here graphs will be read-write
        # print('merging graphs')
        # print(graph1.vertices, graph2.vertices)
        merged_graph= merge_partial_results(graph1,graph2)
        # print('hello')
        merged_graph= refining_causal_graphs_CP(merged_graph)
        return merged_graph

"""##EXP main"""

# CP

def EXP(graph):

    V = graph.vertices

    if len(V) <= graph.size_thres:
      # print('basic solver needed bcz of size less than threshold')
      result_graph= discover_causal_relation(graph)
      return result_graph


    # graph will go as parameter as read-only
    # V1,V3,V2 = find_causal_partition(graph)
    V1,V2 = find_causal_partition_cp(graph)
    # print('final causal partitions',V1,V2)
    if max(len(V1), len(V2))== len(V):
        # print('basic solver needed  since can not partition anymore')
        result_graph= discover_causal_relation(graph)
        return result_graph

    else:
        graph1 = EXP(CausalGraph(V1, graph))
        graph2 = EXP(CausalGraph(V2, graph))
        # graph3 = CAPA(CausalGraph(V3, graph))

        # here graphs will be read-write
        # print('merging graphs')
        # print(graph1.vertices, graph2.vertices)
        merged_graph= merge_partial_results(graph1,graph2)
        # print('hello')
        merged_graph= refining_causal_graphs_EXP(merged_graph, graph1, graph2)
        # merged_graph= refining_causal_graphs_EXP_backup(merged_graph, graph1, graph2)
        return merged_graph

"""##CAPA main"""

# CAPA

def CAPA(graph):

    global capa_param
    V = graph.vertices

    if len(V) <= graph.size_thres:
      # print('basic solver needed bcz of size less than threshold')
      result_graph= discover_causal_relation(graph)
      return result_graph


    # graph will go as parameter as read-only
    V1,V3,V2 = find_causal_partition_capa(graph)
    # print('final causal partitions',V1,V3,V2)
    if max(len(V1), len(V2), len(V3))== len(V):
        # print('basic solver needed  since can not partition anymore')
        result_graph= discover_causal_relation(graph)
        return result_graph

    else:



        graph1 = CAPA(CausalGraph(V1, graph))

        graph2 = CAPA(CausalGraph(V2, graph))



        if true_graph.vertices == graph.vertices:
          # print('entering into the Capa partition3')
          capa_param.algo_state = 'CAPA_Partition3'
          tic = time.time()

        graph3 = CAPA(CausalGraph(V3, graph))

        if true_graph.vertices == graph.vertices:
          toc = time.time()
          capa_param.refined_time += (toc-tic)
          capa_param.algo_state = 'capa_end'
          # print('going out from capa partition3')


        # here graphs will be read-write
        # print('merging graphs')
        # print(graph1.vertices, graph2.vertices, graph3.vertices)
        merged_graph= merge_partial_results(graph1,graph2,graph3)
        return merged_graph

"""#Run algo

##Run CP
"""

def run_cp(current_graph1):

  global cp_param


  # print('CP algorithm starting')

  # run cp algorithm
  tic = time.time()
  cp_result_graph = CP(current_graph1)
  toc = time.time()
  cp_time= toc-tic
  # print('This is the final graph')
  cp_number_of_citest = number_of_citest


  cp_param.executed_time = cp_time

  #Result pring section

  # print('True grpah')
  # true_graph.draw_graph()


  # print('Result for CP')
  # print('This is the final graph')
  # cp_result_graph.draw_graph()
  R,P,F1 = get_undirected_score(true_graph, cp_result_graph)



  cp_param.result_graph = cp_result_graph



  cp_param.R= R
  cp_param.P= P
  cp_param.F1= F1



  # print('in run cp',cp_param.refined_edges)

  return cp_param

"""## Run CAPA"""

# run capa algorithm

def run_capa(current_graph2):
  # number_of_citest=0
  global capa_param

  # print('CAPA algorithm starting')

  tic = time.time()
  capa_result_graph = CAPA(current_graph2)
  toc = time.time()
  capa_number_of_citest = number_of_citest
  capa_time= toc-tic
  capa_param.executed_time = capa_time


  # print('Result for CAPA')
  # print('This is the final graph')
  # capa_result_graph.draw_graph()
  R,P,F1 = get_undirected_score(true_graph, capa_result_graph)


  capa_param.result_graph = capa_result_graph

  capa_param.R= R
  capa_param.P= P
  capa_param.F1= F1

  return capa_param

"""## Run SADA"""

# run sada algorithm

def run_sada(current_graph2):
  # number_of_citest=0
  global sada_param

  # print('SADA algorithm starting')

  tic = time.time()
  sada_result_graph = SADA(current_graph2)
  toc = time.time()
  sada_number_of_citest = number_of_citest
  sada_time= toc-tic
  sada_param.executed_time = sada_time


  # sada_result_graph.draw_graph()
  R,P,F1 = get_undirected_score(true_graph, sada_result_graph)


  sada_param.result_graph = sada_result_graph

  sada_param.R= R
  sada_param.P= P
  sada_param.F1= F1

  return sada_param

"""##Run exp"""

def run_exp(current_graph):

  global exp_param

  # print('EXP algorithm starting')


  # run exp algorithm
  tic = time.time()
  exp_result_graph = EXP(current_graph)
  toc = time.time()
  exp_time= toc-tic

  exp_param.executed_time = exp_time


  R,P,F1 = get_undirected_score(true_graph, exp_result_graph)



  exp_param.result_graph = exp_result_graph

  exp_param.R= R
  exp_param.P= P
  exp_param.F1= F1

  return exp_param

"""##Simulator"""

import time


#Data generation
## Load saved data
# true_graph = read_net('graph.net')
true_graph = read_net_hugin('pigs.net')
# true_graph = read_net('insurance.net')
# true_graph = read_net('win95pts.net')
# true_graph = read_net('hailfinder.net')

# generated_dataset = read_dataset('data.csv')

for itr in range(20):
    for num_of_samples in [250, 500, 1000, 1500, 2000]:

        # Global variables
        graph_sep_set = {}
        refined_removed_edge ={}
        exp_removed_edge ={}
        cp_total_checked =0
        iterations =0
        number_of_citest = 0

        # Inputs
        num_of_nodes=10
        # num_of_samples= 5* num_of_nodes
        # num_of_samples= 500             # sample size 100-200  or 25-400
        in_degree = 2  #connected
        edge_prob = 0.2

        capa_param = GraphParameters('CAPA')
        cp_param = GraphParameters('CP')
        exp_param = GraphParameters('EXP')
        sada_param = GraphParameters('SADA')

        ## Random data
        # true_graph = generate_random_DAG(num_of_nodes, edge_prob)
        # true_graph = generate_true_causal_structure(num_of_nodes, in_degree)
        generated_dataset = generate_LiNGAM_data(true_graph, 0.3, true_graph.V, num_of_samples)

        current_graph =  CausalGraph([nd for nd in range(0, true_graph.V)])
        current_graph.load_data(generated_dataset)

        # Write data
        # write_net(true_graph, 'graph.net', generated_dataset, 'data.csv')
        # write_net_hugin(true_graph, 'graph.net', generated_dataset, 'data.csv')

        current_graph.case=0
        current_graph1= copy.deepcopy(current_graph)
        current_graph2= copy.deepcopy(current_graph)
        current_graph3= copy.deepcopy(current_graph)
        current_graph4= copy.deepcopy(current_graph)



        sada_param = run_sada(current_graph1)
        sada_graph_sep_set = copy.deepcopy(graph_sep_set)   # taking the graph sepset of sada
        graph_sep_set.clear()

        cp_param = run_cp(current_graph3)
        cp_graph_sep_set = copy.deepcopy(graph_sep_set)   # taking the graph sepset of cp
        graph_sep_set.clear()

        exp_param = run_exp(current_graph4)
        exp_graph_sep_set = copy.deepcopy(graph_sep_set)   # taking the graph sepset of exp
        graph_sep_set.clear()


        print('Sample size =',num_of_samples)
        # print('true graph')
        # true_graph.draw_graph()
        # sada_param.result_graph.draw_graph()
        # capa_param.result_graph.draw_graph()
        # cp_param.result_graph.draw_graph()
        # exp_param.result_graph.draw_graph()

        ########################### SADA #################################
        sada_param.print_stat()
        sada_param.print_test_time()
        sada_param.print_edges()

        # ########################### CP #################################
        cp_param.print_stat()
        cp_param.print_test_time()
        cp_param.print_edges()

        # ########################### EXP #################################
        exp_param.print_stat()
        exp_param.print_test_time()
        exp_param.print_edges()


        # exp_param.compare(sada_param, capa_param, cp_param)
        print('',end='',flush=True)

"""# Todo




*   Remove redundancy for which k and check if its okay according to matlab codes
"""
