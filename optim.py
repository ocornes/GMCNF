#!/usr/bin/python

# Copyright 2017, Gurobi Optimization, Inc.

# This example formulates and solves the following simple QP model:
#
#    minimize    x + y + x^2 + x*y + y^2 + y*z + z^2
#    subject to  x + 2 y + 3 z >= 4
#                x +   y       >= 1
#
# The example illustrates the use of dense matrices to store A and Q
# (and dense vectors for the other relevant data).  We don't recommend
# that you use dense matrices, but this example may be helpful if you
# already have your data in this format.

import sys
from gurobipy import *
import numpy as np
import sys
sys.path.insert(0,'/home/olivier/NFK')
from Code.networkFlow import netWork
from operator import attrgetter
import matplotlib.pyplot as plt

def dense_optimize_lin(rows, cols, c, A, sense, rhs, lb, ub, vtype, solution):

  model = Model()

  # Add variables to model
  vars = []
  for j in range(cols):
    vars.append(model.addVar(lb=lb[j], ub=ub[j], vtype=vtype[j]))

  # Populate A matrix
  for i in range(rows):
    expr = LinExpr()
    for j in range(cols):
      if A[i][j] != 0:
        expr += A[i][j]*vars[j]
    model.addConstr(expr, sense[i], rhs[i])

  # Populate objective: 
  obj = LinExpr()    
  for j in range(cols):
    if c[j] != 0:
      obj += c[j]*vars[j]
  model.setObjective(obj)

  # Quieting Gurobi in terminal: 
  model.setParam( 'OutputFlag', False )
  
  # Solve
  model.optimize()

  # Write model to a file
  model.write('dense.lp')

  if model.status == GRB.Status.OPTIMAL:
    x = model.getAttr('x', vars)
    for i in range(cols):
      solution[i] = x[i]
    return True
  else:
    return False
                       
def dense_optimize_quad(rows, cols, c, Q, A, sense, rhs, lb, ub, vtype,
                   solution):
  model = Model()

  # Add variables to model
  vars = []
  for j in range(cols):
    vars.append(model.addVar(lb=lb[j], ub=ub[j], vtype=vtype[j]))

  # Populate A matrix
  for i in range(rows):
    expr = LinExpr()
    for j in range(cols):
      if A[i][j] != 0:
        expr += A[i][j]*vars[j]
    model.addConstr(expr, sense[i], rhs[i])

  # Populate objective
  obj = QuadExpr()
  for i in range(cols):
    for j in range(cols):
      if Q[i][j] != 0:
        obj += Q[i][j]*vars[i]*vars[j]
  for j in range(cols):
    if c[j] != 0:
      obj += c[j]*vars[j]
  model.setObjective(obj)

  # Solve
  model.optimize()

  # Write model to a file
  model.write('dense.lp')

  if model.status == GRB.Status.OPTIMAL:
    x = model.getAttr('x', vars)
    for i in range(cols):
      solution[i] = x[i]
    return True
  else:
    return False

def senseGRB(bub, beq): # inputs are the RHS of the network problem
    # get size of total sense vector:
    LE = ['<']*len(bub)
    EQ = ['=']*len(beq);
    # Add the signs into a full list for input to Gurobi:
    sense = LE + EQ
    return sense

def vtypeGRB(c, Ncom, listInts): 
    '''inputs is the number of variables (length of cost function), number of commodities
    and commodities that have to be integers.'''
    # default is continuous:
    vtype = ['C']*len(c)
    # Number of edges:
    itEdge = range(len(c)/Ncom)
    # Changes the Continuous type variables to integers if instructed:
    for ij in itEdge:
        for jk in listInts:
            vtype[ij*Ncom + listInts] = 'I'
    return vtype

def Ncol(c):# extracts number of columns in the problem
    return len(c)
    
def Nrow(Aub,Aeq):## extracts number of columns in the problem
    return Aub.shape[0] + Aeq.shape[0]
    
def bigA(Aub,Aeq):
    return np.concatenate((Aub,Aeq),axis = 0)

def bigRhs(bub,beq):
    return np.concatenate((bub,beq),axis = 0)

def lbounds(bounds):
    return [int(i[0]) for i in bounds]

def ubounds(bounds):
    return [int(i[1]) for i in bounds]    

def xTabPlus(x,Ncom):
    # x must be a list. This is the table for the outgoing flows of the edge
    ''' Builds a matrix whoes rows are commodities and lines are edges
        [[Com1, Com2, Com3 ...],   Edge 1
        [Com1, Com2, Com3 ...]... Edge 2
                                   Edge i
         [Com1, Com2, Com3]]       Edge N
         
         This is only for the outgoing variables (x+_ij)'''
         # Classify the results into a table
    x_plus  = np.array(x[0:len(x)/2])
    n_edge  = len(x_plus)/Ncom # calculates the number of edges
    xTab    = np.reshape(x_plus,(n_edge,Ncom))
    return xTab

def xTabMinus(x,Ncom):
    # x must be a list. This is the table for the incoming flows of the edge
    ''' Builds a matrix whoes rows are commodities and lines are edges
        [[Com1, Com2, Com3 ...],   Edge 1
        [Com1, Com2, Com3 ...]... Edge 2
                                   Edge i
         [Com1, Com2, Com3]]       Edge N
         
         This is only for the outgoing variables (x+_ij)'''
         # Classify the results into a table
    x_minus  = np.array(x[len(x)/2:len(x)])
    n_edge  = len(x_minus)/Ncom # calculates the number of edges
    xTab    = np.reshape(x_minus,(n_edge,Ncom))
    return xTab
    
def flowVsEdge_plus(x, network):
    nCom = len(network.get_comNames())
    # Getting the commodity names:
    comNames = network.get_comNames()
    # producing the ingoing and outgoing flows:
    outFlows = xTabPlus(x,len(comNames))
    # producing the number of edges:
    edgeList = network.get_arcList()
    outNodes = map(attrgetter('outgoingNode'), edgeList)
    inNodes  = map(attrgetter('incomingNode'), edgeList)
    
    outNodeNames = map(attrgetter('name'), outNodes)
    inNodeNames  = map(attrgetter('name'), inNodes)
    
    # Create labels for axes:
    my_xticks = [a + ' -> ' +b for a, b in zip(outNodeNames, inNodeNames)]
    
    newFig = plt.figure() 
    xaxisrange = range(1,len(network.get_arcList())+1)        
    
    for i in range(nCom):

        plt.plot(xaxisrange, outFlows[:,i] ,linewidth=2 ,label=comNames[i])

    plt.xticks(xaxisrange, my_xticks)
    
    plt.legend(loc='best')
    plt.xlim(0, len(network.get_arcList()) + 1)
    plt.ylim(-1, outFlows.max()+2)
    plt.show()
    
    
def flowVsEdge_minus(x, network):
    nCom = len(network.get_comNames())
    # Getting the commodity names:
    comNames = network.get_comNames()
    # producing the ingoing and outgoing flows:
    inFlows  = xTabMinus(x, len(comNames))
    # producing the number of edges:
    edgeList = network.get_arcList()
    outNodes = map(attrgetter('outgoingNode'), edgeList)
    inNodes  = map(attrgetter('incomingNode'), edgeList)
    
    outNodeNames = map(attrgetter('name'), outNodes)
    inNodeNames  = map(attrgetter('name'), inNodes)
    
    # Create labels for axes:
    my_xticks = [a + ' -> ' +b for a, b in zip(outNodeNames, inNodeNames)]
    print my_xticks
    
    newFig = plt.figure()
    xaxisrange = range(1,len(network.get_arcList())+1)        
    
    for i in range(nCom):

        plt.plot(xaxisrange, inFlows[:,i] ,linewidth=2 ,label=comNames[i])

    plt.xticks(xaxisrange, my_xticks)
    
    plt.legend(loc='best')
    plt.xlim(0, len(network.get_arcList()) + 1)
    plt.ylim(-1, inFlows.max()+2)
    plt.show()
    
def buildPbm(network):
    
    c        = network.obj_fun()
    Aub, bub = network.sub_eq_constraints()
    Aeq, beq = network.eq_constraints()
    boundsLol= network.bounds()
    listInts = []
    Ncom     = len(network.get_comNames())
    rows  = Nrow(Aub,Aeq)
    cols  = Ncol(c)
    A     = bigA(Aub,Aeq)
    sense = senseGRB(bub, beq)
    rhs   = bigRhs(bub,beq)
    lb    = lbounds(boundsLol)
    ub    = ubounds(boundsLol)
    vtype = vtypeGRB(c, Ncom, listInts)
    sol   = [0]*cols
    
    success = dense_optimize_lin(rows, cols, c, A, sense, rhs, lb, ub, vtype, sol)
    
    if success:
      x = sol
    else:
          print 'Optimization Failure'

    return x