#!/usr/bin/python
from __future__ import division
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from operator import attrgetter
import math
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Arc, RegularPolygon

import sys
sys.path.insert(0,'/home/olivier/NFK')
from Code.networkFlow import netWork
from Code.computing import optim

def NodeDisplay(nodeList, pltName, commodities, toggle): # displays only the nodes on an x-t diagram
    ''' Nodedisplay does an x-y plot of the nodes in the network:

    Inputs: nodelist    A list of nodes that need to be displayed
        pltname     The name of the plot that is created
        commodities These are the names that may be used for display purposes
        toggle      The toggle affords options, such as
                    0: a plain plot with nodes, location and names is shown
                    1: the 0 plot plus the sources and sinks (all where b_i != 0)
                    2: the 1 plot plus the names of the commodities '''
    # Display of the nodes of the network:    
    size      = len(nodeList)
    x = map(attrgetter('x_location'), nodeList)
    y = map(attrgetter('y_location'), nodeList)
    n = map(attrgetter('name'), nodeList)    
    b = map(attrgetter('b'), nodeList) 
    
    pltName = plt.scatter(x,y,s=300)
    for name,x_c,y_c, in zip(n, x, y):
        plt.annotate(
        name,
        xy=(x_c, y_c), xytext=(-10, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    if toggle != 0:
        for b,x_c,y_c, in zip(b, x, y):
            plt.annotate(
            ComDisplay(b,commodities,toggle),
            xy=(x_c, y_c), xytext=(20, -10),
            textcoords='offset points', ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.1', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))    
    ax = plt.gca()
    # Adjusts axes for cosmetics (includes all the labels)
    ax.set_ylim(ax.get_ylim()[0]*1.1,ax.get_ylim()[1]*1.1)
    ax.set_xlim(ax.get_xlim()[0]*1.1,ax.get_xlim()[1]*1.1)
    
def ComDisplay(b,commodities, toggle):
    ''' ComDisplay serves only the NodeDisplay function by creating a
    string for the display of the sources and sinks.'''
    string = 'b:\n'
    for b_i, c in zip(b,commodities):
        if b_i != 0:
                if toggle == 1:
                    string += str(b_i) + '\n'
                if toggle == 2:
                    string += c + ' : ' + str(b_i) + '\n'
    #if toggle == 2:
    return string
    
#def NetworkDisplay(networkInput, pltName): # displays only the nodes on an x-t diagram
#
#    # Parameters of the NetworkDisplay function:
#    circleRadius = 1
#
#    # Display of the nodes of the network:  
#    nodeList = networkInput.get_nodeList()
#    arcList  = networkInput.get_arcList()
#    
#    x = map(attrgetter('x_location'), nodeList)
#    y = map(attrgetter('y_location'), nodeList)
#    n = map(attrgetter('name'), nodeList)    
#    outNodeList = map(attrgetter('outgoingNode'), arcList)
#    inNodeList  = map(attrgetter('incomingNode'), arcList)    
#    # Display of the arcs of the network:  
#    
#    for index in range(len(arcList)):
#
#        x0 = outNodeList[index].getX_location()
#        y0 = outNodeList[index].getY_location()
#       
#        x1 = inNodeList[index].getX_location()
#        y1 = inNodeList[index].getY_location()
#        pltName = plt.plot([x0, x1],[y0, y1])
#        
#        # displays a circle for a graph loop:
#        if ((x1-x0)**2 + (y1-y0)**2 == 0):
#            
#            circle = plt.Circle((x1-circleRadius, y1), circleRadius,
#                                 color='b', fill=False)
#            ax = plt.gca()
#            ax.add_artist(circle)
#            pltName = plt.annotate(
#            str(index),
#            xy=((x1-2*circleRadius),(y1)), xytext=(-20, 20),        
#            textcoords='offset points', ha='right', va='bottom',
#            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
#            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
#            
#        # displays an arrow for a flow between 2 distinct points:
#        else:
#            pltName = plt.arrow(x0, y0, x1 - x0, y1 - y0,
#                  head_width=0.3, length_includes_head=True)
#            pltName = plt.annotate(
#            str(index),
#            xy=((0.3*x0 + 0.7*x1),(0.3*y0 + 0.7*y1)), xytext=(-20, 20),        
#            textcoords='offset points', ha='right', va='bottom',
#            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
#            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
#    
#    pltName = plt.scatter(x,y,s=400) 
#    for name,x_c,y_c, in zip(n, x, y):
#        pltName = plt.annotate(
#        name,
#        xy=(x_c, y_c), xytext=(-20, 20),
#        textcoords='offset points', ha='right', va='bottom',
#        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
#        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
#    
#    # Setting limits to plot: 
#    x_min = min(x)
#    x_max = max(x)
#    y_min = min(y)
#    y_max = max(y)
#    
#    axes = plt.gca()
#    axes.set_xlim([x_min-(2*circleRadius+1),x_max+(2*circleRadius+1)])
#    axes.set_ylim([y_min-(2*circleRadius+1),y_max+(2*circleRadius+1)])    
#    pltName = plt.show()      
    
def GraphDisplay(G):
    pos=nx.get_node_attributes(G,'pos')  
    nx.draw(G,pos)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos, edge_labels=labels)


def c_1P(p1,p2,rad): # calculates the point C1 (middle point) according to the arc3 definition
        x_0 = p1[0] 
        y_0 = p1[1]
        x_1 = p2[0]
        y_1 = p2[1]
        C1_x = 0.5*(x_1 + x_0) + rad*(y_1 - y_0)
        C1_y = 0.5*(y_1 + y_0) + rad*(x_0 - x_1)
        C1 = (C1_x, C1_y)
        return C1

def Qbezier(C0,C1,C2, **kwargs): # calculates the middle point of a quad-bezier curve
    t = 0.5   
    # create the if condition if the *kwargs do not contain a parameter t
    for key, value in kwargs.items():
        if (key == 't'):
            t = kwargs[key]        
    x = (1-t)*(1-t)*C0[0] + 2*(1-t)*t*C1[0] + t*t*C2[0]
    y = (1-t)*(1-t)*C0[1] + 2*(1-t)*t*C1[1] + t*t*C2[1]
    C_mid = (x,y)
    return C_mid

def drawCirc(ax,radius,centX,centY,angle_,theta2_,color_='black'):
    #========Line
    arc = Arc([centX-0.5*radius,centY],radius,radius,angle=angle_,
          theta1=0,theta2=theta2_,capstyle='round',linestyle='-',lw=1,color=color_)
    ax.add_patch(arc)


    #========Create the arrow head
    endX=centX+(radius/2)*np.cos(np.radians(theta2_+angle_)) #Do trig to determine end position
    endY=centY+(radius/2)*np.sin(np.radians(theta2_+angle_))

    ax.add_patch(                    #Create triangle as arrow head
        RegularPolygon(
            (endX-0.5*radius, endY),            # (x,y)
            3,                       # number of vertices
            radius/15,                # radius
            np.radians(angle_+theta2_),     # orientation
            color=color_
        )
    )
    ax.set_xlim([centX-radius,centY+radius]) and ax.set_ylim([centY-radius,centY+radius]) 
    # Make sure you keep the axes scaled or else arrow will distort
    
###############################################################################    
def getPos(networkInput):
    # Get the nodes out of the nodelist:
    nodeList = networkInput.get_nodeList()
    
    # Get the xy coordinates: 
    x = map(attrgetter('x_location'), nodeList)
    y = map(attrgetter('y_location'), nodeList)
    
    # Put them in tuple form:
    xyPos = zip(x,y)
    
    # Put them in dictionary form:
    n = range(len(xyPos))
    xyPosN = zip(n,xyPos)
    pos = dict(xyPosN)
    return pos

def getEdges(networkInput):
    # Get the edges of the edgelist:
    arcList  = networkInput.get_arcList()
    nodeList = networkInput.get_nodeList()

    # Get the outgoing and incoming nodes of the edges:
    outNodeList = map(attrgetter('outgoingNode'), arcList)
    inNodeList  = map(attrgetter('incomingNode'), arcList) 
    
    # Get the nodes indexes in the initial list:
    inNodeIndexes =  [None] * len(arcList)
    outNodeIndexes = [None] * len(arcList)
    
    for n in range(len(arcList)):
        outNodeIndexes[n] = nodeList.index(outNodeList[n])        
        inNodeIndexes[n]  = nodeList.index(inNodeList[n])

    # Put them into a list of tuples:
    edges = zip(outNodeIndexes, inNodeIndexes)
    return edges
    
def nodeNamesDict(networkInput):
    # Get the nodes out of the nodelist:
    nodeList = networkInput.get_nodeList()
    
    # Get the names of the nodes:
    names = map(attrgetter('name'), nodeList)
    
    # Put them into dictionnary form:
    n = range(len(names))
    names = zip(n,names)
    names = dict(names)
    return names

def buildMultigraph(networkInput):
    # Get nodes and node x-y locations:
    pos = getPos(networkInput)
        
    # Create Multi-Graph:
    G=nx.MultiDiGraph()
    G.add_nodes_from(pos.keys())
    for n, p in pos.iteritems():
        G.node[n]['pos'] = p
        
    # Add the edges to the Multi-Graph:
    edges = getEdges(networkInput)
    G.add_edges_from(edges)  
    return G  

def draw_network(networkInput, ax, sg=None, **kwargs):

    # Parameters:
    Crad    = 0.2 # radius of the circles of the nodes
    loopRad = 0.6 # radius of the loops 
    # Create the Multigraph G used for plotting:
    G = buildMultigraph(networkInput)

    # Get node names:
    node_names = nodeNamesDict(networkInput)
    
    # Get node positions:
    pos = getPos(networkInput)
    
    # Create a range vector with the number of edges in the graph:
    edgeLabel = [x+1 for x in range(len(G.edges()))]
    edgeLabel = np.matrix(edgeLabel)

    
    # create the if condition if the *kwargs do not contain an edgeVector:
    for key, value in kwargs.items():
        if (key == 'edgeLabel'):
            edgeLabel = kwargs[key]
    
    # Create a default "sense" for the display:
    sense = '+'
    # create the if condition if the *kwargs do not contain a plus or minus:
    for key, value in kwargs.items():
        if (key == 'sense'):
            sense = kwargs[key]         

    commoditiesLabel = ['']
    # create the if condition if the *kwargs do not contain legend labels:
    for key, value in kwargs.items():
        if (key == 'commodities'):
            commoditiesLabel = kwargs[key]   

    for n in G: # enumerates the nodes in the network
        c=Circle(pos[n],radius=Crad,alpha=0.5)
        ax.add_patch(c)
        G.node[n]['patch']=c
        x,y=pos[n]
        ax.text(x, y-2*Crad, node_names[n], fontsize=15)
    seen={}

    ijk = -1 # iterator for edge number
    # color map initialize:
    jet = plt.get_cmap('jet') 
    values = range(len(commoditiesLabel))
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    colorVal = scalarMap.to_rgba(0)
    patches = []
    nodeList = networkInput.get_nodeList()
    invertIdx = 0 # idx that goes to 1 if there are loops in the graph. Helps with the plot being reversed.
    #for (u,v,d) in G.edges(data=True):
    idx = -1
    for edge in networkInput.arcList:

        outNode= edge.get_outgoing()
        u = nodeList.index(outNode)
        
        inNode= edge.get_incoming()
        v = nodeList.index(inNode)
         
        
        # Find the proper index for the edge (MultiGraph rearragnes the edge sequence...)
        #idx = getEdges(networkInput).index((u,v))
        idx = idx+ 1 # the order of the edges in the network is static! I must not change! This works for parrallel edges. 
        n1=G.node[u]['patch']
        n2=G.node[v]['patch']
        rad=0.1
        if (u,v) in seen:
            rad=seen.get((u,v))
            rad=(rad+np.sign(rad)*0.1)*-1

        alpha=0.5
        color='k'

        if (n1 == n2):
            invertIdx = 1
            drawCirc(ax,loopRad,n1.center[0],n1.center[1],0, 300,color_='k')     
            n = edgeLabel.shape[0]
            for i in range(n):
                colorVal = scalarMap.to_rgba(values[i])
                ax.text(n1.center[0] - 0.5*loopRad + 0.7*loopRad*math.cos(2*math.pi*(i+1)/(n+1)),
                        n1.center[1]               + 0.7*loopRad*math.sin(2*math.pi*(i+1)/(n+1)),
                        u'%s'%edgeLabel[i,idx], fontsize=15, color = colorVal)
        else:                            
            e = FancyArrowPatch(n1.center,n2.center,patchA=n1,patchB=n2,
                                arrowstyle='-|>',
                                connectionstyle='arc3,rad=%s'%rad,
                                mutation_scale=10.0,
                                lw=2,
                                alpha=alpha,
                                color=color,
                                visible = True)
                                
            n = edgeLabel.shape[0]
            c_1 = c_1P(n1.center,n2.center,rad)       

            for i in range(n):
                t = (i+1)/(n+1)
                Q_p = Qbezier(n1.center,c_1,n2.center, t = t)
                colorVal = scalarMap.to_rgba(values[i])
                ax.text(Q_p[0], Q_p[1], u'%s'%edgeLabel[i,idx], fontsize=15, color=colorVal)
        
        seen[(u,v)]=rad
        ax.add_patch(e)
                            
    idx = -1
    for comLab in commoditiesLabel:
        idx = idx +1
        colorVal = scalarMap.to_rgba(values[idx])
        patches.append(mpatches.Patch(color=colorVal, label=comLab + ' ' + sense[idx])) 
    
    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower right')
    if invertIdx == 0:
        ax.invert_xaxis()
    plt.legend(handles=patches)
    #return e

def networkDisplay(networkInput, pltName, **kwargs): # displays only the nodes on an x-t diagram

    # Create plot and save it:
    ax=plt.gca()
    
    # Call draw_network
    draw_network(networkInput, ax, **kwargs)
    ax.autoscale()
    ax.invert_xaxis()
    plt.axis('equal')
    #plt.axis('off')
    plt.axis('on')
    plt.savefig(pltName + ".pdf")
    plt.figure().canvas.draw()
    #plt.show()
    
def graphTopo(networkInput, pltName):
    networkDisplay(networkInput, pltName)
   
def graphFlow(x, networkInput, pltName, **kwargs):
    # get the number of commodities:
    comNames = networkInput.get_comNames()
    Ncom = len(comNames)
    x_plus = optim.xTabPlus(x,Ncom)
    x_minus = optim.xTabMinus(x,Ncom)
    N_x  = len(x_plus[:,0])
    # Select the proper commodity:
    
    # Create a range vector with the number of edges in the graph:
    commodities = [comNames[0]]
    
    # create the if condition if the *kwargs do not contain a commodity name (defaults to first com):
    for key, value in kwargs.items():
        if (key == 'commodities'):
            commodities= kwargs[key]    
    
    idxCom = []
    for name in commodities:
        idxCom.append(comNames.index(name))
    
    nComDis = len(idxCom)
    
    # Create a default value for the sense of the edge data (in, out or both)
    sense = '+'*nComDis
    # create the if condition if the *kwargs do not contain a plus or minus:
    for key, value in kwargs.items():
        if (key == 'sense'):
            sense = kwargs[key] 
            
    edgeLabel = np.zeros((nComDis, N_x)) # declare empty matrix for storing the x values.
    idxLabel = -1        
    for idx in idxCom:
        idxLabel = idxLabel + 1
        if sense[idxLabel] == '+':
            edgeLabel[idxLabel,:] = x_plus[:,idx]
        elif sense[idxLabel] == '-':
            edgeLabel[idxLabel,:] = x_minus[:,idx]
        else:
            print 'the sense vector is incorrect. Please check.'


    # Create a default value for the digits for rounding:
    rnd = 2   
    # create the if condition if the *kwargs do not contain a plus or minus:
    for key, value in kwargs.items():
        if (key == 'round'):
            rnd = kwargs[key]     
    
    edgeLabel = np.matrix(edgeLabel)
    edgeLabel = edgeLabel.round(rnd)
            
    networkDisplay(networkInput, pltName, edgeLabel = edgeLabel, sense = sense, commodities = commodities)
    
def plotNoth(x, Y, labels, title, xlabel, ylabel):
    # color map initialize:
    jet = plt.get_cmap('jet') 
    values = range(len(labels))
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    colorVal = scalarMap.to_rgba(0)
    patches = []
    # Create plot and save it:
    ax=plt.gca()
    
    idx = -1
    for y in Y:        
        idx = idx+ 1
        colorVal = scalarMap.to_rgba(values[idx])
        plt.plot(x,y, color = colorVal)
        plt.plot(x,y,'o',color = colorVal)
                                
    idx = -1
    for Label in labels:
        idx = idx +1
        colorVal = scalarMap.to_rgba(values[idx])
        patches.append(mpatches.Patch(color=colorVal, label=Label)) 
    
    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower right')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.legend(handles=patches)