# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:36:29 2017

@author: olivier
"""
import numpy as np
from operator import attrgetter
from basicFunctions import placeMat, findName, print_mat

import networkx as nx
import matplotlib.pyplot as plt


class Node(object):
    """ A node is one of the building blocks of a network: it contains a vector b
    of supply/demand for each commodity in the network, the nodes also have an ID
    (name)"""
    
    def __init__(self, name, b, x_location, y_location):
        self.name       = name
        self.b          = b
        self.x_location = x_location
        self.y_location = y_location
    
    def get_name(self): 
        return self.name 
        
    def get_b(self): 
        return self.b         

    def getX_location(self):
        return self.x_location

    def getY_location(self):
        return self.y_location
        
    def set_name(self,name):
            self.name = name
            
    def set_b(self, b):
            self.b = b
        
    def setX_location(self, x):
        self.x_location = x

    def setY_location(self, y):
        self.y_location = y        
        
    def duplicateNode(self):
        name        = self.name
        b           = self.b
        x_location  = self.x_location
        y_location  = self.y_location
        duplicatedNode = Node(name, b, x_location, y_location)
        return duplicatedNode    
        

class Edge(object):
    """ A edge is one of the building blocks of a network: it contains a outgoing
    Node and an incoming Node (defines spatial network), and has a a vector maxFlow
    for the maximum flow capacity of the arc, and a minFlow for the minimum capacity
    of the flow. mu is the vector containing the gainy/lossy aspect of the edge 
    in question. c is the cost of shipping a commodity through the arc
"""
    
    def __init__(self, outgoingNode, incomingNode, maxFlow_plus, minFlow_plus, 
                 maxFlow_mins, minFlow_mins, c_plus, c_minus, A_plus, A_minus, 
                 B, C_plus, C_minus, **kwargs):

        self.outgoingNode   = outgoingNode
        self.incomingNode   = incomingNode
        self.maxFlow_plus   = maxFlow_plus
        self.minFlow_plus   = minFlow_plus
        self.maxFlow_mins   = maxFlow_mins
        self.minFlow_mins   = minFlow_mins
        self.c_plus         = c_plus
        self.c_minus        = c_minus
        self.A_plus         = A_plus
        self.A_minus        = A_minus        
        self.B              = B
        self.C_plus         = C_plus      
        self.C_minus        = C_minus
        self.name           = kwargs.get('name','defaultName')
        
        if self.name == 'defaultName':
            self.name = self.outgoingNode.get_name() + '_' + self.incomingNode.get_name()
        
        if C_plus.shape[1] != B.shape[1]:
            print "ERROR: The + concurrency matrix of edge " + self.name + " size does not match that of the tranformation matrix! "
            raise ValueError
            
        if C_minus.shape[1] != B.shape[1]:
            print "ERROR: The - concurrency matrix of edge " + self.name + " size does not match that of the tranformation matrix!"
            raise ValueError
           
    def get_outgoing(self): 
        return self.outgoingNode
        
    def get_incoming(self): 
        return self.incomingNode
        
    def get_maxFlowPlus(self): 
        return self.maxFlow_plus 
        
    def get_minFlowPlus(self): 
        return self.minFlow_plus

    def get_maxFlowMins(self): 
        return self.maxFlow_mins 
        
    def get_minFlowMins(self): 
        return self.minFlow_mins

    def get_A_plus(self): 
        return self.A_plus
        
    def get_A_minus(self): 
        return self.A_minus        

    def get_B(self): 
        return self.B 

    def get_C_plus(self): 
        return self.C_plus

    def get_C_minus(self): 
        return self.C_minus
        
    def get_c_plus(self): 
        return self.c_plus        

    def get_c_minus(self): 
        return self.c_minus

    def set_outgoing(self, outgoingNode): 
        self.outgoingNode = outgoingNode
        
    def set_incoming(self, incomingNode): 
        self.incomingNode = incomingNode
        
    def set_maxFlowPlus(self, maxFlow_plus): 
        self.maxFlow_plus = maxFlow_plus
        
    def set_minFlowPlus(self,minFlow_plus): 
        self.minFlow_plus = minFlow_plus

    def set_maxFlowMins(self, maxFlow_mins): 
        self.maxFlow_mins = maxFlow_mins
        
    def set_minFlowMins(self, minFlow_mins): 
        self.minFlow_mins = minFlow_mins

    def set_A_plus(self, A_plus): 
        self.A_plus = A_plus
        
    def set_A_minus(self, A_minus): 
        self.A_minus = A_minus       

    def set_B(self, B): 
        self.B = B 

    def set_C_plus(self, C_plus): 
        self.C_plus = C_plus

    def set_C_minus(self, C_minus): 
        self.C_minus = C_minus
        
    def set_c_plus(self, c_plus): 
        self.c_plus = c_plus        

    def set_c_minus(self, c_minus): 
        self.c_minus = c_minus
        
    def duplicateArc(self):
        outgoingNode = self.outgoingNode
        incomingNode = self.incomingNode   
        maxFlow_plus = self.maxFlow_plus   
        minFlow_plus = self.minFlow_plus  
        maxFlow_mins = self.maxFlow_mins
        minFlow_mins = self.minFlow_mins
        c_plus       = self.c_plus
        c_minus      = self.c_minus
        A_plus       = self.A_plus
        A_minus      = self.A_minus
        B            = self.B     
        C_plus       = self.C_plus
        C_minus      = self.C_minus  
        duplicatedArc = Arc(outgoingNode, incomingNode, maxFlow_plus, minFlow_plus, maxFlow_mins, minFlow_mins, c_plus, c_minus, A_plus, A_minus, B, C_plus, C_minus)
        return duplicatedArc  
              
              
    def print_B(self, **kwargs): 
        print_mat(self.B, **kwargs)


class NodeState(object):
    """ A node state is a state that allows the growing of a network. Examples
    are "clean", "dirty", "reaction time" etc. for a vat in chemical processing.
    A node with a state has a list of states it may be connected to (i.e. where
    a network arc might appear). The states that are not allowed to be connected
    are not connected (i.e. an arc may not go back in time)
    """
    def __init__(self, index, name, rel_x, rel_y):
        self.name       = name
        self.index      = index
        self.rel_x      = rel_x
        self.rel_y      = rel_y
 
    def get_name(self):
        return self.name    
  
    def get_index(self):
        return self.index
        
    def get_x(self):
        return self.rel_x

    def get_y(self):
        return self.rel_y

class TransNode(Node):
    ''' TransNodes are nodes that appear using causal links between other nodes
    It is the basis for the growing network algorithms'''
    
    def __init__(self, name, b, x_location, y_location, listOfStates):
        Node.__init__(self, name, b, x_location, y_location)
        # List of states tells what nodes the current node may spawn:
        self.listOfStates = listOfStates
        
    def get_listOfStates(self):
        return self.listOfStates

class Network(object):
    """ A network is the combination of directed arcs and nodes. The main 
        function of this class is to produce the matrices (network maps) used 
        to create the LP problem"""
        
    def __init__(self, name, nodeList, arcList, comNames):
        self.name      = name
        self.nodeList  = nodeList
        self.arcList   = arcList
        self.comNames  = comNames
        
        edgeNames  = map(attrgetter('name'), arcList)
        a = len(edgeNames)
        b = len(list(set(edgeNames)))
        if a > b:
            print 'ERROR: identical names in the edge list!'
            raise ValueError
    
    def get_name(self): 
        return self.name 
        
    def get_nodeList(self):        
        return self.nodeList

    def get_arcList(self):        
        return self.arcList

    def get_comNames(self):        
        return self.comNames
        
    def get_com_index(self, comName):       
        comNames = self.get_comNames()
        idx = comNames.index(comName)
        return idx
        
    def set_nodeList(self, nodeList):        
        self.nodeList = nodeList
  
    def get_arc(self, arcName): 
        arcList = self.get_arcList()  
        arc = next((x for x in arcList if x.name == arcName), None) # the edges' names are unique. So the first instance is the proper one.
        return arc

    def get_arc_index(self, arcNames): 
        arcList = self.get_arcList()  
        arcNameList  = map(attrgetter('name'), arcList)
        idx = []
        for arcName in arcNames:
            idx.append(arcNameList.index(arcName))
        return idx

    def set_arcList(self, arcList): 
        self.arcList = arcList

    def set_comNames(self, comNames): 
        self.comNames = comNames
        
    def get_numCommodity(self):
        return len(self.nodeList[0].get_b())

    def change_arc(self, new_arc, name_arc_to_be_changed):
        
        arcIndex = self.get_arc_index(name_arc_to_be_changed)[0]
        self.change_arc_idx( new_arc, arcIndex)

    def change_arc_idx(self, arc, arcIndex):
        arcList   = self.get_arcList()
        arcList[arcIndex] = arc
        self.arcList = arcList
        
    def printArcs(self):
        arcList = self.get_arcList()  
        print '================================================================'
        print 'Display of edge information:'
        print '================================================================'
        #print 'Name         Edge Idx.           out. Node             in. Node'
        print '%12s  %12s  %12s %12s' % ('Name', 'Edge Idx', 'out. Node', 'in. Node')
        print '----------------------------------------------------------------'
        edgeIdx = 0
        for arc in arcList:
            name    = arc.name
            outNode = arc.outgoingNode
            outNodeName = outNode.name
            inNode  = arc.incomingNode
            inNodeName  = inNode.name
            edgeIdx = edgeIdx + 1
            #print name + ' ' + outNodeName + ' ' + inNodeName + ' ' + str(edgeIdx)
            line_new = '%12s %12s  %12s  %12s' % (name, outNodeName, inNodeName, str(edgeIdx))
            print line_new
            
        print '================================================================'
        
    def printNodeNames(self):
        nodeList = self.get_nodeList()
        for name in nodeList:
            print name.get_name()
            
    def networkMap(self):
        
        size= len(self.nodeList)
        netMap = np.zeros(shape=(size,size))        
        
        '''Creates an ordered list of nodes for constructing the 
        network map:'''
        nodeList = self.get_nodeList()
        nodeNameList = map(attrgetter('name'), nodeList)

        arcList = self.get_arcList()
        outNodeList = map(attrgetter('outgoingNode'), arcList)
        inNodeList  = map(attrgetter('incomingNode'), arcList)
        
        for x in range(0,len(arcList)):
            onode = outNodeList[x]   
            inode = inNodeList[x]            
            index1 = nodeNameList.index(onode.get_name()) # line of matrix = outgoing node
            index2 = nodeNameList.index(inode.get_name()) # row of matrix = incoming node
            netMap[index1,index2] = x+1 # plus one, because index starts at 0, which is not the first arc!

#        print 'netMap'
#        print netMap
        return netMap
        
    def obj_fun(self):
        arcList = self.get_arcList()
        c_plus = map(attrgetter('c_plus'), arcList)
        c_plus = np.asarray(c_plus)
        c_plus = np.asmatrix(c_plus)
        c_plus = np.reshape(c_plus,(1,-1))

        c_mins = map(attrgetter('c_minus'), arcList)
        c_mins = np.asarray(c_mins)
        c_mins = np.asmatrix(c_mins)
        c_mins = np.reshape(c_mins,(1,-1))
        
        c      = np.concatenate((c_plus, c_mins),axis = 1)
        c      = c.tolist()
        c      = c[0]
        return c
        
    def A_plus(self):
        
        ncom     = self.get_numCommodity()
        size1     = ncom*len(self.get_nodeList())    # size of the total A matrices: number of arcs times the number of commodities
        size2     = ncom*len(self.get_arcList())    # size of the total A matrices: number of arcs times the number of commodities
        A_plus   = np.zeros(shape=(size1,size2))     # allocate zero matrices
        arcList  = self.get_arcList()
        nodeList = self.get_nodeList()        
        arcIdx = -1 # initialize the arc index
        for arc in arcList: # Sweep across all edges. Cannot use network maps because of possible multiple edges between 2 nodes.
            arcIdx = arcIdx +1 
            a_plus = arc.A_plus
            outNode= arc.get_outgoing()
            nodeIdx= nodeList.index(outNode)
            A_plus = placeMat(A_plus, a_plus, nodeIdx*ncom, arcIdx*ncom)
        
        return A_plus
            
    def A_minus(self):
        
        ncom     = self.get_numCommodity()
        size1     = ncom*len(self.get_nodeList())    # size of the total A matrices: number of arcs times the number of commodities
        size2     = ncom*len(self.get_arcList())    # size of the total A matrices: number of arcs times the number of commodities
        A_minus   = np.zeros(shape=(size1,size2))     # allocate zero matrices
        arcList  = self.get_arcList()
        nodeList = self.get_nodeList()        
        arcIdx = -1 # initialize the arc index
        for arc in arcList: # Sweep across all edges. Cannot use network maps because of possible multiple edges between 2 nodes.
            arcIdx = arcIdx +1 
            a_minus = arc.A_minus
            inNode= arc.get_incoming()
            nodeIdx= nodeList.index(inNode)
            A_minus = placeMat(A_minus, a_minus, nodeIdx*ncom, arcIdx*ncom)
        
        return A_minus

    def eq_constraints(self):

        # Preparatory functions:     
        arcList   = self.get_arcList()
        ncom      = self.get_numCommodity()
        size1     = ncom*len(self.get_arcList())    # size of the total A matrices: number of arcs times the number of commodities
        B         = np.zeros(shape=(size1,size1))     # allocate zero matrices
        B_mat_List= map(attrgetter('B'), arcList)        
        
        
        for x in range(len(self.get_arcList())):
            b   = B_mat_List[x]
            B   = placeMat(B, b, x*ncom, x*ncom)


        eyeFill       = np.identity(B.shape[0])
        B_eq          = np.concatenate((B,-eyeFill),axis=1) 
        b_eq = np.zeros(shape = (B.shape[0],1))

        return (B_eq, b_eq)

    def C_plus(self):

        # Preparatory functions:     
        arcList   = self.get_arcList()
        ncom      = self.get_numCommodity()
        size2     = ncom*len(self.get_arcList())    # size of the total A matrices: number of arcs times the number of commodities
        C_mat_List= map(attrgetter('C_plus'), arcList)        
        C_plus    = np.zeros(shape=(1,size2)) # creates a line of zeros to which the rest will be appended, then this is removed...
        
        for x in range(len(self.get_arcList())):
            C_element   = C_mat_List[x]
            C_elem_size = C_element.shape            
            C_line      = np.zeros(shape=(C_elem_size[0], size2))
            C_line      = placeMat(C_line, C_element, 0, x*ncom)
            C_plus = np.vstack((C_plus,C_line))
            
        C_plus = C_plus[~np.all(C_plus == 0, axis=1)]    # the zero lines are removed.        

        return C_plus        
 
    def C_minus(self):

        # Preparatory functions:     
        arcList   = self.get_arcList()
        ncom      = self.get_numCommodity()
        size2     = ncom*len(self.get_arcList())    # size of the total A matrices: number of arcs times the number of commodities
        C_mat_List= map(attrgetter('C_minus'), arcList)        
        C_minus    = np.zeros(shape=(1,size2)) # creates a line of zeros to which the rest will be appended, then this is removed...
        
        for x in range(len(self.get_arcList())):
            C_element   = C_mat_List[x]
            C_elem_size = C_element.shape            
            C_line      = np.zeros(shape=(C_elem_size[0], size2))
            C_line      = placeMat(C_line, C_element, 0, x*ncom)
            C_minus = np.vstack((C_minus,C_line))
        
        C_minus = C_minus[~np.all(C_minus == 0, axis=1)]    # the zero lines are removed. 

        return C_minus        
        
    def sub_eq_constraints(self):
        
        # Assembles the matrix of contraints for the LP problem
        a_plus = self.A_plus()
        a_minus= self.A_minus()
        c_plus = self.C_plus()
        c_minus= self.C_minus()
        
        zeroFill_plus = np.zeros(c_plus.shape)
        zeroFill_minus= np.zeros(c_minus.shape)
        a_matrix      = np.concatenate((a_plus, -a_minus),axis =1)
        Matrix        = a_matrix
        if (c_plus.shape[0] != 0): # if the c_plus matrix is not empty, then concatenate it:
            c_matrix_plus = np.concatenate((c_plus, zeroFill_plus),axis =1)
            Matrix        = np.concatenate((Matrix, c_matrix_plus),axis=0)
        if (c_minus.shape[0] != 0): # if the c_minus matrix is not empty, then concatenate it:
            c_matrix_mins = np.concatenate((zeroFill_minus, c_minus),axis =1)
            Matrix        = np.concatenate((Matrix, c_matrix_mins),axis=0)
        
        nodeList = self.get_nodeList()
        b = map(attrgetter('b'), nodeList)
        b = np.asmatrix(b)
        b = np.reshape(b,(1,-1))
        size = Matrix.shape
        B = np.zeros(shape=(size[0],1))
        b_sub_eq = placeMat(B, b.T, 0, 0)              
             
        return (Matrix, b_sub_eq) # the minus sign allows the inequality to be in the correct direction, e.g. the flow can be more than the minimun, but not less.
        
    def bounds(self):
    
        arcList   = self.get_arcList()

        max_flow_plus_List = map(attrgetter('maxFlow_plus'), arcList)
        min_flow_plus_List = map(attrgetter('minFlow_plus'), arcList)
        max_flow_mins_List = map(attrgetter('maxFlow_mins'), arcList)
        min_flow_mins_List = map(attrgetter('minFlow_mins'), arcList)
        x0_plus_bounds     = ()
        x0_mins_bounds     = ()        
        ncom      = self.get_numCommodity()
        
        for x in range(len(self.get_arcList())):

            min_flow_plus = min_flow_plus_List[x]
            max_flow_plus = max_flow_plus_List[x]            

            min_flow_mins = min_flow_mins_List[x]
            max_flow_mins = max_flow_mins_List[x]

            boundsForACommodity_plus = ()
            boundsForACommodity_mins = ()
                        
            for y in range(ncom):            
                
                boundsForACommodity_plus = boundsForACommodity_plus + ((min_flow_plus[y], max_flow_plus[y]),)
                boundsForACommodity_mins = boundsForACommodity_mins + ((min_flow_mins[y], max_flow_mins[y]),)
                
            x0_plus_bounds = x0_plus_bounds + boundsForACommodity_plus
            x0_mins_bounds = x0_mins_bounds + boundsForACommodity_mins

        x0_bounds = x0_plus_bounds + x0_mins_bounds
        return x0_bounds
        
    def displaySol(self,x,commodity_names):

        # Classify the results into a table
        x_plus  = x[0:x.shape[0]/2]
        x_minus = x[x.shape[0]/2:x.shape[0]] # x_minus are the incoming flows...Could be used...
        
        nCom    = self.get_numCommodity()

        table_plus  = np.zeros(shape=(nCom,len(self.get_arcList())))  
        print 'table plus'
        print table_plus
        table_minus = np.zeros(shape=(nCom,len(self.get_arcList())))          
        for x in range(len(self.get_arcList())):
            subMat = x_plus[x*nCom:(x+1)*nCom]
            subMat = np.asmatrix(subMat)
            subMat = subMat.transpose()
            print 'x_plus'
            print x_plus
            print 'submat'
            print subMat
            print 'nCom'
            print nCom
            print 'x'
            print x
            print 'table plus'
            print table_plus
            table_plus = placeMat(table_plus, subMat, 0, x)
            
            subMat = x_minus[x*nCom:(x+1)*nCom]
            subMat = np.asmatrix(subMat)
            subMat = subMat.transpose()
            table_minus = placeMat(table_minus, subMat, 0, x)
            
        # Get names of commodities for display
        # this is included in the arguments of the function...
                
        
        # make the plot            
        N = nCom

        N = 5
        ind = np.arange(N)  # the x locations for the groups
        width = 0.35       # the width of the bars
        newFig = plt.figure()
        cmap = plt.get_cmap('gnuplot')
        colorses = [cmap(i) for i in np.linspace(0, 1, nCom)]   
        xaxisrange = range(1,len(self.get_arcList())+1)        
        
        for i in range(nCom):

            plt.plot(xaxisrange, table_plus[i,:] ,linewidth=2 ,label=commodity_names[i])


        arcList   = self.get_arcList()
        out_List  = map(attrgetter('outgoingNode'), arcList)
        a         = map(attrgetter('name'), out_List)        
        in_List   = map(attrgetter('incomingNode'), arcList)
        b         = map(attrgetter('name'), in_List)
        my_xticks = ["{}{}".format(b_, a_) for a_, b_ in zip(a, b)]
        print my_xticks
        plt.xticks(xaxisrange, my_xticks)
        
        plt.legend(loc='best')
        plt.xlim(0, len(self.get_arcList()) + 1)
        plt.ylim(-1, table_plus.max()+2)
        plt.show()


        return table_plus, table_minus
    
    def findTips(self):
        '''Function that allows to find the extremeties of a network. Returns
        the index of the tip nodes.'''
        networkMap = self.networkMap()
        listTipNodes = np.where(~networkMap.any(axis=1))[0]
        return listTipNodes
        
    def spawnNodes(self):
        '''These must expand the network according to algorithms/heuristics
        according to the problem at hand'''
        
        '''In this case, we pick each node that is a tip node, and add 2 more
        that follow it such that the network branches.'''

        nodeList     = self.get_nodeList()
        listTipNodes = self.findTips()
        
        for nodeIndex in listTipNodes:
            newNode1 = nodeList[nodeIndex].duplicateNode()
            newNode2 = nodeList[nodeIndex].duplicateNode()
            
            # Test to append the new nodes:
            x1 = newNode1.getX_location()
            x2 = newNode2.getX_location()
            y1 = newNode1.getY_location()
            y2 = newNode2.getY_location()
            
            newNode1.setX_location(x1+1)
            newNode1.setY_location(y1+0.25)
            newNode2.setX_location(x2+1)
            newNode2.setY_location(y2-0.25)
    
            nodeList.append(newNode1)
            nodeList.append(newNode2)
            
            self.set_nodeList(nodeList)
        
        
    def spawnArcs(self):  
        '''These must expand the network according to algorithms/heuristics
        according to the problem at hand'''
       
        
        
    def spawnNetwork(self, gen):

        nodeList     = self.get_nodeList()
        arcList = self.get_arcList()
        listTipNodes = self.findTips()
        size1 = len(arcList)
        size2 = len(nodeList)
        iterator = size2
        
        for nodeIndex in listTipNodes:
            newNode1 = nodeList[nodeIndex].duplicateNode()
            newNode2 = nodeList[nodeIndex].duplicateNode()
            newArc1  = arcList[-1].duplicateArc()
            newArc2  = arcList[-1].duplicateArc()
            
            # Test to append the new nodes:
            x1 = newNode1.getX_location()
            x2 = newNode2.getX_location()
            y1 = newNode1.getY_location()
            y2 = newNode2.getY_location()
            
            newNode1.setX_location(x1+1)
            newNode1.setY_location(y1+(2**-gen))
            iterator = iterator + 1
            newNode1.set_name(str(iterator))
            newNode2.setX_location(x2+1)
            newNode2.setY_location(y2-(2**-gen))
            iterator = iterator + 1
            newNode2.set_name(str(iterator))

            newArc1.set_outgoing(nodeList[nodeIndex])
            newArc1.set_incoming(newNode1)
            newArc2.set_outgoing(nodeList[nodeIndex])
            newArc2.set_incoming(newNode2)
            
            nodeList.append(newNode1)
            nodeList.append(newNode2)
            arcList.append(newArc1)
            arcList.append(newArc2)
            
            self.set_nodeList(nodeList)
            self.set_arcList(arcList)
            
                    
        # Test to append the new edges:  
        print '=========================================='            
        self.printNodeNames()

def B_assembly(allCommodityNames, comNames_v, comNames_h, B_sub, **kwargs):
    # Creates a default B_sup matrix if none is given in input:
    B_sup = np.identity(len(allCommodityNames))
    for key, value in kwargs.items():
        if (key == 'B_sup'):
            B_sup= kwargs[key]
            if B_sup.shape[0] != len(allCommodityNames):
                print 'The input B_sup matrix is incorrect!'
    if B_sub.shape[0] != len(comNames_v):
                print 'The input B_sub matrix is incorrect! (too many or too little sub commodity list)' 
    if B_sub.shape[1] != len(comNames_h):
                print 'The input B_sub matrix is incorrect! (too many or too little sub commodity list)'
    # Get the name locations of the elements in the small matrix to the large matrix:
    nameLocB_sub_v = range(len(comNames_v))
    nameLocB_sup_v = findName(allCommodityNames, comNames_v)

    nameLocB_sub_h = range(len(comNames_h))
    nameLocB_sup_h = findName(allCommodityNames, comNames_h)
    
    # Double for loop for placing the elements:
    for lineIdx_b in nameLocB_sub_h: # iterates over the lines:
        for ColIdx_b in nameLocB_sub_v:
            
            # Get the indexes that map from one matrix to another:
            lineIdx_p = nameLocB_sup_h[lineIdx_b]
#            print 'line index sup:'
#            print lineIdx_p
            ColIdx_p  = nameLocB_sup_v[ColIdx_b]
#            print 'col index sup:'
#            print ColIdx_p
#            print 'Element in sup matrix:'
#            print B_sup[lineIdx_p, ColIdx_p]
#            print '================================='
#            print 'line index sub:'
#            print lineIdx_b
#            print 'Col index sub:'
#            print ColIdx_b
#            print 'B_sub'
#            print B_sub
#            print 'B_sub[lineIdx_b, ColIdx_b]'
#            print B_sub[ColIdx_b, lineIdx_b]
            # Allocate the proper elements:
            B_sup[ColIdx_p, lineIdx_p] = B_sub[ColIdx_b, lineIdx_b]
            
    return B_sup
    
def B_prop(allCommodityNames, propulsionName, phi, **kwargs):
    # Creates a default B_sup matrix if none is given in input:
    B_prop = np.identity(len(allCommodityNames))          
    for key, value in kwargs.items():
        if (key == 'B_prop'):
            B_prop= kwargs[key]
    # Find which line we must replace for the propellant:
    p_index = findName(allCommodityNames, [propulsionName])[0] # need to convert the proplusion name to a string array for the 
    # Get the name locations of the elements in the small matrix to the large matrix:
    subMat = B_prop[p_index, :] -np.matrix([phi]*len(allCommodityNames))
    B_prop = placeMat(B_prop, subMat, p_index, 0)
            
    return B_prop