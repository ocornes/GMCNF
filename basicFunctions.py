import numpy as np
import matplotlib.pyplot as plt


def placeMat(supMat, subMat, firstC, secondC):

    sizeSup = supMat.shape
    sizeSub = subMat.shape
    
    endIndexLine = firstC + sizeSub[0]
    endIndexCol  = secondC+ sizeSub[1]
    
    if endIndexLine > sizeSup[0]:
        print 'sub-matrix too large or not correctly located (too many lines)'

    if endIndexCol > sizeSup[1]:
        print 'sub-matrix too large or not correctly located (too many columns)'
    
    supMat[firstC:endIndexLine, secondC:endIndexCol] = subMat 
    return supMat
    
def findName(allComs, coms):
    nameLocations = []
    for com in coms:
        nameLocations.append(allComs.index(com))
    return nameLocations
    
def print_mat(matrix, **kwargs):
    # create the if condition if the *kwargs do not contain a commodity name (defaults to first com):
    pltName = 'plot_name'
    
    h_labels = range(matrix.shape[1])
    v_labels = range(matrix.shape[0])
    
    for key, value in kwargs.items():
        if (key == 'pltName'):
            pltName= kwargs[key]  

    for key, value in kwargs.items():
        if (key == 'h_labels'):
            h_labels = kwargs[key]  

    for key, value in kwargs.items():
        if (key == 'v_labels'):
            v_labels = kwargs[key]   

    fig, ax = plt.subplots()
    # Rounding matrix values to make the intelligeable on graph:
    matrix = matrix.round(decimals = 2)
    ax.matshow(matrix, cmap=plt.cm.Blues)
    
    plt.xticks(range(len(h_labels)), h_labels)
    plt.yticks(range(len(v_labels)), v_labels)    
    for i in xrange(len(h_labels)):
        for j in xrange(len(v_labels)):
            c = matrix[j,i]
            ax.text(i, j, str(c), va='center', ha='center')
    plt.title(pltName)
    plt.savefig(pltName + '.png', format='png')  
    plt.figure().canvas.draw()