#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Authors: Benjamin Renoust (github.com/renoust)
   Date: 2018/02/13
   Description: Loads a Detangler JSON format graph and compute unweighted entanglement analysis with Py3Plex
"""


import networkx as nx
from py3plex.core import multinet
from py3plex.core import random_generators

from scipy.sparse.csgraph import csgraph_from_dense, connected_components
from scipy import spatial
import numpy as np
import itertools
import math


# Build the R and C matrix
def build_occurrence_matrix(network):

    multiedges = network.get_edges()
    layers = []
    edge_list = []
    for e in multiedges:
        (n1,l1),(n2,l2) = e
        if l1 == l2:
            if l1 not in layers:
                layers += [l1]
            edge_list.append([n1,n2,l1])

    edge_list = sorted(edge_list, key=lambda x: [x[0],x[1]])

    nb_layers = len(layers)
    r_matrix = np.zeros((nb_layers,nb_layers))
    
    def count_overlap(overlap):
        prev_layers = []
        for e in overlap:
            layer = e[2]
            layer_index = layers.index(layer)
            r_matrix[layer_index,layer_index] += 1.0
            for l in prev_layers:
                r_matrix[l,layer_index] += 1.0
                r_matrix[layer_index,l] += 1.0

            prev_layers.append(layer_index)
            

    current_edge = None
    flat_pairs = 0.0
    overlap = []

    for e in edge_list:
        node_pair = [e[0],e[1]]
        if current_edge != node_pair:
            flat_pairs += 1.0
            current_edge = node_pair            
            count_overlap(overlap)
            overlap = []
        overlap.append(e)
    count_overlap(overlap)
    flat_pairs += 1

    c_matrix = r_matrix.copy()

    for i in range(nb_layers):
        c_matrix[i,i] /= flat_pairs

    for i,j in itertools.combinations(range(nb_layers),2):
        c_matrix[i,j] /= r_matrix[j][j] 
        c_matrix[j,i] /= r_matrix[i][i]

    return c_matrix, layers


# proceeds with block decomposition
def compute_blocks(c_matrix):
    c_sparse = csgraph_from_dense(c_matrix)
    nb_components, labels = connected_components(c_sparse, directed=False, return_labels=True)
    
    v2i = {}
    for i,v in enumerate(labels):
        v2i[v] = v2i.get(v, []) + [i]


    blocks = []
    indices = []
    for v,i in v2i.iteritems():
        indices.append(i)
        blocks.append(c_matrix[np.ix_(i,i)])

    return indices, blocks


# computes entanglement for one block
def compute_entanglement(block_matrix):
    eigenvals, eigenvects = np.linalg.eig(block_matrix)
    max_eigenval = max(eigenvals.real)
    index_first_eigenvect = np.argmax(eigenvals)

    nb_layers = len(block_matrix)
    # normalizes the max eigenval to dimensions
    entanglement_intensity = max_eigenval/nb_layers

    gamma_layers = []
    for i in range(nb_layers):
        gamma_layers.append(abs(eigenvects[i][index_first_eigenvect].real)) #because of approx.

    # computes entanglement homogeneity, cosine distance with the [1...1] vector
    entanglement_homogeneity = 1 - spatial.distance.cosine(gamma_layers, np.ones(nb_layers))
    # normalizes within the top right quadrant (sorts of flatten the [0-1] value distribution)
    normalized_entanglement_homogeneity = 1 - math.acos(entanglement_homogeneity)/(math.pi/2)

    return [entanglement_intensity, entanglement_homogeneity, normalized_entanglement_homogeneity], gamma_layers


def compute_entanglement_analysis(network):

    matrix, layers = build_occurrence_matrix(network)
    indices, blocks = compute_blocks(matrix)


    analysis = []
    for i,b in enumerate(blocks):
        layer_labels = [layers[x] for x in indices[i]]
        [I,H,H_norm],gamma=compute_entanglement(b)
        block_analysis = { 
            'Entanglement intensity':I,
            'Layer entanglement': {layer_labels[x]:gamma[x] for x in range(len(gamma))},
            'Entanglement homogeneity': H,
            'Normalized homogeneity': H_norm
        }
        analysis.append(block_analysis)
    return analysis


#loads a detangler json file into Py3Plex structure
def load_detangler_json(file_path):
    import json
    network = multinet.multi_layer_network()
    with open(file_path) as f:
        graph = json.load(f)

    id2n = {}
    for n in graph[u'nodes']:
        id2n[n[u'id']] = n
        layers = n[u'descriptors'].split(';')
        node = n[u'label']
        for l in layers:
            node_dict = {'source':node, 'type':l}
            network.add_nodes(node_dict)
        for c in itertools.combinations(layers, 2):
            network.add_edges({'source':node, 'target':node, 'source_type':c[0], 'target_type':c[1]},input_type='dict')


    for e in graph[u'links']:
        s = id2n[e[u'source']][u'label']
        t = id2n[e[u'target']][u'label']
        layers = e[u'descriptors'].split(';')
        for l in layers:
            network.add_edges({'source':s, 'target':t, 'type':l, 'source_type':l, 'target_type':l}, input_type='dict')

    return network


if __name__ == '__main__':
    #net = load_detangler_json('./data/cluster1.json')
    net = random_generators.random_multilayer_ER(500,8,0.05,directed=False)


    analysis = compute_entanglement_analysis(net)

    print ("%d connected components of layers"%len(analysis))
    for i,b in enumerate(analysis):
        print ('--- block %d'%i)
        layer_labels = b['Layer entanglement'].keys()
        print ('Covering layers: %s'%layer_labels)

        print ('Entanglement intensity: %f'%b['Entanglement intensity'])
        print ('Layer entanglement: %s'%b['Layer entanglement'])
        print ('Entanglement homogeneity: %f'%b['Entanglement homogeneity'])
        print ('Normalized homogeneity: %f'%b['Normalized homogeneity'])
