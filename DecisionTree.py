# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:18:11 2018

@author: Daniel
"""

import numpy as np
import pandas as pd
import math
from collections import Counter
from sklearn import datasets
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold, cross_val_score


# A class representing a node in the decision tree
class Node:
    def __init__(self):
        self.attribute = -1   # This is essentially the question
        
        self.available_data = []
        
#        self.features = {}
        
        self.isLeaf = False
        # The tree is represented in the child nodes
        # The keys are the branches and the values are the nodes
        self.child_nodes = {}  
        
        self.avg_entropy = 0
    
class DecisionTreeModel:
    def __init__(self, data):
        self.root_node = Node() # This node is the root node, and holds the whole tree within it
        self.root_node.available_data = data
        self.build_tree(self.root_node)
    
    def calc_entropy(self, targets_list):
        entropy = 0.0
        # Get a map of how many of each target is contained in classes
        counts = Counter(targets_list)
        total_num = len(targets_list)
        
        for target in counts:
            entropy = entropy + (-1 * (counts[target] / total_num) * math.log2(counts[target] / total_num))
        
        return entropy
    
    
    def pick_node(self, parent_node):
        # Map the calculated entropies of each node to the node for each attribute
        avg_node_entropies = {}
               
        # Calculate the entropy of each column (attribute) except the target column
        for col in range(len(parent_node.available_data[0]) - 1):
            # The new node to be created
            node = Node()            

            # A list to contain all the values in the column
            col_entries = []
            
            # Figure out what the branches are within this attribute
            for row in parent_node.available_data:
                # Turn the column into a list for better usability
                col_entries.append(row[col])
                   
            # A map for each branch and how many occurences of that branch in the column
            counts = Counter(col_entries)
            
            # Get the different branches in the column
            branches = counts.keys()
            
            # Hold the entropies of each branch
            branch_entropies = []
            
            # Create a list of targets for each branch 
            for branch in branches:
                # Make a child node for each branch where the data associated with the child can be placed
                parent_node.child_nodes[branch] = Node()
                
                targets_of_branch = []
                
                # Make a list of the targets associated with the branch in the column
                for r in parent_node.available_data:
                    if r[col] == branch:
                        targets_of_branch.append(r[len(r) - 1])
                        
                        # Fill the node with its available data for later use if this node is chosen
                        parent_node.child_nodes[branch].available_data.append(r)
                        
                # Store the entropy of the branch to calculate the avg entropy later
                entropy = self.calc_entropy(targets_of_branch)
                print("Branch Entropy: {}".format(entropy))
                branch_entropies.append(entropy)

            avg_entropy = 0            
            
            # Calculate the weighted average entropy for this node
            e = 0
            for br in branches:
                avg_entropy += (counts[br] / len(parent_node.available_data)) * branch_entropies[e]
                e += 1
            
            print("Weighted Avg E: {}".format(avg_entropy))
            
            # The attribute of this node is the column name
            node.attribute = col
            avg_node_entropies[avg_entropy] = node
            
        # Return the node with the lowest entropy (highest information gain)
        return avg_node_entropies[min(avg_node_entropies.keys())]
    
    def build_tree(self, parent_node):
        #ID3 Algorithm
        
        # If all examples have the same label return a leaf with that label
        # This is done by making a list of all the targets and getting the counts. If there's only one
        # thing to count, they are all the same.
        if len(Counter([value[len(value)-1] for value in parent_node.available_data]).keys()) == 1:    
            parent_node.isLeaf = True
            parent_node.attribute = parent_node.available_data[0][len(parent_node.available_data[0]) - 1]                
            return; 
        elif len(Counter([value[len(value)-1] for value in parent_node.available_data]).keys()) == 1:
            # Else if there are no features left to test, return a leaf with the most common label
            # NOTE: This case is not implemented. This is just a hard-coded value for now.
            parent_node.isLeaf = True
            parent_node.attribute = parent_node.available_data[0][len(parent_node.available_data[0]) - 1]
        else:
            # Consider each available feature
            # Choose the one that maximizes information gain
            # Create a new node for that feature
            node = self.pick_node(parent_node)
            
            # For each possible value of the feature
            #    Create a branch for this value
            #    Create a subset of the examples for each branch
            #    Recursively call the function to create a new node at that branch
            for value in node.child_nodes.keys():
                build_tree(node)
                
                
            
    
    def predict(self, data_test):
        predictions = []
        for i in data_test:
            predictions.append(0)
        return predictions
        
    
class DecisionTreeClassifier:
#    def __init__(self):
        
    def fit(self, data):#_train, targets_train):
        model = DecisionTreeModel(data)#_train, targets_train)
        return model
    
    
def bin_iris_data(data, targets):
    print("Discretizing the iris dataset...")
    
    formatted_data = []
    
    # Take in the iris dataset and discretize it
    for i in range(len(data)):
        
        if data[i][0] < 6.1:
            data[i][0] = 0.0 # Small petal length
        elif data[i][0] >= 6.1:
            data[i][0] = 1.0 # Big petal length
        
        if data[i][1] < 3.2:
            data[i][1] = 0.0 # Small seaple length
        elif data[i][1] >= 3.2:
            data[i][1] = 1.0 # Big seaple length
            
        if data[i][2] < 3.95:
            data[i][2] = 0.0 # Small seaple width  
        elif data[i][2] >= 3.95:
            data[i][2] = 1.0 # Big seaple width
        
        if data[i][3] < 1.3:
            data[i][3] = 0.0 # Small petal width 
        elif data[i][3] >=1.3:
            data[i][3] = 1.0 # Big petal width
        
        # Associate the targets with it's appropriate data
        formatted_data.append(np.append(data[i], targets[i]))
        
    return formatted_data

def main():
    
    # Load the iris dataset
    print("Loading the data...")
    dataset = datasets.load_iris()
    data = dataset.data
    targets = dataset.target
    
    data = bin_iris_data(data, targets)
    
    # Get the predicted targets
    print("Testing...")
          
    # Obtain a classifier
    classifier = DecisionTreeClassifier()
    model = classifier.fit(data) # The data and the targets are together

    # Obtain the accuracy of the classifier
    # Configure k-fold validation
#    k_fold = KFold(len(np.ravel(targets)), n_folds=10, shuffle=True, random_state=18)
#    accuracy_score = cross_val_score(classifier, data, np.ravel(targets), cv=k_fold, n_jobs=1).mean()
    
    # Display the accuracy results
#    print("Total Accuracy: {:.2f}%".format(accuracy_score * 100.0))

main()