import numpy as np
import numpy.random as npr



class Node(object):
    
    '''
    Object representing a node of the tree.
    '''
    
    def __init__(self,value=0, parent=None, is_root = False, leaf_idx = -1):

	self.value = value # vlaue of the Node
	self.leaf_idx = leaf_idx # [0,1,...,M] for leaves

	self.parent = parent # (a Node)
	self.left = None # left child (a Node)
	self.right = None # right child (a Node)
	
class Tree(object):
    
    '''
    Object representing the tree
    Main functions that will be use are:
    1. initialize
    2. sample(gamma)
    3. update(i,f)
    See the Tree box of the paper for their meaning
    '''

    def __init__(self, m):
    
	self.root = Node(is_root = True)
	self.current_key = 1
	self.m = m
	self.height = int(np.ceil(np.log2(self.m))) # defining the height as in the paper

    def build_full_tree(self):
	
	nodes = [self.root]
	
	for i in range(self.height):
	
	    if nodes[0].left is None: # means it's root
		new_layer = nodes
	    else:
		new_layer = [child for n in new_layer for child in [n.left, n.right] ]

	    counter = 0
	    
	    for node in new_layer:
		
		if (i == self.height-1) and (counter <= self.m -1):		    
		    value = 1
		    counter += 1
		else:
		    value = 0
		
		node.left = Node(value = value, parent = node)
		
		
		if (i == self.height-1) and (counter <= self.m -1):		    
		    value = 1
		    counter += 1
		else:
		    value = 0
		
		node.right = Node(value = value, parent = node)
		
		self.current_key += 1
	
	leaves = [child for n in new_layer for child in [n.left, n.right] ]
	return leaves
		
    def initialize_nodes(self, leaves):
	
	parents = [l.parent for l in leaves[::2]]
	
	i = 0
	
	while parents[0] is not None:
	    
	    i += 1
	    
	    for p in parents:
		p.value = p.left.value + p.right.value
	    
	    
	    parents = [l.parent for l in parents[::2]]
	
    def get_leaves_as_list(self):
	return [l.value for l in self.leaves]
	
    # Three methods necessary to use the Tree described in the paper.
	
    def initialize(self):
	
	self.leaves = self.build_full_tree()
	self.initialize_nodes(self.leaves)
	
	# setting leaf_idx valus, needed to return i in sample
	for l,idx in zip(self.leaves,range(self.m)):
	    l.leaf_idx = idx
	
    def sample(self,gamma):
	
	b = npr.binomial(1,1-gamma)
	
	if b == 0:
	    i = npr.randint(self.m)
	    return (i, (gamma /self.m) + (1 - gamma)*self.leaves[i].value)
	
	
	else:
	    node = self.root
	    while node.left is not None: #it's full, left is completely arbitrary here
		node = node.left if node.left.value > node.right.value else node.right 
	    return (node.leaf_idx, (gamma /self.m) + (1 - gamma)*node.value)
	
    def update(self, i, f):
	
	v = self.leaves[i].value
	delta = f*v - v
	
	node = self.leaves[i]
	
	while node.parent is not None:
	    print 'Delta', delta
	    print 'Previous value', node.value 
	    print 'New value', node.value + delta
	    node.value += delta
	    node = node.parent 
	    
	  	
if __name__ == "__main__":
    
    m = 10000
    
    tree = Tree(m)
    tree.initialize()
    
    (i,val) = tree.sample(.5)
    print (i,val)
    tree.update(i,f=0)
    


		    
		    
		
			
		
		
		
		
