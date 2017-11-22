import numpy as np
import numpy.random as npr



class Node(object):
    
    def __init__(self,key=0,value=0, distance_from_root = 0, parent=None, is_root = False, leaf_idx = -1):

	self.key = key
	self.value = value
	self.distance_from_root = distance_from_root
	self.parent = parent
	self.left = None
	self.right = None
	self.leaf_idx = leaf_idx # [0,1,...,M] for leaves

class Tree(object):

    def __init__(self, m):
    
	self.root = Node(is_root = True)
	self.current_key = 1
	self.distance_from_root = 0
	self.m = m
	self.height = int(np.ceil(np.log2(self.m)))

    def build_full_tree(self):
	
	nodes = [self.root]
	
	for i in range(self.height):
	
	    if nodes[0].left is None: # means it's root
		new_layer = nodes
	    else:
		new_layer = [child for n in new_layer for child in [n.left, n.right] ]

	    counter = 0
	    leaf_idx = -1
	    
	    for node in new_layer:
		
		if (i == self.height-1) and (counter <= self.m -1):		    
		    value = 1
		    counter += 1
		else:
		    value = 0
		
		node.left = Node(key = self.current_key, value = value, distance_from_root = i, parent = node, leaf_idx = leaf_idx)
		
		self.current_key += 1
		
		if (i == self.height-1) and (counter <= self.m -1):		    
		    value = 1
		    counter += 1
		else:
		    value = 0
		
		node.right = Node(key = self.current_key, value = value, distance_from_root = i, parent = node, leaf_idx = leaf_idx)
		
		self.current_key += 1
	
	leaves = [child for n in new_layer for child in [n.left, n.right] ]
	return leaves
		
    def initialize_nodes(self, leaves):
	
	print [l.value for l in leaves]
	parents = [l.parent for l in leaves[::2]]
	
	i = 0
	
	while parents[0] is not None:
	    
	    i += 1
	    
	    for p in parents:
		p.value = p.left.value + p.right.value
	    
	    
	    print [p.value for p in parents]
	    parents = [l.parent for l in parents[::2]]
	
    # Three methods necessary to use the Tree described in the paper.
	
    def initialize(self):
	
	self.leaves = self.build_full_tree()
	self.initialize_nodes(self.leaves)
	
    def sample(self,gamma):
	
	b = npr.binomial(1,1-gamma)
	
	if b == 0:
	    i = npr.randint(len(self.leaves))
	    return (i,gamma / (self.m + (1 - gamma)*self.leaves[i].value))
	
	
	else:
	    node = tree.root
	    while node.left is not None: #it's full, left is completely arbitrary here
		node = node.left if node.left.value > node.right.value else node.right 
	    return (node.leaf_idx,gamma / (self.m + (1 - gamma)*node.value))
	
    def update(self, i, f):
	
	v = self.leaves[i].value
	delta = f*v - v
	
	node = self.leaves[i]
	
	while node.parent is not None:
	    
	    node.value += delta
	    node = node.parent 
	    
	  	
if __name__ == "__main__":
    
    m = 200
    
    tree = Tree(m)
    tree.initialize()
    
    (i,val) = tree.sample(.5)
    print (i,val)
    tree.update(i,f=0)


		    
		    
		
			
		
		
		
		
