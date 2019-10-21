from sklearn.neighbors import KDTree
import re, abc, numpy as np

TREE_NAME_REGEX = re.compile(r'^[^\:\+\(\)]+$')
def is_valid_tree_name(name):
    '''
        Validate whether a given string is fit to be the name of a tree in a TreeCollection.
        Must not contain any plus signs, parentheses or colons.
    '''
    # re.match() returns a MatchObject with a boolean value of True, or None
    return not not TREE_NAME_REGEX.match(name)

class TreeCollection:
    '''
        Keeps track of a collection of KDTrees in cooperation with a TreeHandler.
    '''
    trees = {}

    def __init__(self, name, handler):
        self.handler = handler
        self.name = name

    def list_trees(self):
        return self.handler.list_trees(self.name)

    def create_tree(self, name, nodes):
        if not is_valid_tree_name(name):
            raise Exception("Tree names must not contain any : or + signs.")
        if(len(nodes) == 0):
            return
        self.add_tree(name, KDTree(nodes))

    def add_tree(self, name, tree):
        self.trees[name] = tree

    def items(self):
        return self.trees.items()

    def values(self):
        return self.trees.values()

    def keys(self):
        return self.trees.keys()

    def load_tree(self, name):
        try:
            self.trees[name] = self.handler.load_tree(self.name, name)
            return self.trees[name]
        except Exception:
            self.trees[name] = None
            return None

    def get_tree(self, name):
        if not name in self.trees:
            self.load_tree(name)
        return self.trees[name]

    def get_planar_tree(self, name, plane='xyz'):
        if plane == 'xyz':
            return self.get_tree(name)
        planar_name = '{}:{}'.format(plane, name)
        if not planar_name in self.trees:
            self.load_tree(planar_name)
        if self.trees[planar_name] is None:
            self.make_planar_tree(name, plane)
        return self.trees[planar_name]

    def make_planar_tree(self, name, plane):
        full_tree = self.get_tree(name)
        if full_tree is None:
            raise Exception("Cannot make planar tree from unknown tree '{}'".format(name))
        dimensions = ["x", "y", "z"]
        selected_dimensions = [e in plane for e in dimensions]
        planar_tree = KDTree(np.array(full_tree.get_arrays()[0])[:,selected_dimensions])
        self.trees['{}:{}'.format(plane, name)] = planar_tree
        self.save()
        return planar_tree

    def save(self):
        self.handler.store_tree_collections([self])
