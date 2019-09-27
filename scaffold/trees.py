from sklearn.neighbors import KDTree
import re, abc

TREE_NAME_REGEX = re.compile(r'^[^\:\+]+$')
def validate_tree_name(name):
    '''
        Validate whether a given string is fit to be the name of a tree in a TreeCollection.
        Must not contain any plus signs or colons.
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

    def create_tree(self, name, nodes):
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
        self.trees[name] = handler.load_tree(self.name, name)
        return self.trees[name]

    def get_tree(self, name):
        if not name in self.trees:
            self.load_tree(name)
        return self.trees[name]
