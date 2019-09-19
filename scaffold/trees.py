from sklearn.neighbors import KDTree

class TreeCollection:
    trees = {}

    def __init__(self, scaffold, name):
        self.scaffold = scaffold
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
        self.trees[name] = self.scaffold.output_formatter.load_tree(self.name, name)
        return self.trees[name]

    def get_tree(self, name):
        if not name in self.trees:
            self.load_tree(name)
        return self.trees[name]
