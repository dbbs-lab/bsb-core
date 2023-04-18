import abc


class Parser(abc.ABC):
    @abc.abstractmethod
    def parse(self, content, path=None):
        pass

    @abc.abstractmethod
    def generate(self, tree, pretty=False):
        pass
