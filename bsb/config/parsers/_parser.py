import abc


class Parser(abc.ABC):
    @abc.abstractmethod
    def parse(self, content, path=None):
        pass
