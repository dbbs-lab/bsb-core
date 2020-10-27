from . import BaseCommand


class BsbCompile(BaseCommand, name="compile"):
    def handler(self, namespace):
        pass


class BsbSimulate(BaseCommand, name="simulate"):
    def handler(self, namespace):
        pass


# Command plugin
def compile():
    return BsbCompile


# Command plugin
def simulate():
    return BsbSimulate
