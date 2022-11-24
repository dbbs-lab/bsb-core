from ..device import NeuronDevice


class CurrentClamp(NeuronDevice):
    def implement(self, target, location):
        pattern = self.get_pattern(target, location)
        location.section.iclamp(**pattern)

    def validate_specifics(self):
        pass

    def create_patterns(self):
        return self.parameters

    def get_pattern(self, target, cell=None, section=None, synapse=None):
        return self.get_patterns()
