from ..adapter import NeuronDevice


class VoltageClamp(NeuronDevice):
    def implement(self, target, location):
        pattern = self.get_pattern(target, location)
        location.section.vclamp(**pattern)

    def validate_specifics(self):
        pass

    def create_patterns(self):
        return self.parameters

    def get_pattern(self, target, cell=None, section=None, synapse=None):
        return self.get_patterns()
