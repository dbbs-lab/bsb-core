from bsb import config
from bsb.config import types
from bsb.simulation.cell import CellModel
from bsb.exceptions import AttributeMissingError, ConfigurationError


def _merge_params(node, models, model_key, model_params):
    # Take the default set of parameters
    merger = node.parameters.copy()
    # Merge the model specific parameters
    merger.update(model_params)
    # Place the merged dict back in the catch_all dictionary under the model key
    models[model_key] = merger


def _unmerge_params(node, key):
    to_remove = set(node.parameters)
    return {k: v for k, v in node.model_parameters[key].items() if k not in to_remove}


class MapsScaffoldIdentifiers:
    def reset_identifiers(self):
        self.nest_identifiers = []
        self.scaffold_identifiers = []
        self.scaffold_to_nest_map = {}

    def _build_identifier_map(self):
        self.scaffold_to_nest_map = dict(
            zip(self.scaffold_identifiers, self.nest_identifiers)
        )

    def get_nest_ids(self, ids):
        return [self.scaffold_to_nest_map[id] for id in ids]


@config.node
class NestCell(CellModel, MapsScaffoldIdentifiers):
    neuron_model = config.attr(type=str)
    relay = config.attr(default=False)
    parameters = config.dict(type=types.any_())
    model_parameters = config.catch_all(
        type=dict, catch=_merge_params, tree_cb=_unmerge_params
    )

    def boot(self):
        self.receptor_specifications = {}
        self.neuron_model = self.neuron_model or self.simulation.default_neuron_model
        self.reset()
        if self.relay:
            self.neuron_model = "parrot_neuron"

        # Each cell model is loaded with a set of parameters for each nest model that can
        # be used for it. We iterate over them and take out the `receptors` parameter to
        # obtain this model's `receptor_specifications`
        for model_name, model_parameters in self.model_parameters.items():
            if "receptors" in model_parameters:
                # Transfer the receptor specifications
                self.receptor_specifications[model_name] = model_parameters["receptors"]
                del model_parameters["receptors"]

    def validate(self):
        if not self.relay and not hasattr(self, "parameters"):
            raise AttributeMissingError(
                "Required attribute 'parameters' missing from '{}'".format(
                    self.get_config_node()
                )
            )

    def reset(self):
        self.reset_identifiers()

    def get_parameters(self):
        # Get the default synapse parameters
        params = self.parameters.copy()
        # Raise an exception if the requested model is not configured.
        if not hasattr(self, self.neuron_model):
            raise ConfigurationError(
                "Missing parameters for '{}' model in '{}'".format(
                    self.neuron_model, self.name
                )
            )
        # Merge in the model specific parameters
        params.update(self.__dict__[self.neuron_model])
        return params

    def get_receptor_specifications(self):
        if self.neuron_model in self.receptor_specifications:
            return self.receptor_specifications[self.neuron_model]
        else:
            return {}
