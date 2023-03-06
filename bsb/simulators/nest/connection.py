from bsb import config
from bsb.config import types
from bsb.simulation.connection import ConnectionModel
from bsb.exceptions import ReceptorSpecificationError
import numpy as np


@config.node
class NestConnectionSettings:
    rule = config.attr(type=str)
    model = config.attr(type=str)
    weight = config.attr(type=float, required=True)
    delay = config.attr(type=types.distribution(), required=True)


@config.node
class NestSynapseSettings:
    model_settings = config.catch_all(type=dict)


@config.node
class NestConnection(ConnectionModel):
    connection = config.attr(type=NestConnectionSettings, required=True)
    synapse = config.attr(type=NestSynapseSettings, required=True)
    synapse_model = config.attr(type=str)
    plastic = config.attr(default=False)
    hetero = config.attr(default=False)
    teaching = config.attr(type=str)
    is_teaching = config.attr(default=False)

    def boot(self):
        self.synapse_model = self.synapse_model or self.simulation.default_synapse_model

    def validate(self):
        if self.plastic:
            # Set plasticity synapse dict defaults for on each possible model
            synapse_defaults = {
                "A_minus": 0.0,
                "A_plus": 0.0,
                "Wmin": 0.0,
                "Wmax": 4000.0,
            }
            for key, model in self.synapse.model_settings.items():
                self.synapse.model_settings[key] = synapse_defaults.update(model)

    def get_synapse_parameters(self, synapse_model_name):
        # Get the default synapse parameters
        return self.synapse[synapse_model_name]

    def get_connection_parameters(self):
        # Get the default synapse parameters
        params = self.connection.copy()
        # Add the receptor specifications, if required.
        if self.should_specify_receptor_type():
            # If specific receptors are specified, the weight should always be positive.
            # We try to sanitize user data as best we can. If the given weight is a distr
            # (given as a dict) we try to sanitize the `mu` value, if present.
            if type(params["weight"]) is dict:
                if "mu" in params["weight"].keys():
                    params["weight"]["mu"] = np.abs(params["weight"]["mu"])
            else:
                params["weight"] = np.abs(params["weight"])
            if "Wmax" in params:
                params["Wmax"] = np.abs(params["Wmax"])
            if "Wmin" in params:
                params["Wmin"] = np.abs(params["Wmin"])
            params["receptor_type"] = self.get_receptor_type()
        params["model"] = self.simulation.suffixed(self.name)
        return params

    def _get_cell_types(self, key="from"):
        meta = self.scaffold.output_formatter.get_connectivity_set_meta(self.name)
        if key + "_cell_types" in meta:
            cell_types = set()
            for name in meta[key + "_cell_types"]:
                cell_types.add(self.scaffold.get_cell_type(name))
            return list(cell_types)
        connection_types = (
            self.scaffold.output_formatter.get_connectivity_set_connection_types(
                self.name
            )
        )
        cell_types = set()
        for connection_type in connection_types:
            cell_types |= set(connection_type.__dict__[key + "_cell_types"])
        return list(cell_types)

    def get_cell_types(self):
        return self._get_cell_types(key="from"), self._get_cell_types(key="to")

    def should_specify_receptor_type(self):
        _, to_cell_types = self.get_cell_types()
        if len(to_cell_types) > 1:
            raise NotImplementedError(
                "Specifying receptor types of connections consisiting of more than 1 cell type is currently undefined behaviour."
            )
        to_cell_type = to_cell_types[0]
        to_cell_model = self.simulation.cell_models[to_cell_type.name]
        return to_cell_model.neuron_model in to_cell_model.receptor_specifications

    def get_receptor_type(self):
        from_cell_types, to_cell_types = self.get_cell_types()
        if len(to_cell_types) > 1:
            raise NotImplementedError(
                "Specifying receptor types of connections consisiting of more than 1 target cell type is currently undefined behaviour."
            )
        if len(from_cell_types) > 1:
            raise NotImplementedError(
                "Specifying receptor types of connections consisting of more than 1 origin cell type is currently undefined behaviour."
            )
        to_cell_type = to_cell_types[0]
        from_cell_type = from_cell_types[0]
        to_cell_model = self.simulation.cell_models[to_cell_type.name]
        if from_cell_type.name in self.simulation.cell_models.keys():
            from_cell_model = self.simulation.cell_models[from_cell_type.name]
        else:  # For neurons receiving from entities
            from_cell_model = self.simulation.entities[from_cell_type.name]
        receptors = to_cell_model.get_receptor_specifications()
        if from_cell_model.name not in receptors:
            raise ReceptorSpecificationError(
                "Missing receptor specification for cell model '{}' in '{}' while attempting to connect a '{}' to it during '{}'".format(
                    to_cell_model.name, self.node_name, from_cell_model.name, self.name
                )
            )
        return receptors[from_cell_model.name]
