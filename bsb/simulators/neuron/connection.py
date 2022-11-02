from bsb import config
from bsb.config import types
from bsb.simulation.connection import ConnectionModel


_str_list = types.list(type=str)


@config.node
class NeuronConnection(ConnectionModel):
    synapses = config.attr(
        type=types.or_(types.dict(type=_str_list), _str_list), required=True
    )
    source = config.attr(type=str, default=None)

    def validate_prepare(self):
        for connection_model in self.connection_models.values():
            # Get the connectivity set associated with this connection model
            connectivity_set = self.scaffold.get_connectivity_set(connection_model.name)
            from_type = connectivity_set.connection_types[0].presynaptic.type
            to_type = connectivity_set.connection_types[0].postsynaptic.type
            from_cell_model = self.cell_models[from_type.name]
            to_cell_model = self.cell_models[to_type.name]
            if (
                from_type.entity
                or from_cell_model.relay
                or to_type.entity
                or to_cell_model.relay
            ):
                continue
            if not connectivity_set.compartment_set.exists():
                raise IntersectionDataNotFoundError(
                    "No intersection data found for '{}'".format(connection_model.name)
                )

    def resolve_synapses(self):
        return self.synapses
