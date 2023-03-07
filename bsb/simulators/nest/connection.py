from bsb import config
from bsb.config import types, compose_nodes
from bsb.simulation.connection import ConnectionModel


@config.node
class NestSynapseSettings:
    model_settings = config.catch_all(type=dict)


@config.node
class NestConnectionSettings:
    rule = config.attr(type=str)
    model = config.attr(type=str)
    weight = config.attr(type=float, required=True)
    delay = config.attr(type=types.distribution(), required=True)


@config.dynamic(attr_name="model_strategy", required=False)
class NestConnection(compose_nodes(NestConnectionSettings, ConnectionModel)):
    tag = config.attr(type=str)
    synapse = config.attr(type=NestSynapseSettings, required=True)

    def create_connections(self, pre_nodes, post_nodes):
        import nest

        conn_spec = self.get_conn_spec()
        if self.rule is not None:
            synapses = nest.Connect(
                pre_nodes, post_nodes, conn_spec, self.get_syn_spec(), True
            )
        else:
            print("Rule-less connection should be loaded from CS")
            synapses = None
        return synapses

    def get_connectivity_set(self):
        if self.tag is not None:
            return self.scaffold.get_connectivity_set(self.tag)
        else:
            return self.connection_model

    def get_conn_spec(self):
        return {attr: getattr(self, attr) for attr in ("rule", "")}

    def get_syn_spec(self):
        return self.synapse.model_settings.copy()
