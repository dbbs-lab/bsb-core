import numpy as np
from tqdm import tqdm

from bsb import config
from bsb.config import types, compose_nodes
from bsb.simulation.connection import ConnectionModel


@config.node
class NestSynapseSettings:
    model = config.attr(type=str, default="static_synapse")
    weight = config.attr(type=float, required=True)
    delay = config.attr(type=float, required=True)
    receptor_type = config.attr(type=int)
    constants = config.catch_all(type=types.any_())


@config.node
class NestConnectionSettings:
    rule = config.attr(type=str)
    constants = config.catch_all(type=types.any_())


@config.dynamic(attr_name="model_strategy", required=False)
class NestConnection(compose_nodes(NestConnectionSettings, ConnectionModel)):
    tag = config.attr(type=str)
    synapse = config.attr(type=NestSynapseSettings, required=True)

    def create_connections(self, simdata, pre_nodes, post_nodes, cs):
        import nest

        syn_spec = self.get_syn_spec()
        if self.rule is not None:
            connections = [
                nest.Connect(pre_nodes, post_nodes, self.get_conn_spec(), syn_spec, True)
            ]
        else:
            connections = []
            local_chunks = cs.get_local_chunks("out")
            for local_chunk in tqdm(local_chunks, total=len(local_chunks), desc="locals"):
                conns = {}
                itr = cs.load_connections().from_(local_chunk).as_globals()
                for pre_locs, post_locs in tqdm(
                    itr,
                    total=len(cs.get_global_chunks("out", local_chunk)),
                    desc="blocks",
                ):
                    cell_pairs, multiplicity = np.unique(
                        np.column_stack((pre_locs[:, 0], post_locs[:, 0])),
                        return_counts=True,
                        axis=0,
                    )
                    prel = pre_nodes.tolist()
                    postl = post_nodes.tolist()
                    ssw = {**syn_spec}
                    bw = syn_spec["weight"]
                    ssw["weight"] = [bw * m for m in multiplicity]
                    ssw["delay"] = [syn_spec["delay"]] * len(ssw["weight"])
                    nest.Connect(
                        [prel[x] for x in cell_pairs[:, 0]],
                        [postl[x] for x in cell_pairs[:, 1]],
                        "one_to_one",
                        ssw,
                        return_synapsecollection=False,
                    )
        return connections

    def get_connectivity_set(self):
        if self.tag is not None:
            return self.scaffold.get_connectivity_set(self.tag)
        else:
            return self.connection_model

    def get_conn_spec(self):
        return {
            "rule": self.rule,
            **self.constants,
        }

    def get_syn_spec(self):
        return {
            **{
                label: value
                for attr, label in (
                    ("model", "synapse_model"),
                    ["weight"] * 2,
                    ["delay"] * 2,
                    ["receptor_type"] * 2,
                )
                if (value := getattr(self.synapse, attr)) is not None
            },
            **self.synapse.constants,
        }
