import functools
import sys

import numpy as np
import psutil
from tqdm import tqdm

from bsb import config
from bsb.config import types, compose_nodes
from bsb.services import MPI
from bsb.simulation.connection import ConnectionModel
from bsb.exceptions import NestConnectError


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


class LazySynapseCollection:
    def __init__(self, pre, post):
        self._pre = pre
        self._post = post

    def __len__(self):
        return self.collection.__len__()

    def __str__(self):
        return self.collection.__str__()

    def __iter__(self):
        return iter(self.collection)

    def __getattr__(self, attr):
        return getattr(self.collection, attr)

    @functools.cached_property
    def collection(self):
        import nest

        return nest.GetConnections(self._pre, self._post)


@config.dynamic(attr_name="model_strategy", required=False)
class NestConnection(compose_nodes(NestConnectionSettings, ConnectionModel)):
    tag = config.attr(type=str)
    synapse = config.attr(type=NestSynapseSettings, required=True)

    def create_connections(self, simdata, pre_nodes, post_nodes, cs):
        import nest

        syn_spec = self.get_syn_spec()
        if syn_spec["synapse_model"] not in nest.synapse_models:
            raise NestConnectError(
                f"Unknown synapse model '{syn_spec['synapse_model']}'."
            )
        if self.rule is not None:
            nest.Connect(pre_nodes, post_nodes, self.get_conn_spec(), syn_spec)
        else:
            MPI.barrier()
            for pre_locs, post_locs in self.predict_mem_iterator(
                pre_nodes, post_nodes, cs
            ):
                MPI.barrier()
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
            MPI.barrier()
        return LazySynapseCollection(pre_nodes, post_nodes)

    def predict_mem_iterator(self, pre_nodes, post_nodes, cs):
        avmem = psutil.virtual_memory().available
        predicted_all_mem = (
            len(pre_nodes) * 8 * 2 + len(post_nodes) * 8 * 2 + len(cs) * 6 * 8 * (16 + 2)
        ) * MPI.get_size()
        predicted_local_mem = predicted_all_mem / len(cs.get_local_chunks("out"))
        if predicted_local_mem > avmem / 2:
            # Iterate block-by-block
            return self.block_iterator(cs)
        elif predicted_all_mem > avmem / 2:
            # Iterate local hyperblocks
            return self.local_iterator(cs)
        else:
            # Iterate all
            return (cs.load_connections().as_globals().all(),)

    def block_iterator(self, cs):
        locals = cs.get_local_chunks("out")

        def block_iter():
            iter = locals
            if MPI.get_rank() == 0:
                iter = tqdm(iter, desc="hyperblocks", file=sys.stdout)
            for local in iter:
                inner_iter = cs.load_connections().as_globals().from_(local)
                if MPI.get_rank() == 0:
                    yield from tqdm(
                        inner_iter,
                        desc="blocks",
                        total=len(cs.get_global_chunks("out", local)),
                        file=sys.stdout,
                        leave=False,
                    )
                else:
                    yield from inner_iter

        return block_iter()

    def local_iterator(self, cs):
        iter = cs.get_local_chunks("out")
        if MPI.get_rank() == 0:
            iter = tqdm(iter, desc="hyperblocks", file=sys.stdout)
        yield from (
            cs.load_connections().as_globals().from_(local).all() for local in iter
        )

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
