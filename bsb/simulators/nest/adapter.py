import functools

from bsb.simulation.adapter import SimulatorAdapter
from bsb.simulation.results import SimulationRecorder, SimulationResult
from bsb.services import MPI
from bsb.reporting import report, warn
from bsb.exceptions import (
    KernelWarning,
    NestModelError,
    NestModuleError,
    ConnectivityWarning,
    UnknownGIDError,
    ConfigurationError,
    AdapterError,
    KernelLockedError,
    SuffixTakenError,
    NestKernelError,
    SimulationWarning,
)
from .connection import NestConnection
import os
import json
import numpy as np
from copy import deepcopy
import warnings
import h5py
import time


class SimulationData:
    def __init__(self):
        self.chunks = None
        self.cells = dict()
        self.cid_offsets = dict()
        self.connections = dict()
        self.first_gid: int = None
        self.result: "NestResult" = None


class NestResult(SimulationResult):
    def record(self, obj, **annotations):
        from quantities import ms

        def flush(segment):
            segment.analogsignals.append(...)

        self.create_recorder(flush)


class NestAdapter(SimulatorAdapter):
    def __init__(self):
        self.simdata = dict()
        self.loaded_modules = set()

    @property
    @functools.cache
    def nest(self):
        report("Importing  NEST...", level=2)
        import nest

        return nest

    def prepare(self, simulation):
        self.simdata[simulation] = simdata = SimulationData()
        try:
            simdata.result = SimulationResult(simulation)
            report("Installing  NEST modules...", level=2)
            self.load_modules(simulation)
            report("Creating neurons...", level=2)
            self.create_neurons(simulation)
            report("Creating connections...", level=2)
            self.connect_neurons(simulation)
            report("Creating devices...", level=2)
            self.create_devices(simulation)
            return self.simdata[simulation]
        except:
            del self.simdata[simulation]
            raise

    def reset_kernel(self):
        self.nest.set_verbosity(self.verbosity)
        self.nest.ResetKernel()
        # Reset which modules we should consider explicitly loaded by the user
        # to appropriately warn them when they load them twice.
        self.loaded_modules = set()
        self.reset_processes(self.threads)
        self.nest.SetKernelStatus(
            {
                "resolution": self.resolution,
                "overwrite_files": True,
                "data_path": self.scaffold.output_formatter.get_simulator_output_path(
                    self.simulator_name
                ),
            }
        )

    def reset(self):
        self.is_prepared = False
        if hasattr(self, "nest"):
            self.reset_kernel()
        self.global_identifier_map = {}
        for cell_model in self.cell_models.values():
            cell_model.reset()

    def run(self, simulation):
        report("Simulating...", level=2)
        tick = time.time()
        with self.nest.RunManager():
            for oi, i in self.step_progress(self.duration, step=1):
                self.nest.Run(i - oi)
                self.progress(i)
        report(f"Simulation done. {time.time() - tick:.2f}s elapsed.", level=2)

    def step(self, simulator, dt):
        if not self.is_prepared:
            warn("Adapter has not been prepared", SimulationWarning)
        report("Simulating...", level=2)
        tick = time.time()
        with simulator.RunManager():
            for oi, i in self.step_progress(self.duration, step=dt):
                simulator.Run(i - oi)
                yield self.progress(i)

        report(f"Simulation done. {time.time() - tick:.2f}s elapsed.", level=2)
        if self.has_lock:
            self.release_lock()

    def collect_output(self, simulation):
        report("Collecting output...", level=2)
        tick = time.time()
        rank = MPI.get_rank()

        result = self.simdata[simulation].result
        if rank == 0:
            with h5py.File(result_path, "a") as f:
                f.attrs["configuration_string"] = self.scaffold.configuration._raw
                for path, data, meta in self.result.safe_collect():
                    try:
                        path = "/".join(path)
                        if path in f:
                            data = np.vstack((f[path][()], data))
                            del f[path]
                        d = f.create_dataset(path, data=data)
                        for k, v in meta.items():
                            d.attrs[k] = v
                    except Exception as e:
                        import traceback

                        traceback.print_exc()
                        if not isinstance(data, np.ndarray):
                            warn(
                                "Recorder {} numpy.ndarray expected, got {}".format(
                                    path, type(data)
                                )
                            )
                        else:
                            warn(
                                "Recorder {} processing errored out: {}".format(
                                    path, "{} {}".format(data.dtype, data.shape)
                                )
                            )
        MPI.bcast(result_path, root=0)
        report(
            f"Output collected in '{result_path}'. "
            + f"{time.time() - tick:.2f}s elapsed.",
            level=2,
        )
        return result_path

    def load_modules(self, simulation):
        for module in simulation.modules:
            try:
                self.nest.Install(module)
                self.loaded_modules.add(module)
            except Exception as e:
                if e.errorname == "DynamicModuleManagementError":
                    if "loaded already" in e.message:
                        # Modules stay loaded in between `ResetKernel` calls. We
                        # assume that there's nothing to warn the user about if
                        # the adapter installs the modules each
                        # `reset`/`prepare` cycle.
                        if module in self.loaded_modules:
                            warn(f"Already loaded '{module}'.", KernelWarning)
                    elif "file not found" in e.message:
                        raise NestModuleError(f"Module {module} not found") from None
                    else:
                        raise
                else:
                    raise

    def create_neurons(self, simulation):
        """
        Create a population of nodes in the NEST simulator based on the cell model
        configurations.
        """
        simdata = self.simdata[simulation]
        for cell_model in simulation.cell_models.values():
            simdata.populations[cell_model] = cell_model.create_population()

    def connect_neurons(self, simulation):
        """
        Connect the cells in NEST according to the connection model configurations
        """
        simdata = self.simdata[simulation]
        for connection_model in simulation.connection_models.values():
            simdata.synapses[connection_model] = connection_model.create_connections()

    def create_devices(self):
        """
        Create the configured NEST devices in the simulator
        """
        pass
