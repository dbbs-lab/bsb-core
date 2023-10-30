import sys
from neo import SpikeTrain
import typing
import time
import functools
from tqdm import tqdm

from ...simulation.adapter import SimulatorAdapter, SimulationData
from ...simulation.results import SimulationResult
from ...reporting import report, warn
from ...exceptions import (
    KernelWarning,
    NestModuleError,
    NestModelError,
    NestConnectError,
)
from ...services import MPI

if typing.TYPE_CHECKING:
    from ...simulation import Simulation


class NestResult(SimulationResult):
    def record(self, nc, **annotations):
        import nest

        recorder = nest.Create("spike_recorder", params={"record_to": "memory"})
        nest.Connect(nc, recorder)

        def flush(segment):
            events = recorder.events[0]

            segment.spiketrains.append(
                SpikeTrain(
                    events["times"],
                    waveforms=events["senders"],
                    t_stop=nest.biological_time,
                    units="ms",
                    **annotations,
                )
            )

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

        self.check_comm()

        return nest

    def simulate(self, simulation):
        try:
            self.reset_kernel()
            return super().simulate(simulation)
        finally:
            self.reset_kernel()

    def prepare(self, simulation, comm=None):
        self.simdata[simulation] = SimulationData(
            simulation, result=NestResult(simulation)
        )
        try:
            report("Installing  NEST modules...", level=2)
            self.load_modules(simulation)
            self.set_settings(simulation)
            report("Creating neurons...", level=2)
            self.create_neurons(simulation)
            report("Creating connections...", level=2)
            self.connect_neurons(simulation)
            report("Creating devices...", level=2)
            self.create_devices(simulation)
            return self.simdata[simulation]
        except Exception:
            del self.simdata[simulation]
            raise

    def reset_kernel(self):
        self.nest.ResetKernel()
        # Reset which modules we should consider explicitly loaded by the user
        # to appropriately warn them when they load them twice.
        self.loaded_modules = set()

    def run(self, simulation):
        if simulation not in self.simdata:
            raise RuntimeError("Can't run unprepared simulation")
        report("Simulating...", level=2)
        tick = time.time()
        simulation.start_progress(simulation.duration)
        try:
            with self.nest.RunManager():
                for oi, i in simulation.step_progress(simulation.duration, step=1):
                    self.nest.Run(i - oi)
                    simulation.progress(i)
        finally:
            result = self.simdata[simulation].result
            del self.simdata[simulation]
        report(f"Simulation done. {time.time() - tick:.2f}s elapsed.", level=2)
        return result

    def load_modules(self, simulation):
        for module in simulation.modules:
            try:
                self.nest.Install(module)
                self.loaded_modules.add(module)
            except Exception as e:
                if e.errorname == "DynamicModuleManagementError":
                    if "loaded already" in e.message:
                        # Modules stay loaded in between `ResetKernel` calls. If the module
                        # is not in the `loaded_modules` set, then it's the first time this
                        # `reset`/`prepare` cycle, and there is no user-side issue.
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
            simdata.populations[cell_model] = cell_model.create_population(simdata)

    def connect_neurons(self, simulation):
        """
        Connect the cells in NEST according to the connection model configurations
        """
        simdata = self.simdata[simulation]
        iter = simulation.connection_models.values()
        if MPI.get_rank() == 0:
            iter = tqdm(iter, desc="", file=sys.stdout)
        for connection_model in iter:
            try:
                iter.set_description(connection_model.name)
            except AttributeError:
                # Only rank 0 should report progress bar
                pass
            cs = simulation.scaffold.get_connectivity_set(
                connection_model.tag or connection_model.name
            )
            try:
                pre_nodes = simdata.populations[simulation.get_model_of(cs.pre_type)]
            except KeyError:
                raise NestModelError(f"No model found for {cs.pre_type}")
            try:
                post_nodes = simdata.populations[simulation.get_model_of(cs.post_type)]
            except KeyError:
                raise NestModelError(f"No model found for {cs.post_type}")
            try:
                simdata.connections[
                    connection_model
                ] = connection_model.create_connections(
                    simdata, pre_nodes, post_nodes, cs
                )
            except Exception as e:
                raise NestConnectError(f"{connection_model} error during connect.")

    def create_devices(self, simulation):
        simdata = self.simdata[simulation]
        for device_model in simulation.devices.values():
            device_model.implement(self, simulation, simdata)

    def set_settings(self, simulation: "Simulation"):
        self.nest.set_verbosity(simulation.verbosity)
        self.nest.resolution = simulation.resolution
        self.nest.overwrite_files = True

    def check_comm(self):
        import nest

        if nest.NumProcesses() != MPI.get_size():
            raise RuntimeError(
                f"NEST is managing {nest.NumProcesses()} processes, but {MPI.get_size()}"
                " were detected. Please check your MPI setup."
            )
