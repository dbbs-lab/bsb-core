from ..reporting import warn
import traceback


class SimulationResult:
    def __init__(self, simulation):
        from neo import Block

        self.block = neo.Block(name=simulation.name, config=simulation.__tree__())
        self.recorders = []

    def add(self, recorder):
        self.recorders.append(recorder)

    def create_recorder(self, path_func, data_func, meta_func=None):
        recorder = ClosureRecorder(path_func, data_func, meta_func)
        self.add(recorder)
        return recorder

    def _collect(self, recorder):
        return recorder.get_path(), recorder.get_data(), recorder.get_meta()

    def flush(self):
        from neo import Segment

        segment = Segment()
        self.block.segments.append(segment)
        for recorder in self.recorders:
            try:
                recorder.flush(segment)
            except Exception as e:
                traceback.print_exc()
                warn("Recorder errored out!")

    def write(self):
        from neo import io

        io.NixIO("test.nix", mode="ow").write(self.block)


class SimulationRecorder:
    def flush(self):
        raise NotImplementedError("Recorders need to implement the `flush` function.")


class ClosureRecorder(SimulationRecorder):
    def __init__(self, flush_func):
        super().__init__()
        self.flush = flush_func
