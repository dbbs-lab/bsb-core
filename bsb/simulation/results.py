from ..reporting import warn
import traceback


class SimulationResult:
    def __init__(self, simulation):
        from neo import Block

        tree = simulation.__tree__()
        try:
            del tree["post_prepare"]
        except KeyError:
            pass
        self.block = Block(name=simulation.name, config=tree)
        self.recorders = []

    def add(self, recorder):
        self.recorders.append(recorder)

    def create_recorder(self, flush):
        recorder = SimulationRecorder()
        recorder.flush = flush
        self.add(recorder)
        return recorder

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

    def write(self, filename, mode):
        from neo import io

        io.NixIO(filename, mode=mode).write(self.block)


class SimulationRecorder:
    def flush(self):
        raise NotImplementedError("Recorders need to implement the `flush` function.")
