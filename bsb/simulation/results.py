from ..reporting import warn
import traceback


class SimulationResult:
    def __init__(self):
        self.recorders = []

    def add(self, recorder):
        self.recorders.append(recorder)

    def create_recorder(self, path_func, data_func, meta_func=None):
        recorder = ClosureRecorder(path_func, data_func, meta_func)
        self.add(recorder)
        return recorder

    def collect(self):
        for recorder in self.recorders:
            yield recorder.get_path(), recorder.get_data(), recorder.get_meta()

    def safe_collect(self):
        gen = iter(self.collect())
        while True:
            try:
                yield next(gen)
            except StopIteration:
                break
            except Exception as e:
                traceback.print_exc()
                warn("Recorder errored out!")


class SimulationRecorder:
    def get_path(self):
        raise NotImplementedError("Recorders need to implement the `get_path` function.")

    def get_data(self):
        raise NotImplementedError("Recorders need to implement the `get_data` function.")

    def get_meta(self):
        return {}


class ClosureRecorder(SimulationRecorder):
    def __init__(self, path_func, data_func, meta_func=None):
        super().__init__()
        self.get_path = path_func
        self.get_data = data_func
        if meta_func:
            self.get_meta = meta_func


class PresetPathMixin:
    def get_path(self):
        return self.path


class PresetMetaMixin:
    def get_meta(self):
        return self.meta
