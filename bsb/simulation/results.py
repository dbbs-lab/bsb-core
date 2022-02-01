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

    def _collect(self, recorder):
        return recorder.get_path(), recorder.get_data(), recorder.get_meta()

    def collect(self):
        for recorder in self.recorders:
            if hasattr(recorder, "multi_collect"):
                yield from (
                    self._collect(subrecorder) for subrecorder in recorder.multi_collect()
                )
            else:
                yield self._collect(recorder)

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


class MultiRecorder(SimulationRecorder):
    def get_data(*args, **kwargs):
        raise RuntimeError("Multirecorder data should be collected from children.")

    def multi_collect(self, *args, **kwargs):
        raise NotImplementedError("Multirecorders need to implement `multi_collect`.")


class PresetPathMixin:
    def get_path(self):
        return self.path


class PresetMetaMixin:
    def get_meta(self):
        return self.meta
