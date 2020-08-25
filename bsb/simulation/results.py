class SimulationResult:
    def __init__(self):
        self.recorders = []

    def add(self, recorder):
        self.recorders.append(recorder)

    def create_recorder(self, path_func, data_func, meta_func=None):
        recorder = SimulationRecorder.create(path_func, data_func, meta_func)
        self.add(recorder)
        return recorder

    def collect():
        for recorder in self.recorders:
            yield recorder.get_path(), recorder.get_data(), recorder.get_meta()


class SimulationRecorder:
    def get_path(self):
        raise NotImplementedError("Recorders need to implement the `get_path` function.")

    def get_data(self):
        raise NotImplementedError("Recorders need to implement the `get_data` function.")

    def get_meta(self):
        return {}

    @classmethod
    def create(cls, path_func, data_func, meta_func=None):
        instance = cls()
        instance.get_path = path_func
        instance.get_data = data_func
        if meta_func:
            instance.get_meta = meta_func
        return instance
