from .component import SimulationComponent


class SimulationDevice(SimulationComponent):
    def _bootstrap(self, device_cls):
        self.__class__ = device_cls
        # Re-initialise the self
        # TODO: Switch to better config in v4
        self.initialise(self.scaffold)
        self.validate_specifics()
        self.initialise_targets()
        self.initialise_patterns()

    def create_patterns(self):
        raise NotImplementedError(
            "The "
            + self.__class__.__name__
            + " device does not implement any `create_patterns` function."
        )

    def get_pattern(self, target, cell=None, section=None, synapse=None):
        raise NotImplementedError(
            "The "
            + self.__class__.__name__
            + " device does not implement any `get_pattern` function."
        )

    def implement(self, target, location):
        raise NotImplementedError(
            "The "
            + self.__class__.__name__
            + " device does not implement any `implement` function."
        )

    def validate_specifics(self):
        raise NotImplementedError(
            "The "
            + self.__class__.__name__
            + " device does not implement any `validate_specifics` function."
        )

    def get_patterns(self):
        """
        Return the patterns of the device.
        """
        if hasattr(self, "_patterns"):
            return self._patterns
        raise ParallelIntegrityError(
            f"MPI process %rank% failed a checkpoint."
            + " `initialise_patterns` should always be called before `get_patterns` on all MPI processes.",
            self.adapter.get_rank(),
        )

    def initialise_patterns(self):
        if self.adapter.get_rank() == 0:
            # Have root 0 prepare the possibly random patterns.
            patterns = self.create_patterns()
        else:
            patterns = None
        # Broadcast to make sure all the nodes have the same patterns for each device.
        self._patterns = self.scaffold.MPI.COMM_WORLD.bcast(patterns, root=0)


class PatternlessDevice:
    def create_patterns(*args, **kwargs):
        pass

    def get_pattern(*args, **kwargs):
        pass
