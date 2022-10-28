from .. import config
from ..config import refs, types
from ..exceptions import ReificationError
import abc


@config.dynamic(
    attr_name="type", default="value", auto_classmap=True, classmap_entry="value"
)
class Parameter(abc.ABC):
    population = config.ref(refs.sim_cell_model_ref)
    gid = config.attr(type=types.list(type=int))
    value = config.attr(type=types.distribution())

    def get_values(self, n, **re):
        return self.reify(**re)(n)

    def reify(self, **re):
        if self.population and re.get("population", None) != self.population.name:
            raise ReificationError(
                f"Parameter applies only to the `{self.population.name}` population."
            )
        if self.gid and re.get("gid", None) != self.gid:
            raise ReificationError(
                f"Parameter applies only to GID `{self.population.name}`."
            )

        def produce_parameter_values(n):
            return self.value.draw(n)

    def applies_to(self, **re):
        return all(
            self[factor] is None or self[factor] == value for factor, value in re.items()
        )
