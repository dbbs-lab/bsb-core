from ...interfaces import Filter as IFilter
from .resource import Resource
import numpy as np


class Filter(IFilter):
    @classmethod
    def get_filter_types(cls):
        return ["label"]
