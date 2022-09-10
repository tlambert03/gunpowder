# flake8: noqa
from typing import Dict, Iterator, Optional, Tuple, Union, overload
from .freezable import Freezable
from .profiling import ProfilingStats
from .array import Array, ArrayKey
from .graph import Graph, GraphKey

Key = Union[ArrayKey, GraphKey]

class Batch(Freezable):
    """Contains the requested batch as a collection of :class:`Arrays<Array>`
    and :class:`Graph` that is passed through the pipeline from sources to
    sinks.
    """

    id: int
    profiling_stats: ProfilingStats
    arrays: Dict[ArrayKey, Array]
    graphs: Dict[GraphKey, Graph]
    affinity_neighborhood: Optional[ArrayKey]
    loss: Optional[float]
    iteration: Optional[int]

    @staticmethod
    def get_next_id() -> int: ...
    def __init__(self) -> None:

        self.id = Batch.get_next_id()
        self.profiling_stats = ProfilingStats()
        self.arrays = {}
        self.graphs = {}
        self.affinity_neighborhood = None
        self.loss = None
        self.iteration = None

        self.freeze()
    @overload
    def __setitem__(self, key: ArrayKey, value: Array) -> None: ...
    @overload
    def __setitem__(self, key: GraphKey, value: Graph) -> None: ...
    @overload
    def __getitem__(self, key: ArrayKey) -> Array: ...
    @overload
    def __getitem__(self, key: GraphKey) -> Graph: ...
    def __len__(self) -> int:
        ...
    def __contains__(self, key: Key) -> bool:
        ...
    def __delitem__(self, key: Key) -> None:
        ...
    def items(self) -> Iterator[Tuple[Key, Union[Array, Graph]]]:
        """Provides a generator iterating over key/value pairs."""
    def get_total_roi(self) -> Optional[Roi]:
        """Get the union of all the array ROIs in the batch."""
        

        total_roi = None

        for _, array in self.arrays.items():
            if not array.spec.nonspatial:
                if total_roi is None:
                    total_roi = array.spec.roi
                else:
                    total_roi = total_roi.union(array.spec.roi)

        for _, graph in self.graphs.items():
            if total_roi is None:
                total_roi = graph.spec.roi
            else:
                total_roi = total_roi.union(graph.spec.roi)

        return total_roi
    def __repr__(self):

        r = "\n"
        for collection_type in [self.arrays, self.graphs]:
            for (key, obj) in collection_type.items():
                r += "\t%s: %s\n" % (key, obj.spec)
        return r
    def crop(self, request, copy=False):
        """Crop batch to meet the given request."""

        cropped = Batch()
        cropped.profiling_stats = self.profiling_stats
        cropped.loss = self.loss
        cropped.iteration = self.iteration

        for key, val in request.items():
            assert key in self, "%s not contained in this batch" % key
            if val.roi is None:
                cropped[key] = self[key]
            else:
                if isinstance(key, GraphKey):
                    cropped[key] = self[key].crop(val.roi)
                else:
                    cropped[key] = self[key].crop(val.roi, copy)

        return cropped
    def merge(self, batch, merge_profiling_stats=True):
        """Merge this batch (``a``) with another batch (``b``).

        This creates a new batch ``c`` containing arrays and graphs from
        both batches ``a`` and ``b``:

            * Arrays or Graphs that exist in either ``a`` or ``b`` will be
              referenced in ``c`` (not copied).

            * Arrays or Graphs that exist in both batches will keep only
              a reference to the version in ``b`` in ``c``.

        All other cases will lead to an exception.
        """

        merged = shallow_copy(self)

        for key, val in batch.items():
            # TODO: What is the goal of `val.spec.roi is None`? Why should that
            # mean that the key in merged gets overwritten?
            if key not in merged or val.spec.roi is None:
                merged[key] = val
            elif key in merged:
                merged[key] = val

        if merge_profiling_stats:
            merged.profiling_stats.merge_with(batch.profiling_stats)
        if batch.loss is not None:
            merged.loss = batch.loss
        if batch.iteration is not None:
            merged.iteration = batch.iteration

        return merged
