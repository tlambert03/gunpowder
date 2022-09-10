# flake8: noqa
from typing import Dict, Iterator, Optional, Tuple, Union, overload

from gunpowder.batch_request import BatchRequest
from gunpowder.roi import Roi

from .array import Array, ArrayKey
from .freezable import Freezable
from .graph import Graph, GraphKey
from .profiling import ProfilingStats

Key = Union[ArrayKey, GraphKey]

class Batch(Freezable):
    """Contains the requested batch as a collection of :class:`Arrays<Array>`
    and :class:`Graph` that is passed through the pipeline from sources to sinks.
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
    def __len__(self) -> int: ...
    def __contains__(self, key: Key) -> bool: ...
    def __delitem__(self, key: Key) -> None: ...
    def items(self) -> Iterator[Tuple[Key, Union[Array, Graph]]]:
        """Provides a generator iterating over key/value pairs."""
    def get_total_roi(self) -> Optional[Roi]:
        """Get the union of all the array ROIs in the batch."""
    def __repr__(self) -> str: ...
    def crop(self, request: BatchRequest, copy: bool = False) -> Batch:
        """Crop batch to meet the given request."""
    def merge(self, batch: Batch, merge_profiling_stats: bool = True) -> Batch:
        """Merge this batch (``a``) with another batch (``b``)."""
