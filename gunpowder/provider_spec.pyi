# flake8: noqa
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, overload, Union
from gunpowder.coordinate import Coordinate
from gunpowder.array import ArrayKey
from gunpowder.array_spec import ArraySpec
from gunpowder.graph import GraphKey
from gunpowder.graph_spec import GraphSpec
from gunpowder.roi import Roi
from .freezable import Freezable

Key = Union[ArrayKey, GraphKey]
Spec = Union[ArraySpec, GraphSpec]

class ProviderSpec(Freezable):
    """A collection of (possibly partial) :class:`ArraySpecs<ArraySpec>` and
    :class:`GraphSpecs<GraphSpec>` describing a
    :class:`BatchProvider's<BatchProvider>` offered arrays and graphs.
    """

    array_specs: Dict[ArrayKey, ArraySpec]
    graph_specs: Dict[GraphKey, GraphSpec]

    def __init__(
        self,
        array_specs: Optional[Dict[ArrayKey, ArraySpec]] = None,
        graph_specs: Optional[Dict[GraphKey, GraphSpec]] = None,
        points_specs: Optional[Dict[GraphKey, GraphSpec]] = None,
    ) -> None: ...
    @property
    def points_specs(self) -> Dict[GraphKey, GraphSpec]: ...
    @overload
    def __setitem__(self, key: ArrayKey, spec: Union[Roi, ArraySpec]) -> None: ...
    @overload
    def __setitem__(self, key: GraphKey, spec: Union[Roi, GraphSpec]) -> None: ...
    @overload
    def __getitem__(self, key: ArrayKey) -> ArraySpec: ...
    @overload
    def __getitem__(self, key: GraphKey) -> GraphSpec: ...
    def __len__(self) -> int: ...
    def __contains__(self, key: Key) -> bool: ...
    def __delitem__(self, key: Key) -> None: ...
    def remove_placeholders(self) -> None: ...
    def items(self) -> Iterator[Tuple[Key, Spec]]:
        """Provides a generator iterating over key/value pairs."""
    def get_total_roi(self) -> Optional[Roi]:
        """Get the union of all the ROIs."""
    def get_common_roi(self) -> Optional[Roi]:
        """Get the intersection of all the requested ROIs."""
    def get_lcm_voxel_size(
        self, array_keys: Optional[Sequence[ArrayKey]] = None
    ) -> Optional[Coordinate]:
        """Get the least common multiple of the voxel sizes in this spec."""
    def __eq__(self, other: Any) -> bool: ...
    def __ne__(self, other: Any) -> bool: ...
    def __repr__(self) -> str: ...
