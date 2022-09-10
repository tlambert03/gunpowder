from __future__ import annotations
import copy
from typing import Dict, Optional

from gunpowder.coordinate import Coordinate
from .provider_spec import ProviderSpec, Key
from .array import ArrayKey
from .array_spec import ArraySpec
from .graph import GraphKey
from .graph_spec import GraphSpec

class BatchRequest(ProviderSpec):
    """A collection of (possibly partial) :class:`ArraySpec` and
    :class:`GraphSpec` forming a request.
    """

    def __init__(
        self,
        array_specs: Optional[Dict[ArrayKey, ArraySpec]] = None,
        graph_specs: Optional[Dict[GraphKey, GraphSpec]] = None,
        points_specs: Optional[Dict[GraphKey, GraphSpec]] = None,
        random_seed: Optional[int] = None,
    ) -> None: ...
    def add(
        self,
        key: Key,
        shape: Coordinate,
        voxel_size: Optional[Coordinate] = None,
        directed: Optional[bool] = None,
        placeholder: Optional[bool] = False,
    ) -> None:
        """Convenience method to add an array or graph spec by providing only
        the shape of a ROI (in world units)."""
    def copy(self) -> BatchRequest:
        """Create a copy of this request."""
        return copy.deepcopy(self)
    @property
    def random_seed(self) -> int: ...
    def update_with(self, request: BatchRequest) -> BatchRequest:
        """Update current request with another"""
    def merge(self, request: BatchRequest) -> BatchRequest:
        """Merge another request with current request"""
    def __eq__(self, other: object) -> bool:
        """Override equality check to allow batche requests with different
        seeds to still be checked. Otherwise equality check should
        never succeed."""
