# flake8: noqa
from typing import Optional, NoReturn

from gunpowder.array_spec import ArraySpec
from .freezable import Freezable
from gunpowder.roi import Roi
import numpy as np

class Array(Freezable):
    """A numpy array with a specification describing the data."""

    data: np.ndarray
    spec: Optional[ArraySpec]
    attrs: dict

    def __init__(
        self,
        data: np.ndarray,
        spec: Optional[ArraySpec] = None,
        attrs: Optional[dict] = None,
    ): ...
    def crop(self, roi: Roi, copy: bool = True) -> Array:
        """Create a cropped copy of this Array."""
    def merge(
        self, array: Array, copy_from_self: bool = False, copy: bool = False
    ) -> NoReturn:
        """Merge this array with another one. The resulting array will have the
        size of the larger one, with values replaced from ``array``.

        Do not use.
        """
    def __repr__(self) -> str: ...
    def copy(self) -> Array:
        """Create a copy of this array."""

class ArrayKey(Freezable):
    """A key to identify arrays in requests, batches, and across nodes."""

    identifier: str
    hash: int

    def __init__(self, identifier: str) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

class ArrayKeys:
    """Convenience access to all created :class:``ArrayKey``s."""

    def __getattribute__(self, name: str) -> ArrayKey: ...
