# flake8: noqa
from typing import Optional, Union

import numpy as np

from gunpowder.coordinate import Coordinate
from gunpowder.roi import Roi

from .freezable import Freezable


class ArraySpec(Freezable):
    """Contains meta-information about an array. This is used by
    :class:`BatchProviders<BatchProvider>` to communicate the arrays they
    offer, as well as by :class:`Arrays<Array>` to describe the data they
    contain.
    """

    roi: Optional[Roi]
    voxel_size: Optional[Coordinate]
    interpolatable: Optional[bool]
    nonspatial: Optional[bool]
    dtype: Optional[np.dtype]
    placeholder: Optional[bool]

    def __init__(
        self,
        roi: Optional[Roi] = None,
        voxel_size: Union[Coordinate, tuple[int, ...], None] = None,
        interpolatable: Optional[bool] = None,
        nonspatial: Optional[bool] = False,
        dtype: Optional[np.dtype] = None,
        placeholder: Optional[bool] = False,
    ): ...
    def update_with(self, spec: ArraySpec) -> None: ...
    def copy(self) -> ArraySpec: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
