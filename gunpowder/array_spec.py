from __future__ import annotations

import copy
from typing import TYPE_CHECKING
from .coordinate import Coordinate
from .freezable import Freezable

if TYPE_CHECKING:
    from gunpowder.roi import Roi
    import numpy as np


class ArraySpec(Freezable):
    '''Contains meta-information about an array. This is used by
    :class:`BatchProviders<BatchProvider>` to communicate the arrays they
    offer, as well as by :class:`Arrays<Array>` to describe the data they
    contain.

    Attributes:

        roi (:class:`Roi`):

            The region of interested represented by this array spec. Can be
            ``None`` for :class:`BatchProviders<BatchProvider>` that allow
            requests for arrays everywhere, but will always be set for array
            specs that are part of a :class:`Array`.

        voxel_size (:class:`Coordinate`):

            The size of the spatial axises in world units.

        interpolatable (``bool``):

            Whether the values of this array can be interpolated.

        nonspatial (``bool``, optional):

            If set, this array does not represent spatial data (e.g., a list of
            labels for samples in a batch). ``roi`` and ``voxel_size`` have to
            be ``None``. No consistency checks will be performed.

        dtype (``np.dtype``):

            The data type of the array.
    '''

    def __init__(
            self,
            roi: Roi | None = None,
            voxel_size: Coordinate | None = None,
            interpolatable: bool | None = None,
            nonspatial: bool | None = False,
            dtype: np.dtype | None = None,
            placeholder: bool | None = False):

        self.roi = roi
        self.voxel_size = None if voxel_size is None else Coordinate(voxel_size)
        self.interpolatable = interpolatable
        self.nonspatial = nonspatial
        self.dtype = dtype
        self.placeholder = placeholder

        if nonspatial:
            assert roi is None, "Non-spatial arrays can not have a ROI"
            assert voxel_size is None, "Non-spatial arrays can not " \
                "have a voxel size"

        self.freeze()

    def update_with(self, spec: ArraySpec) -> None:
        if self.roi is not None and \
           spec.roi is not None:
            self.roi = self.roi.union(spec.roi)
        elif spec.roi is not None:
            self.roi = spec.roi

        if spec.voxel_size is not None:
            self.voxel_size = spec.voxel_size

        if spec.interpolatable is not None:
            self.interpolatable = spec.interpolatable

        if spec.nonspatial is not None:
            self.nonspatial = spec.nonspatial

        if spec.dtype is not None:
            self.dtype = spec.dtype

        if spec.placeholder is not None:
            self.placeholder = spec.placeholder

    def copy(self) -> ArraySpec:
        '''Create a copy of this spec.'''
        return copy.deepcopy(self)

    def __eq__(self, other: object) -> bool:

        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other: object) -> bool:

        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __repr__(self) -> str:
        r = ""
        r += "ROI: " + str(self.roi) + ", "
        r += "voxel size: " + str(self.voxel_size) + ", "
        r += "interpolatable: " + str(self.interpolatable) + ", "
        r += "non-spatial: " + str(self.nonspatial) + ", "
        r += "dtype: " + str(self.dtype) + ", "
        r += "placeholder: " + str(self.placeholder)
        return r
