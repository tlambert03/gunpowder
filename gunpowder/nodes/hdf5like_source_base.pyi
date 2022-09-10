# flake8: noqa
from typing import Any, ContextManager, Dict, Mapping, Optional, Protocol
import numpy as np
from gunpowder.batch import Batch
from gunpowder.batch_request import BatchRequest

from gunpowder.coordinate import Coordinate
from gunpowder.roi import Roi
from gunpowder.array import ArrayKey
from gunpowder.array_spec import ArraySpec
from .batch_provider import BatchProvider

class SupportsAttrs(Protocol):
    attrs: dict

class DataSet(SupportsAttrs, Protocol):
    shape: tuple[int, ...]
    dtype: np.dtype
    def __getitem__(self, key: Any) -> np.ndarray: ...

DataFile = Mapping[str, DataSet]

class Hdf5LikeSource(BatchProvider):
    """An HDF5-like data source."""

    filename: str
    datasets: Dict[ArrayKey, str]
    array_specs: Dict[ArrayKey, ArraySpec]
    ndims: Optional[int]
    channels_first: bool
    def __init__(
        self,
        filename: str,
        datasets: Dict[ArrayKey, str],
        array_specs: Optional[Dict[ArrayKey, ArraySpec]] = None,
        channels_first: bool = True,
    ): ...
    def _open_file(self, filename: str) -> ContextManager[DataFile]: ...
    def _get_voxel_size(self, dataset: SupportsAttrs) -> Optional[Coordinate]: ...
    def _get_offset(self, dataset: SupportsAttrs) -> Optional[Coordinate]: ...
    def __read_spec(self, array_key: ArrayKey, data_file: DataFile, ds_name: str): ...
    def __read(self, data_file: DataFile, ds_name: str, roi: Roi) -> np.ndarray: ...
    def name(self) -> str: ...
