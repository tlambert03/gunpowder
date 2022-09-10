from typing import ContextManager, Union

import zarr

from .hdf5like_source_base import Hdf5LikeSource

class ZarrSource(Hdf5LikeSource):
    """A `zarr <https://github.com/zarr-developers/zarr>`_ data source."""

    def _open_file(
        self, filename: str
    ) -> ContextManager[Union[zarr.Array, zarr.Group]]: ...
