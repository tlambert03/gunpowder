# flake8: noqa
from __future__ import annotations

from typing import Optional

import numpy as np

from gunpowder.roi import Roi

from .freezable import Freezable


class GraphSpec(Freezable):
    """Contains meta-information about a graph. This is used by
    :class:`BatchProviders<BatchProvider>` to communicate the graphs they
    offer, as well as by :class:`Graph` to describe the data they contain.
    """

    roi: Optional[Roi]
    directed: Optional[bool]
    dtype: np.dtype
    placeholder: Optional[bool]

    def __init__(
        self,
        roi: Optional[Roi] = None,
        directed: Optional[bool] = None,
        dtype: np.dtype = np.float32,
        placeholder: Optional[bool] = False,
    ): ...
    def update_with(self, spec: GraphSpec) -> None: ...
    def copy(self) -> GraphSpec: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
