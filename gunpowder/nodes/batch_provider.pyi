from __future__ import annotations
from abc import abstractmethod

from typing import List, Optional, Tuple, Union
import numpy as np
from gunpowder.pipeline import Pipeline

from gunpowder.provider_spec import Key, ProviderSpec, Spec
from gunpowder.batch import Batch
from gunpowder.batch_request import BatchRequest
from collections.abc import ABC

class BatchRequestError(Exception):
    provider: BatchProvider
    request: BatchRequest
    batch: Batch
    def __init__(
        self, provider: BatchProvider, request: BatchRequest, batch: Batch
    ): ...

class BatchProvider(ABC):
    """Superclass for all nodes in a `gunpowder` graph."""

    _provided_items: List[Key]

    @abstractmethod
    def setup(self):
        """To be implemented in subclasses.

        Called during initialization of the DAG. Callees can assume that all
        upstream providers are set up already.

        In setup, call :func:`provides` to announce the arrays and points
        provided by this node.
        """
    @abstractmethod
    def provide(self, request: BatchRequest) -> Batch:
        """To be implemented in subclasses.

        This function takes a :class:`BatchRequest` and should return the
        corresponding :class:`Batch`.
        """
    def teardown(self) -> None:
        """To be implemented in subclasses.

        Called during destruction of the DAG. Subclasses should use this to
        stop worker processes, if they used some.
        """
    def add_upstream_provider(self, provider: BatchProvider) -> BatchProvider: ...
    def remove_upstream_providers(self) -> None: ...
    def get_upstream_providers(self) -> List[BatchProvider]: ...
    @property
    def remove_placeholders(self) -> bool: ...
    def provides(self, key: Key, spec: Spec) -> None:
        """Introduce a new output provided by this :class:`BatchProvider`.

        Implementations should call this in their :func:`setup` method, which
        will be called when the pipeline is build.
        """
    def internal_teardown(self) -> None: ...
    @property
    def spec(self) -> Optional[ProviderSpec]:
        """Get the :class:`ProviderSpec` of this :class:`BatchProvider`."""
    @property
    def provided_items(self) -> List[Key]:
        """Get a list of the keys provided by this :class:`BatchProvider`."""
    def remove_provided(self, request: BatchRequest) -> None:
        """Remove keys from `request` that are provided by this
        :class:`BatchProvider`.
        """
    def request_batch(self, request: BatchRequest) -> Optional[Batch]:
        """Request a batch from this provider."""
    def set_seeds(self, request: BatchRequest) -> None: ...
    def check_request_consistency(self, request: BatchRequest) -> None: ...
    def check_batch_consistency(self, batch: Batch, request: BatchRequest) -> None: ...
    def remove_unneeded(self, batch: Batch, request: BatchRequest) -> None: ...
    def enable_placeholders(self) -> None: ...
    def name(self) -> str: ...
    def __repr__(self) -> str: ...
    def __add__(self, other: Union[BatchProvider, Pipeline]) -> Pipeline:
        """Support ``self + other`` operator. Return a :class:`Pipeline`."""
    def __radd__(
        self, other: Union[BatchProvider, Tuple[Union[BatchProvider, Pipeline], ...]]
    ) -> Pipeline:
        """Support ``other + self`` operator. Return a :class:`Pipeline`."""
