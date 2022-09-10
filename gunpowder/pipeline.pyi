# flake8: noqa
from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, TypeVar, Union
from gunpowder.nodes import BatchProvider
from gunpowder.batch_request import BatchRequest
from gunpowder.batch import Batch
from gunpowder.provider_spec import ProviderSpec

T = TypeVar("T")

class PipelineSetupError(Exception):
    def __init__(self, provider: BatchProvider) -> None: ...

class PipelineTeardownError(Exception):
    def __init__(self, provider: BatchProvider) -> None: ...

class PipelineRequestError(Exception):
    def __init__(self, pipeline: Pipeline, request: BatchRequest) -> None: ...

class Pipeline:
    children: List[Pipeline]
    output: BatchProvider
    initialized: bool

    def __init__(self, node: BatchProvider) -> None: ...
    def traverse(
        self, callback: Callable[[Pipeline], T], reverse: bool = False
    ) -> List[T]:
        """Visit every node in the pipeline recursively (either from root to
        leaves of from leaves to the root if ``reverse`` is true). ``callback``
        will be called for each node encountered."""
    def copy(self) -> Pipeline:
        """Make a shallow copy of the pipeline."""
    def setup(self) -> None:
        """Connect all batch providers in the pipeline and call setup for
        each, from source to sink."""
    def internal_teardown(self) -> None:
        """Call teardown on each batch provider in the pipeline and disconnect
        all nodes."""
    def request_batch(self, request: BatchRequest) -> Batch:
        """Request a batch from the pipeline."""
    @property
    def spec(self) -> Optional[ProviderSpec]: ...
    def __add__(self, other: Union[BatchProvider, Pipeline]) -> Pipeline: ...
    def __radd__(
        self, other: Tuple[Union[BatchProvider, Pipeline], ...]
    ) -> Pipeline: ...
    def __repr__(self) -> str: ...
