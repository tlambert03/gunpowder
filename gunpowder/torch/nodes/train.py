import logging
import numpy as np

from gunpowder.array import ArrayKey, Array
from gunpowder.ext import torch, tensorboardX, NoSuchModule
from gunpowder.nodes.generic_train import GenericTrain

from typing import Dict, Union, Optional

logger = logging.getLogger(__name__)


class Train(GenericTrain):
    """Torch implementation of :class:`gunpowder.nodes.GenericTrain`.

    Args:

        model (subclass of ``torch.nn.Module``):

            The model to train.

        loss:

            The torch loss to use.

        optimizer:

            The torch optimizer to use.

        inputs (``dict``, ``string`` -> :class:`ArrayKey`):

            Dictionary from the names of input tensors (argument names of the
            ``forward`` method) in the model to array keys.

        loss_inputs (``dict``, ``string`` or ``int`` -> :class:`ArrayKey`):

            Dictionary with the names of input variables to the loss function as
            keys, and ArrayKeys containing the desired data as values. Keys can
            be either strings or integers. If the key is an integer, it will
            be treated as a positional argument to the loss function, a
            string will be used as a named argument

        outputs (``dict``, ``string`` or ``int`` -> :class:`ArrayKey`):

            Dictionary from the names of tensors in the network to array
            keys. If the key is a string, the tensor will be retrieved
            by checking the model for an attribute with they key as its name.
            If the key is an integer, it is interpreted as a tuple index of
            the outputs of the network.
            New arrays will be generated by this node for each entry (if
            requested downstream).

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`, optional):

            Used to set the specs of generated arrays (at the moment only
            ``output``). This is useful to set the ``voxel_size``, for example,
            if they differ from the voxel size of the input arrays. Only fields
            that are not ``None`` in the given :class:`ArraySpec` will be used.

        checkpoint_basename (``string``, optional):

            The basename used for checkpoint files. Defaults to ``model``.

        save_every (``int``, optional):

            After how many iterations to create a checkpoint to store the
            learnt weights.

        log_dir (``string``, optional):

            Directory for saving tensorboard summaries.

        log_every (``int``, optional):

            After how many iterations to write out tensorboard summaries.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss,
        optimizer,
        inputs: Dict[str, ArrayKey],
        outputs: Dict[Union[int, str], ArrayKey],
        loss_inputs: Dict[Union[int, str], ArrayKey],
        gradients: Dict[Union[int, str], ArrayKey] = {},
        array_specs: Optional[Dict[Union[int, str], ArrayKey]] = None,
        checkpoint_basename: str = "model",
        save_every: int = 2000,
        log_dir: str = None,
        log_every: int = 1,
    ):

        # not yet implemented
        gradients = gradients

        super(Train, self).__init__(
            inputs, outputs, gradients, array_specs, spawn_subprocess=False
        )

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.loss_inputs = loss_inputs
        self.checkpoint_basename = checkpoint_basename
        self.save_every = save_every

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = self.model.to(self.device)
        self.iteration = 0

        if not isinstance(tensorboardX, NoSuchModule) and log_dir is not None:
            self.summary_writer = tensorboardX.SummaryWriter(log_dir)
            self.log_every = log_every
        else:
            self.summary_writer = None
            if log_dir is not None:
                logger.warn("log_dir given, but tensorboardX is not installed")

        self.intermediate_layers = {}
        self.register_hooks()

    def register_hooks(self):
        for key in self.outputs:
            if isinstance(key, str):
                layer = getattr(self.model, key)
                layer.register_forward_hook(self.create_hook(key))

    def create_hook(self, key):
        def save_layer(module, input, output):
            self.intermediate_layers[key] = output

        return save_layer

    def retain_gradients(self, request, outputs):
        for array_name, array_key in self.gradients.items():
            if array_key not in request:
                continue
            if isinstance(array_name, int):
                tensor = outputs[array_name]
            elif isinstance(array_name, str):
                tensor = getattr(self.model, array_name)
            else:
                raise RuntimeError(
                    "only ints and strings are supported as gradients keys"
                )
            tensor.retain_grad()

    def start(self):

        checkpoint, self.iteration = self._get_latest_checkpoint(
            self.checkpoint_basename
        )

        if checkpoint is not None:

            logger.info("Resuming training from iteration %d", self.iteration)
            logger.info("Loading %s", checkpoint)

            checkpoint = torch.load(checkpoint)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        else:

            logger.info("Starting training from scratch")

        logger.info("Using device %s", self.device)

    def train_step(self, batch, request):

        inputs = self.__collect_provided_inputs(batch)
        requested_outputs = self.__collect_requested_outputs(request)

        device_inputs = {
            k: torch.as_tensor(v, device=self.device) for k, v in inputs.items()
        }

        self.optimizer.zero_grad()
        model_outputs = self.model(**device_inputs)
        if isinstance(model_outputs, tuple):
            outputs = {i: model_outputs[i] for i in range(len(model_outputs))}
        elif isinstance(model_outputs, torch.Tensor):
            outputs = {0: model_outputs}
        else:
            raise RuntimeError(
                "Torch train node only supports return types of tuple",
                f"and torch.Tensor from model.forward(). not {type(model_outputs)}",
            )
        outputs.update(self.intermediate_layers)

        for array_key, array_name in requested_outputs.items():
            spec = self.spec[array_key].copy()
            spec.roi = request[array_key].roi
            batch.arrays[array_key] = Array(
                outputs[array_name].cpu().detach().numpy(), spec
            )

        loss_inputs = self.__collect_provided_targets(batch)

        device_loss_inputs = {
            k: torch.as_tensor(v, device=self.device) for k, v in loss_inputs.items()
        }
        device_loss_args = []
        for i in range(len(device_loss_inputs)):
            if i in device_loss_inputs:
                device_loss_args.append(device_loss_inputs.pop(i))
            else:
                break
        device_loss_kwargs = {}
        for k, v in device_loss_inputs:
            if isinstance(k, str):
                device_loss_kwargs[k] = device_loss_inputs.pop(k)
        assert (
            len(device_loss_inputs) == 0
        ), f"Not all loss inputs could be interpreted. Failed keys: {device_loss_inputs.keys()}"

        self.retain_gradients(request, outputs)

        logger.debug("model outputs: %s", outputs)
        logger.debug(f"loss_inputs: {device_loss_args}, {device_loss_kwargs}")
        loss = self.loss(*device_loss_args, **device_loss_kwargs)
        loss.backward()
        self.optimizer.step()

        for array_name, array_key in self.gradients.items():
            if array_key not in request:
                continue
            if isinstance(array_name, int):
                tensor = outputs[array_name]
            elif isinstance(array_name, str):
                tensor = getattr(self.model, array_name)
            else:
                raise RuntimeError(
                    "only ints and strings are supported as gradients keys"
                )
            spec = self.spec[array_key].copy()
            spec.roi = request[array_key].roi
            batch.arrays[array_key] = Array(tensor.grad.cpu().numpy(), spec)

        for array_key, array_name in requested_outputs.items():
            spec = self.spec[array_key].copy()
            spec.roi = request[array_key].roi
            batch.arrays[array_key] = Array(
                outputs[array_name].cpu().detach().numpy(), spec
            )

        batch.loss = loss.cpu().detach().numpy()
        self.iteration += 1
        batch.iteration = self.iteration

        if batch.iteration % self.save_every == 0:

            checkpoint_name = self._checkpoint_name(
                self.checkpoint_basename, batch.iteration
            )

            logger.info("Creating checkpoint %s", checkpoint_name)

            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                checkpoint_name,
            )

        if self.summary_writer and batch.iteration % self.log_every == 0:
            self.summary_writer.add_scalar("loss", batch.loss, batch.iteration)

    def __collect_requested_outputs(self, request):

        array_outputs = {}

        for output_name, array_key in self.outputs.items():
            if array_key in request:
                array_outputs[array_key] = output_name

        return array_outputs

    def __collect_provided_inputs(self, batch):

        return self.__collect_provided_arrays(self.inputs, batch)

    def __collect_provided_targets(self, batch):

        return self.__collect_provided_arrays(self.targets, batch)

    def __collect_provided_arrays(self, reference, batch):

        arrays = {}

        for array_name, array_key in reference.items():
            if isinstance(array_key, ArrayKey):
                if array_key in batch.arrays:
                    arrays[array_name] = batch.arrays[array_key].data
                else:
                    logger.warn(
                        "batch does not contain %s, array %s will not " "be set",
                        array_key,
                        array_name,
                    )
            elif isinstance(array_key, np.ndarray):
                arrays[array_name] = array_key
            elif isinstance(array_key, str):
                arrays[array_name] = getattr(batch, array_key)
            else:
                raise Exception(
                    "Unknown network array key {}, can't be given to "
                    "network".format(array_key)
                )

        return arrays
