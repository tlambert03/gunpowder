from gunpowder import BatchProvider
import numpy as np
from gunpowder.nodes.hdf5like_source_base import Hdf5LikeSource
import copy
import logging
logger = logging.getLogger(__name__)


class RandomSourceGenerator:
    '''Create random points in a provided ROI with the given density.

    Args:

        repetitions (int):

            How many times the generator will be used in a pipeline.
            Only the first request to the RandomSource will have a 
            random source chosen. Future calls will use the same source.

        probabilities (`numpy array`):

           The probability of each branch getting chosen. Any probabilities
           will be normalized to the sum of one. This way you can give the
           volume of each source and it will each volume proportionatly to
           it's size.
    '''
    def __init__(self, num_sources, probabilities=None, repetitions=1):
        self.repetitions = repetitions
        self.num_sources = num_sources
        self.probabilities = probabilities

        # automatically normalize probabilities to sum to 1
        if self.probabilities is not None:
            self.probabilities = [float(x) / np.sum(probabilities) 
                                  for x in self.probabilities]

        if self.probabilities is not None:
            assert self.num_sources == len(
                self.probabilities), "if probabilities are specified, they " \
                                     "need to be given for each batch " \
                                     "provider added to the RandomProvider"
        self.iteration = 0

    def get_random_source(self):
        '''Get a randomly chosen source. If `repetitions` is larger than 1,
        the previously chosen source will be given. 
        '''
        if self.iteration % self.repetitions == 0:
            
            self.choice = np.random.choice(list(range(self.num_sources)),
                                           p=self.probabilities)
        self.iteration += 1
        return self.choice


class RandomMultiBranchSource(BatchProvider):
    '''Randomly selects one of the upstream providers based on a RandomSourceGenerator::
        (a + b + c) + RandomProvider()
    will create a provider that randomly relays requests to providers ``a``,
    ``b``, or ``c``. Array and point keys of ``a``, ``b``, and ``c`` should be
    the same. 

    The RandomSourceGenerator will ensure that multiple branches of
    inputs will choose the same source.

    Args:

        random_source_generator (:class: RandomSourceGenerator):

            The random source generator to sync random choice.  
    '''

    def __init__(self, random_source_generator):
        self.random_source_generator = random_source_generator

    def setup(self):

        assert len(self.get_upstream_providers()) > 0,\
            "at least one batch provider must be added to the RandomProvider"

        common_spec = None

        # advertise outputs only if all upstream providers have them
        for provider in self.get_upstream_providers():

            if common_spec is None:
                common_spec = copy.deepcopy(provider.spec)
            else:
                for key, spec in list(common_spec.items()):
                    if key not in provider.spec:
                        del common_spec[key]

        for key, spec in common_spec.items():
            self.provides(key, spec)

    def provide(self, request):
        source_idx = self.random_source_generator.get_random_source()
        source = self.get_upstream_providers()[source_idx]
        if isinstance(source, Hdf5LikeSource):
            logger.debug(f"Dataset chosen: {source.datasets}, {self.random_source_generator.iteration}")
        return source.request_batch(request)
