import gunpowder as gp
import numpy as np


class UnsqueezeChannelDim(gp.BatchFilter):

    """
        Adds a spatial dim of size 1 to the specified axis.

        Args:
            
            array (:class: ArrayKey):
                
                The array key to modify.
    """

    def __init__(self, array, axis=0):
        self.array = array
        self.axis = axis

    def process(self, batch, request):

        if self.array not in batch:
            return
        batch[self.array].data = np.expand_dims(batch[self.array].data, self.axis)
