import gunpowder as gp
import numpy as np
import skimage.filters as filters
from collections.abc import Iterable


class Blur(gp.BatchFilter):
    '''Add random noise to an array. Uses the scikit-image function
    skimage.filters.gaussian
    See scikit-image documentation for more information.

    Args:

        array (:class:`ArrayKey`):

            The array to blur.

        sigma (``scalar or list``):

            The st. dev to use for the gaussian filter. If scalar it will be 
            projected to match the number of ROI dims. If give an list or numpy
            array, it must match the number of ROI dims.

    '''

    def __init__(self, array, sigma=1):
        self.array = array
        self.sigma = sigma.copy()
        self.filter_radius = np.ceil(np.array(self.sigma) * 3)

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def prepare(self, request):
        deps = gp.BatchRequest()
        spec = request[self.array].copy()
        
        if isinstance(self.sigma, Iterable):
            assert spec.roi.dims() == len(self.sigma), \
                   ("Dimensions given for sigma (" 
                   + str(len(self.sigma)) + ") is not equal to the ROI dims (" 
                   + str(spec.roi.dims()) + ")")
        else:
            self.filter_radius = [self.filter_radius * spec.voxel_size[dim]
                                  for dim in range(spec.roi.dims())]

        self.grow_amount = gp.Coordinate([radius
                                          for radius in self.filter_radius])

        grown_roi = spec.roi.grow(
            self.grow_amount,
            self.grow_amount)
        grown_roi = grown_roi.snap_to_grid(self.spec[self.array].voxel_size)

        spec.roi = grown_roi
        deps[self.array] = spec
        return deps

    def process(self, batch, request):

        raw = batch.arrays[self.array]
        roi = raw.spec.roi
        
        if not isinstance(self.sigma, Iterable):
            sigma = [0 for dim in range(len(raw.data.shape) - roi.dims())] \
                + [self.sigma for dim in range(roi.dims())]
        else: 
            sigma = [0 for dim in range(len(raw.data.shape) - roi.dims())] \
                + self.sigma 

        raw.data = filters.gaussian(raw.data, sigma=sigma,
                                    mode='constant', preserve_range=True,
                                    multichannel=False)
        
        batch[self.array].crop(request[self.array].roi)
