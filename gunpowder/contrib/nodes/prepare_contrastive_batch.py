import gunpowder as gp
import numpy as np


class PrepareContrastiveBatch(gp.BatchFilter):
    """
        Prepares a batch of points for contrastive training.
        
        Finds the intersecting nodes, and then converts them
        to unit locations by subtracting the offset and dividing
        by raw's voxel size. Also adds the batch shape to the points.

        If the locations should be 2D but aren't the is_2d flag
        should be true and will only return the last 2 dims of 
        the locations. 
    """
    def __init__(
            self,
            raw_0, raw_1,
            points_0, points_1,
            locations_0, locations_1,
            is_2d):
        self.raw_0 = raw_0
        self.raw_1 = raw_1
        self.points_0 = points_0
        self.points_1 = points_1
        self.locations_0 = locations_0
        self.locations_1 = locations_1
        self.is_2d = is_2d

    def setup(self):
        self.provides(
            self.locations_0,
            gp.ArraySpec(nonspatial=True))
        self.provides(
            self.locations_1,
            gp.ArraySpec(nonspatial=True))

    def process(self, batch, request):

        ids_0 = set([n.id for n in batch[self.points_0].nodes])
        ids_1 = set([n.id for n in batch[self.points_1].nodes])
        common_ids = ids_0.intersection(ids_1)

        locations_0 = []
        locations_1 = []
        # get list of only xy locations
        # locations are in voxels, relative to output roi
        points_roi = request[self.points_0].roi
        voxel_size = batch[self.raw_0].spec.voxel_size
        for i in common_ids:
            location_0 = np.array(batch[self.points_0].node(i).location)
            location_1 = np.array(batch[self.points_1].node(i).location)

            location_0 -= points_roi.get_begin()
            location_1 -= points_roi.get_begin()
            location_0 /= voxel_size
            location_1 /= voxel_size
            locations_0.append(location_0)
            locations_1.append(location_1)
        
        locations_0 = np.array(locations_0, dtype=np.float32)
        locations_1 = np.array(locations_1, dtype=np.float32)
        if self.is_2d:
            locations_0 = locations_0[:, 1:]
            locations_1 = locations_1[:, 1:]
        locations_0 = locations_0[np.newaxis]
        locations_1 = locations_1[np.newaxis]

        # create point location arrays (with batch dimension)
        batch[self.locations_0] = gp.Array(
            locations_0, self.spec[self.locations_0])
        batch[self.locations_1] = gp.Array(
            locations_1, self.spec[self.locations_1])

        # add batch dimension to raw
        batch[self.raw_0].data = batch[self.raw_0].data[np.newaxis, :]
        batch[self.raw_1].data = batch[self.raw_1].data[np.newaxis, :]

        # make sure raw is float32
        batch[self.raw_0].data = batch[self.raw_0].data.astype(np.float32)
        batch[self.raw_1].data = batch[self.raw_1].data.astype(np.float32)
