import gunpowder as gp
import numpy as np


class FillLocations(gp.BatchFilter):
    """
        Helper method to transfer points into a non-spatial array.
        Converts locations into unit locations that start at 0.
        Ex: [120, 500, 360] with voxel_size (10, 10, 10) and ponit
        roi (20:140, 300:600, 300:600)

        Then the unit location will be:

                ([120, 500, 360] - (20, 300, 300)) / (10, 10, 10)
            =   (10, 20, 6)

        This is useful because we often want to use the points directly
        with the underlying data they are points of eg Training. There
        we no longer worry about voxel_size and so it is useful to deal
        with it here.

        Args:

            points (:class: `gp.Graph`):
                
                The graph containing the points that will be transfered.

            locations (:class: `gp.Graph`):

                The non-spatial array to transfer the point locations to.

            voxel_size (:class: `gp.Array`):

                The voxel size to use to get unit locations of the points.

            is_2d (:class: `bool`):

                Whether the given point locations should actually be 2d.
                This is only useful when the points in the graph have 3
                dims, but the 1st dim is a placeholder or represents the 
                sample dimension. Seting is_2d to true will only add the
                last two coordinates to the locations array.

            max_points (:class: `int`):
                
                The maximum number of points to add to the locations array.
                If max < the number of points they will be randomly chosen.

    """

    def __init__(
            self,
            points,
            locations,
            voxel_size=None,
            is_2d=False,
            max_points=None):
        self.points = points
        self.locations = locations
        self.voxel_size = voxel_size
        self.is_2d = is_2d
        self.max_points = max_points

    def setup(self):
        self.provides(
            self.locations,
            gp.ArraySpec(nonspatial=True))

    def process(self, batch, request):

        locations = []
        # get list of only xy locations
        # locations are in voxels, relative to output roi
        points_roi = request[self.points].roi
        if self.voxel_size is None:
            self.voxel_size = (1,) * points_roi.dims()

        for i, node in enumerate(batch[self.points].nodes):
            if self.max_points is not None and i > self.max_points - 1:
                break

            location = node.location
            location -= points_roi.get_begin()
            location /= self.voxel_size
            locations.append(location)
        
        locations = np.array(locations, dtype=np.float32)
        print(locations)
        if self.is_2d:
            locations = locations[:, 1:]

        # create point location arrays 
        batch[self.locations] = gp.Array(
            locations, self.spec[self.locations])
