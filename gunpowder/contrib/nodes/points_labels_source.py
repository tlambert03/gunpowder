import gunpowder as gp
from gunpowder import BatchProvider
import numpy as np
from gunpowder.profiling import Timing
import logging
logger = logging.getLogger(__name__)

class PointsLabelsSource(BatchProvider):
    ''' Using a gicen numpy array of pionts provide random points 
    to either a non-spatial Array or Graph and if wanted, provide
    the correseponding labels. This has similar functinailty to 
    CSVPointsSource, but the points are not read from a csv, can 
    give points can be in a non-spatial array, and can provide labels.

    Args:

        points (:class: `ArrayKey` or `GraphKey`):

            The key of the points correseponding to the gunpowder array
            or graph key. If points is an array key it will directly load the given 
            number of points into the array. If points is a graph key it will
            provide points in a ROI, in similar functinailty to CSVPointsSource.

        data (:class:`numpy array`):
            
            The data correseponding to the points. If points is an ArrayKey,
            data should be the actual data values (not the point locations). If
            points is a graphkey the data should be the point locations.

        labels (:class:`ArrayKey`, optional):
            
            The gunpowder ArrayKey for the labels for each point.

        label_data (:class: `Numpy Array`, optional):

            The actual label for each point, will be loaded into the labels key.

        num_points (:class: `int`, default=1):

            The number of points to return. If given an array key this will 
            specify the number of points that will be randomly selected to be 
            put into the points ArrayKey. If points is a GraphKey it does not 
            affect the number of points. 

        points_spec (:class:`GraphSpec` or `ArraySpec`, optional):

            An optional :class:`GraphSpec` or :class:`ArraySpec` to overwrite the points specs
            automatically determined from the points data. This is useful to set
            the :class:`Roi` manually.

        labels_spec (`ArraySpec`, optional):

            An optional :class:`ArraySpec` to overwrite the labels specs
            automatically given a voxel size of 1. This is useful to set
            the voxel_size manually.

        scale (scalar or array-like):

            An optional scaling to apply to the coordinates of the given points data.
            This is useful if the points refer to voxel positions to convert them to world units.
    '''

    def __init__(self,
                 points, 
                 data, 
                 labels=None, 
                 label_data=None, 
                 num_points=1, 
                 points_spec=None, 
                 labels_spec=None, 
                 scale=None):

        self.points = points
        self.labels = labels
        self.data = data
        self.label_data = label_data
        self.num_points = num_points
        self.points_spec = points_spec
        self.labels_spec = labels_spec
        self.scale = scale
        
        # Apply scale to given data
        if scale is not None:
            self.data = self.data * scale

    def setup(self):
        
        self.ndims = self.data.shape[1]

        if self.points_spec is not None:
            self.provides(self.points, self.points_spec)
        elif isinstance(self.points, gp.ArrayKey):
            self.provides(self.points, gp.ArraySpec(voxel_size=((1,))))
        elif isinstance(self.points, gp.GraphKey):
            print(self.ndims)
            min_bb = gp.Coordinate(np.floor(np.amin(self.data[:, :self.ndims], 0)))
            max_bb = gp.Coordinate(np.ceil(np.amax(self.data[:, :self.ndims], 0)) + 1)

            roi = gp.Roi(min_bb, max_bb - min_bb)
            logger.debug(f"Bounding Box: {roi}")

            self.provides(self.points, gp.GraphSpec(roi=roi))

        if self.labels is not None:
            assert isinstance(self.labels, gp.ArrayKey), \
                   f"Label key must be an ArrayKey, \
                     was given {type(self.labels)}"

            if self.labels_spec is not None:
                self.provides(self.labels, self.labels_spec)
            else:
                self.provides(self.labels, gp.ArraySpec(voxel_size=((1,))))

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = gp.Batch()

        # If a Array is requested then we will randomly choose
        # the number of requested points
        if isinstance(self.points, gp.ArrayKey):
            points = np.random.choice(self.data.shape[0], self.num_points)
            data = self.data[points][np.newaxis]
            if self.scale is not None:
                data = data * self.scale
            if self.label_data is not None:
                labels = self.label_data[points]
            batch[self.points] = gp.Array(data, self.spec[self.points])

        else:
            # If a graph is request we must select points within the 
            # request ROI

            min_bb = request[self.points].roi.get_begin()
            max_bb = request[self.points].roi.get_end()

            logger.debug(
                "Points source got request for %s",
                request[self.points].roi)

            point_filter = np.ones((self.data.shape[0],), dtype=np.bool)
            for d in range(self.ndims):
                point_filter = np.logical_and(point_filter, self.data[:, d] >= min_bb[d])
                point_filter = np.logical_and(point_filter, self.data[:, d] < max_bb[d])

            points_data, labels = self._get_points(point_filter)
            logger.debug(f"Found {len(points_data)} points")
            points_spec = gp.GraphSpec(roi=request[self.points].roi.copy())
            batch.graphs[self.points] = gp.Graph(points_data, [], points_spec)
        
        # Labels will always be an Array
        if self.label_data is not None:
            batch[self.labels] = gp.Array(labels, self.spec[self.labels])

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def _get_points(self, point_filter):
        filtered = self.data[point_filter]

        if self.label_data is not None:
            filtered_labels = self.labels[point_filter]
        else:
            filtered_labels = None

        ids = np.arange(len(self.data))[point_filter]

        return (
            [gp.Node(id=i, location=p)
                for i, p in zip(ids, filtered)],
            filtered_labels
        )
