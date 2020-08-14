import gunpowder as gp


class SqueezeSpatialDim(gp.BatchFilter):
    """
        Squeezes the first spatial dim from an array or a graph.
        Will modify request for an extra spatial dim in order to satisfy the
        removal.

        Args:
            
            key (:class: ArrayKey):
                
                The key to modify
    """
    def __init__(self, key):
        self.key = key
        # TODO: add axis argument to specify which dim

    def setup(self):

        upstream_spec = self.get_upstream_provider().spec[self.key]

        spec = upstream_spec.copy()
        if spec.roi is not None:
            spec.roi = gp.Roi(self.__remove_dim(spec.roi.get_begin()),
                              self.__remove_dim(spec.roi.get_shape()))

        if isinstance(self.key, gp.ArrayKey):
            if spec.voxel_size is not None:
                spec.voxel_size = self.__remove_dim(spec.voxel_size)

        self.spec[self.key] = spec
        self.updates(self.key, self.spec[self.key])

    def prepare(self, request):

        if self.key not in request:
            return

        request[self.key].roi = gp.Roi(
            self.__insert_dim(request[self.key].roi.get_begin(), 0),
            self.__insert_dim(request[self.key].roi.get_shape(), 1))

        if isinstance(self.key, gp.ArrayKey):
            if request[self.key].voxel_size is not None:
                request[self.key].voxel_size = self.__insert_dim(
                    request[self.key].voxel_size, 1)

    def process(self, batch, request):
        if self.key not in batch:
            return

        if isinstance(self.key, gp.ArrayKey):
            data = batch[self.key].data
            shape = data.shape
            roi = batch[self.key].spec.roi
            assert shape[-roi.dims()] == 1, "Channel to delete must be size 1," \
                                           "but given shape " + str(shape)

            shape = self.__remove_dim(shape, len(shape) - roi.dims())
            batch[self.key].data = data.reshape(shape)
            batch[self.key].spec.roi = gp.Roi(
                self.__remove_dim(roi.get_begin()),
                self.__remove_dim(roi.get_shape()))
            batch[self.key].spec.voxel_size = \
                self.__remove_dim(batch[self.key].spec.voxel_size)

        if isinstance(self.key, gp.GraphKey):
            roi = batch[self.key].spec.roi

            batch[self.key].spec.roi = gp.Roi(
                self.__remove_dim(roi.get_begin()),
                self.__remove_dim(roi.get_shape()))

            graph = gp.Graph([], [], spec=batch[self.key].spec)
            for node in batch[self.key].nodes:
                new_node = gp.Node(node.id,
                                   node.location[1:],
                                   temporary=node.temporary,
                                   attrs=node.attrs)
                graph.add_node(new_node)

            batch[self.key] = graph

    def __remove_dim(self, a, dim=0):
        return a[:dim] + a[dim + 1:]

    def __insert_dim(self, a, s, dim=0):
        return a[:dim] + (s, ) + a[dim:]
