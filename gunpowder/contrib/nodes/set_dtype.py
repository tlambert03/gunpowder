import gunpowder as gp

class SetDtype(gp.BatchFilter):
    """
       Set the datatype of a Array or Graph with the given type.

       Args:
            
            key (:class: `ArrayKey` or `GraphKey`):

                The key to modify

            dtype (:class: `Numpy Dtype`):

                The dtype to change to.
                
    """

    def __init__(self, key, dtype):
        self.key = key
        self.dtype = dtype

    def setup(self):
        self.enable_autoskip()
        spec = self.spec[self.key].copy()
        spec.dtype = self.dtype
        self.updates(self.key, spec) 

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.key] = request[self.key].copy()
        return deps

    def process(self, batch, request):
        data = batch[self.key]
        data.data = data.data.astype(self.dtype)
        data.spec.dtype = self.dtype
