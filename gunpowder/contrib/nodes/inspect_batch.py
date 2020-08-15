import gunpowder as gp
import logging
logger = logging.getLogger(__name__)


class InspectBatch(gp.BatchFilter):

    def __init__(self, prefix="InspectBatch"):
        self.prefix = prefix
    
    def prepare(self, request):
        for key, v in request.items():
            logger.info(f"{self.prefix} ======== {key} ROI: {self.spec[key].roi}")

    def process(self, batch, request):
        for key, array in batch.arrays.items():
            logger.info(f"{self.prefix} ======== {key}: \
                          SHAPE: {array.data.shape} ROI: {array.spec.roi}")
        for key, graph in batch.graphs.items():
            logger.info(f"{self.prefix} ======== {key}: \
                          ROI: {graph.spec.roi} GRAPH: {graph}")
