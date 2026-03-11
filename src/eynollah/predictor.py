import threading
from contextlib import ExitStack
from functools import lru_cache
from typing import List
import logging
import logging.handlers
import multiprocessing as mp
import numpy as np

from .utils.shm import share_ndarray, ndarray_shared

QSIZE = 200


class Predictor(mp.context.SpawnProcess):
    """
    singleton subprocess solely responsible for prediction with TensorFlow,
    communicates with any number of worker processes,
    acts as a shallow replacement for EynollahModelZoo
    """
    class SingleModelPredictor:
        """
        acts as a shallow replacement for EynollahModelZoo
        """
        def __init__(self, predictor: 'Predictor', model: str):
            self.predictor = predictor
            self.model = model
        @property
        def name(self):
            return self.model
        @property
        def output_shape(self):
            return self.predictor(self.model, {})
        def predict(self, data: dict, verbose=0):
            return self.predictor(self.model, data)

    def __init__(self, logger, model_zoo):
        self.logger = logger
        self.loglevel = logger.level
        self.model_zoo = model_zoo
        ctxt = mp.get_context('spawn')
        self.jobid = ctxt.Value('i', 0)
        self.taskq = ctxt.Queue(maxsize=QSIZE)
        self.resultq = ctxt.Queue(maxsize=QSIZE)
        self.logq = ctxt.Queue(maxsize=QSIZE * 100)
        log_listener = logging.handlers.QueueListener(
            self.logq, *self.logger.handlers,
            respect_handler_level=True).start()
        self.stopped = ctxt.Event()
        ctxt = mp.get_context('fork') # ocrd.Processor will fork workers
        self.results = ctxt.Manager().dict()
        self.closable = ctxt.Manager().list()
        super().__init__(name="EynollahPredictor", daemon=True)

    @lru_cache
    def get(self, model: str):
        return Predictor.SingleModelPredictor(self, model)

    def __call__(self, model: str, data: dict):
        with self.jobid.get_lock():
            self.jobid.value += 1
            jobid = self.jobid.value
        if not len(data):
            self.taskq.put((jobid, model, data))
            return self.result(jobid)
        with share_ndarray(data) as shared_data:
            self.taskq.put((jobid, model, shared_data))
            #self.logger.debug("sent task '%d'", jobid)
            return self.result(jobid)

    def result(self, jobid):
        while not self.stopped.is_set():
            if jobid in self.results:
                #self.logger.debug("received result for '%d'", jobid)
                result = self.results.pop(jobid)
                if isinstance(result, Exception):
                    raise Exception(f"predictor failed for {jobid}") from result
                elif isinstance(result, dict):
                    with ndarray_shared(result) as shared_result:
                        result = np.copy(shared_result)
                    self.closable.append(jobid)
                return result
            try:
                jobid, result = self.resultq.get(timeout=0.7)
            except mp.queues.Empty:
                continue
            #self.logger.debug("storing results for '%d'", jobid)
            self.results[jobid] = result
        raise Exception(f"predictor terminated while waiting on results for {jobid}")

    def run(self):
        try:
            self.setup() # fill model_zoo etc
        except Exception as e:
            self.logger.exception("setup failed")
            self.stopped.set()
        closing = {}
        def close_all():
            for jobid in list(self.closable):
                self.closable.remove(jobid)
                closing.pop(jobid).close()
                #self.logger.debug("closed shm for '%d'", jobid)
        while not self.stopped.is_set():
            close_all()
            try:
                jobid, model, shared_data = self.taskq.get(timeout=1.1)
            except mp.queues.Empty:
                continue
            #self.logger.debug("predicting '%d'", jobid)
            try:
                model = self.model_zoo.get(model)
                if not len(shared_data):
                    # non-input msg: model query
                    result = model.output_shape
                else:
                    with ndarray_shared(shared_data) as data:
                        result = model.predict(data, verbose=0)
                    #self.logger.debug("sharing result array for '%d'", jobid)
                    with ExitStack() as stack:
                        # we don't know when the result will be received,
                        # but don't want to wait either, so 
                        result = stack.enter_context(share_ndarray(result))
                        closing[jobid] = stack.pop_all()
            except Exception as e:
                self.logger.error("prediction failed: %s", e.__class__.__name__)
                result = e
            self.resultq.put((jobid, result))
            #self.logger.debug("sent result for '%d'", jobid)
        close_all()
        #self.logger.debug("predictor terminated")

    def load_models(self, *loadable: List[str]):
        self.loadable = loadable
        self.start()
        # parent context here
        del self.model_zoo

    def setup(self):
        logging.root.handlers = [logging.handlers.QueueHandler(self.logq)]
        self.logger.setLevel(self.loglevel)
        self.model_zoo.load_models(*self.loadable)

    def shutdown(self):
        # do not terminate from forked processor instances
        if mp.parent_process() is None:
            self.stopped.set()
            self.terminate()
            self.logq.close()
            self.taskq.close()
            self.taskq.cancel_join_thread()
            self.resultq.close()
            self.resultq.cancel_join_thread()
        else:
            self.model_zoo.shutdown()

    def __del__(self):
        #self.logger.debug(f"deinit of {self} in {mp.current_process().name}")
        self.shutdown()
