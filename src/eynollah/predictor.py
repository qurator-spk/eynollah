from contextlib import ExitStack
from typing import List, Dict
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
    acting as a shallow replacement for various model types in EynollahModelZoo's
    _loaded dict for each single model
    """
    def __init__(self, logger, model_zoo):
        self.logger = logger
        self.loglevel = logger.parent.level
        self.model_zoo = model_zoo
        ctxt = mp.get_context('spawn')
        self.taskq = ctxt.Queue(maxsize=QSIZE)
        self.resultq = ctxt.Queue(maxsize=QSIZE)
        self.logq = ctxt.Queue(maxsize=QSIZE * 100)
        logging.handlers.QueueListener(
            self.logq, *(
                # as per ocrd_utils.initLogging():
                logging.root.handlers +
                # as per eynollah_cli.main():
                self.logger.parent.handlers
            ), respect_handler_level=False).start()
        self.stopped = ctxt.Event()
        self.closable = ctxt.Manager().list()
        super().__init__(name="EynollahPredictor", daemon=True)

    @property
    def output_shape(self):
        return self({})

    def predict(self, data: dict, verbose=0):
        return self(data)

    def __call__(self, data: dict):
        # unusable as per python/cpython#79967
        #with self.jobid.get_lock():
        # would work, but not public:
        #with self.jobid._mutex:
        with self.joblock:
            self.jobid.value += 1
            jobid = self.jobid.value
        if not len(data):
            self.taskq.put((jobid, data))
            #self.logger.debug("sent shape query task '%d' for model '%s'", jobid, self.name)
            return self.result(jobid)
        with share_ndarray(data) as shared_data:
            self.taskq.put((jobid, shared_data))
            #self.logger.debug("sent prediction task '%d' for model '%s': %s", jobid, self.name, shared_data)
            return self.result(jobid)

    def result(self, jobid):
        while not self.stopped.is_set():
            if jobid in self.results:
                #self.logger.debug("received result for '%d'", jobid)
                result = self.results.pop(jobid)
                if isinstance(result, Exception):
                    raise Exception(f"predictor {self.name} failed for {jobid}") from result
                elif isinstance(result, dict):
                    with ndarray_shared(result) as shared_result:
                        result = np.copy(shared_result)
                    self.closable.append(jobid)
                return result
            try:
                jobid0, result = self.resultq.get(timeout=0.7)
            except mp.queues.Empty:
                continue
            #self.logger.debug("storing results for '%d': '%s'", jobid0, result)
            self.results[jobid0] = result
        raise Exception(f"predictor {self.name} terminated while waiting on results for {jobid}")

    def run(self):
        try:
            self.setup() # fill model_zoo etc
        except Exception as e:
            self.logger.exception("setup failed")
            self.stopped.set()
            return
        closing = {}
        def close_all():
            for jobid in list(self.closable):
                self.closable.remove(jobid)
                closing.pop(jobid).close()
                #self.logger.debug("closed shm for '%d'", jobid)
        while not self.stopped.is_set():
            close_all()
            try:
                TIMEOUT = 4.5 # 1.1 too is greedy - not enough for rebatching
                jobid, shared_data = self.taskq.get(timeout=TIMEOUT)
            except mp.queues.Empty:
                continue
            try:
                # up to what batch size fits into small (8GB) VRAM?
                # (notice we are not listing _resized/_patched models here,
                #  because its inputs/outputs will have varying shapes)
                REBATCH_SIZE = {
                    # small models (448x448)...
                    "col_classifier": 2,
                    "page": 2,
                    "binarization": 4,
                    "enhancement": 4,
                    "reading_order": 4,
                    # medium size (672x672)...
                    "textline": 2,
                    # large models...
                    "table": 1,
                    "region_1_2": 1,
                    "region_fl_np": 1,
                    "region_fl": 1,
                }.get(self.name, 1)
                if not len(shared_data):
                    #self.logger.debug("getting '%d' output shape of model '%s'", jobid, self.name)
                    result = self.model.output_shape
                    self.resultq.put((jobid, result))
                    #self.logger.debug("sent result for '%d': %s", jobid, result)
                else:
                    tasks = [(jobid, shared_data)]
                    batch_size = shared_data['shape'][0]
                    while (not self.taskq.empty() and
                           # climb to target batch size
                           batch_size * len(tasks) < REBATCH_SIZE):
                        jobid0, shared_data0 = self.taskq.get()
                        if len(shared_data0):
                            # add to our batch
                            tasks.append((jobid0, shared_data0))
                        else:
                            # immediately anser
                            self.resultq.put((jobid0, self.model.output_shape))
                    if len(tasks) > 1:
                        self.logger.debug("rebatching %d '%s' tasks of batch size %d",
                                          len(tasks), self.name, batch_size)
                    with ExitStack() as stack:
                        data = []
                        jobs = []
                        for jobid, shared_data in tasks:
                            #self.logger.debug("predicting '%d' with model '%s': %s", jobid, self.name, shared_data)
                            jobs.append(jobid)
                            data.append(stack.enter_context(ndarray_shared(shared_data)))
                        data = np.concatenate(data)
                        result = self.model.predict(data, verbose=0)
                    results = np.split(result, len(jobs))
                    #self.logger.debug("sharing result array for '%d'", jobid)
                    with ExitStack() as stack:
                        for jobid, result in zip(jobs, results):
                            # we don't know when the result will be received,
                            # but don't want to wait either, so track closing
                            # context per job, and wait for closable signal
                            # from client
                            result = stack.enter_context(share_ndarray(result))
                            closing[jobid] = stack.pop_all()
                            self.resultq.put((jobid, result))
                            #self.logger.debug("sent result for '%d': %s", jobid, result)
            except Exception as e:
                self.logger.error("prediction for %s failed: %s", self.name, e.__class__.__name__)
                result = e
                self.resultq.put((jobid, result))
        close_all()
        #self.logger.debug("predictor terminated")

    def load_model(self, *load_args, **load_kwargs):
        assert len(load_args)
        self.name = '_'.join(list(load_args[:1]) + list(load_kwargs))
        self.load_args = load_args
        self.load_kwargs = load_kwargs
        self.start() # call run() in subprocess
        # parent context here
        del self.model_zoo # only in subprocess
        ctxt = mp.get_context('fork') # ocrd.Processor will fork workers
        mngr = ctxt.Manager()
        self.jobid = mngr.Value('i', 0)
        self.joblock = mngr.Lock()
        self.results = mngr.dict()

    def setup(self):
        logging.root.handlers = [logging.handlers.QueueHandler(self.logq)]
        self.logger.setLevel(self.loglevel)
        self.model = self.model_zoo.load_model(*self.load_args, **self.load_kwargs)

    def shutdown(self):
        # do not terminate from forked processor instances
        if mp.parent_process() is None:
            self.stopped.set()
            self.taskq.close()
            self.taskq.cancel_join_thread()
            self.resultq.close()
            self.resultq.cancel_join_thread()
            self.logq.close()
            self.terminate()
        else:
            del self.model

    def __del__(self):
        #self.logger.debug(f"deinit of {self} in {mp.current_process().name}")
        self.shutdown()
