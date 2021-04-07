from collections import Counter

REGION_ID_TEMPLATE = 'region_%04d'
LINE_ID_TEMPLATE = 'region_%04d_line_%04d'

class EynollahIdCounter():

    def __init__(self, region_idx=0, line_idx=0):
        self._counter = Counter()
        self._inital_region_idx = region_idx
        self._inital_line_idx = line_idx
        self.reset()

    def reset(self):
        self.set('region', self._inital_region_idx)
        self.set('line', self._inital_line_idx)

    def inc(self, name, val=1):
        self._counter.update({name: val})

    def get(self, name):
        return self._counter[name]

    def set(self, name, val):
        self._counter[name] = val

    def region_id(self, region_idx=None):
        if region_idx is None:
            region_idx = self._counter['region']
        return REGION_ID_TEMPLATE % region_idx

    def line_id(self, region_idx=None, line_idx=None):
        if region_idx is None:
            region_idx = self._counter['region']
        if line_idx is None:
            line_idx = self._counter['line']
        return LINE_ID_TEMPLATE % (region_idx, line_idx)

    @property
    def next_region_id(self):
        self.inc('region')
        self.set('line', 0)
        return self.region_id()

    @property
    def next_line_id(self):
        self.inc('line')
        return self.line_id()
