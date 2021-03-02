from collections import Counter

REGION_ID_TEMPLATE = 'region_%04d'
LINE_ID_TEMPLATE = 'region_%04d_line_%04d'

class EynollahIdCounter():

    def __init__(self, region_idx=0, line_idx=0):
        self._counter = Counter()
        self.set('region', region_idx)
        self.set('line', line_idx)

    def inc(self, name, val=1):
        self._counter.update({name: val})

    def get(self, name):
        return self._counter[name]

    def set(self, name, val):
        self._counter[name] = val

    @property
    def next_region_id(self):
        self.inc('region')
        self.set('line', 0)
        return REGION_ID_TEMPLATE % self._counter['region']

    @property
    def next_line_id(self):
        self.inc('line')
        return LINE_ID_TEMPLATE % (self._counter['region'], self._counter['line'])
