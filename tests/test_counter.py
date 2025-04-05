from eynollah.utils.counter import EynollahIdCounter

def test_counter_string():
    c = EynollahIdCounter()
    assert c.next_region_id == 'region_0001'
    assert c.next_region_id == 'region_0002'
    assert c.next_line_id == 'region_0002_line_0001'
    assert c.next_region_id == 'region_0003'
    assert c.next_line_id == 'region_0003_line_0001'
    assert c.region_id(999) == 'region_0999'
    assert c.line_id(999, 888) == 'region_0999_line_0888'

def test_counter_init():
    c = EynollahIdCounter(region_idx=2)
    assert c.get('region') == 2
    c.inc('region')
    assert c.get('region') == 3
    c.reset()
    assert c.get('region') == 2

def test_counter_methods():
    c = EynollahIdCounter()
    assert c.get('region') == 0
    c.inc('region', 5)
    assert c.get('region') == 5
    c.set('region', 10)
    assert c.get('region') == 10
    c.inc('region', -9)
    assert c.get('region') == 1

