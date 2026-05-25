from gateway.metrics import MetricsCollector


def test_counter_accumulates_per_label_set():
    m = MetricsCollector()
    m.inc_counter("reqs_total", backend="b0")
    m.inc_counter("reqs_total", backend="b0")
    m.inc_counter("reqs_total", backend="b1")
    out = m.render()
    assert 'reqs_total{backend="b0"} 2.0' in out
    assert 'reqs_total{backend="b1"} 1.0' in out


def test_gauge_overwrites():
    m = MetricsCollector()
    m.set_gauge("inflight", 5, backend="b0")
    m.set_gauge("inflight", 2, backend="b0")
    assert 'inflight{backend="b0"} 2.0' in m.render()


def test_histogram_buckets_cumulative_sum_count():
    m = MetricsCollector()
    for v in (0.0003, 0.002, 0.02):
        m.observe("lat_seconds", v)
    out = m.render()
    # le="0.001" should include only the 0.0003 observation (cumulative)
    assert 'lat_seconds_bucket{le="0.001"} 1' in out
    # le="0.0025" should include 0.0003 and 0.002
    assert 'lat_seconds_bucket{le="0.0025"} 2' in out
    assert 'lat_seconds_bucket{le="+Inf"} 3' in out
    assert "lat_seconds_count 3" in out
    assert "lat_seconds_sum " in out


def test_render_has_type_headers():
    m = MetricsCollector()
    m.inc_counter("c_total", help="a counter")
    m.set_gauge("g", 1.0)
    m.observe("h", 0.01)
    out = m.render()
    assert "# TYPE c_total counter" in out
    assert "# HELP c_total a counter" in out
    assert "# TYPE g gauge" in out
    assert "# TYPE h histogram" in out
