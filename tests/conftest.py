import pytest

from harmony_proxy import metrics, trace


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.configure(True)
    metrics.reset_for_test()
    yield
    metrics.reset_for_test()
    metrics.configure(True)


@pytest.fixture(autouse=True)
def reset_trace():
    trace.reset_for_test()
    yield
    trace.reset_for_test()
