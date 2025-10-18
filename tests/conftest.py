import pytest

from harmony_proxy import metrics


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.configure(True)
    metrics.reset_for_test()
    yield
    metrics.reset_for_test()
    metrics.configure(True)
