import pytest
from right_shape.invoices import build, dataset


@pytest.fixture(scope="session")
def invoices():
    return build(dataset())
