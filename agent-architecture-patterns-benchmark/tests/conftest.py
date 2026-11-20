import logging

import pytest
from four_shapes.kestrel_api import KestrelAPI
from four_shapes.scenarios import SCALE, SEED, build_scenarios, dataset, directory
from kestrel.services import serve_in_thread

logging.getLogger("httpx").setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def api():
    server, url = serve_in_thread(seed=SEED, scale=SCALE)
    client = KestrelAPI(url)
    yield client
    client.close()
    server.shutdown()


@pytest.fixture(scope="session")
def data():
    return dataset()


@pytest.fixture(scope="session")
def scenarios(data):
    return build_scenarios(data)


@pytest.fixture(scope="session")
def customers(data):
    return directory(data)
