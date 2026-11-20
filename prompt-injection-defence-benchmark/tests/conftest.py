import logging

import pytest
from kestrel.services import serve_in_thread
from poisoned_ticket.kestrel_api import KestrelAPI
from poisoned_ticket.scenarios import SCALE, SEED, build_scenarios, dataset

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
