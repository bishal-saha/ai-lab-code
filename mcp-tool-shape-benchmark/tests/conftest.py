import logging

import pytest
from kestrel.services import serve_in_thread
from tool_shape.kestrel_api import KestrelAPI
from tool_shape.servers import endpoints, tasks
from tool_shape.tasks import SCALE, SEED, build_tasks, dataset

logging.getLogger("httpx").setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def api():
    server, url = serve_in_thread(seed=SEED, scale=SCALE)
    client = KestrelAPI(url)
    yield client
    client.close()
    server.shutdown()


@pytest.fixture(scope="session")
def servers(api):
    return {"endpoints": endpoints.build_server(api), "tasks": tasks.build_server(api)}


@pytest.fixture(scope="session")
def task_list():
    return build_tasks(dataset())
