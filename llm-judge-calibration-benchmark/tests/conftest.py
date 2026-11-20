import pytest
from judge_the_judge.cases import build


@pytest.fixture(scope="session")
def built():
    return build()


@pytest.fixture(scope="session")
def cases(built):
    return built[0]


@pytest.fixture(scope="session")
def pairs(built):
    return built[1]
