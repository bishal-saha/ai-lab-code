import pytest
from cut_here.corpus import build, dataset
from cut_here.questions import build_questions


@pytest.fixture(scope="session")
def corpus():
    return build(dataset())


@pytest.fixture(scope="session")
def docs(corpus):
    return corpus[0]


@pytest.fixture(scope="session")
def questions(docs):
    return build_questions(docs)
