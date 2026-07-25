# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/Jott2121/rag-guard/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                          |    Stmts |     Miss |   Cover |   Missing |
|------------------------------ | -------: | -------: | ------: | --------: |
| rag\_guard/\_\_init\_\_.py    |        4 |        0 |    100% |           |
| rag\_guard/cli.py             |       22 |        3 |     86% | 27-28, 32 |
| rag\_guard/config.py          |       25 |        5 |     80% |29-31, 45-46 |
| rag\_guard/corpus.py          |       45 |        3 |     93% | 50, 53-54 |
| rag\_guard/evaluate.py        |       24 |        0 |    100% |           |
| rag\_guard/guard.py           |       29 |        0 |    100% |           |
| rag\_guard/hooklog.py         |       45 |        0 |    100% |           |
| rag\_guard/index.py           |       54 |        7 |     87% |16-18, 39-40, 63-64 |
| rag\_guard/pipeline.py        |       26 |        0 |    100% |           |
| rag\_guard/providers.py       |        8 |        0 |    100% |           |
| rag\_guard/rebuild\_health.py |       46 |        0 |    100% |           |
| rag\_guard/reindex.py         |       45 |        3 |     93% | 76-77, 81 |
| rag\_guard/retriever.py       |       50 |        0 |    100% |           |
| rag\_guard/service.py         |       22 |        1 |     95% |        48 |
| rag\_guard/sqlite\_index.py   |      164 |       13 |     92% |52, 115-116, 206-207, 209-210, 214-217, 240-241 |
| rag\_guard/stamps.py          |       15 |        0 |    100% |           |
| rag\_guard/webverify.py       |       44 |        0 |    100% |           |
| **TOTAL**                     |  **668** |   **35** | **95%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/Jott2121/rag-guard/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/Jott2121/rag-guard/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Jott2121/rag-guard/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/Jott2121/rag-guard/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FJott2121%2Frag-guard%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/Jott2121/rag-guard/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.