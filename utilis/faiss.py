#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/2/22 22:32
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   faiss.py
# @Desc     :   

from faiss import Index, index_factory, METRIC_L2, write_index, read_index
from numpy import random, ndarray, array
from os import path
from pandas import DataFrame


def faiss_index_creator(dimensions: int, method: str = "Flat") -> Index:
    """ Creates a Faiss index based on the given dimension and method. """
    index = index_factory(dimensions, method, METRIC_L2)
    # index = IndexFlatL2(dimension)
    print(f"The dimensions of the index are {index.d}.")
    return index


def faiss_index_adder(index, vectors: DataFrame) -> None:
    """ Adds vectors to a Faiss index. """
    print(f"The shape of the param vectors with Dataframe is {vectors.shape}.")
    index.add(vectors)


def faiss_index_search(index, vectors: DataFrame, top_n: int) -> tuple[ndarray, ndarray]:
    """ Searches a Faiss index for the k nearest neighbors of the given query vectors. """
    query = vectors.iloc[0].values.reshape(1, -1)
    print(f"The shape of the extracted query is {query.shape}.")

    distances, indices = index.search(query, top_n)
    return distances, indices


def faiss_index_remover(index, ids: list[int]) -> None:
    """ Removes vectors from a Faiss index based on their ids. """
    index.remove_ids(array(ids))


def faiss_index_storager(index, file_name: str):
    """ Saves a Faiss index to a file. """
    write_index(index, f"{file_name}.faiss")


def faiss_index_dropper(index) -> None:
    """ Drops a Faiss index. """
    index.reset()


def faiss_index_loader(file_name: str) -> Index:
    """ Loads a Faiss index from a file. """
    return read_index(f"{file_name}.faiss")


class SeedRandom(object):
    """ A class to seed the random number controller. """

    def __init__(self, seed: int):
        self._seed = seed

    def __enter__(self):
        self._state = random.get_state()
        random.seed(self._seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        random.set_state(self._state)

    def __repr__(self):
        return f"Random seed is {self._seed}."


class SeedNumpy(object):
    """ A class to seed the numpy random number controller. """

    def __init__(self, seed: int):
        self._seed = seed

    def __enter__(self):
        self._state = random.get_state()
        random.seed(self._seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        random.set_state(self._state)

    def __repr__(self):
        return f"Numpy seed is {self._seed}."


def file_size_getter(file_path: str) -> str:
    """ Returns the size of a file in bytes. """
    file_path = f"{file_path}.faiss"
    size: int = path.getsize(file_path)

    # Convert bytes to KB, MB, GB, etc.
    if size < 1024:
        return f"{size} Bytes"
    elif size < 1024 ** 2:
        return f"{size / 1024:.2f} KB"
    elif size < 1024 ** 3:
        return f"{size / 1024 ** 2:.2f} MB"
    else:
        return f"{size / 1024 ** 3:.2f} GB"
