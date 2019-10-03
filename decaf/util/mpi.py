"""
mpi implements common util functions based on mpi4py
"""

import logging
import numpy as np
import os
import random
import socket
import time

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
except ImportError as error:
    logging.warning('Warning: I cannot import mpi4py. Using a dummy single noded implementation instead. The program '
                    'will run in single node mode even if you executed me with mpirun or mpiexec.\n'
                    'We STRONGLY recommend you to try to install mpi and mpi4py.\n')

    logging.warning('mpi4py exception message is {}'.format(error))
    from decaf.util._mpi_dummy import COMM

RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
HOST = socket.gethostname()

_MPI_PRINT_MESSAGE_TAG = 560710
_MPI_BUFFER_LIMIT = 2 ** 30

# we need to set the random seed different for each mpi instance
logging.info('util.mpi: seting different random seeds for each node.')
random.seed(time.time() * RANK)


def mkdir(dirname):
    """
    make a directory safely.

    This function avoids race conditions when writing to a common location.
    """
    try:
        os.makedirs(dirname)
    except OSError as error:
        if not os.path.exists(dirname):
            raise error


def mpi_any(decision):
    """
    the logical any() over all instances
    """
    return any(COMM.allgather(decision))


def mpi_all(decision):
    """
    the logical all() over all instances
    """
    return all(COMM.allgather(decision))


def elect():
    """
    elect() randomly chooses a node from all the nodes as the president.

    Output:
        the rank of the president
    """
    president = COMM.bcast(np.random.randint(SIZE))
    return president


def is_president():
    """
    Return true if I am the president, otherwise return false
    """
    return (RANK == elect())


def is_root():
    """
    returns if the current node is root
    """
    return RANK == 0


def barrier(tag=0, sleep=0.01):
    """
    The original MPI.comm.barrier() may cause idle processes to still occupy the CPU, while this barrier waits.
    """
    if SIZE == 1:
        return
    mask = 1
    while mask < SIZE:
        dst = (RANK + mask) % SIZE
        src = (RANK - mask + SIZE) % SIZE
        req = COMM.isend(None, dst, tag)
        while not COMM.Iprobe(src, tag):
            time.sleep(sleep)
        COMM.recv(None, src, tag)
        req.Wait()
        mask <<= 1


def root_log_level(level, name=None):
    """
    set the log level on root.
    Input:
        level: the logging level, such as logging.DEBUG
        name: (optional) the logger name
    """
    if is_root():
        log_level(level, name)


def log_level(level, name=None):
    """
    set the log level on all nodes.
    Input:
        level: the logging level, such as logging.DEBUG
        name: (optional) the logger name
    """
    logging.getLogger(name).setLevel(level)


if __name__ == '__main__':
    pass
