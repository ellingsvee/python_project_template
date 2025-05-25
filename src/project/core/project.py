import logging

from project import ArrayLike, NDArray, backend_flags, comm_rank, comm_size, xp
from project.configs.project_config import ProjectConfig

if backend_flags["mpi_avail"]:
    from mpi4py import MPI

import time

xp.set_printoptions(precision=8, suppress=True, linewidth=150)


class Project:
    """
    Project
    """

    def __init__(self, config: ProjectConfig):
        self.config = config
