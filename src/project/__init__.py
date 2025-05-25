import os
from typing import Any, TypeAlias, TypeVar
from warnings import warn

from project_name.__about__ import __version__

import numpy as np
from numpy.typing import ArrayLike

backend_flags = {
    "array_module": None,
    "cupy_avail": False,
    "mpi_avail": False,
}

__all__ = [
    "__version__",
]

# Allows user to specify the array module via an environment variable.
backend_flags["array_module"] = os.environ.get("ARRAY_MODULE")

if backend_flags["array_module"] is not None:
    if backend_flags["array_module"] == "numpy":
        import numpy as xp
        import scipy as sp

        xp_host = xp
    else:
        raise ValueError(f"Unrecognized ARRAY_MODULE '{backend_flags['array_module']}'")

else:
    # If the user does not specify the array module, prioritize numpy.
    warn("No `ARRAY_MODULE` specified, defaulting to 'NumPy'.")
    import numpy as xp
    import scipy as sp

    xp_host = xp

try:
    # Check if mpi4py is available
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Create a small GPU array
    array = np.array([comm_rank], dtype=np.float32)

    # Perform an MPI operation to check working
    if comm_size > 1:
        if comm_rank == 0:
            comm.Send([array, MPI.FLOAT], dest=1)
        elif comm_rank == 1:
            comm.Recv([array, MPI.FLOAT], source=0)

    backend_flags["mpi_avail"] = True
    if backend_flags["cupy_avail"] and os.environ.get("MPI_CUDA_AWARE", "0") == "1":
        backend_flags["mpi_cuda_aware"] = True
except (ImportError, ImportWarning, ModuleNotFoundError) as w:
    warn(f"No 'MPI' backend detected. ({w})")
    comm_rank = 0
    comm_size = 1

# Some type aliases for the array module.
_ScalarType = TypeVar("ScalarType", bound=xp.generic, covariant=True)
_DType = xp.dtype[_ScalarType]
NDArray: TypeAlias = xp.ndarray[Any, _DType]

__all__ = [
    "__version__",
    "xp",
    "xp_host",
    "sp",
    "ArrayLike",
    "NDArray",
    "comm_rank",
    "comm_size",
    "backend_flags",
]
