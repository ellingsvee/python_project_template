import inspect
from mpi4py import MPI

from project import NDArray, backend_flags, xp

if backend_flags["cupy_avail"]:
    import cupy as cp

# def get_available_devices() -> list:
#     """
#     Return a list of available GPU devices.
#
#     Returns
#     -------
#     device_properties : list
#         A list of available GPU devices. Returns None if CuPy is not available.
#     """
#     device_properties = None
#
#     if backend_flags["cupy_avail"]:
#         n_gpus = cp.cuda.runtime.getDeviceCount()
#         device_properties = [
#             cp.cuda.runtime.getDeviceProperties(i)["name"] for i in range(n_gpus)
#         ]
#
#     return device_properties
#
#
# def set_device(comm_rank: int, comm_size: int) -> None:
#     """
#     Set the device to use.
#
#     Parameters
#     ----------
#     device_id : int
#         The device id to use.
#     """
#     if backend_flags["cupy_avail"]:
#         available_devices = get_available_devices()
#         device_id = comm_rank % len(available_devices)
#         cp.cuda.Device(device_id).use()
#         # TOLOG: COMPUTE INFOS
#         print(f"Rank {comm_rank} is using device {device_id}.")
#
#
# def get_array_module_name(arr: NDArray) -> str:
#     """Given an array, returns the array's module name.
#     This works for `numpy` even when `cupy` is not available.
#
#     Parameters
#     ----------
#     arr : NDArray
#         The array to check.
#
#     Returns
#     -------
#     submodule_name : str
#         The array module name used by the array.
#
#     """
#     submodule = inspect.getmodule(type(arr))
#     return submodule.__name__.split(".")[0]
#
#
# def get_host(arr: NDArray) -> NDArray:
#     """Returns the host array of the given array.
#
#     Parameters
#     ----------
#     arr : NDArray
#         The array to convert.
#
#     Returns
#     -------
#     host_arr : np.ndarray
#         The equivalent numpy array.
#     """
#     if get_array_module_name(arr) == "numpy":
#         return arr
#     return arr.get()
#
#
# def get_device(arr: NDArray) -> NDArray:
#     """Returns the device array of the given array.
#
#     Parameters
#     ----------
#     arr : NDArray
#         The array to convert.
#
#     Returns
#     -------
#     device_arr : NDArray
#         The equivalent cupy array.
#     """
#     if get_array_module_name(arr) == "cupy":
#         return arr
#     return xp.asarray(arr)
#
#
# def format_size(size_bytes):
#     for unit in ['B', 'KB', 'MB', 'GB']:
#         if size_bytes < 1024:
#             return f"{size_bytes:.2f} {unit}"
#         size_bytes /= 1024
#     return f"{size_bytes:.2f} TB"
#
#
# # query memory usage GPU and free unused memory
# def free_unused_gpu_memory(verbose: bool = False) -> int:
#     """Free unused memory on the GPU."""
#
#
#     if backend_flags["cupy_avail"]:
#         mempool = cp.get_default_memory_pool()
#
#         if verbose:
#             if backend_flags["mpi_avail"]:
#                 comm = MPI.COMM_WORLD
#                 rank = comm.Get_rank()
#                 if rank == 0:
#                     print("memory used    : ", format_size(mempool.used_bytes()))
#                     print("mem total bytes: ", format_size(mempool.total_bytes()))
#             else:
#                 print("memory used    : ", format_size(mempool.used_bytes()))
#                 print("mem total bytes: ", format_size(mempool.total_bytes()))
#
#         mempool.free_all_blocks()
#
#         return mempool.total_bytes()
#
#     else:
#         # return dummy value for numpy
#         return 1
