# distutils: language = c++

from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np

# Expose the C++ functions to Python
def single_tier_pipeline_py(vector[size_t] t1_layers,
                            size_t in_flight,
                            size_t max_iters,
                            double mid_comp,
                            double last_comp,
                            double transit,
                            double latency):
    cdef vector[double] result = single_tier_pipeline(t1_layers, in_flight, max_iters, mid_comp, last_comp, transit, latency)

    # Convert C++ vector to numpy array
    cdef np.ndarray[np.float64_t, ndim=1] result_np = np.empty(len(result), dtype=np.float64)
    for i in range(len(result)):
        result_np[i] = result[i]
    return result_np

def two_tier_pipeline_py(vector[size_t] t1_layers,
                         size_t in_flight,
                         size_t max_iters,
                         double mid_t1_comp,
                         double last_t1_comp,
                         double t2_comp,
                         double t1_to_t2_transit,
                         double t1_to_t2_latency,
                         double t2_to_t1_transit,
                         double t2_to_t1_latency):
    cdef vector[size_t] t1_layers_vec = t1_layers
    cdef vector[double] result = two_tier_pipeline(t1_layers, in_flight, max_iters, mid_t1_comp, last_t1_comp, t2_comp, t1_to_t2_transit, t1_to_t2_latency, t2_to_t1_transit, t2_to_t1_latency)

    # Convert C++ vector to numpy array
    cdef np.ndarray[np.float64_t, ndim=1] result_np = np.empty(len(result), dtype=np.float64)
    for i in range(len(result)):
        result_np[i] = result[i]
    return result_np