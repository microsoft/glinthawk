# pipe_sim.pxd

from libcpp.vector cimport vector

# Declare the C++ functions
cdef extern from "pipeline_simulation.hh":
    vector[double] single_tier_pipeline(vector[size_t] t1_layers,
                                        size_t in_flight,
                                        size_t max_iters,
                                        double mid_comp,
                                        double last_comp,
                                        double transit,
                                        double latency)

    vector[double] two_tier_pipeline(vector[size_t] t1_layers,
                                     size_t in_flight,
                                     size_t max_iters,
                                     double mid_t1_comp,
                                     double last_t1_comp,
                                     double t2_comp,
                                     double t1_to_t2_transit,
                                     double t1_to_t2_latency,
                                     double t2_to_t1_transit,
                                     double t2_to_t1_latency)