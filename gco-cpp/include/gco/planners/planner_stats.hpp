#pragma once

#include <gco/types.hpp>
#include <chrono>

namespace gco {

// Structure to hold planning statistics
struct PlannerStats {
    // The computed paths
    MultiRobotPaths paths;
    
    // Planning time (excluding heuristic computation)
    double planning_time_seconds;
    
    // Time per iteration (if applicable)
    double time_per_iteration_seconds;
    
    // Number of iterations performed
    int num_iterations;
    
    // Maximum iterations reached
    bool max_iterations_reached;
    
    // Timeout reached
    bool timeout_reached;
    
    // Heuristic computation time (separate from planning time)
    double heuristic_time_seconds;
    
    // Total time (planning + heuristic)
    double total_time_seconds;
    
    // Success status
    bool success;
    
    // Error message if planning failed
    std::string error_message;
    
    // Constructor with default values
    PlannerStats() : 
        planning_time_seconds(0.0),
        time_per_iteration_seconds(0.0),
        num_iterations(0),
        max_iterations_reached(false),
        timeout_reached(false),
        heuristic_time_seconds(0.0),
        total_time_seconds(0.0),
        success(false),
        error_message("") {}
    
    // Constructor with paths
    explicit PlannerStats(const MultiRobotPaths& paths) : 
        paths(paths),
        planning_time_seconds(0.0),
        time_per_iteration_seconds(0.0),
        num_iterations(0),
        max_iterations_reached(false),
        timeout_reached(false),
        heuristic_time_seconds(0.0),
        total_time_seconds(0.0),
        success(false),
        error_message("") {}
};

} // namespace gco
