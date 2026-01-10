// Project includes.
#include "gco/planners/base_planner.hpp"
#include "gco/planners/planner_stats.hpp"
#include <chrono>

namespace gco {

BasePlanner::BasePlanner(const gco::WorldPtr& world, const double goal_tolerance, double timeout_seconds)
    : world_(world), goal_tolerance_(goal_tolerance), max_iterations_(100), timeout_seconds_(timeout_seconds), verbose_(false), seed_(42), disregard_orientation_(true), goal_check_tolerance_(-1.0) {
}

bool BasePlanner::checkTimeout(const std::chrono::high_resolution_clock::time_point& start_time, 
                              int current_iterations, 
                              PlannerStats& stats) const {
    auto current_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = current_time - start_time;
    
    if (elapsed_time.count() >= timeout_seconds_) {
        debug() << "Timeout reached (" << timeout_seconds_ << " seconds). Returning empty paths." << std::endl;
        stats = createTimeoutStats(start_time, current_iterations);
        return true;
    }
    return false;
}

PlannerStats BasePlanner::createTimeoutStats(const std::chrono::high_resolution_clock::time_point& start_time, 
                                            int current_iterations) const {
    auto current_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = current_time - start_time;
    
    PlannerStats stats;
    stats.paths = MultiRobotPaths();
    stats.planning_time_seconds = elapsed_time.count();
    stats.num_iterations = current_iterations;
    stats.max_iterations_reached = false;
    stats.timeout_reached = true;
    stats.success = false;
    stats.error_message = "Planning timeout reached (" + std::to_string(timeout_seconds_) + " seconds)";
    return stats;
}

PlannerStats BasePlanner::createSuccessStats(const MultiRobotPaths& paths, 
                                            const std::chrono::high_resolution_clock::time_point& start_time,
                                            int current_iterations,
                                            bool max_iterations_reached,
                                            const std::vector<double>& iteration_times) const {
    auto current_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = current_time - start_time;
    
    PlannerStats stats;
    stats.paths = paths;
    stats.planning_time_seconds = elapsed_time.count();
    stats.num_iterations = current_iterations;
    stats.max_iterations_reached = max_iterations_reached;
    stats.timeout_reached = false;
    stats.success = !paths.empty() && !max_iterations_reached;
    
    // Calculate iteration timing statistics
    if (!iteration_times.empty()) {
        double total_iteration_time = 0.0;
        double min_iteration_time = std::numeric_limits<double>::max();
        double max_iteration_time = 0.0;
        
        for (double time : iteration_times) {
            total_iteration_time += time;
            min_iteration_time = std::min(min_iteration_time, time);
            max_iteration_time = std::max(max_iteration_time, time);
        }
        
        stats.time_per_iteration_seconds = total_iteration_time / iteration_times.size();
        // Store min/max in error message for now (we can extend PlannerStats later if needed)
        if (max_iterations_reached) {
            stats.error_message = "Reached maximum iterations (" + std::to_string(max_iterations_) + "). ";
        } else {
            stats.error_message = "";
        }
        stats.error_message += "Iteration times - Avg: " + std::to_string(stats.time_per_iteration_seconds) + 
                              "s, Min: " + std::to_string(min_iteration_time) + 
                              "s, Max: " + std::to_string(max_iteration_time) + "s";
    } else if (max_iterations_reached) {
        stats.error_message = "Reached maximum iterations (" + std::to_string(max_iterations_) + ")";
    }
    
    return stats;
}

} // namespace gco
