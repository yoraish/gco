#pragma once

#include <chrono>
#include <iostream>
#include <iomanip>

namespace gco {

/**
 * @brief A modular progress tracker for monitoring planning iterations
 * 
 * This class provides progress tracking functionality that can be used
 * across different planners to show iteration progress, timing, and
 * performance metrics.
 */
class ProgressTracker {
public:
    /**
     * @brief Constructor
     * @param max_iterations Maximum number of iterations to track
     * @param description Optional description for the progress display
     */
    ProgressTracker(int max_iterations, const std::string& description = "Planning");
    
    /**
     * @brief Update progress display
     * @param iteration_count Current iteration count
     */
    void updateProgress(int iteration_count);
    
    /**
     * @brief Finalize progress display with completion message
     * @param iteration_count Final iteration count
     */
    void finalize(int iteration_count);
    
    /**
     * @brief Get elapsed time in milliseconds
     * @return Elapsed time since construction
     */
    long getElapsedTimeMs() const;
    
    /**
     * @brief Get average time per iteration
     * @param iteration_count Current iteration count
     * @return Average time per iteration in milliseconds
     */
    double getAverageTimePerIteration(int iteration_count) const;

private:
    int max_iterations_;
    std::string description_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

} // namespace gco
