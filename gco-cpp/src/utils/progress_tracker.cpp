#include "gco/utils/progress_tracker.hpp"

namespace gco {

ProgressTracker::ProgressTracker(int max_iterations, const std::string& description)
    : max_iterations_(max_iterations), description_(description), 
      start_time_(std::chrono::high_resolution_clock::now()) {
    // Initial progress display
    std::cout << description_ << " progress: 0/" << max_iterations_ << " iterations" << std::flush;
}

void ProgressTracker::updateProgress(int iteration_count) {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed_total = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time_);
    
    if (iteration_count > 0) {
        double avg_time_ms = static_cast<double>(elapsed_total.count()) / iteration_count;
        std::cout << "\r" << description_ << " progress: " << iteration_count << "/" << max_iterations_ 
                  << " iterations | Avg: " << std::fixed << std::setprecision(1) << avg_time_ms << "ms/iter" << std::flush;
    } else {
        std::cout << "\r" << description_ << " progress: " << iteration_count << "/" << max_iterations_ << " iterations" << std::flush;
    }
}

void ProgressTracker::finalize(int iteration_count) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_);
    double avg_time_ms = iteration_count > 0 ? static_cast<double>(total_time.count()) / iteration_count : 0.0;
    
    std::cout << "\r" << description_ << " progress: " << iteration_count << "/" << max_iterations_ 
              << " iterations | Total: " << total_time.count() << "ms | Avg: " << std::fixed << std::setprecision(1) << avg_time_ms << "ms/iter - Complete!" << std::endl;
}

long ProgressTracker::getElapsedTimeMs() const {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time_);
    return elapsed.count();
}

double ProgressTracker::getAverageTimePerIteration(int iteration_count) const {
    if (iteration_count <= 0) return 0.0;
    return static_cast<double>(getElapsedTimeMs()) / iteration_count;
}

} // namespace gco
