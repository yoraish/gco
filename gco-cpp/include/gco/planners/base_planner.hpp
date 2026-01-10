#pragma once

// General includes.
#include <chrono>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <ostream>

// Project includes.
#include <gco/types.hpp>
#include <gco/world/world.hpp>
#include <gco/planners/planner_stats.hpp>
#include <gco/heuristics/heuristic.hpp>

namespace gco {

// Forward declarations.
struct PlannerStats;

class BasePlanner {
public:
    BasePlanner(const gco::WorldPtr& world, const double goal_tolerance, double timeout_seconds = 60.0);
    virtual ~BasePlanner() = default;

    // Pure virtual function that derived classes must implement
    virtual PlannerStats plan(const std::map<std::string, Configuration2>& cfgs_start_raw, 
                             const std::map<std::string, Configuration2>& cfgs_goal_raw,
                             const std::map<Configuration2, HeuristicPtr>& goal_heuristics = {},
                             bool is_modify_starts_goals_locally = true) = 0;

    // Common getter/setter methods
    void setVerbose(bool verbose) { verbose_ = verbose; }
    bool getVerbose() const { return verbose_; }
    
    void setWorld(const gco::WorldPtr& world) { world_ = world; }
    gco::WorldPtr getWorld() const { return world_; }
    
    void setTimeout(double timeout_seconds) { timeout_seconds_ = timeout_seconds; }
    double getTimeout() const { return timeout_seconds_; }
    
    void setMaxIterations(int max_iterations) { max_iterations_ = max_iterations; }
    int getMaxIterations() const { return max_iterations_; }
    
    void setGoalTolerance(double goal_tolerance) { goal_tolerance_ = goal_tolerance; }
    double getGoalTolerance() const { return goal_tolerance_; }

protected:
    // Common timeout checking method
    bool checkTimeout(const std::chrono::high_resolution_clock::time_point& start_time, 
                     int current_iterations, 
                     PlannerStats& stats) const;
    
    // Common method to create timeout stats
    PlannerStats createTimeoutStats(const std::chrono::high_resolution_clock::time_point& start_time, 
                                   int current_iterations) const;
    
    // Common method to create success stats
    PlannerStats createSuccessStats(const MultiRobotPaths& paths, 
                                   const std::chrono::high_resolution_clock::time_point& start_time,
                                   int current_iterations,
                                   bool max_iterations_reached,
                                   const std::vector<double>& iteration_times = {}) const;

    // Common member variables
    gco::WorldPtr world_;
    double goal_tolerance_;
    int max_iterations_;
    double timeout_seconds_;
    bool verbose_;
    unsigned int seed_;
    
    // Flag to disregard orientation in goal checking (default true)
    bool disregard_orientation_;
    
    // Tolerance for goal checking (if -1, uses default 0.01)
    double goal_check_tolerance_;

    // Common debug functionality
    class DebugStream {
    private:
        bool verbose_;
        std::ostream& stream_;
        
    public:
        DebugStream(bool verbose, std::ostream& stream = std::cout) : verbose_(verbose), stream_(stream) {}
        
        template<typename T>
        DebugStream& operator<<(const T& value) {
            if (verbose_) {
                stream_ << value;
            }
            return *this;
        }
        
        // Handle std::endl and other stream manipulators
        DebugStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
            if (verbose_) {
                stream_ << manip;
            }
            return *this;
        }
        
        // Handle std::set
        template<typename T>
        DebugStream& operator<<(const std::set<T>& container) {
            if (verbose_) {
                stream_ << "{";
                for (auto it = container.begin(); it != container.end(); ++it) {
                    if (it != container.begin()) stream_ << ", ";
                    stream_ << *it;
                }
                stream_ << "}";
            }
            return *this;
        }
        
        // Handle std::vector
        template<typename T>
        DebugStream& operator<<(const std::vector<T>& container) {
            if (verbose_) {
                stream_ << "[";
                for (size_t i = 0; i < container.size(); ++i) {
                    if (i > 0) stream_ << ", ";
                    stream_ << container[i];
                }
                stream_ << "]";
            }
            return *this;
        }
        
        // Handle std::map
        template<typename K, typename V>
        DebugStream& operator<<(const std::map<K, V>& container) {
            if (verbose_) {
                stream_ << "{";
                for (auto it = container.begin(); it != container.end(); ++it) {
                    if (it != container.begin()) stream_ << ", ";
                    stream_ << it->first << ": " << it->second;
                }
                stream_ << "}";
            }
            return *this;
        }
    };
    
    // Get debug stream for chaining
    DebugStream debug() const {
        return DebugStream(verbose_);
    }
    
    // Get debug stream with custom output
    template<typename Stream>
    DebugStream debug(Stream& stream) const {
        return DebugStream(verbose_, stream);
    }
};

} // namespace gco
