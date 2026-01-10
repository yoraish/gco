#pragma once

// General includes.
#include <vector>
#include <memory>
#include <set>
#include <map>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <queue>

// Project includes.
#include <gco/types.hpp>
#include <gco/world/world.hpp>
#include <gco/planners/ctswap.hpp>
#include <gco/planners/pibt.hpp>
#include <gco/planners/planner_stats.hpp>
#include <gco/planners/base_planner.hpp>
#include <gco/heuristics/heuristic.hpp>

// Forward declarations from other planners
namespace gco {
    struct SearchState;
    using SearchStatePtr = std::shared_ptr<SearchState>;
    struct PIBTState;
    using PIBTStatePtr = std::shared_ptr<PIBTState>;
}

namespace gco {

// Forward declarations.
struct HybridState;

using HybridStatePtr = std::shared_ptr<HybridState>;

// A hybrid state representing a robot's current state
struct HybridState {
    // The current configuration of the robot
    Configuration2 cfg;

    // The priority of this robot (for PIBT stage)
    double priority;

    // The original priority of this robot (between 0 and 1)
    double original_priority;

    // The current path of the robot. Each edge includes the first and last configurations of the transition.
    std::vector<EdgePtr> path;

    // Get the last time of the path. The path includes this time and nothing after it.
    int getPlannedUpToTime(){
        return (int)path.size();
    }

    Configuration2 getLastCfg(){
        return path.back()->back();
    }

    // Constructor.
    HybridState(const Configuration2& cfg, double priority = 0.0) 
        : cfg(cfg), priority(priority), original_priority(priority) {
        Edge initial_wait_edge = {cfg, cfg, cfg, cfg, cfg, cfg};
        path.push_back(std::make_shared<Edge>(initial_wait_edge));
    }
};

// The GSPI planner.
class GSPIPlanner : public BasePlanner {
    public:
    GSPIPlanner(const gco::WorldPtr& world, const double goal_tolerance, const std::map<std::string, std::string>& heuristic_params = {}, unsigned int seed = 42, double timeout_seconds = 60.0, bool disregard_orientation = true, double goal_check_tolerance = -1.0);
    ~GSPIPlanner();

    PlannerStats plan(const std::map<std::string, Configuration2>& cfgs_start_raw, 
                     const std::map<std::string, Configuration2>& cfgs_goal_raw,
                     const std::map<Configuration2, HeuristicPtr>& goal_heuristics = {},
                     bool is_modify_starts_goals_locally = true) override;
    
    // Modify goals locally until they become valid.
    std::map<std::string, Configuration2> modifyCfgsLocallyUntilValid(std::map<std::string, Configuration2> cfgs);

    // Modify goals locally until they become valid (radial approach).
    std::map<std::string, Configuration2> modifyCfgsLocallyUntilValidRadial(std::map<std::string, Configuration2> cfgs);

    // Set heuristic grid resolution.
    void setGridResolutionHeuristic(double resolution) { grid_resolution_heuristic_ = resolution; }

    // Get heuristic grid resolution.
    double getGridResolutionHeuristic() const { return grid_resolution_heuristic_; }

    // Set maximum distance for heuristic computation.
    void setMaxDistanceHeuristic(double max_distance_meters) { max_distance_meters_heuristic_ = max_distance_meters; }

    // Get maximum distance for heuristic computation.
    double getMaxDistanceHeuristic() const { return max_distance_meters_heuristic_; }

    // Check if all robots are at their goals.
    bool areAllRobotsAtGoal(const std::map<std::string, HybridStatePtr>& hybrid_states, 
                           const std::map<std::string, Configuration2>& cfgs_goal) const;

    // Check if a robot is at their goal.
    bool isRobotAtGoal(const std::string& robot_name, const Configuration2& cfg_current, 
                      const std::map<std::string, Configuration2>& cfgs_goal) const;

    // Get the heuristic for a configuration.
    double getHeuristic(const Configuration2& cfg_current, const Configuration2& cfg_goal) const;

    // Set the goal heuristics map.
    void setGoalHeuristics(const std::map<Configuration2, HeuristicPtr>& goal_heuristics);

    // Get the goal heuristics map.
    const std::map<Configuration2, HeuristicPtr>& getGoalHeuristics() const { return goal_heuristics_; }

    // Add a heuristic for a specific goal.
    void addGoalHeuristic(const Configuration2& goal, const HeuristicPtr& heuristic) { goal_heuristics_[goal] = heuristic; }

    private:
    // ====================
    // Private functions.
    // ====================
    // Run one iteration of PIBT movement planning
    bool runPIBTMovement(std::map<std::string, HybridStatePtr>& hybrid_states,
                        const std::map<std::string, Configuration2>& cfgs_goal,
                        int t_plus_1);

    // Clear the processed robots set for PIBT
    void clearProcessedRobotsSet();

    // ====================
    // Private variables.
    // ====================
    // Map from goal configurations to their corresponding heuristic objects.
    std::map<Configuration2, HeuristicPtr> goal_heuristics_;
    
    // The CTSWAP planner.
    std::unique_ptr<CTSWAPPlanner> ctswap_planner_;
    
    // The PIBT planner.
    std::unique_ptr<PIBTPlanner> pibt_planner_;
    
    // Heuristic grid resolution for backwards Dijkstra heuristic.
    double grid_resolution_heuristic_ = 0.01;
    
    // Maximum distance in meters for heuristic computation.
    double max_distance_meters_heuristic_ = 3.0;
    
    // The type of heuristic to use.
    std::string heuristic_type_ = "bwd_dijkstra";
    
    // Flag to disregard orientation in goal checking (default true)
    bool disregard_orientation_;
};

} // namespace gco 