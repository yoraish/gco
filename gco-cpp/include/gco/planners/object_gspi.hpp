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
struct ObjectHybridState;

using ObjectHybridStatePtr = std::shared_ptr<ObjectHybridState>;

// A hybrid state representing an object's current state
struct ObjectHybridState {
    // The current configuration of the object
    Configuration2 cfg;

    // The priority of this object (for PIBT stage)
    double priority;

    // The original priority of this object (between 0 and 1)
    double original_priority;

    // The current path of the object. Each edge includes the first and last configurations of the transition.
    std::vector<EdgePtr> path;

    // Get the last time of the path. The path includes this time and nothing after it.
    int getPlannedUpToTime(){
        return (int)path.size();
    }

    Configuration2 getLastCfg(){
        return path.back()->back();
    }

    // Constructor.
    ObjectHybridState(const Configuration2& cfg, double priority = 0.0) 
        : cfg(cfg), priority(priority), original_priority(priority) {
        Edge initial_wait_edge = {cfg, cfg, cfg, cfg, cfg, cfg};
        path.push_back(std::make_shared<Edge>(initial_wait_edge));
    }
};

// The Object GSPI planner for object-level planning.
class ObjectGSPIPlanner : public BasePlanner {
    public:
    ObjectGSPIPlanner(const gco::WorldPtr& world, const double goal_tolerance, const std::map<std::string, std::string>& heuristic_params = {}, unsigned int seed = 42, double timeout_seconds = 60.0, bool disregard_orientation = true);
    ~ObjectGSPIPlanner();

    // Initialize the planner with object targets (persistent setup)
    void initializeObjectTargets(const std::map<std::string, Configuration2>& object_targets);

    // Update object positions and get next moves (called each iteration)
    std::map<std::string, std::vector<Configuration2>> getNextObjectMoves(
        const std::map<std::string, Configuration2>& current_object_positions,
        int horizon = 3);

    // Check if all objects are at their goals
    bool areAllObjectsAtGoal(const std::map<std::string, Configuration2>& current_positions) const;

    // Check if a specific object is at its goal
    bool isObjectAtGoal(const std::string& object_name, const Configuration2& current_pos) const;

    // Get current object-to-goal assignments
    std::map<std::string, Configuration2> getCurrentAssignments() const;

    // Reset the planner state for new scenarios
    void resetPlannerState();

    // Set heuristic grid resolution.
    void setGridResolutionHeuristic(double resolution) { grid_resolution_heuristic_ = resolution; }

    // Get heuristic grid resolution.
    double getGridResolutionHeuristic() const { return grid_resolution_heuristic_; }

    // Set maximum distance for heuristic computation.
    void setMaxDistanceHeuristic(double max_distance_meters) { max_distance_meters_heuristic_ = max_distance_meters; }

    // Get maximum distance for heuristic computation.
    double getMaxDistanceHeuristic() const { return max_distance_meters_heuristic_; }

    // Get the heuristic for a configuration.
    double getHeuristic(const Configuration2& cfg_current, const Configuration2& cfg_goal) const;

    // Set the goal heuristics map.
    void setGoalHeuristics(const std::map<Configuration2, HeuristicPtr>& goal_heuristics);

    // Get the goal heuristics map.
    const std::map<Configuration2, HeuristicPtr>& getGoalHeuristics() const { return goal_heuristics_; }

    // Add a heuristic for a specific goal.
    void addGoalHeuristic(const Configuration2& goal, const HeuristicPtr& heuristic) { goal_heuristics_[goal] = heuristic; }

    // Update the world state (for dynamic obstacles)
    void updateWorldState(const WorldPtr& new_world);

    // Required by BasePlanner interface
    PlannerStats plan(const std::map<std::string, Configuration2>& cfgs_start_raw, 
                     const std::map<std::string, Configuration2>& cfgs_goal_raw,
                     const std::map<Configuration2, HeuristicPtr>& goal_heuristics = {},
                     bool is_modify_starts_goals_locally = true) override;

    private:
    // ====================
    // Private functions.
    // ====================
    
    // Create a simplified world for object planning (objects as robots, robots as obstacles)
    WorldPtr createObjectPlanningWorld();

    // Stage 1: Run one iteration of CTSWAP goal adaptation (no movement)
    bool runCTSWAPGoalAdaptation(std::map<std::string, ObjectHybridStatePtr>& object_states,
                                std::map<std::string, Configuration2>& cfgs_goal);

    // Stage 2: Run one iteration of PIBT movement planning
    bool runPIBTMovement(std::map<std::string, ObjectHybridStatePtr>& object_states,
                        const std::map<std::string, Configuration2>& cfgs_goal,
                        int t_plus_1);


    RobotPtr createDiskRobotFromObject(const std::string& object_name, const ObjectPtr& object, double radius);
    ObstaclePtr convertObjectToObstacle(const std::string& object_name, const ObjectPtr& object);

    // ====================
    // Private variables.
    // ====================
    
    // Map from goal configurations to their corresponding heuristic objects.
    std::map<Configuration2, HeuristicPtr> goal_heuristics_;
    
    // The CTSWAP planner for objects.
    std::unique_ptr<CTSWAPPlanner> ctswap_planner_;
    
    // The PIBT planner for objects.
    std::unique_ptr<PIBTPlanner> pibt_planner_;
    
    // Object target configurations (persistent)
    std::map<std::string, Configuration2> object_targets_;
    
    // Current object-to-goal assignments (updated during planning)
    std::map<std::string, Configuration2> current_assignments_;
    
    // Current object states (persistent)
    std::map<std::string, ObjectHybridStatePtr> object_states_;
    
    // Object radius for planning (all objects treated as disks)
    double object_radius_ = 0.5;
    
    // Heuristic grid resolution for backwards Dijkstra heuristic.
    double grid_resolution_heuristic_ = 0.01;
    
    // Maximum distance in meters for heuristic computation.
    double max_distance_meters_heuristic_ = 3.0;
    
    // The type of heuristic to use.
    std::string heuristic_type_ = "bwd_dijkstra";
    
    // Track if this is the first call to getNextObjectMoves for this planner instance
    bool first_call_ = true;
    
    // Original world (for reference)
    WorldPtr original_world_;
    
    // Flag to disregard orientation in goal checking (default true)
    bool disregard_orientation_;
};

} // namespace gco
