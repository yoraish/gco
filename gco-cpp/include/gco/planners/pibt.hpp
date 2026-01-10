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
#include <random>

// Project includes.
#include <gco/types.hpp>
#include <gco/world/world.hpp>
#include <gco/heuristics/heuristic.hpp>
#include <gco/planners/planner_stats.hpp>
#include <gco/planners/base_planner.hpp>

namespace gco {

// Forward declarations.
struct PIBTState;

using PIBTStatePtr = std::shared_ptr<PIBTState>;

// A PIBT state representing a robot's current state
struct PIBTState {
    // The current path of the robot. Each edge includes the first and last configurations of the transition.
    std::vector<EdgePtr> path;

    // The priority of this robot.
    double priority;

    // The original priority of this robot (between 0 and 1).
    double original_priority;

    // Get the last time of the path. The path includes this time and nothing after it.
    int getPlannedUpToTime(){
        return (int)path.size();
    }

    Configuration2 getLastCfg(){
        if (path.empty()) {
            throw std::runtime_error("PIBTState path is empty");
        }
        if (!path.back()) {
            throw std::runtime_error("PIBTState path.back() is null");
        }
        if (path.back()->empty()) {
            throw std::runtime_error("PIBTState path.back() is empty");
        }
        return path.back()->back();
    }

    // Constructor.
    PIBTState(const Configuration2& cfg, double priority = 0.0) 
        : priority(priority), original_priority(priority) {
        Edge initial_wait_edge = {cfg, cfg, cfg, cfg, cfg, cfg};
        path.push_back(std::make_shared<Edge>(initial_wait_edge));  
    }
};

// The PIBT planner.
class PIBTPlanner : public BasePlanner {
    public:
    PIBTPlanner(const gco::WorldPtr& world, const double goal_tolerance, const std::map<std::string, std::string>& heuristic_params = {}, unsigned int seed = 42, double timeout_seconds = 60.0, bool disregard_orientation = true, double goal_check_tolerance = -1.0);
    ~PIBTPlanner();

    PlannerStats plan(const std::map<std::string, Configuration2>& cfgs_start_raw, 
                     const std::map<std::string, Configuration2>& cfgs_goal_raw,
                     const std::map<Configuration2, HeuristicPtr>& goal_heuristics = {},
                     bool is_modify_starts_goals_locally = true) override;
    
    // Modify goals locally until they become valid.
    std::map<std::string, Configuration2> modifyCfgsLocallyUntilValid(std::map<std::string, Configuration2> cfgs);

    // Modify goals locally until they become valid (radial approach).
    std::map<std::string, Configuration2> modifyCfgsLocallyUntilValidRadial(std::map<std::string, Configuration2> cfgs);

    // Set maximum time steps for planning.
    void setMaxTimeSteps(int max_time_steps) { max_time_steps_ = max_time_steps; }

    // Get maximum time steps for planning.
    int getMaxTimeSteps() const { return max_time_steps_; }

    // Check if all robots are at their goals.
    bool areAllRobotsAtGoal(const std::map<std::string, PIBTStatePtr>& pibt_states, 
                           const std::map<std::string, Configuration2>& cfgs_goal) const;

    // Check if a robot is at their goal. This means that they are within 0.01 meters of their goal.
    bool isRobotAtGoal(const std::string& robot_name, const Configuration2& cfg_current, 
                      const std::map<std::string, Configuration2>& cfgs_goal) const;

    // Check if a robot is near their goal. This means that they are within the goal tolerance.
    bool isRobotNearGoal(const std::string& robot_name, const Configuration2& cfg_current, 
                        const std::map<std::string, Configuration2>& cfgs_goal, 
                        const double tolerance) const;

    // Get the heuristic for a configuration.
    double getHeuristic(const Configuration2& cfg_current, const Configuration2& cfg_goal) const;

    // Set the goal heuristics map.
    void setGoalHeuristics(const std::map<Configuration2, HeuristicPtr>& goal_heuristics) { goal_heuristics_ = goal_heuristics; }

    // Get the goal heuristics map.
    const std::map<Configuration2, HeuristicPtr>& getGoalHeuristics() const { return goal_heuristics_; }

    // Add a heuristic for a specific goal.
    void addGoalHeuristic(const Configuration2& goal, const HeuristicPtr& heuristic) { goal_heuristics_[goal] = heuristic; }

    // Plan a single step for movement planning
    bool planSingleStep(std::map<std::string, PIBTStatePtr>& pibt_states,
                        const std::map<std::string, Configuration2>& cfgs_goal,
                        int t_plus_1);

    private:
    // ====================
    // Private functions.
    // ====================
    // The PIBT procedure.
    bool pibtProcedure(const std::string& robot_name, 
                       std::map<std::string, PIBTStatePtr>& pibt_states,
                       const std::map<std::string, Configuration2>& cfgs_goal,
                       int t_plus_1,
                       int recursion_depth = 0,
                       std::shared_ptr<std::set<std::string>> affected_robots = nullptr);

    // Get all possible edges for a robot.
    std::vector<ActionSequencePtr> getAllPossibleEdges(const std::string& robot_name, 
                                                       const Configuration2& cfg_current,
                                                       const std::map<std::string, Configuration2>& cfgs_goal);

    // Check if an edge is a wait edge.
    bool isWaitEdge(EdgePtr edge);

    // Get preassigned robots stepped on by an edge.
    std::vector<std::string> getPreassignedRobotsSteppedOnByEdge(const std::string& robot_name,
                                                                 const ActionSequencePtr& edge,
                                                                 const std::map<std::string, PIBTStatePtr>& pibt_states,
                                                                 int t_plus_1);

    // Get unassigned robots stepped on by an edge.
    std::set<std::string> getUnassignedRobotsSteppedOnByEdge(const std::string& robot_name,
                                                             const ActionSequencePtr& edge,
                                                             const std::map<std::string, PIBTStatePtr>& pibt_states,
                                                             int t_plus_1);

    // Create a wait edge.
    ActionSequencePtr createWaitEdge(const Configuration2& cfg_current);

    // Create a snap edge (moving to the goal).
    EdgePtr createSnapEdge(const Configuration2& cfg_current, const Configuration2& cfg_goal);

    // Sort robots by priority.
    std::vector<std::string> sortRobotsByPriority(const std::map<std::string, PIBTStatePtr>& pibt_states);

    // Update priorities based on distance to goal.
    void updatePriorities(std::map<std::string, PIBTStatePtr>& pibt_states,
                         const std::map<std::string, Configuration2>& cfgs_goal);

    // Extend paths to the same length.
    void extendPathsToSameLength(std::map<std::string, PIBTStatePtr>& pibt_states);

    // ====================
    // Private variables.
    // ====================
    // Map from goal configurations to their corresponding heuristic objects.
    std::map<Configuration2, HeuristicPtr> goal_heuristics_;

    // The maximum number of time steps to plan.
    int max_time_steps_ = 100;

    // Heuristic grid resolution for backwards Dijkstra heuristic.
    double grid_resolution_heuristic_ = 0.01;

    // Maximum distance in meters for heuristic computation.
    double max_distance_meters_heuristic_ = 3.0;

    // The type of heuristic to use.
    std::string heuristic_type_ = "bwd_dijkstra";
};

} // namespace gco
