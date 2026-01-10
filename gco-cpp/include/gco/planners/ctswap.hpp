#pragma once

// General includes.
#include <vector>
#include <memory>
#include <set>
#include <map>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <random>
#include <algorithm>
#include <queue>
#include <set>
#include <limits>
#include <fstream>
#include <sstream>
#include <string>

// Project includes.
#include <gco/types.hpp>
#include <gco/world/world.hpp>
#include <gco/heuristics/heuristic.hpp>
#include <gco/heuristics/bwd_dijkstra_heuristic.hpp>
#include <gco/heuristics/euclidean_heuristic.hpp>
#include <gco/planners/planner_stats.hpp>
#include <gco/planners/base_planner.hpp>
#include <gco/utils/progress_tracker.hpp>

namespace gco {

// Forward declarations.
struct SearchState;

using SearchStatePtr = std::shared_ptr<SearchState>;

// A search state.
struct SearchState {
    // The configuration of the robot.
    Configuration2 cfg;

    // The action sequence from the parent search state. Of form [cfg_parent, cfg_1, cfg_2, ..., cfg_this].
    PathPtr action_sequence_from_parent;

    // The parent search state.
    SearchStatePtr parent_search_state;

    // Constructor.
    SearchState(const Configuration2& cfg, 
                const ActionSequencePtr& action_sequence_from_parent, 
                const SearchStatePtr& parent_search_state) : cfg(cfg), action_sequence_from_parent(action_sequence_from_parent), parent_search_state(parent_search_state) {}
};

// The CTSWAP planner.
class CTSWAPPlanner : public BasePlanner {
    public:
    CTSWAPPlanner(const gco::WorldPtr& world, const double goal_tolerance, const std::map<std::string, std::string>& heuristic_params = {}, unsigned int seed = 42, double timeout_seconds = 60.0, bool disregard_orientation = true, double goal_check_tolerance = -1.0);
    ~CTSWAPPlanner();

    PlannerStats plan(const std::map<std::string, Configuration2>& cfgs_start_raw, 
                     const std::map<std::string, Configuration2>& cfgs_goal_raw,
                     const std::map<Configuration2, HeuristicPtr>& goal_heuristics = {},
                     bool is_modify_starts_goals_locally = true) override;
    
    // Modify goals locally until they become valid.
    std::map<std::string, Configuration2> modifyCfgsLocallyUntilValid(std::map<std::string, Configuration2> cfgs);

    // Modify goals locally until they become valid.
    std::map<std::string, Configuration2> modifyCfgsLocallyUntilValidRadial(std::map<std::string, Configuration2> cfgs);

    // Set heuristic grid resolution.
    void setGridResolutionHeuristic(double resolution) { grid_resolution_heuristic_ = resolution; }

    // Get heuristic grid resolution.
    double getGridResolutionHeuristic() const { return grid_resolution_heuristic_; }

    // Set maximum node expansions for heuristic computation.
    void setMaxDistanceHeuristic(double max_distance_meters) { max_distance_meters_heuristic_ = max_distance_meters; }

    // Get maximum node expansions for heuristic computation.
    double getMaxDistanceHeuristic() const { return max_distance_meters_heuristic_; }

    // Set the goal heuristics map.
    void setGoalHeuristics(const std::map<Configuration2, HeuristicPtr>& goal_heuristics) { goal_heuristics_ = goal_heuristics; }

    // Get the goal heuristics map.
    const std::map<Configuration2, HeuristicPtr>& getGoalHeuristics() const { return goal_heuristics_; }

    // Add a heuristic for a specific goal.
    void addGoalHeuristic(const Configuration2& goal, const HeuristicPtr& heuristic) { goal_heuristics_[goal] = heuristic; }

    // Check if all robots are at their goals.
    bool areAllRobotsAtGoal(const std::map<std::string, SearchStatePtr>& search_states_current, const std::map<std::string, Configuration2>& cfgs_goal) const;

    // Check if a robot is at their goal.
    bool isRobotAtGoal(const std::string& robot_name, const Configuration2& cfg_current, const std::map<std::string, Configuration2>& cfgs_goal) const;

    // Check if a robot is near their goal.
    bool isRobotNearGoal(const std::string& robot_name, const Configuration2& cfg_current, 
                        const std::map<std::string, Configuration2>& cfgs_goal, 
                        const double tolerance) const;

    // Plan a single step for goal adaptation (CTSWAP only does goal swapping, no movement)
    bool planSingleStep(std::map<std::string, SearchStatePtr>& search_states_current, 
                         std::map<std::string, Configuration2>& cfgs_goal);

    // Get the best edge according to a heuristic.
    ActionSequencePtr getBestEdge(const std::vector<ActionSequencePtr>& edge_sequences, const Configuration2& cfg_goal) const;

    // Get the best edge for a robot given its current configuration and goal.
    ActionSequencePtr getBestEdgeForRobot(const std::string& robot_name, const Configuration2& cfg_current, const Configuration2& cfg_goal, const std::map<std::string, SearchStatePtr>& search_states_current, bool is_force_swaps_move_away = true, bool is_enforce_closed_list = true);

    // Get the deadlock loops for a robot.
    std::vector<std::vector<std::string>> getDeadlockLoops(const std::string& robot_name, const std::map<std::string, SearchStatePtr>& search_states_current, const std::map<std::string, Configuration2>& cfgs_goal);

    // Get the heuristic for an edge sequence.
    // Uses precomputed grid-based heuristics with small offsets to differentiate
    // between states within the same grid cell while maintaining efficiency.
    double getHeuristic(const Configuration2& cfg_current, const Configuration2& cfg_goal) const;

    // Get the heuristic considering the direction to a neighbor state.
    // Returns the heuristic value of the grid cell in the direction of the neighbor,
    // or infinity if that cell is an obstacle.
    double getHeuristic(const Configuration2& cfg_current, const Configuration2& cfg_neighbor, const Configuration2& cfg_goal) const;

    // Precompute backwards Dijkstra heuristic for all goals.
    void precomputeHeuristics(const std::map<std::string, Configuration2>& cfgs_goal);

    // Save heuristic data to file for visualization.
    void saveHeuristicData(const std::string& filename) const;

    // Round configuration to nearest grid point for heuristic lookup.
    // Used for efficient storage and lookup of precomputed heuristic values.
    Configuration2 roundToGridHeuristic(const Configuration2& cfg) const;

    // Reset the closed list for a specific robot (called when goal changes).
    void resetClosedList(const std::string& robot_name);

    // Check if a configuration has been visited by a robot.
    bool isConfigurationInClosedList(const std::string& robot_name, const Configuration2& cfg) const;

    // Add a configuration to the closed list for a robot.
    void addToClosedList(const std::string& robot_name, const Configuration2& cfg);

    // Reconstruct the path from a search state.
    std::vector<Configuration2> reconstructPath(const SearchStatePtr& search_state) const;

    // Reconstruct the path from a search state.
    std::vector<Configuration2> reconstructPathOnlyVertices(const SearchStatePtr& search_state) const;

    private:
    // ====================
    // Private functions.
    // ====================
    // Process an edge.
    /**
     * Process an edge. This function determines (a) what the actual edge that the robot will take is, (b) potentially a new goal assignment.
     * (a) If the edge does not step on any other robot at their current configuration, then the edge is unchanged.
     * (b) If the edge steps on a single other robot at their current configuration, then the function checks if this will lead to a deadlock. If so, it rotates the goals for the robots involved and changes the edge to a wait edge.
     * (c) If the edge steps on multiple robots at their current configuration, 
           then the function finds all deadlocks (loops) and selects one for rotating. The edge is set to a wait.
     *
     * @param robot_name The name of the robot.
     * @param edge_sequence The edge sequence to process.
     * @param search_states_current The current search states.
     * @param cfgs_goal The goal configurations.
     */
    void processEdge(const std::string& robot_name, 
                     ActionSequencePtr& edge_sequence, 
                     std::map<std::string, SearchStatePtr>& search_states_current, 
                     std::map<std::string, Configuration2>& cfgs_goal);

    // Get the robots stepped on by an edge.
    std::set<std::string> getRobotsSteppedOnByEdge(const std::string& robot_name, const ActionSequencePtr& edge_sequence, const std::map<std::string, SearchStatePtr>& search_states_current) const;
    // std::set<std::string> getRobotsSteppedOnByEdge(const std::string& robot_name, const ActionSequencePtr& edge_sequence, std::map<std::string, Configuration2> cfgs_current) const;

    // Rotate the goals for a set of robots.
    void rotateGoals(std::vector<std::string>& robot_names, std::map<std::string, Configuration2>& cfgs_goal);

    // Get the targets stepped on by an edge.
    std::map<std::string, gco::Configuration2> getTargetsSteppedOnByEdge(const std::string& robot_name, const ActionSequencePtr& edge_sequence, const std::map<std::string, Configuration2>& cfgs_goal);

    // Check if a robot is stepping on a configuration.
    bool isRobotCfgsOverlapping(const std::string& robot_name, const gco::Configuration2& cfg_current, const gco::Configuration2& cfg_goal) const;

    // Create a snap edge (moving to the goal).
    ActionSequencePtr createSnapEdge(const Configuration2& cfg_current, const Configuration2& cfg_goal);

    // Checks if a swap will be reversed on the other robot's turn.
    // Answers the question: "If I swap with these robot goals, will the other robot ask to reverse the swap?"
    bool isSwapProductive(const std::string& robot_name, 
                          const std::string& robot_name_other, 
                          const std::map<std::string, SearchStatePtr>& search_states_current, 
                          const std::map<std::string, Configuration2>& cfgs_goal);  // The goals here are before swapping. It should not matter though.

    // ====================
    // Private variables.
    // ====================
    // Map from goal configurations to their corresponding heuristic objects.
    std::map<Configuration2, HeuristicPtr> goal_heuristics_;

    // The search states.
    std::map<std::string, std::vector<SearchStatePtr>> search_states_;

    // A map noting robots that must step away from another robot.
    std::map<std::string, std::string> robots_must_step_away_from_;

    // The grid resolution for the heuristic. Used for efficient storage of precomputed
    // heuristic values. Smaller values provide finer granularity but require more memory.
    double grid_resolution_heuristic_ = 0.01;  // 0.05
    double max_distance_meters_heuristic_ = 3.0;

    // The type of heuristic to use.
    std::string heuristic_type_ = "bwd_dijkstra";

    // Precomputed heuristic values: map from goal configuration to map of configuration to distance.
    std::map<Configuration2, std::map<Configuration2, double>> heuristic_values_;

    // Closed list per robot to avoid revisiting states when goal changes.
    // Maps robot name to set of visited configurations (rounded to grid).
    std::map<std::string, std::set<Configuration2>> closed_lists_;

    // Set of robots that have recently had their goals swapped.
    // These robots are allowed to wait (stay in place) even if other edges are available.
    std::set<std::string> robots_with_recent_swaps_;
};

} // namespace gco