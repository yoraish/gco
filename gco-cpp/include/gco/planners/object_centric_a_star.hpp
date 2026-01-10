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
#include <limits>

// Project includes.
#include <gco/types.hpp>
#include <gco/world/world.hpp>
#include <gco/utils.hpp>
#include <gco/obstacles/obstacles.hpp>

namespace gco {

class ObjectCentricAStarPlanner {
    public:
    // ==================== 
    // Public methods.
    // ==================== 
    ObjectCentricAStarPlanner(const gco::WorldPtr& world, const double goal_tolerance, const double weight = 1.0);
    ~ObjectCentricAStarPlanner();

    // Plan a path for the object and the robots.
    // The returned path is in the global frame, and is a dictionary of robot+object names to configuration sequences.
    MultiRobotPaths plan(const std::map<std::string, Configuration2>& cfgs_start_robots, 
                         const std::map<std::string, Configuration2>& cfg_start_objects,
                         const std::map<std::string, Configuration2>& cfg_goal_objects,
                         const double goal_tolerance);

    // Save the planner output to a file
    void savePlannerOutput(const std::vector<Configuration2>& path,
                          const Configuration2& cfg_start_object,
                          const Configuration2& cfg_goal_object,
                          const std::string& filename,
                          const std::string& object_type,
                          const double object_size) const;
    
    // Save the planner output to a file (overloaded for rectangles with width and height)
    void savePlannerOutput(const std::vector<Configuration2>& path,
                          const Configuration2& cfg_start_object,
                          const Configuration2& cfg_goal_object,
                          const std::string& filename,
                          const std::string& object_type,
                          const double object_width,
                          const double object_height) const;

    // ==================== 
    // Public variables.
    // ==================== 

    private:
    // ==================== 
    // Private structs.
    // ==================== 

    struct SearchState {
        // The multi-object multi-robot configuration at this state
        MultiObjectMultiRobotConfigurationPtr cfg_multi_object_multi_robot;
        // The configuration sequence from the parent state. This is of form {object_name: [cfg_1, cfg_2, ...], robot_name_1: [cfg_1, cfg_2, ...], robot_name_2: [cfg_1, cfg_2, ...], ...}.
        // cfg1 is the parent configuration, cfgH is the current configuration.
        // MultiRobotPathsPtr edge_from_parent;
        // The parent search state.
        std::shared_ptr<SearchState> parent_search_state;
        // The cost to come to this state.
        double g;
        // The heuristic cost to the goal.
        double h;
        // The total cost of the edge (g + weight * h).
        double f;
        // If open, closed, or neither.
        bool is_open;
        bool is_closed;
    };

    using SearchStatePtr = std::shared_ptr<SearchState>;

    struct CompareSearchState {
        // Return true if a is higher priority than b (lower f value = higher priority).
        bool operator()(const SearchStatePtr& a, const SearchStatePtr& b) const {
            return a->f > b->f; // Note: priority queue puts highest priority first, so we use > for min-heap
        }
    };

    // ==================== 
    // Private variables.
    // ====================
    gco::WorldPtr world_;
    // The goal configuration.
    MultiObjectMultiRobotConfigurationPtr cfg_goal_multi_;
    // The goal tolerance.
    double goal_tolerance_;
    // The weight for weighted A* (w >= 1.0, w=1.0 is regular A*)
    double weight_;
    // The grid resolution.
    double grid_resolution_ = 0.1;
    // The inflation radius.
    double inflation_radius_ = 0.2;
    // Maximum number of iterations to prevent infinite loops
    int max_iterations_ = 100000;

    // All search states, indexed by their multi-object multi-robot configuration.
    std::map<MultiObjectMultiRobotConfigurationPtr, SearchStatePtr, MultiObjectMultiRobotConfigurationComparator> search_states_;

    // The open set. Use priority queue to store the states.
    std::priority_queue<SearchStatePtr, std::vector<SearchStatePtr>, CompareSearchState> open_list_;
    // The closed set. Use a set to store visited multi-object multi-robot configurations.
    std::set<MultiObjectMultiRobotConfigurationPtr, MultiObjectMultiRobotConfigurationComparator> closed_set_;

    // ==================== 
    // Private methods.
    // ==================== 
    // Reconstruct the path for object-only planning.
    MultiRobotPaths reconstructObjectPath(const SearchStatePtr& goal_search_state) const;

    // Reconstruct the path for multi-robot planning.
    MultiRobotPaths reconstructPath(const SearchStatePtr& goal_search_state) const;

    // Get SearchStatePtr from its multi-object multi-robot configuration.
    SearchStatePtr getSearchStateFromConfiguration(const MultiObjectMultiRobotConfigurationPtr& cfg) const;

    // Round a multi-object multi-robot configuration to discrete grid.
    MultiObjectMultiRobotConfigurationPtr roundToGrid(const MultiObjectMultiRobotConfigurationPtr& cfg) const;

    // Get heuristic value for multi-object multi-robot configuration (heuristic computed only on objects).
    double getHeuristic(const MultiObjectMultiRobotConfigurationPtr& cfg_current, const MultiObjectMultiRobotConfigurationPtr& cfg_goal, const std::string& object_name) const;

    // Get successor configurations for the multi-object multi-robot configuration.
    std::vector<MultiObjectMultiRobotConfigurationPtr> getSuccessorConfigurations(const MultiObjectMultiRobotConfigurationPtr& cfg_current) const;

    // Check if multi-object multi-robot configuration is valid (collision-free).
    bool isConfigurationValid(const MultiObjectMultiRobotConfigurationPtr& cfg, bool verbose = false) const;

    // Check if object is at goal.
    bool isObjectAtGoal(const MultiObjectMultiRobotConfigurationPtr& cfg_current, const MultiObjectMultiRobotConfigurationPtr& cfg_goal, const std::string& object_name, const double tolerance) const;

    // Check that the start and goal configurations are valid.
    void checkStartAndGoalConfigurationsOrDie(const std::map<std::string, Configuration2>& cfgs_start_robots, 
        const std::map<std::string, Configuration2>& cfg_start_objects,
        const std::map<std::string, Configuration2>& cfg_goal_objects) const;

};

} // namespace gco