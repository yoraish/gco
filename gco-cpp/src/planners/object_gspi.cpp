// Project includes.
#include "gco/planners/object_gspi.hpp"
#include "gco/heuristics/euclidean_heuristic.hpp"
#include "gco/utils.hpp"
#include "gco/heuristics/bwd_dijkstra_heuristic.hpp"
#include "gco/utils/progress_tracker.hpp"
#include <random>
#include <algorithm>
#include <queue>
#include <cmath>
#include <set>
#include <limits>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <iomanip>

gco::ObjectGSPIPlanner::ObjectGSPIPlanner(const WorldPtr& world, const double goal_tolerance, const std::map<std::string, std::string>& heuristic_params, unsigned int seed, double timeout_seconds, bool disregard_orientation) 
    : BasePlanner(world, goal_tolerance, timeout_seconds), original_world_(world), disregard_orientation_(disregard_orientation) {
    // Set the seed
    seed_ = seed;
    
    // Parse heuristic parameters
    if (heuristic_params.find("grid_resolution") != heuristic_params.end()) {
        grid_resolution_heuristic_ = std::stod(heuristic_params.at("grid_resolution"));
    }
    if (heuristic_params.find("max_distance_meters") != heuristic_params.end()) {
        max_distance_meters_heuristic_ = std::stod(heuristic_params.at("max_distance_meters"));
    }
    if (heuristic_params.find("object_radius") != heuristic_params.end()) {
        object_radius_ = std::stod(heuristic_params.at("object_radius"));
    }

    if (heuristic_params.find("heuristic_type") != heuristic_params.end()) {
        heuristic_type_ = heuristic_params.at("heuristic_type");
    }
    
    // Initialize the CTSWAP and PIBT planners with heuristic parameters
    // Create a world with all objects as robots (consistent obstacle landscape)
    WorldPtr dummy_world = createObjectPlanningWorld();
    ctswap_planner_ = std::make_unique<CTSWAPPlanner>(dummy_world, goal_tolerance, heuristic_params, seed, timeout_seconds);
    pibt_planner_ = std::make_unique<PIBTPlanner>(dummy_world, goal_tolerance, heuristic_params, seed, timeout_seconds, disregard_orientation_, goal_tolerance_);
    ctswap_planner_->setVerbose(verbose_);
    pibt_planner_->setVerbose(verbose_);
}

gco::ObjectGSPIPlanner::~ObjectGSPIPlanner() {
}

void gco::ObjectGSPIPlanner::initializeObjectTargets(const std::map<std::string, Configuration2>& object_targets) {
    object_targets_ = object_targets;
    
    object_states_.clear();
    for (const auto& [object_name, target_cfg] : object_targets_) {
        object_states_[object_name] = std::make_shared<ObjectHybridState>(target_cfg, 0.0);
    }

    WorldPtr object_world = createObjectPlanningWorld();
    
    // Update planners with new world
    ctswap_planner_->setWorld(object_world);
    pibt_planner_->setWorld(object_world);
    world_ = object_world;

    // Handle heuristics - create world for heuristics if needed
    if (goal_heuristics_.empty()) {
        debug() << "[ObjectGSPI] Creating heuristics with type: " << heuristic_type_ << std::endl;
        
        // For object planning, we'll use euclidean heuristics to avoid complexity
        // The BwdDijkstra heuristic requires robot names and proper world setup
        for (const auto& [object_name, cfg_goal] : object_targets_) {
            if (goal_heuristics_.find(cfg_goal) == goal_heuristics_.end()) {
                if (heuristic_type_ == "bwd_dijkstra") {
                    debug() << "[ObjectGSPI] Creating bwd_dijkstra heuristic for goal " << cfg_goal << std::endl;
                    // Use the object name as robot name since objects are converted to robots
                    auto heuristic = std::make_shared<BwdDijkstraHeuristic>(object_world, grid_resolution_heuristic_, max_distance_meters_heuristic_);
                    auto start_time = std::chrono::high_resolution_clock::now();
                    heuristic->precomputeForGoal(cfg_goal, object_name);
                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                    std::cout << GREEN << "Time taken to precompute heuristic for goal " << cfg_goal << " and object " << object_name << " is " << duration.count() << "ms" << RESET << std::endl;
                    goal_heuristics_[cfg_goal] = heuristic;
                }
                else if (heuristic_type_ == "euclidean") {
                    auto heuristic = std::make_shared<EuclideanHeuristic>();
                    goal_heuristics_[cfg_goal] = heuristic;
                }
                else {
                    throw std::runtime_error("Invalid heuristic type: " + heuristic_type_);
                }
            }
        }
    }
    
    // Assign heuristics to the sub-planners
    for (const auto& [cfg_goal, heuristic] : goal_heuristics_) {
        ctswap_planner_->addGoalHeuristic(cfg_goal, heuristic);
        pibt_planner_->addGoalHeuristic(cfg_goal, heuristic);
    }
    
    debug() << "Initialized ObjectGSPI with " << object_targets_.size() << " object targets" << std::endl;
}

std::map<std::string, std::vector<gco::Configuration2>> gco::ObjectGSPIPlanner::getNextObjectMoves(
    const std::map<std::string, gco::Configuration2>& current_object_positions,
    int horizon) {
    
    // Initialize hybrid states for all robots, if needed.
    if (first_call_) {
        first_call_ = false;

        int robot_index = 0;
        for (const auto& [robot_name, cfg_current] : current_object_positions) {
            // Assign unique original priorities between 0 and 1
            double original_priority;
            if (current_object_positions.size() == 1) {
                original_priority = 0.5; // Single robot gets middle priority
            } else {
                original_priority = static_cast<double>(robot_index) / (current_object_positions.size() - 1);
            }
            object_states_[robot_name] = std::make_shared<ObjectHybridState>(cfg_current, original_priority);
            robot_index++;
        }
    }
    else {
        // Just keep the existing hybrid states, but clear the paths, keeping only the last edge.
        for (auto& [object_name, state] : object_states_) {
            state->path = {state->path.back()};
        }
    }
    // The object to be returned.
    std::map<std::string, std::vector<gco::Configuration2>> object_moves;

    
    // Update object states with current positions
    for (const auto& [object_name, current_pos] : current_object_positions) {
        if (object_states_.find(object_name) != object_states_.end()) {
            object_states_[object_name]->cfg = current_pos;
        }
    }
    
    // Check if all objects are at goals
    if (areAllObjectsAtGoal(current_object_positions)) {
        debug() << "All objects are at their goals" << std::endl;
        for (const auto& [object_name, current_pos] : current_object_positions) {
            std::vector<Configuration2> wait_moves;
            // Add horizon + 1 wait moves (staying at current position) for each object
            for (int i = 0; i <= horizon; i++) {
                wait_moves.push_back(current_pos);
            }
            object_moves[object_name] = wait_moves;
        }
        return object_moves; // Return wait edges for all objects
    }
    
    // Prepare start and goal configurations for planning
    std::map<std::string, gco::Configuration2> cfgs_start;
    std::map<std::string, gco::Configuration2> cfgs_goal;
    
    // Initialize current assignments if empty (first call)
    if (current_assignments_.empty()) {
        current_assignments_ = object_targets_;
    }
    
    for (const auto& [object_name, current_pos] : current_object_positions) {
        if (object_targets_.find(object_name) != object_targets_.end()) {
            cfgs_start[object_name] = current_pos;
            cfgs_goal[object_name] = current_assignments_[object_name];
        }
        else {
            throw std::runtime_error("Object " + object_name + " not found in object targets");
        }
    }
    
    // Update object states with current configurations
    for (const auto& [object_name, cfg_start] : cfgs_start) {
        if (object_states_.find(object_name) != object_states_.end()) {
            object_states_[object_name]->cfg = cfg_start;
        }
    }
    
    // Run planning iterations
    int iteration = 1;
    
    while (iteration < horizon) {

        // Each robot r looks through all lower-priority robots r'. If the distance of swapping goals dist(r,g') + dist(r', g) is less than the distance of the current goal dist(r, g) + dist(r', g'), then swap the goals. This means also swap the heuristics.
        for (const auto& [robot_name, object_state] : object_states_) {
            for (const auto& [other_robot_name, other_object_state] : object_states_) {
                if (object_states_[robot_name]->priority < object_states_[other_robot_name]->priority) {
                    continue;
                }
                if (robot_name == other_robot_name) {
                    continue;
                }
                // Check if swapping goals is better.
                auto cfg_goal_robot = cfgs_goal.at(robot_name);
                auto cfg_goal_other_robot = cfgs_goal.at(other_robot_name);
                // Get the heuristics.
                // Distance of the high priority robot to its goal.
                double dist_current = goal_heuristics_[cfg_goal_robot]->getHeuristic(object_states_[robot_name]->getLastCfg(), cfg_goal_robot);

                // Distance of the high priority robot to the low priority robot's goal.
                double dist_current_robot_to_other_goal = goal_heuristics_[cfg_goal_other_robot]->getHeuristic(object_states_[robot_name]->getLastCfg(), cfg_goal_other_robot);

                // Distance of the low priority robot to its goal.
                double dist_current_other = goal_heuristics_[cfg_goal_other_robot]->getHeuristic(object_states_[other_robot_name]->getLastCfg(), cfg_goal_other_robot);

                // Distance of the low priority robot to the high priority robot's goal.
                double dist_current_other_to_robot_goal = goal_heuristics_[cfg_goal_robot]->getHeuristic(object_states_[other_robot_name]->getLastCfg(), cfg_goal_robot);
                // Distance sum current.
                double dist_current_sum = dist_current + dist_current_other;

                // Distance sum current after swap.
                double dist_current_sum_after_swap = dist_current_robot_to_other_goal + dist_current_other_to_robot_goal;

                if (dist_current_robot_to_other_goal < dist_current && dist_current_sum_after_swap <= dist_current_sum) {
                    // Swap the goals.
                    cfgs_goal[robot_name] = cfg_goal_other_robot;   
                    cfgs_goal[other_robot_name] = cfg_goal_robot;
                    // Swap the priorities.
                    double priority_robot = object_states_[robot_name]->priority;
                    double priority_other_robot = object_states_[other_robot_name]->priority;
                    object_states_[robot_name]->priority = priority_other_robot;
                    object_states_[other_robot_name]->priority = priority_robot;
                }
            }
        }  
        std::cout << RED << "cfgs_goal: " << RESET << std::endl;
        for (const auto& [robot_name, cfg_goal] : cfgs_goal) {
            std::cout << " * " << robot_name << " -> " << cfg_goal << std::endl;
        }
        std::cout << RED << "object_states_: " << RESET << std::endl;
        for (const auto& [robot_name, object_state] : object_states_) {
            std::cout << " * " << robot_name << " -> " << object_state->getLastCfg() << "   priority: " << object_state->priority << std::endl;
        }
        std::cout << std::endl;
            
        current_assignments_ = cfgs_goal;
        
        // Stage 2: Run PIBT movement planning
        bool movement_success = runPIBTMovement(object_states_, cfgs_goal, iteration + 1);
        
        // Check if all objects are at their goals
        if (areAllObjectsAtGoal(cfgs_start)) {
            debug() << "All objects reached their goals in " << iteration << " iterations" << std::endl;
            break;
        }
        
        iteration++;
    }
    
    // Extract the path for each object. This is a limited horizon path.
    for (const auto& [object_name, state] : object_states_) {
        std::vector<Configuration2> moves;
        
        // Get the start state for this object
        Configuration2 start_state = cfgs_start[object_name];
        
        // Extract moves from the planned path
        for (int i = 0; i <= horizon; i++) {
            if (i < state->path.size()) {
                moves.push_back(state->path[i]->back());
            } else if (!moves.empty()) {
                // If we have some moves but not enough, repeat the last move
                moves.push_back(moves.back());
            } else {
                // If we have no moves at all, repeat the start state
                moves.push_back(start_state);
            }
        }
        
        // Always add moves (we guarantee horizon + 1 moves)
        object_moves[object_name] = moves;
    }
    
    return object_moves;
}

bool gco::ObjectGSPIPlanner::areAllObjectsAtGoal(const std::map<std::string, gco::Configuration2>& current_positions) const {
    for (const auto& [object_name, current_pos] : current_positions) {
        if (!isObjectAtGoal(object_name, current_pos)) {
            return false;
        }
    }
    return true;
}

bool gco::ObjectGSPIPlanner::isObjectAtGoal(const std::string& object_name, const gco::Configuration2& current_pos) const {
    if (object_targets_.find(object_name) == object_targets_.end()) {
        return true; // Object not in targets, consider it "at goal"
    }
    
    const gco::Configuration2& goal_pos = object_targets_.at(object_name);
    double distance;
    
    if (disregard_orientation_) {
        // Only consider x, y position, ignore orientation
        distance = std::sqrt(std::pow(current_pos.x - goal_pos.x, 2) + std::pow(current_pos.y - goal_pos.y, 2));
    } else {
        // Use full configuration distance including orientation
        distance = configurationDistance(current_pos, goal_pos);
    }
    
    return distance <= goal_tolerance_;
}

std::map<std::string, gco::Configuration2> gco::ObjectGSPIPlanner::getCurrentAssignments() const {
    return current_assignments_;
}

double gco::ObjectGSPIPlanner::getHeuristic(const gco::Configuration2& cfg_current, const gco::Configuration2& cfg_goal) const {
    if (goal_heuristics_.find(cfg_goal) != goal_heuristics_.end()) {
        return goal_heuristics_.at(cfg_goal)->getHeuristic(cfg_current, cfg_goal);
    }
    
    double distance;
    if (disregard_orientation_) {
        // Only consider x, y position, ignore orientation
        distance = std::sqrt(std::pow(cfg_current.x - cfg_goal.x, 2) + std::pow(cfg_current.y - cfg_goal.y, 2));
    } else {
        // Use full configuration distance including orientation
        distance = configurationDistance(cfg_current, cfg_goal);
    }
    
    std::cout << RED << "No heuristic available for goal " << cfg_goal << " returning distance: " << distance << RESET << std::endl;
    return distance;
}

void gco::ObjectGSPIPlanner::setGoalHeuristics(const std::map<gco::Configuration2, HeuristicPtr>& goal_heuristics) {
    goal_heuristics_ = goal_heuristics;
    
    // Update the goal heuristics in the sub-planners
    if (ctswap_planner_) {
        ctswap_planner_->setGoalHeuristics(goal_heuristics);
    }
    if (pibt_planner_) {
        pibt_planner_->setGoalHeuristics(goal_heuristics);
    }
}

void gco::ObjectGSPIPlanner::updateWorldState(const WorldPtr& new_world) {
    original_world_ = new_world;
    // The planners will be updated with the new world in getNextObjectMoves
}

gco::WorldPtr gco::ObjectGSPIPlanner::createObjectPlanningWorld() {
    debug() << "[ObjectGSPI] Creating object planning world..." << std::endl;
    
    // Safety check
    if (!original_world_) {
        debug() << "[ObjectGSPI] ERROR: original_world_ is null!" << std::endl;
        throw std::runtime_error("Original world is null in createObjectPlanningWorld");
    }
    
    // Create a simplified world where:
    // - ALL objects are treated as robots (disks)
    // - Actual robots are disregarded.
    // - Original obstacles are preserved
    
    auto object_world = std::make_shared<World>();
    
    // Add all original obstacles
    for (const auto& [obstacle_name, obstacle_with_pose] : original_world_->getObstacles()) {
        const ObstaclePtr& obstacle = obstacle_with_pose.first;
        const Transform2& pose = obstacle_with_pose.second;
        object_world->addObstacle(obstacle, pose);
    }
    
    // Convert ALL objects to disk robots (not just planned ones)
    for (const auto& [object_name, object_with_pose] : original_world_->getObjects()) {
        const ObjectPtr& object = object_with_pose.first;
        
        // Convert every object to a disk robot
        RobotPtr disk_robot = createDiskRobotFromObject(object_name, object, object_radius_); 
        object_world->addRobot(disk_robot);
    }
    
    return object_world;
}

bool gco::ObjectGSPIPlanner::runCTSWAPGoalAdaptation(std::map<std::string, ObjectHybridStatePtr>& object_states,
                                                        std::map<std::string, gco::Configuration2>& cfgs_goal) {
    debug() << "=== Object planning Stage 1: CTSWAP Goal Adaptation ===" << std::endl;
    debug() << "Object states: " << std::endl;
    for (const auto& [object_name, object_state] : object_states) {
        debug() << " * " << object_name << " at " << object_state->getLastCfg() << std::endl;
    }
    
    // Convert object hybrid states to CTSWAP search states
    std::map<std::string, SearchStatePtr> ctswap_states;
    for (const auto& [object_name, object_state] : object_states) {
        Configuration2 current_cfg = object_state->getLastCfg();
        ctswap_states[object_name] = std::make_shared<SearchState>(current_cfg, nullptr, nullptr);
    }
    
    // Run CTSWAP single step for goal adaptation
    bool ctswap_progress = ctswap_planner_->planSingleStep(ctswap_states, cfgs_goal);
    // Update object states with any changes from CTSWAP
    for (const auto& [object_name, ctswap_state] : ctswap_states) {
        object_states[object_name]->cfg = ctswap_state->cfg;
    }
    
    return ctswap_progress;
}

bool gco::ObjectGSPIPlanner::runPIBTMovement(std::map<std::string, ObjectHybridStatePtr>& object_states,
                                                const std::map<std::string, gco::Configuration2>& cfgs_goal,
                                                int t_plus_1) {
    
    // Safety check
    if (!pibt_planner_) {
        return false;
    }
    
    // Convert object hybrid states to PIBT states
    std::map<std::string, PIBTStatePtr> pibt_states;
    for (const auto& [object_name, object_state] : object_states) {
        // Create PIBT state with the same priority as object state
        pibt_states[object_name] = std::make_shared<PIBTState>(object_state->cfg, object_state->priority);
        pibt_states[object_name]->original_priority = object_state->original_priority;
        // Copy the accumulated path from object state
        pibt_states[object_name]->path = object_state->path;
        debug() << "Copied path from object state for " << object_name << " with path size " << object_state->path.size() << std::endl;
    }
    
    // Run PIBT single step for movement planning
    bool pibt_progress = pibt_planner_->planSingleStep(pibt_states, cfgs_goal, t_plus_1);
    
    // Update object states with any changes from PIBT
    for (const auto& [object_name, pibt_state] : pibt_states) {
        object_states[object_name]->cfg = pibt_state->getLastCfg();
        object_states[object_name]->priority = pibt_state->priority;
        object_states[object_name]->path = pibt_state->path;
        debug() << "Updated object state for " << object_name << " with path size " << pibt_state->path.size() << std::endl;
    }
    return pibt_progress;
}


gco::PlannerStats gco::ObjectGSPIPlanner::plan(const std::map<std::string, gco::Configuration2>& cfgs_start_raw, 
                                                   const std::map<std::string, gco::Configuration2>& cfgs_goal_raw,
                                                   const std::map<gco::Configuration2, HeuristicPtr>& goal_heuristics,
                                                   bool is_modify_starts_goals_locally) {
    PlannerStats stats;
    stats.success = false;
    stats.total_time_seconds = 0.0;
    stats.num_iterations = 0;
    
    std::map<std::string, gco::Configuration2> current_positions = cfgs_start_raw;
    std::map<std::string, std::vector<gco::Configuration2>> object_moves = getNextObjectMoves(current_positions, 3);
    
    if (!object_moves.empty()) {
        stats.success = true;
        stats.num_iterations = 1;
    }
    
    return stats;
}

void gco::ObjectGSPIPlanner::resetPlannerState() {
    // Reset the first call flag to allow reinitialization
    first_call_ = true;
    
    // Clear object states to start fresh
    object_states_.clear();
    
    // Clear current assignments
    current_assignments_.clear();
    
    debug() << "Reset ObjectGSPI planner state" << std::endl;
}

gco::RobotPtr gco::ObjectGSPIPlanner::createDiskRobotFromObject(const std::string& object_name, const ObjectPtr& object, double radius) {
    // Create a disk robot from an object
    // All objects are treated as disks with the specified radius for planning purposes
    
    // Create joint ranges for the robot (using typical values from examples)
    Configuration2 mins(-10.0, -10.0, -M_PI);
    Configuration2 maxs(10.0, 10.0, M_PI);
    Configuration2 discretization(0.01, 0.01, 0.01);
    JointRanges joint_ranges(mins, maxs, discretization);
    
    // Create a RobotDisk with the specified radius
    RobotPtr disk_robot = std::make_shared<RobotDisk>(object_name, joint_ranges, radius);
    
    return disk_robot;
}

gco::ObstaclePtr gco::ObjectGSPIPlanner::convertObjectToObstacle(const std::string& object_name, const ObjectPtr& object) {
    // Convert an object to an obstacle with the same shape
    ObstaclePtr obstacle;
    
    if (auto circle_obj = std::dynamic_pointer_cast<CircleShape>(object->getShape())) {
        obstacle = gco::Obstacle::createCircle(object_name, circle_obj->getRadius());
    } else if (auto square_obj = std::dynamic_pointer_cast<SquareShape>(object->getShape())) {
        obstacle = gco::Obstacle::createSquare(object_name, square_obj->getWidth());
    } else if (auto rect_obj = std::dynamic_pointer_cast<RectangleShape>(object->getShape())) {
        obstacle = gco::Obstacle::createRectangle(object_name, rect_obj->getWidth(), rect_obj->getHeight());
    } else if (auto polygon_obj = std::dynamic_pointer_cast<PolygonShape>(object->getShape())) {
        obstacle = gco::Obstacle::createPolygon(object_name, polygon_obj->getVertices());
    } else {
        // Default to circle if shape type is unknown
        obstacle = gco::Obstacle::createCircle(object_name, 0.1);
    }
    
    return obstacle;
}
