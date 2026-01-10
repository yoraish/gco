// Project includes.
#include "gco/planners/ctswap.hpp"
#include "gco/utils.hpp"
#include <chrono>
#include <limits>

gco::CTSWAPPlanner::CTSWAPPlanner(const WorldPtr& world, const double goal_tolerance, const std::map<std::string, std::string>& heuristic_params, unsigned int seed, double timeout_seconds, bool disregard_orientation, double goal_check_tolerance) 
    : BasePlanner(world, goal_tolerance, timeout_seconds) {
    // Set the seed
    seed_ = seed;
    
    // Store the additional parameters
    disregard_orientation_ = disregard_orientation;
    goal_check_tolerance_ = goal_check_tolerance;
    
    // Parse heuristic parameters
    if (heuristic_params.find("grid_resolution") != heuristic_params.end()) {
        grid_resolution_heuristic_ = std::stod(heuristic_params.at("grid_resolution"));
    }
    if (heuristic_params.find("max_distance_meters") != heuristic_params.end()) {
        max_distance_meters_heuristic_ = std::stod(heuristic_params.at("max_distance_meters"));
    }
    if (heuristic_params.find("heuristic_type") != heuristic_params.end()) {
        heuristic_type_ = heuristic_params.at("heuristic_type");
    }
}



gco::CTSWAPPlanner::~CTSWAPPlanner() {
    // Destructor implementation
}

std::map<std::string, gco::Configuration2> gco::CTSWAPPlanner::modifyCfgsLocallyUntilValid(std::map<std::string, Configuration2> cfgs) {
    std::map<std::string, Configuration2> cfgs_new;
    for (const auto& [robot_name, cfg] : cfgs) {
        Configuration2 cfg_new = cfg;
        // Move the goal in a spiral.
        double radius = 0.0;
        double angle = 0;
        for (int i = 0; i < 10; i++) {
            angle = 0.0;
            int angle_steps = 5;
            radius += 0.01;
            for (int angle_idx = 0; angle_idx < angle_steps; angle_idx++) {
                // Move the goal in a spiral.
                cfg_new.x = cfg_new.x + radius * std::cos(angle);
                cfg_new.y = cfg_new.y + radius * std::sin(angle);
                angle += M_PI / static_cast<double>(angle_steps) * 2.0;
                // Check if the goal is valid. With respect to the world and other robots whose goals we have already modified.
                // Round the goal to the nearest grid point.
                // cfg_new = roundToGridHeuristic(cfg_new);
                cfgs_new[robot_name] = cfg_new;
                CollisionResult collision_result;
                world_->checkCollision(cfgs_new, collision_result);
                if (collision_result.collisions.empty()) {
                    goto valid_cfg;
                }
                else {
                    cfgs_new.erase(robot_name);
                }
            }
        }
        valid_cfg:
        debug() << GREEN << " * Robot " << robot_name << " cfg changed from " << cfg << " to " << cfg_new << RESET << std::endl;
        cfgs_new[robot_name] = cfg_new;
        std::cout << "Robot " << robot_name << " cfg changed (spiral) from " << cfg.x << ", " << cfg.y << " to " << cfg_new.x << ", " << cfg_new.y << std::endl;
    }
    return cfgs_new;
}

std::map<std::string, gco::Configuration2> gco::CTSWAPPlanner::modifyCfgsLocallyUntilValidRadial(std::map<std::string, Configuration2> cfgs) {
    // For each object, find its nearest object. Then move it slowly away from object's centroid until valid.
    int num_steps = 30;
    double step_size = 0.01;
    double object_distance_max = 1.0; // If greater, then just modify locally.

    for (auto& [robot_name, cfg] : cfgs) {
        double min_distance = std::numeric_limits<double>::max();
        Configuration2 nearest_centroid;
        for (const auto& [object_name, object_and_pose] : world_->getAllObstacles()) {
            Transform2 object_pose = object_and_pose.second;
            double distance = std::sqrt((cfg.x - object_pose.x) * (cfg.x - object_pose.x) + (cfg.y - object_pose.y) * (cfg.y - object_pose.y));
            if (distance < min_distance) {
                min_distance = distance;
                nearest_centroid = object_pose;
            }
        }
        
        // if (min_distance > object_distance_max) {
        //     modifyCfgsLocallyUntilValid({cfg});
        //     continue;
        // }

        double step_dx = (cfg.x - nearest_centroid.x);
        double step_dy = (cfg.y - nearest_centroid.y);
        double step_norm = std::sqrt(step_dx * step_dx + step_dy * step_dy);
        step_dx /= step_norm;
        step_dy /= step_norm;

        int num_steps_taken = 0;
        for (; num_steps_taken < num_steps; num_steps_taken++) {
            cfg.x += step_size * step_dx;
            cfg.y += step_size * step_dy;
            CollisionResult collision_result;
            // Round the configuration to the nearest grid point.
            auto cfg_rounded = roundToGridHeuristic(cfg);
            world_->checkCollision({{robot_name, cfg_rounded}}, collision_result);
            if (collision_result.collisions.empty()) {
                break;
            }
        }
        if (num_steps_taken >= num_steps) {
            std::cout << RED << "Robot " << robot_name << " could not find a valid configuration after " << num_steps << " steps." << RESET << std::endl;
            // Return an empty map.
            return {};
        }
        cfgs[robot_name] = cfg;
    }
    return cfgs;
}










//     // return cfgs;

    
    
//     if (cfgs.size() == 1) {
//         return modifyCfgsLocallyUntilValid(cfgs);
//     }
    
//     std::map<std::string, Configuration2> cfgs_new;
//     Configuration2 center_cfg;
//     for (const auto& [robot_name, cfg] : cfgs) {
//         center_cfg.x += cfg.x;
//         center_cfg.y += cfg.y;
//     }
//     center_cfg.x /= cfgs.size();    
//     center_cfg.y /= cfgs.size();

//     // For each robot, move it radially away from the center until valid.
//     std::map<std::string, Configuration2> step_vectors;
//     for (const auto& [robot_name, cfg] : cfgs) {
//         step_vectors[robot_name] = Configuration2(cfg.x - center_cfg.x, cfg.y - center_cfg.y, 0.0);
//         double norm = std::sqrt(step_vectors[robot_name].x * step_vectors[robot_name].x + step_vectors[robot_name].y * step_vectors[robot_name].y);
//         if (norm == 0.0) {
//             debug() << RED << "Robot " << robot_name << " has zero distance to center. Setting step vector to (1, 1)." << RESET << std::endl;
//             step_vectors[robot_name].x = 1.0;
//             step_vectors[robot_name].y = 1.0;
//         }
//         else{
//             step_vectors[robot_name].x /= norm;
//             step_vectors[robot_name].y /= norm;
//         }
//     }

//     // For each robot, move it radially away from the center until valid.
//     for (const auto& [robot_name, cfg] : cfgs) {
//         cfgs_new[robot_name] = cfg;
//         double step_size = 0.01;
//         for (int i = 0; i < 35; i++) {
//             step_size += 0.01;
//             cfgs_new[robot_name].x = cfg.x + step_vectors[robot_name].x * step_size;
//             cfgs_new[robot_name].y = cfg.y + step_vectors[robot_name].y * step_size;
//             // cfgs_new[robot_name] = roundToGridHeuristic(cfgs_new[robot_name]);
//             CollisionResult collision_result;
//             world_->checkCollision(cfgs_new, collision_result);
//             if (collision_result.collisions.empty()) {
//                 debug() << GREEN << "Robot " << robot_name << " cfg changed from " << cfg << " to " << cfgs_new[robot_name] << RESET << std::endl;
//                 break;
//             }
//         }
//         std::cout << "Robot " << robot_name << " cfg changed (radial) from " << cfg << " to " << cfgs_new[robot_name] << std::endl;
//     }
//     return cfgs_new;
// }

gco::PlannerStats gco::CTSWAPPlanner::plan(const std::map<std::string, Configuration2>& cfgs_start_raw, 
                                           const std::map<std::string, Configuration2>& cfgs_goal_raw,
                                           const std::map<Configuration2, HeuristicPtr>& goal_heuristics,
                                           bool is_modify_starts_goals_locally) {
    std::map<std::string, Configuration2> cfgs_start = cfgs_start_raw;
    std::map<std::string, Configuration2> cfgs_goal = cfgs_goal_raw;
    if (is_modify_starts_goals_locally) {
        // Move the goals in spiral until they become valid.
        debug() << "Modifying start cfgs locally until valid." << std::endl;
        // cfgs_start = modifyCfgsLocallyUntilValid(cfgs_start_raw);
        cfgs_start = modifyCfgsLocallyUntilValidRadial(cfgs_start);
        cfgs_goal = modifyCfgsLocallyUntilValidRadial(cfgs_goal);
        if (cfgs_start.empty() || cfgs_goal.empty()) {
            return PlannerStats();
        }
        // debug() << "Modifying goal cfgs locally until valid." << std::endl;
        // cfgs_goal = modifyCfgsLocallyUntilValid(cfgs_goal_raw);
    }
    std::map<Configuration2, Configuration2> cfgs_goal_to_cfg_goal_raw;
    for (const auto& [robot_name, cfg_goal] : cfgs_goal) {
        cfgs_goal_to_cfg_goal_raw[cfg_goal] = cfgs_goal_raw.at(robot_name);
    }
    
    // Handle heuristics - create world for heuristics if needed
    if (!goal_heuristics.empty()) {
        goal_heuristics_ = goal_heuristics;
    } else {
        // Create a copy of the world with objects converted to obstacles for heuristic computation
        WorldPtr world_for_heuristics = world_->createCopyWithObjectsAsObstacles();
        
        // Store the original world and temporarily replace it for heuristic computation
        world_ = world_for_heuristics;

        // Precompute backwards Dijkstra heuristics for all goals
        precomputeHeuristics(cfgs_goal);
    }
    
    // Save heuristic data for visualization
    if (verbose_) {
        std::cout << RED << "Saving heuristic data to heuristic_data.txt" << RESET << std::endl;
        saveHeuristicData("heuristic_data.txt");
    }

    // // Assign each robot the nearest goal to it, in simple greedy way without repetition.
    // std::map<std::string, Configuration2> cfgs_goal_simple = cfgs_goal;
    // for (const auto& [robot_name, cfg_start] : cfgs_start) {
    //     double min_distance = std::numeric_limits<double>::max();
    //     Configuration2 cfg_goal_simple;
    //     std::string robot_name_goal_simple;
    //     for (const auto& [robot_name_goal, cfg_goal] : cfgs_goal) {
    //         double distance = configurationDistance(cfg_start, cfg_goal);
    //         if (distance < min_distance) {
    //             min_distance = distance;
    //             cfg_goal_simple = cfg_goal;
    //             robot_name_goal_simple = robot_name_goal;
    //         }
    //     }
    //     cfgs_goal_simple[robot_name] = cfg_goal_simple;
    //     cfgs_goal.erase(robot_name_goal_simple);
    // }
    // cfgs_goal = cfgs_goal_simple;

    // Start timing for planning (excluding heuristic computation)
    auto planning_start_time = std::chrono::high_resolution_clock::now();
    
    // TEST TEST. Operate in turns. For each robot, try to step toward its goal by greedily following a heuristic. Stop when all robots are at their goals (up to some tolerance).
    // Initialize the current configuration of each robot.
    std::map<std::string, SearchStatePtr> search_states_current;
    for (const auto& [robot_name, cfg_start] : cfgs_start) {
        search_states_current[robot_name] = std::make_shared<SearchState>(cfg_start, nullptr, nullptr);
        // Reset closed list for each robot at the start of planning
        resetClosedList(robot_name);
    }

    int iteration_count = 0;
    
    // Track iteration timing
    std::vector<double> iteration_times;
    auto iteration_start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize progress tracker
    ProgressTracker progress_tracker(max_iterations_, "CTSWAP Planner");
    
    while (!areAllRobotsAtGoal(search_states_current, cfgs_goal) && iteration_count < max_iterations_) {
        // Check for timeout using base class method
        PlannerStats timeout_stats;
        if (checkTimeout(planning_start_time, iteration_count, timeout_stats)) {
            return timeout_stats;
        }
        
        // Update progress display
        progress_tracker.updateProgress(iteration_count);
        
        debug() << "=== Iteration " << iteration_count << " ===" << std::endl;
        // Create a randomized order of robots for this iteration
        std::vector<std::string> robot_names;
        for (const auto& [robot_name, cfg_start] : cfgs_start) {
            robot_names.push_back(robot_name);
        }
        // Use deterministic seed for reproducible results
        // std::mt19937 gen(seed_ + iteration_count);  // Different seed for each iteration
        // std::shuffle(robot_names.begin(), robot_names.end(), gen);
        
        for (const auto& robot_name : robot_names) {
            // Check if any of the robots overlap at the same state.
            for (auto [other_robot_name, search_state_other] : search_states_current) {
                if (other_robot_name == robot_name) {
                    continue;
                }
                if (search_state_other->cfg.x == search_states_current[robot_name]->cfg.x &&
                    search_state_other->cfg.y == search_states_current[robot_name]->cfg.y &&
                    search_state_other->cfg.theta == search_states_current[robot_name]->cfg.theta) {
                    throw std::runtime_error("Robot " + robot_name + " and " + other_robot_name + " are at the same state.");
                }
            }
            ActionSequencePtr edge_sequence;
            // If robot is at goal, skip.
            if (isRobotAtGoal(robot_name, search_states_current[robot_name]->cfg, cfgs_goal)) {
                // Add edge staying in place.
                edge_sequence = std::make_shared<ActionSequence>();
                for (int i = 0; i < 6; i++) {
                    edge_sequence->push_back(search_states_current[robot_name]->cfg);
                }
            }
            else{
                // Get the best edge for this robot.
                edge_sequence = getBestEdgeForRobot(robot_name, search_states_current[robot_name]->cfg, cfgs_goal.at(robot_name), search_states_current, false, false);
                if (edge_sequence == nullptr) {
                    throw std::runtime_error("No edge sequence found for robot " + robot_name);
                }

                // Edge processing determines (a) what the actual edge that the robot will take is, (b) potentially a new goal assignment.
                processEdge(robot_name, edge_sequence, search_states_current, cfgs_goal);  
                // BUG HERE I THINK.
            }
            
            // Step along this edge. Record this in the search states.
            SearchStatePtr search_state_new = std::make_shared<SearchState>(edge_sequence->back(), edge_sequence, search_states_current.at(robot_name));
            search_states_current[robot_name] = search_state_new;

            // Add the new configuration to the closed list to avoid revisiting.
            addToClosedList(robot_name, edge_sequence->back());
        }
        
        // Record iteration time
        auto iteration_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> iteration_duration = iteration_end_time - iteration_start_time;
        iteration_times.push_back(iteration_duration.count());
        iteration_start_time = iteration_end_time; // Reset for next iteration
        
        iteration_count++;
    }
    
    // Finalize progress display
    progress_tracker.finalize(iteration_count);

    // Reconstruct the paths.
    MultiRobotPaths paths;
    for (const auto& [robot_name, search_state_current] : search_states_current) {
        // paths[robot_name] = reconstructPathOnlyVertices(search_state_current);
        paths[robot_name] = reconstructPath(search_state_current);
    }

    // Interpolate to the RAW goal configurations to avoid long edges.
    for (const auto& [robot_name, search_state_current] : search_states_current) {
        if (!paths[robot_name].empty()) {
            auto last_cfg = paths[robot_name].back();
            auto goal_cfg = cfgs_goal.at(robot_name);
            // auto goal_cfg = cfgs_goal_to_cfg_goal_raw.at(cfgs_goal.at(robot_name));

            
            // Interpolate linearly between last position and goal with 6 intermediate points
            for (int i = 1; i <= 6; i++) {
                double t = static_cast<double>(i) / 6.0; // t goes from 1/6 to 1.0
                gco::Configuration2 interpolated_cfg;
                interpolated_cfg.x = last_cfg.x + t * (goal_cfg.x - last_cfg.x);
                interpolated_cfg.y = last_cfg.y + t * (goal_cfg.y - last_cfg.y);
                interpolated_cfg.theta = last_cfg.theta + t * (goal_cfg.theta - last_cfg.theta);
                paths[robot_name].push_back(interpolated_cfg);
            }
        } else {
            // If path is empty, just add the raw goal.
            paths[robot_name].push_back(cfgs_goal_to_cfg_goal_raw.at(cfgs_goal.at(robot_name)));
        }
    }

    // End timing for planning
    auto planning_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> planning_duration = planning_end_time - planning_start_time;
    
    // Check if we reached max iterations
    if (iteration_count >= max_iterations_) {
        std::cout << RED << "Warning: Reached maximum iterations (" << max_iterations_ << "). Returning current paths." << RESET << std::endl;
        // Return empty paths.
        paths = MultiRobotPaths();
    }
    
    // Create and return PlannerStats using base class method
    return createSuccessStats(paths, planning_start_time, iteration_count, iteration_count >= max_iterations_, iteration_times);
}


bool gco::CTSWAPPlanner::areAllRobotsAtGoal(const std::map<std::string, SearchStatePtr>& search_states_current, const std::map<std::string, Configuration2>& cfgs_goal) const {
    // Check if all robots are at their goals.
    for (const auto& [robot_name, search_state_current] : search_states_current) {
        if (!isRobotAtGoal(robot_name, search_state_current->cfg, cfgs_goal)) {
            return false;
        }
    }
    return true;
}

bool gco::CTSWAPPlanner::isRobotAtGoal(const std::string& robot_name, const Configuration2& cfg_current, const std::map<std::string, Configuration2>& cfgs_goal) const {
    // Check if the robot is at their goal.
    // Use a very small threshold for "at goal" (same as PIBT)
    return configurationDistance(cfg_current, cfgs_goal.at(robot_name)) < 0.05;
}

bool gco::CTSWAPPlanner::isRobotNearGoal(const std::string& robot_name, const Configuration2& cfg_current, 
    const std::map<std::string, Configuration2>& cfgs_goal, 
    const double tolerance) const {
    return configurationDistance(cfg_current, cfgs_goal.at(robot_name)) < tolerance;
}

gco::ActionSequencePtr gco::CTSWAPPlanner::getBestEdge(const std::vector<ActionSequencePtr>& edge_sequences, const Configuration2& cfg_goal) const {
    // Get the best edge according to a heuristic.
    double best_heuristic = std::numeric_limits<double>::max();
    ActionSequencePtr best_edge;
    for (const auto& edge_sequence : edge_sequences) {
        // double heuristic = getHeuristic(edge_sequence->front(), edge_sequence->back(), cfg_goal);
        double heuristic = getHeuristic(edge_sequence->back(), cfg_goal);
        if (heuristic < best_heuristic) {
            best_heuristic = heuristic;
            best_edge = edge_sequence;
        }
    }
    return best_edge;
}

gco::ActionSequencePtr gco::CTSWAPPlanner::getBestEdgeForRobot(const std::string& robot_name, 
                    const Configuration2& cfg_current, 
                    const Configuration2& cfg_goal, 
                    const std::map<std::string, SearchStatePtr>& search_states_current, 
                    bool is_force_swaps_move_away,
                    bool is_enforce_closed_list) {
    // Get successor edges for the robot at its current configuration.
    std::vector<ActionSequencePtr> edge_sequences;
    std::vector<std::string> edge_sequences_names;
    world_->getSuccessorEdges(robot_name, cfg_current, edge_sequences, edge_sequences_names);

    // Add a snap edge to the goal, if the robot is near it but not already at goal.
    if (isRobotNearGoal(robot_name, cfg_current, {{robot_name, cfg_goal}}, goal_tolerance_) && 
        !isRobotAtGoal(robot_name, cfg_current, {{robot_name, cfg_goal}})) {
        ActionSequencePtr snap_edge = createSnapEdge(cfg_current, cfg_goal);
        // Check if valid.
        if (world_->isPathValid(robot_name, snap_edge, true)) {
            edge_sequences.push_back(snap_edge);  // TODO: This is a hack.
        }
        else {
            debug() << RED << "--- Snap edge is not valid for robot " << robot_name << RESET << std::endl;
        }
    }

    // Prune all edges that step on the other robot and those that move closer to it.
    if (robots_must_step_away_from_.find(robot_name) != robots_must_step_away_from_.end() && is_force_swaps_move_away) {
        std::string other_robot_name = robots_must_step_away_from_.at(robot_name);
        Configuration2 cfg_other = search_states_current.at(other_robot_name)->cfg;

        double distance_to_other_min = configurationDistance(cfg_current, cfg_other);

        auto edge_sequences_new = edge_sequences;
        for (const auto& edge_sequence : edge_sequences) {
            double distance_to_other = configurationDistance(edge_sequence->back(), cfg_other);
            if (distance_to_other < distance_to_other_min) {
                edge_sequences_new.erase(std::remove(edge_sequences_new.begin(), edge_sequences_new.end(), edge_sequence), edge_sequences_new.end());
                continue;
            }


            std::set<std::string> robots_stepped_on = getRobotsSteppedOnByEdge(robot_name, edge_sequence, search_states_current);
            if (robots_stepped_on.find(other_robot_name) != robots_stepped_on.end()) {
                edge_sequences_new.erase(std::remove(edge_sequences_new.begin(), edge_sequences_new.end(), edge_sequence), edge_sequences_new.end());
                continue;
            }
        }
        edge_sequences = edge_sequences_new;
        // Remove this robot.
        robots_must_step_away_from_.erase(robot_name);
        if (is_force_swaps_move_away) {
            // Remove the other robot if it should stay away from this robot.
            if (robots_must_step_away_from_.find(other_robot_name) != robots_must_step_away_from_.end()) {
                robots_must_step_away_from_.erase(other_robot_name);
                debug() << "Removed other robot " << other_robot_name << " from robots_must_step_away_from_ for robot " << robot_name << std::endl;
            }
        }
    }

    if (is_enforce_closed_list) {
    // Filter out edges that lead to visited states (closed list)
    auto edge_sequences_filtered = edge_sequences;
        for (const auto& edge_sequence : edge_sequences) {
            if (isConfigurationInClosedList(robot_name, edge_sequence->back())) {
                edge_sequences_filtered.erase(std::remove(edge_sequences_filtered.begin(), edge_sequences_filtered.end(), edge_sequence), edge_sequences_filtered.end());
                debug() << "Removed edge " << edge_sequence << " from closed list for robot " << robot_name << std::endl;
            }
        }
        edge_sequences = edge_sequences_filtered;
    }

    // Check if this robot has had a recent swap and allow waiting
    bool allow_waiting = robots_with_recent_swaps_.find(robot_name) != robots_with_recent_swaps_.end();
    
    // If no edges are left, create a wait edge
    if (edge_sequences.empty()) {
        // Create a wait edge.
        edge_sequences.push_back(std::make_shared<ActionSequence>());
        for (int i = 0; i < 6; i++) {
            edge_sequences.back()->push_back(cfg_current);
        }
        return edge_sequences.front();
    }
    
    // If robot had a recent swap, consider adding a wait edge as an option
    if (allow_waiting) {
        // Create a wait edge
        auto wait_edge = std::make_shared<ActionSequence>();
        for (int i = 0; i < 6; i++) {
            wait_edge->push_back(cfg_current);
        }
        edge_sequences.push_back(wait_edge);
        
        // Remove the robot from recent gxm set after considering waiting
        robots_with_recent_swaps_.erase(robot_name);
    }

    // Get the best edge according to a heuristic.
    return getBestEdge(edge_sequences, cfg_goal);
}

gco::Configuration2 gco::CTSWAPPlanner::roundToGridHeuristic(const Configuration2& cfg) const {
    // Round to the nearest grid point (0.1 resolution)
    
    Configuration2 rounded_cfg;
    rounded_cfg.x = std::round(cfg.x / grid_resolution_heuristic_) * grid_resolution_heuristic_;
    rounded_cfg.y = std::round(cfg.y / grid_resolution_heuristic_) * grid_resolution_heuristic_;
    rounded_cfg.theta = 0.0;  // All configurations have theta = 0 in our coarse discretization
    
    return rounded_cfg;
}

double gco::CTSWAPPlanner::getHeuristic(const Configuration2& cfg_current, const Configuration2& cfg_goal) const {
    // Get heuristic from the goal heuristics map
    auto it = goal_heuristics_.find(cfg_goal);
    if (it != goal_heuristics_.end()) {
        return it->second->getHeuristic(cfg_current, cfg_goal);
    }
    std::stringstream ss;
    ss << "No heuristic available for goal " << cfg_goal;
    throw std::runtime_error(ss.str());
}

double gco::CTSWAPPlanner::getHeuristic(const Configuration2& cfg_current, const Configuration2& cfg_neighbor, const Configuration2& cfg_goal) const {
    // If the neighbor configuration is extremely close to the goal, return 0.
    if (configurationDistance(cfg_neighbor, cfg_goal) < 0.01) {
        return 0.0;
    }
    // Get heuristic from the goal heuristics map
    auto it = goal_heuristics_.find(cfg_goal);
    if (it != goal_heuristics_.end()) {
        return it->second->getHeuristic(cfg_current, cfg_neighbor, cfg_goal);
    }
    std::stringstream ss;
    ss << "No heuristic available for goal " << cfg_goal;
    throw std::runtime_error(ss.str());
    
}

void gco::CTSWAPPlanner::resetClosedList(const std::string& robot_name) {
    closed_lists_[robot_name].clear();
    debug() << "Reset closed list for robot " << robot_name << std::endl;
}

bool gco::CTSWAPPlanner::isConfigurationInClosedList(const std::string& robot_name, const Configuration2& cfg) const {
    auto it = closed_lists_.find(robot_name);
    if (it != closed_lists_.end()) {
        return it->second.find(cfg) != it->second.end();
    }
    return false;
}

void gco::CTSWAPPlanner::addToClosedList(const std::string& robot_name, const Configuration2& cfg) {
    // Configuration2 rounded_cfg = roundToGridHeuristic(cfg);
    closed_lists_[robot_name].insert(cfg);
    debug() << "Added configuration " << cfg << " to closed list for robot " << robot_name << std::endl;
}

std::vector<gco::Configuration2> gco::CTSWAPPlanner::reconstructPath(const SearchStatePtr& search_state) const {
    // Reconstruct the path.
    std::vector<Configuration2> path;
    SearchStatePtr search_state_current = search_state;
    
    // Collect all search states in reverse order
    std::vector<SearchStatePtr> search_states;
    while (search_state_current != nullptr) {
        search_states.push_back(search_state_current);
        search_state_current = search_state_current->parent_search_state;
    }
    
    // Reverse to get chronological order
    std::reverse(search_states.begin(), search_states.end());
    
    // Build the path including intermediate configurations from action sequences
    for (size_t i = 0; i < search_states.size(); i++) {
        const auto& current_state = search_states[i];
        
        // Add intermediate configurations from the action sequence (if it exists)
        if (current_state->action_sequence_from_parent != nullptr) {
            // Skip the first configuration if it's the same as the previous state's final configuration
            // to avoid duplicates (except for the first state)
            size_t start_idx = (i == 0) ? 0 : 1;
            for (size_t j = start_idx; j < current_state->action_sequence_from_parent->size(); j++) {
                path.push_back((*current_state->action_sequence_from_parent)[j]);
            }
        } else {
            // If no action sequence, just add the current configuration
            path.push_back(current_state->cfg);
        }
    }
    
    return path;
}

std::vector<gco::Configuration2> gco::CTSWAPPlanner::reconstructPathOnlyVertices(const SearchStatePtr& search_state) const {
    // Reconstruct the path only including the vertices (no intermediate configurations).
    std::vector<Configuration2> path;
    SearchStatePtr search_state_current = search_state;
    while (search_state_current != nullptr) {
        path.push_back(search_state_current->cfg);
        search_state_current = search_state_current->parent_search_state;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

void gco::CTSWAPPlanner::processEdge(const std::string& robot_name, ActionSequencePtr& edge_sequence, std::map<std::string, SearchStatePtr>& search_states_current, std::map<std::string, Configuration2>& cfgs_goal) {

    std::set<std::string> robot_names_stepped_on = getRobotsSteppedOnByEdge(robot_name, edge_sequence, search_states_current);

    // The robot steps on (at least) one other robot at their current configuration. Set the edge to a wait edge.
    if (robot_names_stepped_on.empty()) {
        return;
    }

    int edge_length = edge_sequence->size();
    edge_sequence->clear();
    for (int i = 0; i < edge_length; i++) {
        edge_sequence->push_back(edge_sequence->front());
    }

    // If the robot steps on any other robot that is at their goal, then we switch between their goals and return.
    for (const auto& robot_name_coll : robot_names_stepped_on) {
        if (isRobotAtGoal(robot_name_coll, search_states_current[robot_name_coll]->cfg, cfgs_goal)) {
            // Rotate the goals.
            std::vector<std::string> robot_names = {robot_name, robot_name_coll};
            rotateGoals(robot_names, cfgs_goal);
            
            // Reset closed lists for robots whose goals changed
            for (const auto& robot_name_rotated : robot_names) {
                resetClosedList(robot_name_rotated);
                // Add current configuration to closed list
                addToClosedList(robot_name_rotated, search_states_current[robot_name_rotated]->cfg);
                // Mark robot as having had a recent swap
                robots_with_recent_swaps_.insert(robot_name_rotated);
            }
            return;
        }
    }

    // Check if the edge creates any deadlock loops. Each deadlock loop is a vector of ordered robot names. All loops begin with robot_name.
    std::vector<std::vector<std::string>> deadlock_loops = getDeadlockLoops(robot_name, search_states_current, cfgs_goal);

    if (!deadlock_loops.empty()) {
        // if (deadlock_loops.front().size() == 2) {
        //     // Check if the swap is productive.
        //     std::string robot_name_other = deadlock_loops.front().back();
        //     if (!isSwapProductive(robot_name, robot_name_other, search_states_current, cfgs_goal)) {
        //         std::cout << RED << "!!! Warning: Swap is not productive. !!!" << RESET << std::endl;
        //         // Make the other robot step away from this robot.
        //         robots_must_step_away_from_[robot_name_other] = robot_name;
        //         return;
        //     }
        // }
        rotateGoals(deadlock_loops.front(), cfgs_goal);
        
        // Reset closed lists for robots whose goals changed
        for (const auto& robot_name_rotated : deadlock_loops.front()) {
            resetClosedList(robot_name_rotated);
            // Add current configuration to closed list
            addToClosedList(robot_name_rotated, search_states_current[robot_name_rotated]->cfg);
            // Mark robot as having had a recent swap
            robots_with_recent_swaps_.insert(robot_name_rotated);
        }
        return;
    }
}

std::vector<std::vector<std::string>> gco::CTSWAPPlanner::getDeadlockLoops(const std::string& robot_name, 
                                                                           const std::map<std::string, SearchStatePtr>& search_states_current, 
                                                                           const std::map<std::string, Configuration2>& cfgs_goal) {
    // Get the edge sequence for the robot.
    ActionSequencePtr edge_sequence = getBestEdgeForRobot(robot_name, search_states_current.at(robot_name)->cfg, cfgs_goal.at(robot_name), search_states_current, false, false);

    // Get the deadlock loops for a robot.
    std::vector<std::vector<std::string>> deadlock_loops;

    // Get the current configuration of each robot.
    std::map<std::string, Configuration2> cfgs_current;
    for (const auto& [other_robot_name, search_state_current] : search_states_current) {
        cfgs_current[other_robot_name] = search_state_current->cfg;
    }

    // Initialize the stepping chains. We add a two-element vector for each robot that the robot steps on {robot_name, other_robot_name}.
    std::vector<std::vector<std::string>> stepping_chains;
    std::set<std::string> robot_names_stepped_on = getRobotsSteppedOnByEdge(robot_name, edge_sequence, search_states_current);
    // std::set<std::string> robot_names_stepped_on = getRobotsSteppedOnByEdge(robot_name, edge_sequence, cfgs_current);
    for (const auto& robot_name_stepped_on : robot_names_stepped_on) {
        stepping_chains.push_back(std::vector<std::string>{robot_name, robot_name_stepped_on});
    }

    // Extend the stepping chains until they either end (i.e., a robot does not step on another robot) or revisit a robot already in the chain.
    // If the chain ends, we remove it as it is not a deadlock loop.
    // If the chain revisits a robot already in the chain, AND the revisited robot is the first robot in the chain, then we have a deadlock loop. We record it and remove it.
    // A chain extension is all robots that the last robot steps on. If more than one, then duplicate the chain and add all options stepped on.
    while (!stepping_chains.empty()) {
        std::vector<std::vector<std::string>> stepping_chains_new;
        for (const auto& stepping_chain : stepping_chains) {
            std::string robot_name_last = stepping_chain.back();
            ActionSequencePtr edge_sequence_for_last_robot = getBestEdgeForRobot(robot_name_last, cfgs_current[robot_name_last], cfgs_goal.at(robot_name_last), search_states_current, false, false);
            std::set<std::string> robot_names_stepped_on = getRobotsSteppedOnByEdge(robot_name_last, edge_sequence_for_last_robot, search_states_current);
            // std::set<std::string> robot_names_stepped_on = getRobotsSteppedOnByEdge(robot_name_last, edge_sequence_for_last_robot, cfgs_current);
            for (const auto& robot_name_stepped_on : robot_names_stepped_on) {
                std::vector<std::string> stepping_chain_new = stepping_chain;

                // If the robot is already in the chain, then we have a deadlock loop. Only add it if it includes the first robot in the chain.
                if (std::find(stepping_chain_new.begin(), stepping_chain_new.end(), robot_name_stepped_on) != stepping_chain_new.end()) {
                    if (stepping_chain_new.front() == robot_name_stepped_on) {
                        deadlock_loops.push_back(stepping_chain_new);
                    }
                    else{
                        continue;
                    }
                } else {
                    stepping_chain_new.push_back(robot_name_stepped_on);
                    stepping_chains_new.push_back(stepping_chain_new);
                }
            }
        }
        stepping_chains = stepping_chains_new;
    }

    return deadlock_loops;
}

std::set<std::string> gco::CTSWAPPlanner::getRobotsSteppedOnByEdge(const std::string& robot_name, const ActionSequencePtr& edge_sequence, const std::map<std::string, SearchStatePtr>& search_states_current) const {
    // Get the robots stepped on by an edge (edge-to-edge collisions only).
    std::set<std::string> robot_names_stepped_on;
    
    for (const auto& [other_robot_name, search_state_other] : search_states_current) {
        if (other_robot_name == robot_name) {
            continue; // Skip self
        }

        // Get the edge of the other robot.
        ActionSequencePtr edge_sequence_other = search_state_other->action_sequence_from_parent;
        if (edge_sequence_other == nullptr) {
            // Populate with repeated current configuration.
            edge_sequence_other = std::make_shared<ActionSequence>();
            for (int i = 0; i < edge_sequence->size(); i++) {
                edge_sequence_other->push_back(search_state_other->cfg);
            }
        }
        if (edge_sequence_other->size() != edge_sequence->size()) {
            throw std::runtime_error("Edge sequence size mismatch for robot " + robot_name + " and " + other_robot_name);
        }

        // Check for collisions.
        for (int i = 0; i < edge_sequence->size(); i++) {
            Configuration2 cfg1 = (*edge_sequence).at(i);
            Configuration2 cfg2 = (*edge_sequence_other).at(i);

            std::map<std::string, Configuration2> cfgs;
            cfgs[robot_name] = cfg1;
            cfgs[other_robot_name] = cfg2;
            CollisionResult collision_result;
            world_->checkCollision(cfgs, collision_result);
            if (!collision_result.collisions.empty()) {
                debug() << "Robot " << robot_name << " is at configuration " << cfg1 << " and is stepping on robot " << other_robot_name << " at configuration " << cfg2 << std::endl;
                robot_names_stepped_on.insert(other_robot_name);
                break;
            }
        }
    }
    return robot_names_stepped_on;
}


// std::set<std::string> gco::CTSWAPPlanner::getRobotsSteppedOnByEdge(const std::string& robot_name, const ActionSequencePtr& edge_sequence, std::map<std::string, Configuration2> cfgs_current) const {
//     // Get the robots stepped on by an edge.
//     std::set<std::string> robot_names_stepped_on;
//     for (const auto& cfg : *edge_sequence) {
//         cfgs_current[robot_name] = cfg;
//         CollisionResult collision_result;
//         world_->checkCollision(cfgs_current, collision_result);
//         for (const auto& [other_robot_name, collisions] : collision_result.collisions) {
//             for (const auto& collision : collisions) {
//                 std::string other_robot_name = collision.name_entity1 == robot_name ? collision.name_entity2 : collision.name_entity1;
//                 robot_names_stepped_on.insert(other_robot_name);
//             }
//         }       
//     }
//     return robot_names_stepped_on;
// }

void gco::CTSWAPPlanner::rotateGoals(std::vector<std::string>& robot_names, std::map<std::string, Configuration2>& cfgs_goal) {
    // Rotate the goals for a set of robots. Each one gets the goal of the previous robot.
    int N = static_cast<int>(robot_names.size());
    
    // Print debug information about goal rotation
    debug() << "Rotating goals for robots: ";
    for (size_t i = 0; i < robot_names.size(); ++i) {
        if (i > 0) debug() << ", ";
        debug() << robot_names[i];
    }
    debug() << std::endl;
    
    debug() << "Old goals:" << std::endl;
    for (const auto& robot_name : robot_names) {
        debug() << "    Robot " << robot_name << ": [" 
                << cfgs_goal[robot_name].x << ", " 
                << cfgs_goal[robot_name].y << ", " 
                << cfgs_goal[robot_name].theta << "]" << std::endl;
    }
    
    std::map<std::string, Configuration2> cfgs_goal_old = cfgs_goal;
    for (int i = 0; i < N; i++) {
        cfgs_goal[robot_names[i]] = cfgs_goal_old[robot_names[(i-1 + N) % N]]; // Previous robot's goal.
    }
    debug() << "New goals: " << std::endl;
    for (const auto& robot_name : robot_names) {
        debug() << "    Robot " << robot_name << ": [" << cfgs_goal[robot_name].x << ", " << cfgs_goal[robot_name].y << ", " << cfgs_goal[robot_name].theta << "]" << std::endl;
    }

    // Mark the robots that must step away from each other.
    for (int i = 0; i < N; i++) {
        robots_must_step_away_from_[robot_names[i]] = robot_names[(i-1 + N) % N];
    }
}

std::map<std::string, gco::Configuration2> gco::CTSWAPPlanner::getTargetsSteppedOnByEdge(const std::string& robot_name, const ActionSequencePtr& edge_sequence, const std::map<std::string, Configuration2>& cfgs_goal) {
    // Get the targets stepped on by an edge.
    std::map<std::string, gco::Configuration2> targets_stepped_on;

    for (const auto& cfg : *edge_sequence) {
        for (const auto& [other_robot_name, cfg_goal] : cfgs_goal) {
            if (isRobotCfgsOverlapping(other_robot_name, cfg, cfg_goal)) {
                targets_stepped_on[other_robot_name] = cfgs_goal.at(other_robot_name);
                break;
            }
        }
    }
    return targets_stepped_on;
}

bool gco::CTSWAPPlanner::isRobotCfgsOverlapping(const std::string& robot_name, const gco::Configuration2& cfg_current, const gco::Configuration2& cfg_goal) const {
    // Check if a robot is stepping on a configuration.
    return configurationDistance(cfg_current, cfg_goal) < 0.2; // TODO: This is a hack.
}

gco::ActionSequencePtr gco::CTSWAPPlanner::createSnapEdge(const Configuration2& cfg_current, const Configuration2& cfg_goal) {
    auto snap_edge = std::make_shared<ActionSequence>();
    // Create an interpolated edge with 6 steps (same as other edges).
    for (int i = 0; i < 6; i++) {
        Configuration2 cfg_interp = Configuration2(cfg_current.x + (cfg_goal.x - cfg_current.x) * i/5.0,
                                                   cfg_current.y + (cfg_goal.y - cfg_current.y) * i/5.0,
                                                   cfg_current.theta + (cfg_goal.theta - cfg_current.theta) * i/5.0);
        snap_edge->push_back(cfg_interp);
    }
    return snap_edge;
}

void gco::CTSWAPPlanner::precomputeHeuristics(const std::map<std::string, Configuration2>& cfgs_goal) {
    auto time_start = std::chrono::high_resolution_clock::now();
    debug() << "[CTSWAP] Precomputing heuristics for " << cfgs_goal.size() << " goals..." << std::endl;
    
    std::vector<BwdDijkstraHeuristic> all_heuristics_for_viz;
    // Precompute heuristics for all goals using the goal_heuristics_ map
    for (const auto& [robot_name, goal_cfg] : cfgs_goal) {
        debug() << "[CTSWAP] Precomputing heuristic for goal " << robot_name << " at " << goal_cfg << std::endl;
        
        // Check if we have a heuristic for this goal
        auto it = goal_heuristics_.find(goal_cfg);
        if (it != goal_heuristics_.end()) {
            // Precompute the existing heuristic
            it->second->precomputeForGoal(goal_cfg, robot_name);
        } else {
            // If no heuristic is available, create the prescribed heuristic.
            if (heuristic_type_ == "bwd_dijkstra") {
                    auto heuristic = std::make_shared<BwdDijkstraHeuristic>(world_, grid_resolution_heuristic_, max_distance_meters_heuristic_);
                    goal_heuristics_[goal_cfg] = heuristic;
                    heuristic->precomputeForGoal(goal_cfg, robot_name);
                    debug() << "[CTSWAP] Created BwdDijkstra heuristic for goal " << robot_name << " at " << goal_cfg << std::endl;
                    all_heuristics_for_viz.push_back(*heuristic);
            }
            else if (heuristic_type_ == "euclidean") {
                    auto heuristic = std::make_shared<EuclideanHeuristic>();
                    goal_heuristics_[goal_cfg] = heuristic;
                    debug() << "[CTSWAP] Created Euclidean heuristic for goal " << robot_name << " at " << goal_cfg << std::endl;
            }
            else {
                throw std::runtime_error("Invalid heuristic type: " + heuristic_type_);
            }
        }
        if (verbose_) {
            BwdDijkstraHeuristic::saveHeuristicData("heuristic_data.txt", all_heuristics_for_viz);
        }
    }
    
    auto time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = time_end - time_start;
    debug() << "Finished precomputing heuristics in " << duration.count() << " seconds" << std::endl;
}

void gco::CTSWAPPlanner::saveHeuristicData(const std::string& filename) const {
    std::ofstream file(filename); 
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    // Write header
    file << "# Heuristic data for CTSWAP planner" << std::endl;
    file << "# Format: goal_x goal_y goal_theta cfg_x cfg_y cfg_theta distance" << std::endl;
    
    // Write data
    for (const auto& [goal_cfg, distances] : heuristic_values_) {
        for (const auto& [cfg, distance] : distances) {
            file << goal_cfg.x << " " << goal_cfg.y << " " << goal_cfg.theta << " "
                 << cfg.x << " " << cfg.y << " " << cfg.theta << " " << distance << std::endl;
        }
    }
    
    file.close();
    debug() << "Heuristic data saved to " << filename << std::endl;
}

bool gco::CTSWAPPlanner::isSwapProductive(const std::string& robot_name, 
                                          const std::string& robot_name_other, 
                                          const std::map<std::string, SearchStatePtr>& search_states_current, 
                                          const std::map<std::string, Configuration2>& cfgs_goal) {
    // Check if the other robot would want to step on this robot when stepping towards its new goal.
    // New goal for the other robot is this robot's goal.
    Configuration2 cfg_goal_new_other = cfgs_goal.at(robot_name);
    // Get the best edge for the other robot.
    ActionSequencePtr edge_sequence_to_new_goal_other = getBestEdgeForRobot(robot_name_other, search_states_current.at(robot_name_other)->cfg, cfg_goal_new_other, search_states_current, false, false);
    // Get the robots stepped on by this edge.
    std::set<std::string> robots_stepped_on_other = getRobotsSteppedOnByEdge(robot_name_other, edge_sequence_to_new_goal_other, search_states_current);
    // If the first robot is not in the set of robots stepped on, then the swap is productive.
    if (robots_stepped_on_other.find(robot_name) == robots_stepped_on_other.end()) {
        return true;
    }

    // If the other robot is stepping on this robot, then check if this robot would want to step on the other robot when stepping towards its new goal.
    // New goal for this robot is the other robot's goal.
    Configuration2 cfg_goal_new = cfgs_goal.at(robot_name_other);
    // If this robot, in an effort to get to its new goal, does not step on the other robot (at its current position), then the swap is productive.
    // Get the best edge for this robot.
    ActionSequencePtr edge_sequence_to_new_goal = getBestEdgeForRobot(robot_name, search_states_current.at(robot_name)->cfg, cfg_goal_new, search_states_current, false, false);
    // Get the targets stepped on by this edge.
    std::set<std::string> robots_stepped_on = getRobotsSteppedOnByEdge(robot_name, edge_sequence_to_new_goal, search_states_current);
    // If the other robot is not in the set of robots stepped on, then the swap is productive.
    if (robots_stepped_on.find(robot_name_other) == robots_stepped_on.end()) {
        return true;
    }

    // Otherwise, the swap is not productive.
    return false;
}

bool gco::CTSWAPPlanner::planSingleStep(std::map<std::string, SearchStatePtr>& search_states_current, 
                                          std::map<std::string, Configuration2>& cfgs_goal) {
    debug() << "=== CTSWAP Single Step: Goal Adaptation ===" << std::endl;
    
    bool any_goal_swaps = false;
    
    // Create a randomized order of robots for this iteration
    std::vector<std::string> robot_names;
    for (const auto& [robot_name, search_state] : search_states_current) {
        robot_names.push_back(robot_name);
    }
    
    // Use deterministic seed for reproducible results
    std::mt19937 gen(seed_);
    // std::shuffle(robot_names.begin(), robot_names.end(), gen);
    
    for (const auto& robot_name : robot_names) {
        // Skip if robot is already at goal
        if (isRobotAtGoal(robot_name, search_states_current[robot_name]->cfg, cfgs_goal)) {
            continue;
        }
        
        // Get the best edge for this robot
        ActionSequencePtr edge_sequence = getBestEdgeForRobot(robot_name, 
            search_states_current[robot_name]->cfg, 
            cfgs_goal.at(robot_name), 
            search_states_current, 
            false, 
            false);
        
        if (edge_sequence == nullptr) {
            debug() << "No valid edge found for robot " << robot_name << std::endl;
            continue;
        }
        
        // Check if this edge steps on any other robots
        std::set<std::string> robot_names_stepped_on = getRobotsSteppedOnByEdge(robot_name, edge_sequence, search_states_current);
        
        if (robot_names_stepped_on.empty()) {
            // No collision, no need for goal swapping.
            debug() << "No collision, no need for goal swapping for robot " << robot_name << std::endl;
            continue;
        }
        
        // Check for goal swapping opportunities
        for (const auto& robot_name_coll : robot_names_stepped_on) {
            if (isRobotAtGoal(robot_name_coll, search_states_current[robot_name_coll]->cfg, cfgs_goal)) {
                // Swap goals between these robots
                std::vector<std::string> robot_names_to_swap = {robot_name, robot_name_coll};

                rotateGoals(robot_names_to_swap, cfgs_goal);
                
                // Reset closed lists for robots whose goals changed
                for (const auto& robot_name_rotated : robot_names_to_swap) {
                    resetClosedList(robot_name_rotated);
                    addToClosedList(robot_name_rotated, search_states_current[robot_name_rotated]->cfg);
                    robots_with_recent_swaps_.insert(robot_name_rotated);
                }
                
                debug() << "Goal swap between robots " << robot_name << " and " << robot_name_coll << " (at goal)" << std::endl;
                any_goal_swaps = true;
                break; // Only do one swap per robot per iteration
            }
        }
        
        // Check for deadlock loops
        if (!any_goal_swaps) {
        std::vector<std::vector<std::string>> deadlock_loops = getDeadlockLoops(robot_name, search_states_current, cfgs_goal);
        
        if (!deadlock_loops.empty()) {

            // If this is just a pair in the deadlock loop, only rotate if this swap is productive.
            // if (deadlock_loops.front().size() == 2){
            //     // Check if this is a productive swap. Do not swap if it is not.
            //     if (!isSwapProductive(deadlock_loops.front().front(), deadlock_loops.front().back(), search_states_current, cfgs_goal)){
            //         continue;
            //     }
            // }

            // Rotate goals for the first deadlock loop
            rotateGoals(deadlock_loops.front(), cfgs_goal);
            
            // Reset closed lists for robots whose goals changed.
            for (const auto& robot_name_rotated : deadlock_loops.front()) {
                resetClosedList(robot_name_rotated);
                addToClosedList(robot_name_rotated, search_states_current[robot_name_rotated]->cfg);
                robots_with_recent_swaps_.insert(robot_name_rotated);
            }
            
            debug() << "Deadlock loop detected and resolved for robots: ";
            for (const auto& robot_name_loop : deadlock_loops.front()) {
                debug() << robot_name_loop << " ";
            }
            debug() << std::endl;
            any_goal_swaps = true;
        }
        }
    }
    
    return any_goal_swaps;
}