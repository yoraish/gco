// Project includes.
#include "gco/planners/gspi.hpp"
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

gco::GSPIPlanner::GSPIPlanner(const WorldPtr& world, const double goal_tolerance, const std::map<std::string, std::string>& heuristic_params, unsigned int seed, double timeout_seconds, bool disregard_orientation, double goal_check_tolerance) 
    : BasePlanner(world, goal_tolerance, timeout_seconds), disregard_orientation_(disregard_orientation) {
    // Set the seed
    seed_ = seed;
    
    // Store the additional parameter
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
    
    // Initialize the CTSWAP and PIBT planners with heuristic parameters
    ctswap_planner_ = std::make_unique<CTSWAPPlanner>(world, goal_tolerance, heuristic_params, seed, timeout_seconds, disregard_orientation_, goal_check_tolerance_);
    pibt_planner_ = std::make_unique<PIBTPlanner>(world, goal_tolerance, heuristic_params, seed, timeout_seconds, disregard_orientation_, goal_tolerance_/2.0);
}



gco::GSPIPlanner::~GSPIPlanner() {
    // Destructor implementation
}

std::map<std::string, gco::Configuration2> gco::GSPIPlanner::modifyCfgsLocallyUntilValid(std::map<std::string, Configuration2> cfgs) {
    return ctswap_planner_->modifyCfgsLocallyUntilValid(cfgs);
}

std::map<std::string, gco::Configuration2> gco::GSPIPlanner::modifyCfgsLocallyUntilValidRadial(std::map<std::string, Configuration2> cfgs) {
    return ctswap_planner_->modifyCfgsLocallyUntilValidRadial(cfgs);
}

gco::PlannerStats gco::GSPIPlanner::plan(const std::map<std::string, Configuration2>& cfgs_start_raw, 
                                             const std::map<std::string, Configuration2>& cfgs_goal_raw,
                                             const std::map<Configuration2, HeuristicPtr>& goal_heuristics,
                                             bool is_modify_starts_goals_locally) {
    std::map<std::string, Configuration2> cfgs_start = cfgs_start_raw;
    std::map<std::string, Configuration2> cfgs_goal = cfgs_goal_raw;

    // Create a copy of the world with objects converted to obstacles for heuristic computation
    WorldPtr world_for_heuristics = world_->createCopyWithObjectsAsObstacles();
    ctswap_planner_->setWorld(world_for_heuristics);
    pibt_planner_->setWorld(world_for_heuristics);
    world_ = world_for_heuristics;

    // Modify the start and goal configurations locally until valid.
    if (is_modify_starts_goals_locally) {
        debug() << "Modifying start cfgs locally until valid." << std::endl;
        cfgs_start = modifyCfgsLocallyUntilValidRadial(cfgs_start);
        cfgs_goal = modifyCfgsLocallyUntilValidRadial(cfgs_goal);
        if (cfgs_start.empty() || cfgs_goal.empty()) {
            return PlannerStats();
        }
    }

    std::vector<std::pair<std::string, Configuration2>> cfgs_start_vec(cfgs_start.begin(), cfgs_start.end());

    // Handle heuristics - create world for heuristics if needed
    if (!goal_heuristics.empty()) {
        goal_heuristics_ = goal_heuristics;
    } else {
        
        // Create heuristics for all goals
        std::map<std::string, std::shared_ptr<BwdDijkstraHeuristic>> all_heuristics_for_viz;
        for (const auto& [robot_name, cfg_goal] : cfgs_goal) {
            if (goal_heuristics_.find(cfg_goal) == goal_heuristics_.end()) {
                if (heuristic_type_ == "bwd_dijkstra") {
                    debug() << "[Hybrid] Creating heuristic for goal " << cfg_goal << std::endl;
                    auto heuristic = std::make_shared<BwdDijkstraHeuristic>(world_for_heuristics, grid_resolution_heuristic_, max_distance_meters_heuristic_);
                    auto start_time = std::chrono::high_resolution_clock::now();
                    heuristic->precomputeForGoal(cfg_goal, robot_name);
                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                    goal_heuristics_[cfg_goal] = heuristic;
                    all_heuristics_for_viz[robot_name] = heuristic;
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
        if (verbose_) {
            if (heuristic_type_ == "bwd_dijkstra") {
                for (const auto& [robot_name_viz, heuristic_ptr] : all_heuristics_for_viz) {
                    BwdDijkstraHeuristic::saveHeuristicData("heuristic_data_" + robot_name_viz + ".txt", {*heuristic_ptr});
                }
            }
        }

    }

    // Assign heuristics to the sub-planners
    for (const auto& [cfg_goal, heuristic] : goal_heuristics_) {
        ctswap_planner_->addGoalHeuristic(cfg_goal, heuristic);
        pibt_planner_->addGoalHeuristic(cfg_goal, heuristic);
    }

    // Start timing for planning (excluding heuristic computation)
    auto planning_start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize hybrid states for all robots
    std::map<std::string, HybridStatePtr> hybrid_states;
    int robot_index = 0;
    for (const auto& [robot_name, cfg_start] : cfgs_start) {
        // Assign unique original priorities between 0 and 1
        double original_priority;
        if (cfgs_start.size() == 1) {
            original_priority = 0.5; // Single robot gets middle priority
        } else {
            original_priority = static_cast<double>(robot_index) / (cfgs_start.size() - 1);
        }
        hybrid_states[robot_name] = std::make_shared<HybridState>(cfg_start, original_priority);
        robot_index++;
    }

    // Set verbose mode for sub-planners
    ctswap_planner_->setVerbose(verbose_);
    pibt_planner_->setVerbose(verbose_);

    int iteration_count = 0;
    int t_plus_1 = 1; // Start planning for time step 1
    
    // Track iteration timing
    std::vector<double> iteration_times;
    auto iteration_start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize progress tracker
    ProgressTracker progress_tracker(max_iterations_);
    
    while (!areAllRobotsAtGoal(hybrid_states, cfgs_goal) && iteration_count < max_iterations_) {
        // Check for timeout using base class method
        PlannerStats timeout_stats;
        if (checkTimeout(planning_start_time, iteration_count, timeout_stats)) {

            // Reconstruct the paths from the hybrid states.
            MultiRobotPaths paths;
            double soc = 0;
            int makespan = 0;
            for (const auto& [robot_name, hybrid_state] : hybrid_states) {
                for (size_t edge_idx = 0; edge_idx < hybrid_state->path.size(); edge_idx++) {
                    makespan = std::max(makespan, (int)edge_idx);
                    EdgePtr edge = hybrid_state->path[edge_idx];
                    
                    // For the first edge, include all configurations
                    // For subsequent edges, skip the first configuration to avoid duplication
                    int start_idx = (edge_idx == 0) ? 0 : 1;
                    
                    for (int i = start_idx; i < edge->size(); i++){
                        // Print if there is a big jump.
                        if (!paths[robot_name].empty()) {
                            soc += configurationDistance(edge->at(i), paths[robot_name].back());
                        }
        
                        Configuration2 cfg = edge->at(i);
                        paths[robot_name].push_back(cfg);
                    }
                }
            }
            timeout_stats.paths = paths;
            return timeout_stats;
        }
        
        // Update progress indicator
        progress_tracker.updateProgress(iteration_count);
        
        debug() << "\n\n=== Hybrid Iteration " << iteration_count << " ===\n\n\n" << std::endl;

        for (const auto& [robot_name, hybrid_state] : hybrid_states) {
            for (const auto& [other_robot_name, other_hybrid_state] : hybrid_states) {
                if (hybrid_states[robot_name]->priority < hybrid_states[other_robot_name]->priority) {
                    continue;
                }
                else {
                    // Check if swapping goals is better.
                    auto cfg_goal_robot = cfgs_goal.at(robot_name);
                    auto cfg_goal_other_robot = cfgs_goal.at(other_robot_name);
                    double dist_current = goal_heuristics_[cfg_goal_robot]->getHeuristic(hybrid_states[robot_name]->getLastCfg(), cfg_goal_robot) + goal_heuristics_[cfg_goal_other_robot]->getHeuristic(hybrid_states[other_robot_name]->getLastCfg(), cfg_goal_other_robot);
                    double dist_swap = goal_heuristics_[cfg_goal_other_robot]->getHeuristic(hybrid_states[robot_name]->getLastCfg(), cfg_goal_other_robot) + goal_heuristics_[cfg_goal_robot]->getHeuristic(hybrid_states[other_robot_name]->getLastCfg(), cfg_goal_robot);
                    double dist_current_robot_after_swap = goal_heuristics_[cfg_goal_robot]->getHeuristic(hybrid_states[other_robot_name]->getLastCfg(), cfg_goal_robot) + goal_heuristics_[cfg_goal_other_robot]->getHeuristic(hybrid_states[robot_name]->getLastCfg(), cfg_goal_other_robot);
                    if (dist_swap < dist_current && dist_current_robot_after_swap <= dist_current) {
                        // Swap the goals.
                        cfgs_goal[robot_name] = cfg_goal_other_robot;   
                        cfgs_goal[other_robot_name] = cfg_goal_robot;
                        // Swap the priorities.
                        double priority_robot = hybrid_states[robot_name]->priority;
                        double priority_other_robot = hybrid_states[other_robot_name]->priority;
                        hybrid_states[robot_name]->priority = priority_other_robot;
                        hybrid_states[other_robot_name]->priority = priority_robot;
                    }
                }
            }
        }          
        
        // Run one iteration of PIBT movement planning
        debug() << "--- Stage 2: PIBT Movement Planning ---" << std::endl;
        bool pibt_progress = runPIBTMovement(hybrid_states, cfgs_goal, t_plus_1);

        // Record iteration time
        auto iteration_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> iteration_duration = iteration_end_time - iteration_start_time;
        iteration_times.push_back(iteration_duration.count());
        iteration_start_time = iteration_end_time; // Reset for next iteration
        
        // Increment time step for next iteration
        t_plus_1++;
        iteration_count++;
    }
    
    // Final progress update and newline with timing summary
    progress_tracker.finalize(iteration_count);

    // Reconstruct the paths from the hybrid states
    MultiRobotPaths paths;
    double soc = 0;
    int makespan = 0;
    for (const auto& [robot_name, hybrid_state] : hybrid_states) {
        for (size_t edge_idx = 0; edge_idx < hybrid_state->path.size(); edge_idx++) {
            makespan = std::max(makespan, (int)edge_idx);
            EdgePtr edge = hybrid_state->path[edge_idx];
            
            // For the first edge, include all configurations
            // For subsequent edges, skip the first configuration to avoid duplication
            int start_idx = (edge_idx == 0) ? 0 : 1;
            
            for (int i = start_idx; i < edge->size(); i++){
                // Print if there is a big jump.
                if (!paths[robot_name].empty()) {
                    soc += configurationDistance(edge->at(i), paths[robot_name].back());
                }

                Configuration2 cfg = edge->at(i);
                paths[robot_name].push_back(cfg);
            }
        }
    }

    if (areAllRobotsAtGoal(hybrid_states, cfgs_goal)) {
        debug() << GREEN << "All robots are at their goals." << RESET << std::endl;
    }

    // Check if we reached max iterations
    if (iteration_count >= max_iterations_) {
        std::cout << RED << "[GSPI] Warning: Reached maximum iterations (" << max_iterations_ << "). Returning current paths." << RESET << std::endl;
        // Return the first three states of each robot.
        auto paths_copy = paths;
        for (const auto& [robot_name, hybrid_state] : hybrid_states) {
            for (int i = 0; i < 3; i++) {
                paths_copy[robot_name].push_back(hybrid_state->path[i]->back());
            }
        }
        paths = paths_copy;
        // Returned empty paths.
        paths = MultiRobotPaths();
    }

    debug() << GREEN << "SOC  = " << soc << RESET << std::endl;
    debug() << GREEN << "MKSP = " << makespan << RESET << std::endl;

    // End timing for planning
    auto planning_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> planning_duration = planning_end_time - planning_start_time;
    
    // Create and return PlannerStats using base class method
    return createSuccessStats(paths, planning_start_time, iteration_count, iteration_count >= max_iterations_, iteration_times);
}

bool gco::GSPIPlanner::runPIBTMovement(std::map<std::string, HybridStatePtr>& hybrid_states,
                                                  const std::map<std::string, Configuration2>& cfgs_goal,
                                                  int t_plus_1) {
    debug() << "=== PIBT Movement Planning ===" << std::endl;
    
    // Convert hybrid states to PIBT states
    std::map<std::string, PIBTStatePtr> pibt_states;
    for (const auto& [robot_name, hybrid_state] : hybrid_states) {
        // Create PIBT state with the same priority as hybrid state
        pibt_states[robot_name] = std::make_shared<PIBTState>(hybrid_state->cfg, hybrid_state->priority);
        pibt_states[robot_name]->original_priority = hybrid_state->original_priority;
        // Copy the accumulated path from hybrid state
        pibt_states[robot_name]->path = hybrid_state->path;
        debug() << "Copied path from hybrid state for " << robot_name << " with path size " << hybrid_state->path.size() << std::endl;
    }
    
    // Run PIBT single step for movement planning
    bool pibt_progress = pibt_planner_->planSingleStep(pibt_states, cfgs_goal, t_plus_1);
    
    // Update hybrid states with any changes from PIBT
    for (const auto& [robot_name, pibt_state] : pibt_states) {
        hybrid_states[robot_name]->cfg = pibt_state->getLastCfg();
        hybrid_states[robot_name]->priority = pibt_state->priority;
        hybrid_states[robot_name]->path = pibt_state->path;
        debug() << "Updated hybrid state for " << robot_name << " with path size " << pibt_state->path.size() << std::endl;
    }
    
    return pibt_progress;
}

bool gco::GSPIPlanner::areAllRobotsAtGoal(const std::map<std::string, HybridStatePtr>& hybrid_states, 
                                                      const std::map<std::string, Configuration2>& cfgs_goal) const {
    for (const auto& [robot_name, hybrid_state] : hybrid_states) {
        if (!isRobotAtGoal(robot_name, hybrid_state->getLastCfg(), cfgs_goal)) {
            return false;
        }
    }
    return true;
}

bool gco::GSPIPlanner::isRobotAtGoal(const std::string& robot_name, const Configuration2& cfg_current, 
                                                 const std::map<std::string, Configuration2>& cfgs_goal) const {
    const Configuration2& goal_pos = cfgs_goal.at(robot_name);
    double distance;
    
    if (disregard_orientation_) {
        // Only consider x, y position, ignore orientation
        distance = std::sqrt(std::pow(cfg_current.x - goal_pos.x, 2) + std::pow(cfg_current.y - goal_pos.y, 2));
    } else {
        // Use full configuration distance including orientation
        distance = configurationDistance(cfg_current, goal_pos);
    }
    
    return distance < goal_tolerance_/2.0;
}

double gco::GSPIPlanner::getHeuristic(const Configuration2& cfg_current, const Configuration2& cfg_goal) const {
    // Get heuristic from the goal heuristics map
    auto it = goal_heuristics_.find(cfg_goal);
    if (it != goal_heuristics_.end()) {
        return it->second->getHeuristic(cfg_current, cfg_goal);
    }
    
    // If no heuristic is available, raise an error.
    std::stringstream ss;
    ss << "No heuristic available for goal " << cfg_goal;
    throw std::runtime_error(ss.str());
}

void gco::GSPIPlanner::setGoalHeuristics(const std::map<Configuration2, HeuristicPtr>& goal_heuristics) {
    goal_heuristics_ = goal_heuristics;
    
    // Update the goal heuristics in the sub-planners
    if (ctswap_planner_) {
        ctswap_planner_->setGoalHeuristics(goal_heuristics);
    }
    if (pibt_planner_) {
        pibt_planner_->setGoalHeuristics(goal_heuristics);
    }
}



