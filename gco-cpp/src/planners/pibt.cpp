// Project includes.
#include "gco/planners/pibt.hpp"
#include "gco/utils.hpp"
#include "gco/heuristics/bwd_dijkstra_heuristic.hpp"
#include "gco/heuristics/euclidean_heuristic.hpp"
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
#include <chrono>

gco::PIBTPlanner::PIBTPlanner(const WorldPtr& world, const double goal_tolerance, const std::map<std::string, std::string>& heuristic_params, unsigned int seed, double timeout_seconds, bool disregard_orientation, double goal_check_tolerance) 
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

gco::PlannerStats gco::PIBTPlanner::plan(const std::map<std::string, Configuration2>& cfgs_start_raw, 
                                         const std::map<std::string, Configuration2>& cfgs_goal_raw,
                                         const std::map<Configuration2, HeuristicPtr>& goal_heuristics,
                                         bool is_modify_starts_goals_locally) {
    std::cout << "[PIBT] Planning is not yet implemented in this snippet." << std::endl;
                                            return PlannerStats();
}

gco::PIBTPlanner::~PIBTPlanner() {
    // Destructor implementation
}

bool gco::PIBTPlanner::areAllRobotsAtGoal(const std::map<std::string, PIBTStatePtr>& pibt_states, 
                                          const std::map<std::string, Configuration2>& cfgs_goal) const {
    for (const auto& [robot_name, pibt_state] : pibt_states) {
        if (!isRobotAtGoal(robot_name, pibt_state->path.back()->back(), cfgs_goal)) {
            return false;
        }
    }
    return true;
}

bool gco::PIBTPlanner::isRobotAtGoal(const std::string& robot_name, const Configuration2& cfg_current, 
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
    
    double tolerance = (goal_check_tolerance_ > 0) ? goal_check_tolerance_ : 0.01;

    return distance < tolerance;
}

bool gco::PIBTPlanner::isRobotNearGoal(const std::string& robot_name, const Configuration2& cfg_current, 
    const std::map<std::string, Configuration2>& cfgs_goal, 
    const double tolerance) const {
    const Configuration2& goal_pos = cfgs_goal.at(robot_name);
    double distance;
    
    if (disregard_orientation_) {
        // Only consider x, y position, ignore orientation
        distance = std::sqrt(std::pow(cfg_current.x - goal_pos.x, 2) + std::pow(cfg_current.y - goal_pos.y, 2));
    } else {
        // Use full configuration distance including orientation
        distance = configurationDistance(cfg_current, goal_pos);
    }
    
    return distance < tolerance;
}


double gco::PIBTPlanner::getHeuristic(const Configuration2& cfg_current, const Configuration2& cfg_goal) const {
    // Get heuristic from the goal heuristics map.
    auto it = goal_heuristics_.find(cfg_goal);
    if (it != goal_heuristics_.end()) {
        return it->second->getHeuristic(cfg_current, cfg_goal);
    }
    
    std::stringstream ss;
    ss << "No heuristic available for goal " << cfg_goal;
    throw std::runtime_error(ss.str());
}

bool gco::PIBTPlanner::pibtProcedure(const std::string& robot_name, 
                                     std::map<std::string, PIBTStatePtr>& pibt_states,
                                     const std::map<std::string, Configuration2>& cfgs_goal,
                                     int t_plus_1,
                                     int recursion_depth,
                                     std::shared_ptr<std::set<std::string>> affected_robots) {
    // ==============================
    // Create the indent string for the debug output.
    // ==============================
    std::string indent = "[PIBT] ";
    for (int i = 0; i < recursion_depth; i++) {
        indent += " >  ";
    }
    debug() << indent << "[" << robot_name << "] " << "PIBT procedure for robot " << robot_name << " at time step [" << t_plus_1-1 << ", " << t_plus_1 << "]" << std::endl;
    
    // Add this robot to the set of affected robots
    if (affected_robots) {
        affected_robots->insert(robot_name);
    }
    
    // ==============================
    // Check if the robot planned for this timestep [t, t+1] already.
    // ==============================
    // For πi[t], collect neighbors and wait edges in C:={(ai,ej)| ej∈E}
    // Get the current configuration at time t_plus_1-1
    Configuration2 current_cfg;
    if (t_plus_1 <= pibt_states[robot_name]->path.size()) {
        // We already have an edge for this time step, use its first configuration.
        current_cfg = pibt_states[robot_name]->path[t_plus_1-1]->front();
        debug() << indent << "[" << robot_name << "] " << "  Already has an assigned edge for this time step." << std::endl;
    } 
    else {
        // We don't have an edge for this time step yet, use the last configuration of the previous edge
        current_cfg = pibt_states[robot_name]->path.back()->back();
    }
    
    // ==============================
    // Get all possible edges.
    // ==============================
    std::vector<ActionSequencePtr> all_edges = getAllPossibleEdges(robot_name, current_cfg, cfgs_goal);
    
    // Sort C based on heuristic
    std::sort(all_edges.begin(), all_edges.end(), 
              [this, &cfgs_goal, &robot_name](const ActionSequencePtr& a, const ActionSequencePtr& b) {
                  return getHeuristic(a->back(), cfgs_goal.at(robot_name)) < 
                         getHeuristic(b->back(), cfgs_goal.at(robot_name));
              });

    // Print the sorted edges' last state with associated heuristics.
    debug() << indent << "[" << robot_name << "] " << "  Sorted edges' last states with associated heuristics:" << std::endl;
    for (const auto& edge : all_edges) {
        debug() << indent << "[" << robot_name << "] " << "    Edge ending at " << edge->back() << " with heuristic " << getHeuristic(edge->back(), cfgs_goal.at(robot_name)) << std::endl;
    }

    // ==============================
    // Try each of the sorted edges.
    // ==============================
    // for (ai,e) ∈ C, do:
    for (const auto& edge : all_edges) {
        debug() << indent << "[" << robot_name << "] " << "  Trying edge from " << edge->front() << " to " << edge->back() << std::endl;
        
        // If ∃ak s.t. πk[t,t+1] ∩ e ≠ ∅, continue
        std::vector<std::string> preassigned_robot_names_stepped_on = getPreassignedRobotsSteppedOnByEdge(robot_name, edge, pibt_states, t_plus_1);
        debug() << indent << "[" << robot_name << "] " << "    Edge collides with preassigned agents (";
        for (auto robot_name_stepped_on : preassigned_robot_names_stepped_on){
            debug() << robot_name_stepped_on << ", ";
        }
        debug() << ")" << std::endl;

        if (!preassigned_robot_names_stepped_on.empty()) {
            debug() << indent << "[" << robot_name << "] " << "    Skipping edge." << std::endl;
            continue;
        }
        
        // ==============================
        // Set the tentative edge for the robot.
        // ==============================
        // Set πi[t,t+1] ← e (Tentative edge for ai)
        // Store the current state to restore if needed
        std::vector<EdgePtr> old_path = pibt_states[robot_name]->path;
        
        // Store the original state of all robots before making changes
        std::map<std::string, std::vector<EdgePtr>> original_paths;
        for (const auto& [name, state] : pibt_states) {
            original_paths[name] = state->path;
        }
        
        // Update the robot's state with the new edge
        if (t_plus_1 == pibt_states[robot_name]->getPlannedUpToTime()) {
            pibt_states[robot_name]->path[t_plus_1-1] = edge;
        } else {
            pibt_states[robot_name]->path.push_back(edge);
        }
        
        // ==============================
        // Collect all robots that the tentative edge steps on.
        // ==============================
        // Collect all ak s.t. e steps on πk[t] and πk[t,t+1] = ⊥
        std::set<std::string> stepped_on_robots = getUnassignedRobotsSteppedOnByEdge(robot_name, edge, pibt_states, t_plus_1);
        debug() << indent << "[" << robot_name << "] " << "   Unassigned robots-stepped-on " << stepped_on_robots << std::endl;
        
        // ==============================
        // Call PIBT for all unassigned robots that the tentative edge steps on.
        // Try all permutations of stepped-on robots until one succeeds.
        // ==============================
        
        // Convert set to vector for permutation
        std::vector<std::string> stepped_robots_vec(stepped_on_robots.begin(), stepped_on_robots.end());
        
        // Try all permutations of stepped-on robots
        bool found_valid_permutation = false;
        do {
            debug() << indent << "[" << robot_name << "] " << "    Trying permutation: ";
            for (const auto& name : stepped_robots_vec) {
                debug() << name << " ";
            }
            debug() << std::endl;
            
            // Store the state before trying this permutation
            std::map<std::string, std::vector<EdgePtr>> permutation_start_paths;
            for (const auto& [name, state] : pibt_states) {
                permutation_start_paths[name] = state->path;
            }
            
            std::vector<bool> results;
            bool permutation_failed = false;
            
            // Try this specific permutation
            for (const auto& stepped_robot_name : stepped_robots_vec) {
                debug() << indent << "[" << robot_name << "] " << "      Calling PIBT for stepped-on robot " << stepped_robot_name << std::endl;
                bool result = pibtProcedure(stepped_robot_name, pibt_states, cfgs_goal, t_plus_1, recursion_depth + 1, affected_robots);
                results.push_back(result);
                
                // If this robot failed, this permutation failed
                if (!result) {
                    debug() << indent << "[" << robot_name << "] " << "      Robot " << stepped_robot_name << " failed in this permutation" << std::endl;
                    permutation_failed = true;
                    break;
                }
            }
            
            // Check if all robots in this permutation succeeded
            if (!permutation_failed && std::all_of(results.begin(), results.end(), [](bool r) { return r; })) {
                debug() << indent << "[" << robot_name << "] " << "    All stepped-on robots succeeded in this permutation, returning valid" << std::endl;
                found_valid_permutation = true;
                break;
            }
            
            // This permutation failed, restore state and try next permutation
            debug() << indent << "[" << robot_name << "] " << "    This permutation failed, restoring state" << std::endl;
            for (const auto& [name, original_path] : permutation_start_paths) {
                pibt_states[name]->path = original_path;
            }

            // TEST TEST TEST. Break after the first permutation.
            break;
            // END TEST TEST TEST.
            
        } while (std::next_permutation(stepped_robots_vec.begin(), stepped_robots_vec.end()));
        
        // If we found a valid permutation, return success
        if (found_valid_permutation) {
            return true;
        }
        
        // // Else continue (restore state and try next edge)
        debug() << indent << "[" << robot_name << "] " << "    Some stepped-on robots failed, restoring state and trying next edge" << std::endl;
        
        // Reset paths of all affected robots by this recursive call
        if (affected_robots) {
            for (const auto& affected_robot : *affected_robots) {
                if (affected_robot != robot_name) {
                    // Restore the original path for affected robots
                    pibt_states[affected_robot]->path = original_paths[affected_robot];
                    debug() << indent << "[" << robot_name << "] " << "    Reset path for affected robot " << affected_robot << std::endl;
                }
            }
        }
        
        pibt_states[robot_name]->path = old_path;
    }
    
    // ==============================
    // If got here, all options of edges failed. Set πi[t,t+1] ← WaitAt(πi[t]).
    // ==============================
    // if (recursion_depth == 0){
        debug() << RED << indent << "[" << robot_name << "] " << "  All edges failed, setting wait edge at " << current_cfg << " and returning invalid" << RESET << std::endl;
        EdgePtr wait_edge = createWaitEdge(current_cfg);
        if (pibt_states[robot_name]->getPlannedUpToTime() == t_plus_1) {
            pibt_states[robot_name]->path[t_plus_1-1] = wait_edge;
        } else {
            pibt_states[robot_name]->path.push_back(wait_edge);
        }
    // }
    
    // Wait edges are always valid (they don't step on anyone)
    return false;
}

std::vector<gco::ActionSequencePtr> gco::PIBTPlanner::getAllPossibleEdges(const std::string& robot_name, 
                                                                          const Configuration2& cfg_current,
                                                                          const std::map<std::string, Configuration2>& cfgs_goal) {
    std::vector<EdgePtr> all_edges;
    
    // Get successor edges from the world
    std::vector<EdgePtr> successor_edges;
    std::vector<std::string> successor_edges_names;
    world_->getSuccessorEdges(robot_name, cfg_current, successor_edges, successor_edges_names);
    
    // Add all successor edges
    all_edges.insert(all_edges.end(), successor_edges.begin(), successor_edges.end());

    // Add a snap edge to the goal, if the robot is near it.
    if (isRobotNearGoal(robot_name, cfg_current, cfgs_goal, goal_tolerance_)) {
        auto snap_edge = createSnapEdge(cfg_current, cfgs_goal.at(robot_name));
        if (world_->isPathValid(robot_name, snap_edge)) {
            all_edges.push_back(snap_edge);
            debug() << GREEN << "--- Snap edge is added, valid for robot " << robot_name << ". It ends at " << snap_edge->back() << " which is " << configurationDistance(snap_edge->back(), cfgs_goal.at(robot_name)) << " away from the goal." << RESET << std::endl;
        }
        else {
            debug() << RED << "--- Snap edge is not valid for robot " << robot_name << RESET << std::endl;
        }
    }
    else {
        debug() << RED << "--- Snap edge is not added for robot " << robot_name << ". It is " << configurationDistance(cfg_current, cfgs_goal.at(robot_name)) << " away from the goal. Tolerance is " << goal_tolerance_ << RESET << std::endl;
    }

    // NOTE(yoraish): not adding wait edge when at goal since the snap edge would effectively do that.
    
    return all_edges;
}

bool gco::PIBTPlanner::isWaitEdge(EdgePtr edge){
    for (int i = 1; i < edge->size(); i++){
        if (edge->at(0) != edge->at(i)){
            return false;
        }
    }
    return true;
}

std::vector<std::string> gco::PIBTPlanner::getPreassignedRobotsSteppedOnByEdge(const std::string& robot_name,
                                                      const ActionSequencePtr& edge,
                                                      const std::map<std::string, PIBTStatePtr>& pibt_states,
                                                      int t_plus_1) {
    std::vector<std::string> robot_names_stepped_on;
    // Check if the edge collides with any robot that already has a path assigned for this time step
    for (const auto& [other_robot_name, other_state] : pibt_states) {
        if (other_robot_name == robot_name) continue;

        
        // Check if the other robot has a path assigned for this time step
        if (other_state->getPlannedUpToTime() == t_plus_1) {
            // Get the other robot's edge for this time step transition ending at t+1.
            EdgePtr other_edge = other_state->path.at(t_plus_1-1);
            
            // Check collision by stepping along both edges synchronously
            if (other_edge && other_edge->size() == edge->size()) {
                for (size_t i = 1; i < edge->size(); ++i) {
                    Configuration2 edge_cfg = (*edge)[i];
                    Configuration2 other_cfg = (*other_edge)[i];

                    // Check collision for all configurations - don't skip any
                    // The collision detection will handle the actual collision logic
                    
                    std::map<std::string, Configuration2> cfgs;
                    cfgs[robot_name] = edge_cfg;
                    cfgs[other_robot_name] = other_cfg;
                    
                    CollisionResult collision_result;
                    world_->checkCollision(cfgs, collision_result);
                    if (!collision_result.collisions.empty()) {
                        robot_names_stepped_on.push_back(other_robot_name);
                        break;
                    }
                }
            }
            else {
                throw std::runtime_error("Error in edge collision checking.");
            }
        }
    }
    return robot_names_stepped_on;
}

std::set<std::string> gco::PIBTPlanner::getUnassignedRobotsSteppedOnByEdge(const std::string& robot_name,
                                                                 const ActionSequencePtr& edge,
                                                                 const std::map<std::string, PIBTStatePtr>& pibt_states,
                                                                 int t_plus_1) {
    // The time t+1 is the time of the end of the edge.
    std::set<std::string> stepped_on_robots;
    
    for (const auto& [other_robot_name, other_state] : pibt_states) {
        if (other_robot_name == robot_name) continue;
        
        // Check if the other robot has a path assigned for this time step
        if (other_state->getPlannedUpToTime() >= t_plus_1) {
            continue;
        } else {
            assert (other_state->getPlannedUpToTime() == t_plus_1-1);
            // Check against the current position of the other agent
            Configuration2 other_current_cfg = other_state->path.back()->back();
            
            for (const auto& edge_cfg : *edge) {
                // Check for collisions at all distances - let the collision detection handle it
                std::map<std::string, Configuration2> cfgs;
                cfgs[robot_name] = edge_cfg;
                cfgs[other_robot_name] = other_current_cfg;
                
                CollisionResult collision_result;
                world_->checkCollision(cfgs, collision_result, true, false);
                if (!collision_result.collisions.empty()) {
                    stepped_on_robots.insert(other_robot_name);
                    break;
                }
            }
        }
    }
    
    return stepped_on_robots;
}

gco::ActionSequencePtr gco::PIBTPlanner::createWaitEdge(const Configuration2& cfg_current) {
    auto wait_edge = std::make_shared<ActionSequence>();
    // Create a wait edge with 6 steps (same as other edges)
    for (int i = 0; i < 6; i++) {
        wait_edge->push_back(cfg_current);
    }
    return wait_edge;
}

gco::EdgePtr gco::PIBTPlanner::createSnapEdge(const Configuration2& cfg_current, const Configuration2& cfg_goal) {
    auto snap_edge = std::make_shared<Edge>();
    // Create an interpolated edge with 6 steps (same as other edges).
    for (int i = 0; i < 6; i++) {
        Configuration2 cfg_interp = Configuration2(cfg_current.x + (cfg_goal.x - cfg_current.x) * i/5.0,
                                                   cfg_current.y + (cfg_goal.y - cfg_current.y) * i/5.0,
                                                   cfg_current.theta + (cfg_goal.theta - cfg_current.theta) * i/5.0);
        snap_edge->push_back(cfg_interp);
    }
    return snap_edge;
}

std::vector<std::string> gco::PIBTPlanner::sortRobotsByPriority(const std::map<std::string, PIBTStatePtr>& pibt_states) {
    std::vector<std::string> robot_names;
    for (const auto& [robot_name, state] : pibt_states) {
        robot_names.push_back(robot_name);
    }
    
    // Sort by priority (higher priority first)
    std::sort(robot_names.begin(), robot_names.end(),
              [&pibt_states](const std::string& a, const std::string& b) {
                  return pibt_states.at(a)->priority > pibt_states.at(b)->priority;
              });
    
    debug() << "Sorted A" << std::endl;
    for (std::string a : robot_names){
        debug() << "Name: " << a << " priority " << pibt_states.at(a)->priority << std::endl;
    }
    return robot_names;
}

void gco::PIBTPlanner::updatePriorities(std::map<std::string, PIBTStatePtr>& pibt_states,
                                        const std::map<std::string, Configuration2>& cfgs_goal) {
    /**
    Set each robot priority as eps_i if at goal, otherwise increment by one.
    */
    for (auto& [robot_name, state] : pibt_states) {
        // Check if robot is at its goal
        bool at_goal = isRobotAtGoal(robot_name, state->path.back()->back(), cfgs_goal);
        
        if (at_goal) {
            // Reset to original priority when at goal
            state->priority = state->original_priority;
        } else {
            // Increment priority by 1 when not at goal
            state->priority += 1.0;
        }
    }
}


bool gco::PIBTPlanner::planSingleStep(std::map<std::string, PIBTStatePtr>& pibt_states,
                                        const std::map<std::string, Configuration2>& cfgs_goal,
                                        int t_plus_1) {
    debug() << "=== PIBT Single Step: Movement Planning for time [" << t_plus_1-1 << ", " << t_plus_1 << "] ===" << std::endl;
    
    bool any_movement = false;
    
    // Sort robots by priority
    std::vector<std::string> robot_names = sortRobotsByPriority(pibt_states);
    
    debug() << "Positions: " << std::endl;
    for (std::string robot_name : robot_names) {
        debug() << " * " << robot_name << " at " << pibt_states[robot_name]->getLastCfg() << std::endl;    
    }
    
    // Process each robot in priority order
    for (const auto& robot_name : robot_names) {
        debug() << GREEN << "Robot " << robot_name << RESET;
        
        // Only try to plan for the current time step if it is not already planned.
        if (pibt_states[robot_name]->getPlannedUpToTime() >= t_plus_1) {
            debug() << RED << "Robot " << robot_name << " is already planned for time step [" << pibt_states[robot_name]->getPlannedUpToTime() << ", " << t_plus_1 << "]" << RESET << std::endl;
            continue;
        }

        debug() << RED << " needs planning" << RESET << " It is now at " << pibt_states[robot_name]->getLastCfg() << std::endl;
        debug() << "Calling PIBT for robot " << robot_name << " (path size " << pibt_states[robot_name]->path.size() << ") extending time step [" << t_plus_1-1 << ", " << t_plus_1 << "]" << std::endl;
        
        auto affected_robots = std::make_shared<std::set<std::string>>();
        bool success = pibtProcedure(robot_name, pibt_states, cfgs_goal, t_plus_1, 0, affected_robots);
        if (success) {
            any_movement = true;
            debug() << GREEN << "PIBT succeeded for robot " << robot_name << RESET << " It is now at " << pibt_states[robot_name]->getLastCfg() << std::endl;
        } else {
            debug() << RED << "PIBT failed for robot " << robot_name << RESET << std::endl;
        }
    }
    
    // Update priorities for next iteration
    updatePriorities(pibt_states, cfgs_goal);
    
    return any_movement;
}
