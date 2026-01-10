// Project includes.
#include "gco/heuristics/bwd_dijkstra_heuristic.hpp"
#include "gco/utils.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <queue>
#include <sstream>

namespace gco {

BwdDijkstraHeuristic::BwdDijkstraHeuristic(const WorldPtr& world, 
                                           double grid_resolution, 
                                           double max_distance_meters)
    : world_(world), grid_resolution_(grid_resolution), max_distance_meters_(max_distance_meters) {
}

double BwdDijkstraHeuristic::getHeuristic(const Configuration2& cfg_current, const Configuration2& cfg_goal) const {
    // If the current configuration is extremely close to the goal, return 0.
    if (configurationDistance(cfg_current, cfg_goal) < 0.01) {
        return 0.0;
    }

    // Otherwise, return the cell value with a small epsilon.
    double epsilon = 0.01;

    // Round the current configuration to the nearest grid point
    Configuration2 rounded_cfg = roundToGrid(cfg_current);
    
    // Round the goal configuration to the same grid resolution for consistent lookup
    Configuration2 rounded_goal = roundToGrid(cfg_goal);
    
    // Check if we have precomputed heuristic values for this goal
    auto goal_it = heuristic_values_.find(rounded_goal);
    if (goal_it != heuristic_values_.end()) {
        auto cell_it = goal_it->second.find(rounded_cfg);
        if (cell_it != goal_it->second.end()) {
            return cell_it->second + epsilon;
        }
        else { 
            // Find nearest cell that does exist. Sorry about this gigantor hack.
            double min_dist = std::numeric_limits<double>::max();
            Configuration2 nearest_cfg;
            for (const auto& [cell_cfg, heuristic_value] : goal_it->second) {
                double dist = configurationDistance(cell_cfg, rounded_cfg);
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest_cfg = cell_cfg;
                }
            }
            if (min_dist > 2 * grid_resolution_) {
                std::cout << "Heuristic could not find (rounded) cell " << rounded_cfg << " for (rounded) goal " << rounded_goal <<". Searching for nearest cell... ";
                std::cout << "Found nearest cell " << nearest_cfg << " with distance " << min_dist << " is too far. Aborting." << std::endl;
                throw std::runtime_error("Heuristic could not find (rounded) cell " + std::to_string(rounded_cfg.x) + ", " + std::to_string(rounded_cfg.y) + " for (rounded) goal " + std::to_string(rounded_goal.x) + ", " + std::to_string(rounded_goal.y) + ". Searching for nearest cell... Found nearest cell " + std::to_string(nearest_cfg.x) + ", " + std::to_string(nearest_cfg.y) + " with distance " + std::to_string(min_dist) + " is too far. Aborting.");
            }
            return getHeuristic(nearest_cfg, cfg_goal) + epsilon;
        }
    }
    // Error if no precomputed value is available.
    std::stringstream ss;
    ss << "No heuristic available for goal " << cfg_goal << " (rounded): " << rounded_goal << " and current " << cfg_current << " rounded to " << rounded_cfg;
    throw std::runtime_error(ss.str());
}

double BwdDijkstraHeuristic::getHeuristic(const Configuration2& cfg_current, 
                                         const Configuration2& cfg_neighbor, 
                                         const Configuration2& cfg_goal) const {
    // Get the current grid cell
    Configuration2 neighbor_cell = roundToGrid(cfg_neighbor);
    return getHeuristic(neighbor_cell, cfg_goal);
}

void BwdDijkstraHeuristic::precomputeForGoal(const Configuration2& cfg_goal, const std::string& robot_name) {
    auto time_start = std::chrono::high_resolution_clock::now();

    // Round the goal configuration to the grid resolution for consistent storage and lookup
    Configuration2 rounded_goal_cfg = roundToGrid(cfg_goal);
    
    // Use a regular queue for BFS (FIFO)
    std::queue<Configuration2> q;
    
    // Map to store distances from goal to each configuration
    std::map<Configuration2, double> distances;
    
    // Set to track visited configurations
    std::set<Configuration2> visited;
    
    // Initialize: goal has distance 0
    q.push(rounded_goal_cfg);
    distances[rounded_goal_cfg] = 0.0;
    visited.insert(rounded_goal_cfg);
    
    // Configuration discretization parameters for exploration
    const double pos_step = grid_resolution_;  // Grid resolution steps
    
    int nodes_expanded = 0;
    const double max_distance_meters = max_distance_meters_;  // Limit to prevent excessive computation
    
    while (!q.empty()) {
        Configuration2 current_cfg = q.front();
        q.pop();
        
        double current_dist = distances[current_cfg];
        nodes_expanded++;
        
        // Check if we've reached the maximum exploration distance
        if (current_dist > max_distance_meters) {
            continue;
        }
        
        // Explore neighbors in 8 directions
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (dx == 0 && dy == 0) continue; // Skip current position
                if (std::abs(dx) == 1 && std::abs(dy) == 1) continue; // Skip diagonal neighbors.
                
                Configuration2 neighbor_cfg;
                neighbor_cfg.x = current_cfg.x + dx * pos_step;
                neighbor_cfg.y = current_cfg.y + dy * pos_step;
                neighbor_cfg.theta = 0.0; // All configurations have theta = 0 in our coarse discretization

                neighbor_cfg = roundToGrid(neighbor_cfg);
                
                // Skip if already visited
                if (visited.find(neighbor_cfg) != visited.end()) {
                    continue;
                }
                
                // Check if neighbor is valid (not in collision)
                std::map<std::string, Configuration2> cfgs;
                cfgs[robot_name] = neighbor_cfg;
                
                CollisionResult collision_result;
                world_->checkCollision(cfgs, collision_result);
                
                if (collision_result.collisions.empty()) {
                    // Calculate distance to neighbor
                    double edge_cost = std::sqrt(dx * dx + dy * dy) * pos_step;
                    double new_dist = current_dist + edge_cost;

                    if (new_dist > max_distance_meters) {
                        continue;
                    }
                    
                    // Mark as visited and add to queue
                    visited.insert(neighbor_cfg);
                    distances[neighbor_cfg] = new_dist;
                    q.push(neighbor_cfg);
                }
            }
        }
    }
    
    // Store the computed heuristic values
    heuristic_values_[rounded_goal_cfg] = distances;
    
    auto time_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
    std::cout << "\rBFS for " << cfg_goal << " Res " << grid_resolution_ << " max dist " << max_distance_meters << " m"
              << " in " << duration.count() << "ms, expanded " << nodes_expanded << " nodes" << std::flush;
}

void BwdDijkstraHeuristic::clearPrecomputedValues() {
    heuristic_values_.clear();
}

Configuration2 BwdDijkstraHeuristic::roundToGrid(const Configuration2& cfg) const {
    // Round to the nearest grid point
    Configuration2 rounded_cfg;
    rounded_cfg.x = std::round(cfg.x / grid_resolution_) * grid_resolution_;
    rounded_cfg.y = std::round(cfg.y / grid_resolution_) * grid_resolution_;
    rounded_cfg.theta = 0.0;  // All configurations have theta = 0 in our coarse discretization
    
    return rounded_cfg;
}

void BwdDijkstraHeuristic::saveHeuristicData(const std::string& filename, const std::vector<BwdDijkstraHeuristic>& heuristics) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    file << "# goal_x goal_y cell_x cell_y heuristic_value" << std::endl;

    // Save this heuristic as well as concatenated other heuristics.
    for (const auto& heuristic : heuristics) {
        for (const auto& [goal_cfg, cell_values] : heuristic.heuristic_values_) {
            for (const auto& [cell_cfg, heuristic_value] : cell_values) {
                file << goal_cfg.x << " " << goal_cfg.y << " " << goal_cfg.theta << " "
                     << cell_cfg.x << " " << cell_cfg.y << " " << cell_cfg.theta << " "
                     << heuristic_value << std::endl;
            }
        }
    }
    
    file.close();
    std::cout << "Heuristic data saved to " << filename << std::endl;
}

} // namespace gco 