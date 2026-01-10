#pragma once

// General includes.
#include <map>
#include <set>
#include <queue>
#include <chrono>

// Project includes.
#include "gco/heuristics/heuristic.hpp"
#include "gco/world/world.hpp"

namespace gco {

// Backwards BFS heuristic that precomputes shortest paths on a grid.
class BwdDijkstraHeuristic : public Heuristic {
public:
    BwdDijkstraHeuristic(const WorldPtr& world, 
                         double grid_resolution = 0.1, 
                         double max_distance_meters = 4);
    ~BwdDijkstraHeuristic() override = default;

    // Get the heuristic value using precomputed grid-based values.
    double getHeuristic(const Configuration2& cfg_current, const Configuration2& cfg_goal) const override;

    // Get the heuristic value considering the direction to a neighbor state.
    // Returns the heuristic value of the grid cell in the direction of the neighbor.
    double getHeuristic(const Configuration2& cfg_current, 
                       const Configuration2& cfg_neighbor, 
                       const Configuration2& cfg_goal) const override;

    // Precompute backwards BFS heuristic for a given goal.
    void precomputeForGoal(const Configuration2& cfg_goal, const std::string& robot_name) override;

    // Clear all precomputed heuristic values.
    void clearPrecomputedValues() override;

    // Get the name of this heuristic type.
    std::string getName() const override { return "BwdBFS"; }

    // Set grid resolution for heuristic computation.
    void setGridResolution(double resolution) { grid_resolution_ = resolution; }

    // Get grid resolution for heuristic computation.
    double getGridResolution() const { return grid_resolution_; }

    // Set maximum node expansions for heuristic computation.
    void setMaxDistance(double max_distance_meters) { max_distance_meters_ = max_distance_meters; }

    // Get maximum node expansions for heuristic computation.
    double getMaxDistance() const { return max_distance_meters_; }

    // Round configuration to nearest grid point for heuristic lookup.
    Configuration2 roundToGrid(const Configuration2& cfg) const;

    // Save heuristic data to file for visualization.
    static void saveHeuristicData(const std::string& filename, const std::vector<BwdDijkstraHeuristic>& heuristics = {});

private:
    WorldPtr world_;
    double grid_resolution_;
    double max_distance_meters_;
    
    // Map from goal configuration to precomputed heuristic values
    // The inner map is from grid cell to heuristic value
    std::map<Configuration2, std::map<Configuration2, double>> heuristic_values_;
};

} // namespace gco 