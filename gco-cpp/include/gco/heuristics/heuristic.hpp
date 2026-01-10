#pragma once

// General includes.
#include <memory>
#include <map>
#include <string>

// Project includes.
#include <gco/types.hpp>

namespace gco {

// Forward declarations.
class World;

// Base heuristic interface that all heuristic implementations must inherit from.
class Heuristic {
public:
    virtual ~Heuristic() = default;

    // Get the heuristic value from current configuration to goal configuration.
    virtual double getHeuristic(const Configuration2& cfg_current, const Configuration2& cfg_goal) const = 0;

    // Get the heuristic value considering the direction to a neighbor state.
    // This is useful for directional heuristics that consider movement direction.
    virtual double getHeuristic(const Configuration2& cfg_current, 
                               const Configuration2& cfg_neighbor, 
                               const Configuration2& cfg_goal) const = 0;

    // Precompute heuristic values for a given goal configuration.
    // This allows for efficient lookup of precomputed values.
    virtual void precomputeForGoal(const Configuration2& cfg_goal, const std::string& robot_name) = 0;

    // Clear all precomputed heuristic values.
    virtual void clearPrecomputedValues() = 0;

    // Get the name of this heuristic type.
    virtual std::string getName() const = 0;
};

using HeuristicPtr = std::shared_ptr<Heuristic>;

} // namespace gco 