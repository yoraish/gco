#pragma once

// Project includes.
#include "gco/heuristics/heuristic.hpp"

namespace gco {

// Simple Euclidean distance heuristic.
class EuclideanHeuristic : public Heuristic {
public:
    EuclideanHeuristic() = default;
    ~EuclideanHeuristic() override = default;

    // Get the heuristic value using Euclidean distance.
    double getHeuristic(const Configuration2& cfg_current, const Configuration2& cfg_goal) const override;

    // Get the heuristic value considering the direction to a neighbor state.
    // For Euclidean heuristic, this is the same as the regular heuristic.
    double getHeuristic(const Configuration2& cfg_current, 
                       const Configuration2& cfg_neighbor, 
                       const Configuration2& cfg_goal) const override;

    // No precomputation needed for Euclidean heuristic.
    void precomputeForGoal(const Configuration2& cfg_goal, const std::string& robot_name) override {}

    // No precomputed values to clear for Euclidean heuristic.
    void clearPrecomputedValues() override {}

    // Get the name of this heuristic type.
    std::string getName() const override { return "Euclidean"; }
};

} // namespace gco 