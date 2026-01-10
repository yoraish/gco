#pragma once

// General includes.
#include <iostream>
#include <set>

// Project includes.
#include "gco/robot.hpp"
#include "gco/types.hpp"

// ====================
// Prints.
// ====================
// Vector to stream <<.
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i < vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

// Similar, but for vectors of vectors of T.
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<T>>& vec) {
    os << "[\n";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << " * " << vec[i] << "\n";
    }
    os << "]\n";
    return os;
}

// Similar, but for sets of T.
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::set<T>& set) {
    os << "{";
    for (auto it = set.begin(); it != set.end(); ++it) {
        const auto& item = *it;
        os << item;
        if (std::next(it) != set.end()) {
            os << ", ";
        }
    }
    os << "}";
    return os;
}

// Similar, but for action sequences.
std::ostream& operator<<(std::ostream& os, const gco::ActionSequencePtr& action_sequence);

namespace gco {

// Normalize and discretize a configuration.
void normalizeDiscretizeConfiguration(Configuration2& cfg, const JointRanges& joint_ranges);

// Normalize and discretize a path.
void normalizeDiscretizePath(PathPtr path, const JointRanges& joint_ranges);

// Compute the distance between two configurations.
double configurationDistance(const Configuration2& cfg1, const Configuration2& cfg2);

// Transform a configuration to the world frame.
Configuration2 transformConfigurationToWorld(const Configuration2& cfg_local, const Configuration2& x_world_local);

// Transform an action sequence to the world frame.
ActionSequencePtr transformActionSequenceToWorld(const ActionSequencePtr& action_sequence_local, const Configuration2& x_world_local);

} // namespace gco