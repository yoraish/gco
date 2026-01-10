// General includes.
#include <iostream>
#include <cmath>

// Project includes.
#include "gco/utils.hpp"
#include "gco/types.hpp"
#include "gco/spatial/transforms.hpp"

// ====================
// Implementation of the utils.
// ====================
void gco::normalizeDiscretizeConfiguration(Configuration2& cfg, const JointRanges& joint_ranges) {
    // Normalize the configuration. This only applies to the theta dimension. Theta may be arbitrary so we must use
    double theta_range = joint_ranges.maxs.theta - joint_ranges.mins.theta;
    cfg.theta = fmod(cfg.theta - joint_ranges.mins.theta, theta_range);
    if (cfg.theta < 0.0) {
        cfg.theta += theta_range;
    }
    cfg.theta += joint_ranges.mins.theta;

    // Discretize the configuration.
    cfg.x = std::round(cfg.x / joint_ranges.discretization.x) * joint_ranges.discretization.x;
    cfg.y = std::round(cfg.y / joint_ranges.discretization.y) * joint_ranges.discretization.y;
    cfg.theta = std::round(cfg.theta / joint_ranges.discretization.theta) * joint_ranges.discretization.theta;
}

void gco::normalizeDiscretizePath(PathPtr path, const JointRanges& joint_ranges) {
    // Normalize and discretize the path.
    for (auto& cfg : *path) {
        normalizeDiscretizeConfiguration(cfg, joint_ranges);
    }
}

double gco::configurationDistance(const Configuration2& cfg1, const Configuration2& cfg2) {
    // Compute the distance between two configurations.
    // For theta, we need to handle the wrapping around [-pi, pi]
    double theta_diff = cfg1.theta - cfg2.theta;
    
    // Normalize theta difference to [-pi, pi]
    while (theta_diff > M_PI) {
        theta_diff -= 2 * M_PI;
    }
    while (theta_diff < -M_PI) {
        theta_diff += 2 * M_PI;
    }
    
    return std::sqrt(std::pow(cfg1.x - cfg2.x, 2) + std::pow(cfg1.y - cfg2.y, 2) + std::pow(theta_diff, 2) * 0.1);
}

gco::Configuration2 gco::transformConfigurationToWorld(const Configuration2& cfg_local, const Configuration2& x_world_local) {
    // Transform a configuration to the world frame.
    return x_world_local * cfg_local;
}

gco::ActionSequencePtr gco::transformActionSequenceToWorld(const ActionSequencePtr& action_sequence_local, const Configuration2& x_world_local) {
    // Transform action sequences to the world frame.
    
    gco::ActionSequencePtr action_sequence_world = std::make_shared<gco::ActionSequence>();
    
    for (size_t i = 0; i < action_sequence_local->size(); ++i) {
        const auto& cfg_local = (*action_sequence_local)[i];
        
        // Transform the configuration to the world frame.
        Configuration2 cfg_world = transformConfigurationToWorld(cfg_local, x_world_local);
        
        action_sequence_world->push_back(cfg_world);
    }
    
    return action_sequence_world;
}

// Implementation of the operator<< for ActionSequencePtr
std::ostream& operator<<(std::ostream& os, const gco::ActionSequencePtr& action_sequence) {
    os << "[";  
    for (size_t i = 0; i < action_sequence->size(); ++i) {
        os << "[" << (*action_sequence)[i].x << ", " << (*action_sequence)[i].y << ", " << (*action_sequence)[i].theta << "]";
        if (i < action_sequence->size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}
