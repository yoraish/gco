/*
    This file contains the types for the CTSWAP algorithm.
*/

#pragma once

// General includes.
#include <vector>
#include <memory>
#include <map>

// Project includes.
#include "gco/spatial/transforms.hpp"

namespace gco {
// A configuration is a vector of robot IDs.
using Configuration2 = Transform2;
using State = std::pair<Configuration2, int>;

// A sequence of actions, each entry representing a change in the configuration. Of form [cfg_parent, cfg_1, cfg_2, ..., cfg_this].
using ActionSequence = std::vector<Configuration2>;
using ActionSequencePtr = std::shared_ptr<ActionSequence>;

// A multi-robot configuration is a map of robot names to configurations.
using MultiRobotConfiguration = std::map<std::string, Configuration2>;
using MultiRobotConfigurationPtr = std::shared_ptr<MultiRobotConfiguration>;

// A multi-object multi-robot configuration separates robots and objects
struct MultiObjectMultiRobotConfiguration {
    std::map<std::string, Configuration2> robots;
    std::map<std::string, Configuration2> objects;
    
    // Constructor
    MultiObjectMultiRobotConfiguration() = default;
    
    // Constructor from separate robot and object maps
    MultiObjectMultiRobotConfiguration(const std::map<std::string, Configuration2>& robots, 
                                      const std::map<std::string, Configuration2>& objects)
        : robots(robots), objects(objects) {}
};
using MultiObjectMultiRobotConfigurationPtr = std::shared_ptr<MultiObjectMultiRobotConfiguration>;

// Comparator for MultiObjectMultiRobotConfiguration
struct MultiObjectMultiRobotConfigurationComparator {
    bool operator()(const MultiObjectMultiRobotConfigurationPtr& a, const MultiObjectMultiRobotConfigurationPtr& b) const {
        // Compare robots first
        if (a->robots != b->robots) {
            return std::lexicographical_compare(a->robots.begin(), a->robots.end(), b->robots.begin(), b->robots.end(), 
            [](const std::pair<std::string, Configuration2>& a, const std::pair<std::string, Configuration2>& b) {
                if (a.first != b.first) {
                    return a.first < b.first;
                }
                return a.second < b.second;
            });
        }
        // If robots are equal, compare objects
        return std::lexicographical_compare(a->objects.begin(), a->objects.end(), b->objects.begin(), b->objects.end(), 
        [](const std::pair<std::string, Configuration2>& a, const std::pair<std::string, Configuration2>& b) {
            if (a.first != b.first) {
                return a.first < b.first;
            }
            return a.second < b.second;
        });
    }
};

// Hash function for MultiRobotConfiguration.
struct MultiRobotConfigurationHash {
    std::size_t operator()(const MultiRobotConfigurationPtr& cfg) const {
        return std::hash<MultiRobotConfigurationPtr>{}(cfg);
    }
};

// A multi-robot path is a map of robot names to action sequences.
using Path = std::vector<Configuration2>;
using Edge = std::vector<Configuration2>;
using PathPtr = std::shared_ptr<Path>;
using EdgePtr = std::shared_ptr<Edge>;
using MultiRobotPaths = std::map<std::string, Path>;
using MultiRobotPathsPtr = std::shared_ptr<MultiRobotPaths>;

// Colors.
const std::string GREEN = "\033[92m"; // Green.
const std::string RED = "\033[91m"; // Red.
const std::string YELLOW = "\033[93m"; // Yellow.
const std::string CYAN = "\033[96m"; // Blue.

const std::string RESET = "\033[0m"; // Reset.


} // namespace gco