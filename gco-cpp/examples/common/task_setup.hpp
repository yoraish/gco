#pragma once

#include <map>
#include <vector>
#include <string>

#include "gco/types.hpp"
#include "gco/obstacles/obstacles.hpp"
#include "configuration_generators.hpp"

namespace gco_examples {

// Task configuration structure
struct TaskConfig {
    std::map<std::string, gco::Configuration2> cfgs_start;
    std::map<std::string, gco::Configuration2> cfgs_goal;
    std::map<std::string, gco::Configuration2> cfgs_post_goal;
    std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>> obstacles;
    double radius_robot = 0.13;
    double goal_tolerance = 0.03;
    std::map<std::string, std::string> extra_info;
};

TaskConfig setupTask(int task_type, const std::vector<std::string>& robot_names);

void printConfigurations(const std::map<std::string, gco::Configuration2>& cfgs_start,
                        const std::map<std::string, gco::Configuration2>& cfgs_goal);

void addPostGoalConfigurations(gco::MultiRobotPaths& paths,
                              const std::map<std::string, gco::Configuration2>& cfgs_post_goal,
                              const std::map<std::string, gco::Transform2>& cfgs_goal);

std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>> createObstaclesFromLines(
    const std::vector<std::pair<std::pair<double, double>, std::pair<double, double>>>& lines,
    double line_radius,
    const std::string& line_name_prefix = "obstacle_circle");

} // namespace gco_examples 