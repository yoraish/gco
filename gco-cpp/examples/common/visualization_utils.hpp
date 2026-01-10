#pragma once

#include <map>
#include <string>
#include <fstream>

#include "gco/types.hpp"
#include "gco/robot.hpp"
#include "gco/obstacles/obstacles.hpp"

namespace gco_examples {

void saveVisualizationJSON(const std::map<std::string, gco::RobotPtr>& robots,
                          const std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>>& obstacles,
                          const std::map<std::string, gco::Configuration2>& cfgs_start,
                          const std::map<std::string, gco::Configuration2>& cfgs_goal,
                          const gco::MultiRobotPaths& paths,
                          const std::string& filename);

} // namespace gco_examples 