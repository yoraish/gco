#pragma once

#include <map>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

#include "gco/types.hpp"
#include "gco/obstacles/obstacles.hpp"

namespace gco_examples {

// Configuration generation functions
std::map<std::string, gco::Configuration2> create_cfgs_square(std::vector<std::string> robot_names, double spacing = 0.5, bool reverse = false);
std::map<std::string, gco::Configuration2> create_cfgs_square_shifted(std::vector<std::string> robot_names, double dx = 0.0, double dy = 0.0, double spacing = 0.5, bool reverse = false);
std::map<std::string, gco::Configuration2> create_cfgs_square_staggered(std::vector<std::string> robot_names, double spacing = 0.5, double shift_x = 0.0, double shift_y = 0.0, bool reverse = false);
std::map<std::string, gco::Configuration2> create_cfgs_circle(std::vector<std::string> robot_names, double radius = 0.8, double robot_radius = 0.12, bool reverse = false);
std::map<std::string, gco::Configuration2> create_cfgs_line_vertical(std::vector<std::string> robot_names, double x = 0.5, double spacing = 0.5, bool reverse = false);
std::map<std::string, gco::Configuration2> create_cfgs_line_horizontal(std::vector<std::string> robot_names, double y = 0.5, double spacing = 0.3, bool reverse = false);
std::map<std::string, gco::Configuration2> create_cfgs_CMU_staggered(std::vector<std::string> robot_names, double spacing = 0.5, bool reverse = false);
std::map<std::string, gco::Configuration2> create_cfgs_curtain(std::vector<std::string> robot_names, double spacing, double width, double top_left_x, double top_left_y, bool reverse);

// Random configuration generation
gco::Configuration2 generateRandomConfig(std::mt19937& rng, double min_x = -1.8, double max_x = 1.8, 
                                        double min_y = -1.8, double max_y = 1.8);
bool checkCollisionFree(const std::map<std::string, gco::Configuration2>& cfgs, 
                       double robot_radius, 
                       const std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>>& obstacles);
std::map<std::string, gco::Configuration2> create_cfgs_random(std::vector<std::string> robot_names, 
    double min_x = -1.8, 
    double max_x = 1.8, 
    double min_y = -1.8, 
    double max_y = 1.8, 
    double robot_radius = 0.12, 
    int seed = 42,
    const std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>>& obstacles = std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>>());

} // namespace gco_examples 