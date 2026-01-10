#include "visualization_utils.hpp"
#include <iostream>

namespace gco_examples {

void saveVisualizationJSON(const std::map<std::string, gco::RobotPtr>& robots,
                          const std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>>& obstacles,
                          const std::map<std::string, gco::Configuration2>& cfgs_start,
                          const std::map<std::string, gco::Configuration2>& cfgs_goal,
                          const gco::MultiRobotPaths& paths,
                          const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open visualization file for writing." << std::endl;
        return;
    }
    
    file << "{" << std::endl;
    
    // Save robot radii
    file << "    \"robot_radii\": {" << std::endl;
    for (auto it = robots.begin(); it != robots.end(); ++it) {
        auto robot_disk = std::dynamic_pointer_cast<gco::RobotDisk>(it->second);
        if (robot_disk) {
            file << "        \"" << it->first << "\": " << robot_disk->getRadius();
        } else {
            file << "        \"" << it->first << "\": 0.0";
        }
        if (std::next(it) != robots.end()) {
            file << ",";
        }
        file << std::endl;
    }
    file << "    }," << std::endl;
    
    // Save obstacle positions
    file << "    \"obstacle_positions\": {" << std::endl;
    for (auto it = obstacles.begin(); it != obstacles.end(); ++it) {
        std::string obstacle_name = it->first;
        gco::Transform2 transform = it->second.second;
        file << "        \"" << obstacle_name << "\": [" << transform.x << ", " << transform.y << "]";
        if (std::next(it) != obstacles.end()) {
            file << ",";
        }
        file << std::endl;
    }
    file << "    }," << std::endl;
    
    // Save obstacle radii
    file << "    \"obstacle_radii\": {" << std::endl;
    for (auto it = obstacles.begin(); it != obstacles.end(); ++it) {
        auto obstacle = it->second.first;
        auto obstacle_circle = std::dynamic_pointer_cast<gco::CircleShape>(obstacle->getShape());
        auto obstacle_square = std::dynamic_pointer_cast<gco::SquareShape>(obstacle->getShape());
        auto obstacle_rectangle = std::dynamic_pointer_cast<gco::RectangleShape>(obstacle->getShape());
        if (obstacle_circle) {
            file << "        \"" << it->first << "\": " << obstacle_circle->getRadius();
        } else if (obstacle_square) {
            file << "        \"" << it->first << "\": " << obstacle_square->getWidth();
        } else if (obstacle_rectangle) {
            file << "        \"" << it->first << "\": [" << obstacle_rectangle->getWidth() << ", " << obstacle_rectangle->getHeight() << "]";
        } else {
            file << "        \"" << it->first << "\": 0.0";
        }
        if (std::next(it) != obstacles.end()) {
            file << ",";
        }
        file << std::endl;
    }
    file << "    }," << std::endl;
    
    // Save start configurations
    file << "    \"cfgs_start\": {" << std::endl;
    for (auto it = cfgs_start.begin(); it != cfgs_start.end(); ++it) {
        file << "        \"" << it->first << "\": [" << it->second.x << ", " << it->second.y << ", " << it->second.theta << "]";
        if (std::next(it) != cfgs_start.end()) {
            file << ",";
        }
        file << std::endl;
    }
    file << "    }," << std::endl;
    
    // Save goal configurations
    file << "    \"cfgs_goal\": {" << std::endl;
    for (auto it = cfgs_goal.begin(); it != cfgs_goal.end(); ++it) {
        file << "        \"" << it->first << "\": [" << it->second.x << ", " << it->second.y << ", " << it->second.theta << "]";
        if (std::next(it) != cfgs_goal.end()) {
            file << ",";
        }
        file << std::endl;
    }
    file << "    }," << std::endl;
    
    // Save paths
    file << "    \"paths\": {" << std::endl;
    for (auto it = paths.begin(); it != paths.end(); ++it) {
        file << "        \"" << it->first << "\": [" << std::endl;
        for (size_t i = 0; i < it->second.size(); ++i) {
            file << "            [" << it->second[i].x << ", " << it->second[i].y << ", " << it->second[i].theta << "]";
            if (i < it->second.size() - 1) {
                file << ",";
            }
            file << std::endl;
        }
        file << "        ]";
        if (std::next(it) != paths.end()) {
            file << ",";
        }
        file << std::endl;
    }
    file << "    }" << std::endl;
    file << "}" << std::endl;
    
    file.close();
}

} // namespace gco_examples 