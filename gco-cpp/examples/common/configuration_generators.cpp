#include "configuration_generators.hpp"
#include <iostream>

namespace gco_examples {

std::map<std::string, gco::Configuration2> create_cfgs_square(std::vector<std::string> robot_names, double spacing, bool reverse) {
    // Organize the robots in a square/rectangle formation next to each other.
    std::map<std::string, gco::Configuration2> cfgs_start;
    int num_robots = robot_names.size();
    
    // Calculate grid dimensions to make it as square as possible
    int cols = static_cast<int>(std::ceil(std::sqrt(num_robots)));
    int rows = static_cast<int>(std::ceil(static_cast<double>(num_robots) / cols));
        
    // Calculate total width and height to center the formation
    double total_width = (cols - 1) * spacing;
    double total_height = (rows - 1) * spacing;
    
    // Starting position (top-left corner)
    double start_x = -total_width / 2.0;
    double start_y = total_height / 2.0;
    
    if (reverse) {
        std::reverse(robot_names.begin(), robot_names.end());
    }
    
    for (int i = 0; i < num_robots; i++) {
        int row = i / cols;
        int col = i % cols;
        
        double x = start_x + col * spacing;
        double y = start_y - row * spacing;
        double theta = 0.0; // All robots face the same direction
        
        cfgs_start[robot_names[i]] = gco::Configuration2(x, y, theta);
    }
    return cfgs_start;
}

std::map<std::string, gco::Configuration2> create_cfgs_square_shifted(std::vector<std::string> robot_names, double dx, double dy, double spacing, bool reverse) {
    // Get square, and then shift all x's and y's by dx dy.
    std::map<std::string, gco::Configuration2> cfgs_start = create_cfgs_square(robot_names, spacing, reverse);
    for (auto& [robot_name, cfg] : cfgs_start) {
        cfg.x += dx;
        cfg.y += dy;
    }
    return cfgs_start;
}

std::map<std::string, gco::Configuration2> create_cfgs_square_staggered(std::vector<std::string> robot_names, double spacing, double shift_x, double shift_y, bool reverse) {
    // Each row is shifted by half a spacing.
    std::map<std::string, gco::Configuration2> cfgs_start;
    int num_robots = robot_names.size();
    // Calculate grid dimensions to make it as square as possible
    int cols = static_cast<int>(std::ceil(std::sqrt(num_robots)));
    int rows = static_cast<int>(std::ceil(static_cast<double>(num_robots) / cols));
        
    // Calculate total width and height to center the formation
    double total_width = (cols - 1) * spacing;
    double total_height = (rows - 1) * spacing;
    
    // Starting position (top-left corner)
    double start_x = -total_width / 2.0;
    double start_y = total_height / 2.0;
    
    if (reverse) {
        std::reverse(robot_names.begin(), robot_names.end());
    }
    
    for (int i = 0; i < num_robots; i++) {
        int row = i / cols;
        int col = i % cols;
        
        double x = start_x + col * spacing + (row % 2) * spacing / 2.0 + shift_x;
        double y = start_y - row * spacing * 0.88 + shift_y;
        double theta = 0.0; // All robots face the same direction
        
        cfgs_start[robot_names[i]] = gco::Configuration2(x, y, theta);
    }
    
    return cfgs_start;
}

std::map<std::string, gco::Configuration2> create_cfgs_circle(std::vector<std::string> robot_names, double radius, double robot_radius, bool reverse) {
    // Organize the robots in a circle along the perimeter.
    // If the circumference is too small, create larger outer circles.
    std::map<std::string, gco::Configuration2> cfgs_start;
    int num_robots = robot_names.size();
    
    if (reverse) {
        std::reverse(robot_names.begin(), robot_names.end());
    }
    
    // Calculate minimum spacing needed between robots (2 * robot_radius for safety)
    double min_spacing = 4.0 * robot_radius;
    
    // Calculate if we can fit all robots on the outer circle
    double circumference = 2.0 * M_PI * radius;
    double required_spacing = num_robots * min_spacing;
    
    if (required_spacing <= circumference) {
        // All robots can fit on the outer circle
        for (int i = 0; i < num_robots; i++) {
            double angle = 2.0 * M_PI * i / num_robots;
            double x = radius * cos(angle);
            double y = radius * sin(angle);
            
            cfgs_start[robot_names[i]] = gco::Configuration2(x, y, 0.0);
        }
    } else {
        // Need to create larger outer circles
        int robots_placed = 0;
        double current_radius = radius;
        
        while (robots_placed < num_robots) {
            // Calculate how many robots can fit on current circle
            double current_circumference = 2.0 * M_PI * current_radius;
            int robots_on_circle = static_cast<int>(current_circumference / min_spacing);
            
            // Limit to remaining robots
            robots_on_circle = std::min(robots_on_circle, num_robots - robots_placed);
            
            if (robots_on_circle > 0) {
                // Place robots on current circle
                for (int i = 0; i < robots_on_circle; i++) {
                    double angle = 2.0 * M_PI * i / robots_on_circle;
                    double x = current_radius * cos(angle);
                    double y = current_radius * sin(angle);
                    
                    cfgs_start[robot_names[robots_placed + i]] = gco::Configuration2(x, y, 0.0);
                }
                robots_placed += robots_on_circle;
            }
            
            // Move to larger outer circle (increase radius by robot diameter)
            current_radius += 3.0 * robot_radius;
        }
    }
    
    return cfgs_start;
}

std::map<std::string, gco::Configuration2> create_cfgs_line_vertical(std::vector<std::string> robot_names, double x, double spacing, bool reverse) {
    // Organize the robots in a line.
    std::map<std::string, gco::Configuration2> cfgs_start;
    int num_robots = robot_names.size();

    if (reverse) {
        std::reverse(robot_names.begin(), robot_names.end());
    }
    
    for (int i = 0; i < num_robots; i++) {
        double y = i * spacing - (num_robots - 1) * spacing / 2.0;
        double theta = 0.0;
        
        cfgs_start[robot_names[i]] = gco::Configuration2(x, y, theta);
    }
    return cfgs_start;
}

std::map<std::string, gco::Configuration2> create_cfgs_line_horizontal(std::vector<std::string> robot_names, double y, double spacing, bool reverse) {  
    // Organize the robots in a line.
    std::map<std::string, gco::Configuration2> cfgs_start;
    int num_robots = robot_names.size();
    
    if (reverse) {
        std::reverse(robot_names.begin(), robot_names.end());
    }
    
    for (int i = 0; i < num_robots; i++) {
        double x = i * spacing - (num_robots - 1) * spacing / 2.0;
        double theta = 0.0;
        
        cfgs_start[robot_names[i]] = gco::Configuration2(x, y, theta);
    }
    
    return cfgs_start;
}

gco::Configuration2 generateRandomConfig(std::mt19937& rng, double min_x, double max_x, 
                                        double min_y, double max_y) {
    std::uniform_real_distribution<double> dist_x(min_x, max_x);
    std::uniform_real_distribution<double> dist_y(min_y, max_y);
    // std::uniform_real_distribution<double> dist_theta(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<double> dist_theta(0.0, 0.0);
    
    return gco::Configuration2(dist_x(rng), dist_y(rng), dist_theta(rng));
}

bool checkCollisionFree(const std::map<std::string, gco::Configuration2>& cfgs, 
                       double robot_radius, 
                       const std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>>& obstacles) {
    // Check robot-robot collisions
    std::vector<gco::Configuration2> configs;
    for (const auto& [name, cfg] : cfgs) {
        configs.push_back(cfg);
    }
    
    for (size_t i = 0; i < configs.size(); ++i) {
        for (size_t j = i + 1; j < configs.size(); ++j) {
            double dx = configs[i].x - configs[j].x;
            double dy = configs[i].y - configs[j].y;
            double distance = std::sqrt(dx * dx + dy * dy);
            if (distance < 2.0 * robot_radius) {
                return false;
            }
        }
    }
    
    // Check robot-obstacle collisions
    for (const auto& [robot_name, cfg] : cfgs) {
        for (const auto& [obstacle_name, obstacle_pair] : obstacles) {
            auto obstacle = obstacle_pair.first;
            auto transform = obstacle_pair.second;
            
            // Simple circle-circle collision check
            auto obstacle_circle = std::dynamic_pointer_cast<gco::CircleShape>(obstacle->getShape());
            if (obstacle_circle) {
                double dx = cfg.x - transform.x;
                double dy = cfg.y - transform.y;
                double distance = std::sqrt(dx * dx + dy * dy);
                if (distance < (robot_radius + obstacle_circle->getRadius())) {
                    return false;
                }
            }
        }
    }
    
    return true;
}

std::map<std::string, gco::Configuration2> create_cfgs_random(std::vector<std::string> robot_names, 
    double min_x, 
    double max_x, 
    double min_y, 
    double max_y, 
    double robot_radius, 
    int seed,
    const std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>>& obstacles) {
    
    std::map<std::string, gco::Configuration2> cfgs_start;
    std::mt19937 rng(seed);
    for (const auto& robot_name : robot_names) {
        for (int i = 0; i < 100; i++) {
            auto tmp = generateRandomConfig(rng, min_x, max_x, min_y, max_y);
            cfgs_start[robot_name] = tmp;
            // Check collision free.
            if (checkCollisionFree(cfgs_start, robot_radius, obstacles)) {
                cfgs_start[robot_name] = tmp;
                break;
            }
        }
        if (cfgs_start.find(robot_name) == cfgs_start.end()) {
            std::cout << "Failed to find a collision-free configuration for robot " << robot_name << std::endl;
            break;
        }
    }
    return cfgs_start;
}

std::map<std::string, gco::Configuration2> create_cfgs_CMU_staggered(std::vector<std::string> robot_names, double spacing, bool reverse) {
    // Each row is shifted by half a spacing.
    std::map<std::string, gco::Configuration2> cfgs_start;
    int num_robots = robot_names.size();
    // Compute goals for the letters. Each line is a staggered rectangle.
    // First third for the first letter.
    std::vector<std::string> robot_names_1 = std::vector<std::string>(robot_names.begin(), robot_names.begin() + num_robots / 3);
    std::map<std::string, gco::Configuration2> cfgs_goal_1 = create_cfgs_square_staggered(robot_names_1, spacing, -1.3, 0.8, reverse);
    // Second third for the second letter.
    std::vector<std::string> robot_names_2 = std::vector<std::string>(robot_names.begin() + num_robots / 3, robot_names.begin() + 2 * num_robots / 3);
    std::map<std::string, gco::Configuration2> cfgs_goal_2 = create_cfgs_square_staggered(robot_names_2, spacing, 0.0, 0.0, reverse);
    // Third third for the third letter.
    std::vector<std::string> robot_names_3 = std::vector<std::string>(robot_names.begin() + 2 * num_robots / 3, robot_names.end());
    std::map<std::string, gco::Configuration2> cfgs_goal_3 = create_cfgs_square_staggered(robot_names_3, spacing, 1.3, -0.8, reverse);

    // Combine the goals.
    cfgs_start.insert(cfgs_goal_1.begin(), cfgs_goal_1.end());
    cfgs_start.insert(cfgs_goal_2.begin(), cfgs_goal_2.end());
    cfgs_start.insert(cfgs_goal_3.begin(), cfgs_goal_3.end());
    return cfgs_start;
}

std::map<std::string, gco::Configuration2> create_cfgs_curtain(std::vector<std::string> robot_names, double spacing, double width, double top_left_x, double top_left_y, bool reverse) {
    // Organize the robots in a rectangle of width (given) and height dictated by the number of robots.
    std::map<std::string, gco::Configuration2> cfgs_start;
    int num_robots = robot_names.size();
    
    // Calculate grid dimensions to make it as square as possible
    int cols = static_cast<int>(std::ceil(width / spacing));
    int rows = static_cast<int>(std::ceil(static_cast<double>(num_robots) / cols));
    std::cout << "num_robots: " << num_robots << ", cols: " << cols << ", rows: " << rows << std::endl;
        
    // Starting position (top-left corner)
    double start_x = top_left_x;
    double start_y = top_left_y;
    
    if (reverse) {
        std::reverse(robot_names.begin(), robot_names.end());
    }
    
    for (int i = 0; i < num_robots; i++) {
        int row = i / cols;
        int col = i % cols;
        
        double x = start_x + col * spacing;
        double y = start_y - row * spacing;
        double theta = 0.0; // All robots face the same direction
        
        cfgs_start[robot_names[i]] = gco::Configuration2(x, y, theta);
    }
    return cfgs_start;
}


} // namespace gco_examples 