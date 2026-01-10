#include "task_setup.hpp"
#include <iostream>
#include <numeric>
#include <limits>
#include "common/configuration_generators.hpp"
#include "gco/utils.hpp"

namespace gco_examples {

std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>> createObstaclesFromLines(
    const std::vector<std::pair<std::pair<double, double>, std::pair<double, double>>>& lines,
    double line_radius,
    const std::string& line_name_prefix) {
    
    std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>> obstacles;
    
    int k = 0;
    for (auto line : lines) {
        double x0 = line.first.first;
        double y0 = line.first.second;
        double x1 = line.second.first;
        double y1 = line.second.second;
        
        // Calculate line properties
        double dx = x1 - x0;
        double dy = y1 - y0;
        double length = std::sqrt(dx * dx + dy * dy);
        double angle = std::atan2(dy, dx);
        
        // Create obstacles along the line
        for (double t = 0; t < length; t += line_radius * 2) {
            double x = x0 + t * std::cos(angle);
            double y = y0 + t * std::sin(angle);
            
            obstacles[line_name_prefix + "_" + std::to_string(k)] = std::make_pair(
                gco::Obstacle::createCircle(line_name_prefix + "_" + std::to_string(k), line_radius), 
                gco::Transform2(x, y, 0.0));
            k++;
        }
    }
    
    return obstacles;
}


std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>> createMazeObstacles() {
    // Create a maze with obstacles.
    std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>> obstacles;
    double radius = 0.1;
    std::vector<std::pair<std::pair<double, double>, std::pair<double, double>>> lines;
    // lines.push_back(std::make_pair(std::make_pair(1.0, -1.0), std::make_pair(1.0, 1.0))); // center to left, top.  
    lines.push_back(std::make_pair(std::make_pair(0.0, -1.0), std::make_pair(0.0, 1.0))); // center to left, top.  
    // lines.push_back(std::make_pair(std::make_pair( 0.5,  0.25), std::make_pair( 0.5,  5.0))); // center to left, top.  
    lines.push_back(std::make_pair(std::make_pair( 0.5, -0.25), std::make_pair( 0.5, -5.0))); // center to left, top.  
    lines.push_back(std::make_pair(std::make_pair(-0.5, -0.25), std::make_pair(-0.5, -5.0))); // center to left, top.  
    lines.push_back(std::make_pair(std::make_pair(-0.5,  0.25), std::make_pair(-0.5,  5.0))); // center to left, top.  

    obstacles = createObstaclesFromLines(lines, radius, "maze_circle");
    return obstacles;
}

TaskConfig setupTask(int task_type, const std::vector<std::string>& robot_names) {
    TaskConfig config;
    
    switch (task_type) {

        // ==============================
        // Free space tasks.
        // ==============================
        case 0:
        {
            config.radius_robot = 0.1;
            config.goal_tolerance = 0.035;
            config.cfgs_start = create_cfgs_circle(robot_names, 1.5, config.radius_robot);
            config.cfgs_goal = create_cfgs_square_staggered(robot_names, 0.3);
            // config.cfgs_post_goal = create_cfgs_square_staggered(robot_names, 0.215);
            config.extra_info["heuristic_type"] = "euclidean";
            break;
        }
        // Square to circle.
        case 1:
        {
            config.radius_robot = 0.1;
            config.goal_tolerance = 0.035;
            config.cfgs_start = create_cfgs_square(robot_names, 0.3);
            config.cfgs_goal = create_cfgs_circle(robot_names, 1.5, config.radius_robot);
            config.extra_info["heuristic_type"] = "euclidean";
            break;
        }
        // Left to right.
        case 2:
        {
            config.radius_robot = 0.1;
            config.goal_tolerance = 0.035;
            config.cfgs_start = create_cfgs_square_staggered(robot_names, 0.3, -1.5, 0.0);
            config.cfgs_goal = create_cfgs_square_staggered(robot_names, 0.3, 1.5, 0.0, true);
            config.extra_info["heuristic_type"] = "euclidean";
            break;
        }

        // ==============================
        // Light obstacle tasks.
        // ==============================
        // Left to right with small wall.
        case 3:
        {
            std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>> obstacles;
            double radius = 0.1;
            std::vector<std::pair<std::pair<double, double>, std::pair<double, double>>> lines;
            lines.push_back(std::make_pair(std::make_pair(0.0, -1.0), std::make_pair(0.0, 1.0)));
 
            obstacles = createObstaclesFromLines(lines, radius, "wall_circle");
            config.obstacles = obstacles;
            config.radius_robot = 0.1;
            config.cfgs_start = create_cfgs_square_shifted(robot_names, -1.5, 0.0, 0.3);
            config.cfgs_goal = create_cfgs_square_shifted(robot_names, 1.5, 0.0, 0.3);
            // config.cfgs_post_goal = create_cfgs_square_staggered(robot_names, 0.215, 1.5);
            config.goal_tolerance = 0.035;
            config.extra_info["heuristic_type"] = "bwd_dijkstra";
            config.extra_info["heuristic_grid_resolution"] = "0.05";
            break;
        }
        // Circle to square with left V obstacle.
        case 4:
        {
            config.radius_robot = 0.1;
            config.goal_tolerance = 0.055;
            double spacing_min = 0.3;
            double spacing_max = 0.8;
            double spacing = spacing_min + (spacing_max - spacing_min) * (50 - robot_names.size()) / 50.0;
            config.cfgs_start = create_cfgs_circle(robot_names, 1.8, config.radius_robot);
            config.cfgs_goal = create_cfgs_square_staggered(robot_names, spacing, 1.5, 1.0);
            // config.cfgs_post_goal = create_cfgs_square_staggered(robot_names, 0.216, 0.5, 0.5);
            config.obstacles["obstacle_circle_1"] = std::make_pair(gco::Obstacle::createCircle("obstacle_circle_1", 0.3), gco::Transform2( -1.2, 0.3, 0.0));
            config.obstacles["obstacle_circle_2"] = std::make_pair(gco::Obstacle::createCircle("obstacle_circle_2", 0.25), gco::Transform2(-1.0, 0.1, 0.0));
            config.obstacles["obstacle_circle_3"] = std::make_pair(gco::Obstacle::createCircle("obstacle_circle_3", 0.2), gco::Transform2( -0.8, 0.0, 0.0));
            config.obstacles["obstacle_circle_4"] = std::make_pair(gco::Obstacle::createCircle("obstacle_circle_4", 0.2), gco::Transform2( -0.5, -0.1, 0.0));
            config.obstacles["obstacle_circle_5"] = std::make_pair(gco::Obstacle::createCircle("obstacle_circle_5", 0.25), gco::Transform2(-1.0, 0.5, 0.0));
            config.obstacles["obstacle_circle_6"] = std::make_pair(gco::Obstacle::createCircle("obstacle_circle_6", 0.2), gco::Transform2( -0.8, 0.6, 0.0));
            config.obstacles["obstacle_circle_7"] = std::make_pair(gco::Obstacle::createCircle("obstacle_circle_7", 0.2), gco::Transform2( -0.5, 0.7, 0.0));

            break;
        }
        // Circle to bounded circle with one entrance.
        case 5: 
        {
            // Create a circle of obstacle, radius 1.8.
            std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>> obstacles;
            double circle_radius = 1.8;
            double circle_center_x = 0.0;
            double circle_center_y = 0.0;
            double obstacle_radius = 0.1;
            int num_obstacles = 30;
            for (int i = 0; i < num_obstacles; i++) {
                double angle = 1.2 * M_PI * i / num_obstacles;
                double x = circle_center_x + circle_radius * std::cos(angle);
                double y = circle_center_y + circle_radius * std::sin(angle);
                obstacles["obstacle_circle_" + std::to_string(i)] = std::make_pair(gco::Obstacle::createCircle("obstacle_circle_" + std::to_string(i), obstacle_radius), gco::Transform2(x, y, 0.0));
            }


            // Enclose the whole thing in a 2.7 by 2.7 box.
            std::vector<std::pair<std::pair<double, double>, std::pair<double, double>>> lines;
            lines.push_back(std::make_pair(std::make_pair(-2.7, -2.7), std::make_pair(2.7, -2.7))); // Bottom.
            lines.push_back(std::make_pair(std::make_pair(-2.7, -2.7), std::make_pair(-2.7, 2.7))); // Left.
            lines.push_back(std::make_pair(std::make_pair(2.7, -2.7), std::make_pair(2.7, 2.8))); // Top.
            lines.push_back(std::make_pair(std::make_pair(-2.7, 2.7), std::make_pair(2.7, 2.7))); // Right.
            double line_radius = 0.1;
            auto boundary_obstacles = createObstaclesFromLines(lines, line_radius, "boundary_circle");
            obstacles.insert(boundary_obstacles.begin(), boundary_obstacles.end());

            
            config.radius_robot = 0.1;
            config.goal_tolerance = 0.05;
            config.obstacles = obstacles;
            config.extra_info["heuristic_grid_resolution"] = "0.05";
            config.cfgs_start = create_cfgs_circle(robot_names, 2.2, config.radius_robot);
            config.cfgs_goal = create_cfgs_square_staggered(robot_names, 0.35, 0.0, 0.0);
            config.extra_info["max_distance_meters"] = "12";
            config.extra_info["heuristic_type"] = "bwd_dijkstra";
            config.extra_info["heuristic_grid_resolution"] = "0.05";
            // config.cfgs_post_goal = create_cfgs_square_staggered(robot_names, 0.13, -1.3);
            break;
        }

        // ==============================
        // Heavy obstacle tasks.
        // ==============================
        // // CMU.
        // case 6:
        // {
        //     std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>> obstacles;
        //     double radius = 0.05;
        //     std::vector<std::pair<std::pair<double, double>, std::pair<double, double>>> lines;
        //     double factorx = 1.7;
        //     double factory = 1.0;
        //     lines.push_back(std::make_pair(std::make_pair(-1.5 * factorx, -1.0 * factory), std::make_pair(-1.0 * factorx, -1.0 * factory))); // C bottom.
        //     lines.push_back(std::make_pair(std::make_pair(-1.5 * factorx, -1.0 * factory), std::make_pair(-1.5 * factorx, 1.0 * factory))); // C left.
        //     lines.push_back(std::make_pair(std::make_pair(-1.5 * factorx, 1.0 * factory), std::make_pair(-1.0 * factorx, 1.0 * factory))); // C top.
        //     lines.push_back(std::make_pair(std::make_pair(-0.5 * factorx, -1.0 * factory), std::make_pair(-0.5 * factorx, 1.0 * factory))); // M left.
        //     lines.push_back(std::make_pair(std::make_pair(-0.5 * factorx, 1.0 * factory), std::make_pair(0.0 * factorx, 0.0 * factory))); // M center.
        //     lines.push_back(std::make_pair(std::make_pair(0.0 * factorx, 0.0 * factory), std::make_pair(0.5 * factorx, 1.0 * factory))); // M center 2.
        //     lines.push_back(std::make_pair(std::make_pair(0.5 * factorx, -1.0 * factory), std::make_pair(0.5 * factorx, 1.0 * factory))); // M right.
        //     lines.push_back(std::make_pair(std::make_pair(1.0 * factorx, -1.0 * factory), std::make_pair(1.0 * factorx, 1.0 * factory))); // U left.
        //     lines.push_back(std::make_pair(std::make_pair(1.0 * factorx, -1.0 * factory), std::make_pair(1.5 * factorx, -1.0 * factory))); // U left.
        //     lines.push_back(std::make_pair(std::make_pair(1.5 * factorx, -1.0 * factory), std::make_pair(1.5 * factorx, 1.0 * factory))); // U left.

        //     obstacles = createObstaclesFromLines(lines, radius, "cmu_circle");


        //     // // Enclose the whole thing in a box.
        //     // std::vector<std::pair<std::pair<double, double>, std::pair<double, double>>> lines_box;
        //     // lines_box.push_back(std::make_pair(std::make_pair(-2.5, -2.5), std::make_pair(2.5, -2.5))); // Bottom.
        //     // lines_box.push_back(std::make_pair(std::make_pair(-2.5, -2.5), std::make_pair(-2.5, 2.5))); // Left.
        //     // lines_box.push_back(std::make_pair(std::make_pair(2.5, -2.5), std::make_pair(2.5, 2.5))); // Top.
        //     // lines_box.push_back(std::make_pair(std::make_pair(-2.5, 2.5), std::make_pair(2.5, 2.5))); // Right.
        //     // double line_radius = 0.1;
        //     // auto boundary_obstacles = createObstaclesFromLines(lines, line_radius, "boundary_circle");
        //     // obstacles.insert(boundary_obstacles.begin(), boundary_obstacles.end());


            
        //     config.radius_robot = 0.1;
        //     config.goal_tolerance = 0.05;
        //     config.obstacles = obstacles;            
        //     config.cfgs_start = create_cfgs_square_shifted(robot_names, -2.0, 2.0, 0.3);
        //     config.cfgs_goal = create_cfgs_square_shifted(robot_names, 2.0, -2.0, 0.26);
        //     // config.cfgs_post_goal = create_cfgs_square_staggered(robot_names, 0.13, -1.3);

        //     config.extra_info["max_distance_meters"] = "12";
        //     config.extra_info["heuristic_type"] = "bwd_dijkstra";
        //     config.extra_info["heuristic_grid_resolution"] = "0.05";
        //     break;
        // }

        // Funnel.
        case 6:
        {
            std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>> obstacles;

            // Enclose the whole thing in a 2.7 by 2.7 box.
            std::vector<std::pair<std::pair<double, double>, std::pair<double, double>>> lines;
            lines.push_back(std::make_pair(std::make_pair(-2.7, -2.7), std::make_pair(2.7, -2.7))); // Bottom.
            lines.push_back(std::make_pair(std::make_pair(-2.7, -2.7), std::make_pair(-2.7, 2.7))); // Left.
            lines.push_back(std::make_pair(std::make_pair(2.7, -2.7), std::make_pair(2.7, 2.8))); // Top.
            lines.push_back(std::make_pair(std::make_pair(-2.7, 2.7), std::make_pair(2.7, 2.7))); // Right.
            lines.push_back(std::make_pair(std::make_pair(0.0, -2.7), std::make_pair(0.6, -0.18)));
            lines.push_back(std::make_pair(std::make_pair(0.0, 2.7), std::make_pair(0.6, 0.18)));
            // lines.push_back(std::make_pair(std::make_pair(-0.5, -1.0), std::make_pair(0.6, -0.25)));
            // lines.push_back(std::make_pair(std::make_pair(-0.5, 1.0), std::make_pair(0.6, 0.15)));

            
            double line_radius = 0.1;
            auto boundary_obstacles = createObstaclesFromLines(lines, line_radius, "boundary_circle");
            obstacles.insert(boundary_obstacles.begin(), boundary_obstacles.end());
            

            config.obstacles = obstacles;
            config.radius_robot = 0.1;
            config.goal_tolerance = 0.05;
            double spacing = 0.3;
            config.cfgs_start = create_cfgs_curtain(robot_names, spacing, 0.6, -2.0, 2.2, true);
            config.cfgs_goal = create_cfgs_curtain(robot_names, spacing, 0.6, 1.6, 2.2, true);
            config.extra_info["heuristic_type"] = "bwd_dijkstra";
            config.extra_info["heuristic_grid_resolution"] = "0.05";
            config.extra_info["max_distance_meters"] = "20";
            config.extra_info["heuristic_grid_resolution"] = "0.05";
            break;
        }        
        // Circle around X to right column.
        case 7:
        {
            std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>> obstacles;

            // Enclose the whole thing in a 2.7 by 2.7 box.
            std::vector<std::pair<std::pair<double, double>, std::pair<double, double>>> lines;
            lines.push_back(std::make_pair(std::make_pair(-2.7, -2.7), std::make_pair(2.7, -2.7))); // Bottom.
            lines.push_back(std::make_pair(std::make_pair(-2.7, -2.7), std::make_pair(-2.7, 2.7))); // Left.
            lines.push_back(std::make_pair(std::make_pair(2.7, -2.7), std::make_pair(2.7, 2.8))); // Top.
            lines.push_back(std::make_pair(std::make_pair(-2.7, 2.7), std::make_pair(2.7, 2.7))); // Right.
            lines.push_back(std::make_pair(std::make_pair(-0.8, 0.8), std::make_pair(0.8, -0.8)));
            lines.push_back(std::make_pair(std::make_pair(-0.8, -0.8), std::make_pair(0.8, 0.8)));
            
            double line_radius = 0.1;
            auto boundary_obstacles = createObstaclesFromLines(lines, line_radius, "boundary_circle");
            obstacles.insert(boundary_obstacles.begin(), boundary_obstacles.end());
            

            config.obstacles = obstacles;
            config.radius_robot = 0.1;
            config.goal_tolerance = 0.05;
            double spacing = 0.24;
            config.cfgs_start = create_cfgs_circle(robot_names, 1.5, config.radius_robot);
            config.cfgs_goal = create_cfgs_curtain(robot_names, spacing, 0.45, 1.6, 2.2, true);
            config.extra_info["heuristic_type"] = "bwd_dijkstra";
            config.extra_info["heuristic_grid_resolution"] = "0.05";
            config.extra_info["max_distance_meters"] = "12";
            config.extra_info["heuristic_grid_resolution"] = "0.05";
            break;
        }
        // Slalom.
        case 8:
        {
            std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>> obstacles;

            // Enclose the whole thing in a 2.7 by 2.7 box.
            std::vector<std::pair<std::pair<double, double>, std::pair<double, double>>> lines;
            lines.push_back(std::make_pair(std::make_pair(-2.7, -2.7), std::make_pair(2.7, -2.7))); // Bottom.
            lines.push_back(std::make_pair(std::make_pair(-2.7, -2.7), std::make_pair(-2.7, 2.7))); // Left.
            lines.push_back(std::make_pair(std::make_pair(2.7, -2.7), std::make_pair(2.7, 2.8))); // Top.
            lines.push_back(std::make_pair(std::make_pair(-2.7, 2.7), std::make_pair(2.7, 2.7))); // Right.
            lines.push_back(std::make_pair(std::make_pair(0.0, -2.7), std::make_pair(0.0, 1.8)));
            lines.push_back(std::make_pair(std::make_pair(-1.0, 2.7), std::make_pair(-1.0, -1.8)));
            // lines.push_back(std::make_pair(std::make_pair(1.0, 2.7), std::make_pair(1.0, -1.8)));
            
            double line_radius = 0.1;
            auto boundary_obstacles = createObstaclesFromLines(lines, line_radius, "boundary_circle");
            obstacles.insert(boundary_obstacles.begin(), boundary_obstacles.end());
            

            config.obstacles = obstacles;
            config.radius_robot = 0.1;
            config.goal_tolerance = 0.05;
            double spacing = 0.24;
            config.cfgs_start = create_cfgs_curtain(robot_names, spacing, 0.45, -2.0, 2.2, true);
            config.cfgs_goal = create_cfgs_curtain(robot_names, spacing, 0.45, 1.6, 2.2, true);
            config.extra_info["heuristic_type"] = "bwd_dijkstra";
            config.extra_info["heuristic_grid_resolution"] = "0.05";
            config.extra_info["max_distance_meters"] = "20";
            config.extra_info["heuristic_grid_resolution"] = "0.05";
            break;
        }
        
        // ==============================
        // Special tasks.
        // ==============================
        // Box tight pack.
        case 9: 
        {
            // Number of robots must be 9.
            if (robot_names.size() != 9) {
                throw std::invalid_argument("Number of robots must be 9");
            }
            // Create a box with a small opening.
            std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>> obstacles;
            double radius = 0.11;
            std::vector<std::pair<std::pair<double, double>, std::pair<double, double>>> lines;
            lines.push_back(std::make_pair(std::make_pair(-1.9, -0.55), std::make_pair(-1.9, 0.55))); // Left.
            lines.push_back(std::make_pair(std::make_pair(-0.7, -0.55), std::make_pair(-0.7, 0.55))); // Right.
            lines.push_back(std::make_pair(std::make_pair(-1.9,  0.55), std::make_pair(-0.5,  0.55))); // Top.
            lines.push_back(std::make_pair(std::make_pair(-1.9, -0.55), std::make_pair(-0.5, -0.55))); // Bottom.
 
            obstacles = createObstaclesFromLines(lines, radius, "box_circle");

            config.radius_robot = 0.12;
            config.goal_tolerance = 0.05;
            config.obstacles = obstacles;
            config.extra_info["heuristic_grid_resolution"] = "0.01";
            config.cfgs_start = create_cfgs_square_shifted(robot_names, -1.3, 0.0, 0.25);
            config.cfgs_goal = create_cfgs_square_shifted(robot_names, -1.3, 0.0, 0.25, true);
            break;
        }
        // Box with small opening.
        case 10:
        {
            // Number of robots must be 9.
            if (robot_names.size() != 9) {
                throw std::invalid_argument("Number of robots must be 9");
            }
            // Create a box with a small opening.
            std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>> obstacles;
            double radius = 0.1;
            std::vector<std::pair<std::pair<double, double>, std::pair<double, double>>> lines;
            lines.push_back(std::make_pair(std::make_pair(-1.9, -0.55), std::make_pair(-1.9, 0.55))); // Left.
            lines.push_back(std::make_pair(std::make_pair(-0.6, -0.55), std::make_pair(-0.6, 0.15))); // Right.
            lines.push_back(std::make_pair(std::make_pair(-0.6, -0.55), std::make_pair(-0.6, 0.15))); // Right.
            lines.push_back(std::make_pair(std::make_pair(-1.9,  0.55), std::make_pair(-0.3,  0.55))); // Top.
            lines.push_back(std::make_pair(std::make_pair(-1.9, -0.55), std::make_pair(-0.3, -0.55))); // Bottom.
 
            obstacles = createObstaclesFromLines(lines, radius, "box_circle");

            
            config.radius_robot = 0.12;
            config.goal_tolerance = 0.05;
            config.obstacles = obstacles;
            config.extra_info["heuristic_grid_resolution"] = "0.05";
            config.cfgs_start = create_cfgs_square_shifted(robot_names, -1.3, 0.0, 0.27);
            config.cfgs_goal = create_cfgs_square_shifted(robot_names, 2.0, 0.0, 0.27, true);
            // config.cfgs_post_goal = create_cfgs_square_staggered(robot_names, 0.13, -1.3);
            break;
        }
        // Single long corridor.
        case 11: 
        {
            // Must be 9 robots.
            if (robot_names.size() != 9) {
                throw std::invalid_argument("Number of robots must be 9");
            }
            std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>> obstacles;
            // Enclose the whole thing in a 2.7 by 2.7 box.
            std::vector<std::pair<std::pair<double, double>, std::pair<double, double>>> lines;
            lines.push_back(std::make_pair(std::make_pair(-2.7, -0.5), std::make_pair(2.7, -0.25))); // Bottom.
            lines.push_back(std::make_pair(std::make_pair(-2.7, -0.5), std::make_pair(-2.7, 0.5))); // Left.
            lines.push_back(std::make_pair(std::make_pair(2.7, 0.25), std::make_pair(2.7, -0.25))); // Top.
            lines.push_back(std::make_pair(std::make_pair(-2.7, 0.5), std::make_pair(2.7, 0.25))); // Right.
            double line_radius = 0.1;
            auto boundary_obstacles = createObstaclesFromLines(lines, line_radius, "boundary_circle");
            obstacles.insert(boundary_obstacles.begin(), boundary_obstacles.end());

            config.radius_robot = 0.1;
            config.goal_tolerance = 0.05;
            config.obstacles = obstacles;
            config.extra_info["heuristic_grid_resolution"] = "0.05";
            
            config.cfgs_start;
            for (int i = 0; i < robot_names.size(); i++) {
                config.cfgs_start[robot_names[i]] = gco::Configuration2(-2.3 + i * 0.25, 0.0, 0.0);
            }
            config.cfgs_goal;
            for (int i = 0; i < robot_names.size(); i++) {
                config.cfgs_goal[robot_names[i]] = gco::Configuration2(2.3 - i * 0.25, 0.0, 0.0);
            }

            config.extra_info["max_distance_meters"] = "12";
            config.extra_info["heuristic_type"] = "bwd_dijkstra";
            config.extra_info["heuristic_grid_resolution"] = "0.05";
            // config.cfgs_post_goal = create_cfgs_square_staggered(robot_names, 0.13, -1.3);
            break;
            break;
        }

        // NEW SCENES.
        // Get out of triangle requiring concurrent motion.
        case 12: 
        {
            // Must be 3 robots.
            if (robot_names.size() > 3) {
                throw std::invalid_argument("Number of robots must be at most 3.");
            }
            std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>> obstacles;
            // Enclose the whole thing in a triangle.
            std::vector<std::pair<std::pair<double, double>, std::pair<double, double>>> lines;
            double scale = 0.7;
            lines.push_back(std::make_pair(std::make_pair(-0.5 * scale, 0.0 * scale), std::make_pair(0.5 * scale, 0.0 * scale))); // Bottom.
            lines.push_back(std::make_pair(std::make_pair(-0.5 * scale, 0.0 * scale), std::make_pair((-0.5+0.125) * scale, 0.25 * scale))); // Left bottom.
            lines.push_back(std::make_pair(std::make_pair(-0.125 * scale, 0.75 * scale), std::make_pair(0.0 * scale,1.0 * scale))); // Left.
            lines.push_back(std::make_pair(std::make_pair(0.5 * scale, 0.0 * scale), std::make_pair(0.0 * scale, 1.0 * scale))); // Right.
            double line_radius = 0.04;
            auto boundary_obstacles = createObstaclesFromLines(lines, line_radius, "boundary_circle");
            obstacles.insert(boundary_obstacles.begin(), boundary_obstacles.end());

            config.radius_robot = 0.1;
            config.goal_tolerance = 0.05;
            config.obstacles = obstacles;
            config.extra_info["heuristic_grid_resolution"] = "0.02";
            
            std::vector<gco::Configuration2> cfgs_start_full = {
                                    gco::Configuration2(-0.1, 0.2, 0.0),
                                    gco::Configuration2(0.1, 0.2, 0.0),
                                    gco::Configuration2(0.0, 0.4, 0.0)};
            // Take the first N only.
            config.cfgs_start.clear();
            for (int i = 0; i < robot_names.size(); i++) {
                config.cfgs_start[robot_names[i]] = cfgs_start_full[i];
            }
            

            config.cfgs_goal;
            for (int i = 0; i < robot_names.size(); i++) {
                config.cfgs_goal[robot_names[i]] = gco::Configuration2(2.3 - i * 0.25, 0.0, 0.0);
            }

            std::cout << "Planning with N cfgs start, goal " << config.cfgs_start.size() << ", " << config.cfgs_goal.size() << std::endl;

            config.extra_info["max_distance_meters"] = "12";
            config.extra_info["heuristic_type"] = "bwd_dijkstra";
            config.extra_info["heuristic_grid_resolution"] = "0.05";
            // config.cfgs_post_goal = create_cfgs_square_staggered(robot_names, 0.13, -1.3);
            break;
        }

        // Packed starts with goals being between starts. Needs rotation.
        case 13: 
        {
            // Must be at most 3 robots.
            if (robot_names.size() > 3) {
                throw std::invalid_argument("Number of robots must be at most 3.");
            }
            // Create a circle of obstacle, radius 1.8.
            std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>> obstacles;
            double circle_radius = 0.3;
            double circle_center_x = 0.0;
            double circle_center_y = -0.01;
            double obstacle_radius = 0.05;
            int num_obstacles = 30;
            for (int i = 0; i < num_obstacles; i++) {
                double angle = 2 * M_PI * i / num_obstacles;
                double x = circle_center_x + circle_radius * std::cos(angle);
                double y = circle_center_y + circle_radius * std::sin(angle);
                obstacles["obstacle_circle_" + std::to_string(i)] = std::make_pair(gco::Obstacle::createCircle("obstacle_circle_" + std::to_string(i), obstacle_radius), gco::Transform2(x, y, 0.0));
            }
            
            config.radius_robot = 0.10;
            config.goal_tolerance = 0.03;
            config.obstacles = obstacles;
            config.extra_info["heuristic_grid_resolution"] = "0.02";
            config.cfgs_start = create_cfgs_circle(robot_names, 0.4, config.radius_robot);
            config.cfgs_goal;
          
            // Create a circle of obstacle, radius 1.8.
            std::vector<gco::Configuration2> cfgs_start_full = {
                                    gco::Configuration2(-0.1, -0.09, 0.0),
                                    gco::Configuration2(0.1, -0.09, 0.0),
                                    gco::Configuration2(0.0, 0.08, 0.0)};
            // Take the first N only.
            config.cfgs_start.clear();
            for (int i = 0; i < robot_names.size(); i++) {
                config.cfgs_start[robot_names[i]] = cfgs_start_full[i];
            }
            std::vector<gco::Configuration2> cfgs_goal_full = {
                                    gco::Configuration2(-0.11, 0.065, 0.0),
                                    gco::Configuration2(0.11, 0.065, 0.0),
                                    gco::Configuration2(0.0, -0.13, 0.0)};
            // Take the first N only.
            config.cfgs_goal.clear();
            for (int i = 0; i < robot_names.size(); i++) {
                config.cfgs_goal[robot_names[i]] = cfgs_goal_full[i];
            }

            config.extra_info["max_distance_meters"] = "12";
            config.extra_info["heuristic_type"] = "bwd_dijkstra";
            config.extra_info["heuristic_grid_resolution"] = "0.05";
            // config.cfgs_post_goal = create_cfgs_square_staggered(robot_names, 0.13, -1.3);
            break;
        }


        // Packed starts with goals being between starts. Needs rotation.
        case 14: 
        {
            std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>> obstacles;

            // Enclose the whole thing in a 5 by 5 box.
            std::vector<std::pair<std::pair<double, double>, std::pair<double, double>>> lines;
            lines.push_back(std::make_pair(std::make_pair(-3, -3), std::make_pair(3, -3))); // Bottom.
            lines.push_back(std::make_pair(std::make_pair(-3, -3), std::make_pair(-3, 3))); // Left.
            lines.push_back(std::make_pair(std::make_pair(3, -3), std::make_pair(3, 3))); // Top.
            lines.push_back(std::make_pair(std::make_pair(-3, 3), std::make_pair(3, 3))); // Right.
             
            double line_radius = 0.1;
            auto boundary_obstacles = createObstaclesFromLines(lines, line_radius, "boundary_circle");
            obstacles.insert(boundary_obstacles.begin(), boundary_obstacles.end());

            // Add a few sparse obstacle lines.
            int obstacle_counter = 0;
            for (double obsx = -1.0; obsx <= 1.0; obsx += 0.4) {
                for (double obsy = -2.9; obsy <= 2.9; obsy += 0.4) {
                    // Create obstacle circle.
                    double y_offset = obstacle_counter % 2 == 0 ? 0.0 : 0.2;
                    auto obs = gco::Obstacle::createCircle("obstacle_circle_" + std::to_string(obstacle_counter), 0.05);
                    obstacles["obstacle_circle_" + std::to_string(obstacle_counter)] = std::make_pair(obs, gco::Transform2(obsx, obsy + y_offset, 0.0));
                    obstacle_counter++;
                }
            }

            config.obstacles = obstacles;
            config.radius_robot = 0.1;
            config.goal_tolerance = 0.05;
            double spacing = 0.3;
            config.cfgs_start = create_cfgs_curtain(robot_names, spacing, 0.8, -2.75, 2.7, true);
            config.cfgs_goal = create_cfgs_curtain(robot_names, spacing, 0.8, 2.0, 2.7, true);
            config.extra_info["heuristic_type"] = "bwd_dijkstra";
            config.extra_info["heuristic_grid_resolution"] = "0.05";
            config.extra_info["max_distance_meters"] = "20";
            config.extra_info["heuristic_grid_resolution"] = "0.05";
            break;
        }


        // Packed starts with goals being between starts. Needs rotation.
        case 15: 
        {
            std::map<std::string, std::pair<gco::ObstaclePtr, gco::Transform2>> obstacles;

            // Enclose the whole thing in a 5 by 5 box.
            std::vector<std::pair<std::pair<double, double>, std::pair<double, double>>> lines;
            lines.push_back(std::make_pair(std::make_pair(-3, -6), std::make_pair(3, -6))); // Bottom.
            lines.push_back(std::make_pair(std::make_pair(-3, -6), std::make_pair(-3, 6))); // Left.
            lines.push_back(std::make_pair(std::make_pair(3, -6), std::make_pair(3, 6))); // Top.
            lines.push_back(std::make_pair(std::make_pair(-3, 6), std::make_pair(3, 6))); // Right.
             
            double line_radius = 0.1;
            auto boundary_obstacles = createObstaclesFromLines(lines, line_radius, "boundary_circle");
            obstacles.insert(boundary_obstacles.begin(), boundary_obstacles.end());

            // Add a few sparse obstacle lines.
            int obstacle_counter = 0;
            for (double obsx = -1.0; obsx <= 1.0; obsx += 0.4) {
                for (double obsy = -5.9; obsy <= 5.9; obsy += 0.4) {
                    // Create obstacle circle.
                    // Random offset to stagger obstacles.
                    auto rng = std::mt19937{std::random_device{}()};
                    std::uniform_int_distribution<int> dist(-2, 2);
                    int rand_offset = dist(rng);
                    double y_offset = rand_offset * 0.1;
                    auto obs = gco::Obstacle::createCircle("obstacle_circle_" + std::to_string(obstacle_counter), 0.05);
                    obstacles["obstacle_circle_" + std::to_string(obstacle_counter)] = std::make_pair(obs, gco::Transform2(obsx, obsy + y_offset, 0.0));
                    obstacle_counter++;
                }
            }

            config.obstacles = obstacles;
            config.radius_robot = 0.1;
            config.goal_tolerance = 0.05;
            double spacing = 0.3;
            config.cfgs_start = create_cfgs_curtain(robot_names, spacing, 0.8, -2.75, 5.7, true);
            config.cfgs_goal = create_cfgs_curtain(robot_names, spacing, 0.8, 2.0, 5.7, true);
            config.extra_info["heuristic_type"] = "bwd_dijkstra";
            config.extra_info["heuristic_grid_resolution"] = "0.05";
            config.extra_info["max_distance_meters"] = "20";
            config.extra_info["heuristic_grid_resolution"] = "0.05";
            break;
        }

        default:
        {
            std::cerr << "Invalid task type: " << task_type << std::endl;
            throw std::invalid_argument("Invalid task type");
        }
    }
    
    return config;
}

void printConfigurations(const std::map<std::string, gco::Configuration2>& cfgs_start,
                        const std::map<std::string, gco::Configuration2>& cfgs_goal) {
    // Print the start and goal configurations in a way that is easy to copy and paste into Python.
    std::cout << "cfgs_start = {" << std::endl;
    for (auto [robot_name, cfg] : cfgs_start) {
        std::cout << "    '" << robot_name << "': [" << cfg.x << ", " << cfg.y << ", " << cfg.theta << "]," << std::endl;
    }
    std::cout << "}" << std::endl;
    
    std::cout << "cfgs_goal = {" << std::endl;
    for (auto [robot_name, cfg] : cfgs_goal) {
        std::cout << "    '" << robot_name << "': [" << cfg.x << ", " << cfg.y << ", " << cfg.theta << "]," << std::endl;
    }
    std::cout << "}" << std::endl;
}

void addPostGoalConfigurations(gco::MultiRobotPaths& paths,
                              const std::map<std::string, gco::Configuration2>& cfgs_post_goal,
                              const std::map<std::string, gco::Configuration2>& cfgs_goal) {
    if (cfgs_post_goal.empty()) {
        return;
    }
    
    for (auto& [robot_name, cfg] : cfgs_post_goal) {
        // Find the last configuration of the path that is closest to the post-goal configuration.
        auto last_cfg = paths[robot_name].back();
        std::string robot_name_originally_assigned_to_this_goal = "";
        double min_distance = std::numeric_limits<double>::max();
        for (auto& [robot_name_original, cfg_original] : cfgs_goal) {
            double distance = gco::configurationDistance(cfg_original, last_cfg);
            if (distance < min_distance) {
                min_distance = distance;
                robot_name_originally_assigned_to_this_goal = robot_name_original;
            }
        }
        
        // Get the goal and post-goal configurations for this robot
        auto goal_cfg = last_cfg;
        auto post_goal_cfg = cfgs_post_goal.at(robot_name_originally_assigned_to_this_goal);
        
        // Interpolate linearly between goal and post-goal with 6 intermediate points
        for (int i = 1; i <= 6; i++) {
            double t = static_cast<double>(i) / 6.0; // t goes from 1/6 to 1.0
            gco::Configuration2 interpolated_cfg;
            interpolated_cfg.x = goal_cfg.x + t * (post_goal_cfg.x - goal_cfg.x);
            interpolated_cfg.y = goal_cfg.y + t * (post_goal_cfg.y - goal_cfg.y);
            interpolated_cfg.theta = goal_cfg.theta + t * (post_goal_cfg.theta - goal_cfg.theta);
            paths[robot_name].push_back(interpolated_cfg);
        }
    }
}

} // namespace gco_examples 