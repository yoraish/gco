#pragma once

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <iostream>

#include "gco/robot.hpp"
#include "gco/types.hpp"
#include "gco/obstacles/obstacles.hpp"
#include "gco/spatial/transforms.hpp"
#include "gco/world/world.hpp"
#include "gco/utils.hpp"
#include "gco/heuristics/bwd_dijkstra_heuristic.hpp"
#include "gco/heuristics/euclidean_heuristic.hpp"

#include "configuration_generators.hpp"
#include "visualization_utils.hpp"
#include "task_setup.hpp"

namespace gco_examples {

template<typename PlannerType>
void runPlannerExample(int task_type, int num_robots, const std::string& output_filename = "amrmp_solution.json") {
    // Create robot names
    std::vector<std::string> robot_names;
    for (int i = 0; i < num_robots; i++) {
        robot_names.push_back("robot_" + std::to_string(i));
    }

    // Setup task configuration
    TaskConfig task_config = setupTask(task_type, robot_names);

    // Create the robots
    gco::JointRanges joint_ranges({-2.0, -2.0, 0.0}, {2.0, 2.0, 1.0}, {0.01, 0.01, 0.01});

    std::map<std::string, gco::RobotPtr> robots;
    for (const auto& robot_name : robot_names) {
        robots[robot_name] = std::make_shared<gco::RobotDisk>(robot_name, joint_ranges, task_config.radius_robot);
    }

    // Create the world
    gco::WorldPtr world = std::make_shared<gco::World>();
    for (const auto& robot : robots) {
        world->addRobot(robot.second);
        std::cout << "Added robot " << robot.first << std::endl;
    }

    // Add the prescribed obstacles
    for (const auto& [obstacle_name, obstacle_pair] : task_config.obstacles) {
        world->addObstacle(obstacle_pair.first, obstacle_pair.second);
        std::cout << "Added obstacle " << obstacle_name << std::endl;
    }

    // Print configurations
    printConfigurations(task_config.cfgs_start, task_config.cfgs_goal);

    // // Create goal heuristics.
    // std::map<gco::Configuration2, gco::HeuristicPtr> goal_heuristics;
    // for (const auto& [robot_name, cfg_goal] : task_config.cfgs_goal) {
    //     auto heuristic = std::make_shared<gco::BwdDijkstraHeuristic>(world, 0.02, 100000);
    //     heuristic->precomputeForGoal(cfg_goal);
    //     // auto heuristic = std::make_shared<gco::EuclideanHeuristic>();
    //     goal_heuristics[cfg_goal] = heuristic;
    // }

    // Create planner and run planning
    auto start_time = std::chrono::high_resolution_clock::now();
    
    PlannerType planner(world, task_config.goal_tolerance);
    planner.setVerbose(false);
    
    gco::MultiRobotPaths paths;
    
    // Call the plan function with heuristics - the planner will handle the interface differences
    paths = planner.plan(task_config.cfgs_start, task_config.cfgs_goal, {}, false);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    // Add post-goal configurations if any
    addPostGoalConfigurations(paths, task_config.cfgs_post_goal, task_config.cfgs_goal);

    // Save the solution to a JSON file
    saveVisualizationJSON(robots, task_config.obstacles, task_config.cfgs_start, task_config.cfgs_goal, paths, output_filename);
    std::cout << "Solution saved to " << output_filename << std::endl;

    // Call visualization script
    std::string command = "python3 /home/yoraish/code/gco-dev/trials/visualize_trajectories.py";
    system(command.c_str());
}

void printUsage(const std::string& program_name, const std::string& planner_name) {
    std::cerr << "Usage: " << program_name << " <mode> [options...]" << std::endl;
    std::cerr << "Modes:" << std::endl;
    std::cerr << "  <task_type> <N robots>  - Run specific task (0-8) with " << planner_name << std::endl;
    std::cerr << "Examples:" << std::endl;
    std::cerr << "  " << program_name << " 0 4                    - Run task 0 with 4 robots" << std::endl;
    std::cerr << "  " << program_name << " 7 2                    - Run task 7 with 2 robots" << std::endl;
}

bool parseArguments(int argc, char* argv[], int& task_type, int& num_robots) {
    if (argc < 3) {
        std::cerr << "Error: Task type and number of robots required" << std::endl;
        return false;
    }
    
    try {
        task_type = std::stoi(argv[1]);
        num_robots = std::stoi(argv[2]);
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid arguments. Task type and number of robots must be integers." << std::endl;
        return false;
    }
    
    if (task_type < 0 || task_type > 8) {
        std::cerr << "Error: Task type must be between 0 and 8" << std::endl;
        return false;
    }
    
    if (num_robots <= 0) {
        std::cerr << "Error: Number of robots must be positive" << std::endl;
        return false;
    }
    
    return true;
}

} // namespace gco_examples 