#include <iostream>
#include <string>
#include <map>
#include <memory>

// Include common utilities first
#include "common/example_template.hpp"
#include "gco/planners/planner_stats.hpp"

// Forward declarations to avoid redefinition issues
namespace gco {
    class GSPIPlanner;
}

// Include all planner headers
#include "gco/planners/pibt.hpp"
#include "gco/planners/ctswap.hpp"
#include "gco/planners/gspi.hpp"

class PlannerInterface {
public:
    virtual ~PlannerInterface() = default;
    virtual gco::PlannerStats plan(const std::map<std::string, gco::Configuration2>& cfgs_start,
                                  const std::map<std::string, gco::Configuration2>& cfgs_goal,
                                  const std::map<gco::Configuration2, gco::HeuristicPtr>& goal_heuristics,
                                  bool is_modify_starts_goals_locally = true) = 0;
    virtual void setVerbose(bool verbose) = 0;
    virtual void setMaxIterations(int max_iterations) = 0;
};

template<typename PlannerType>
class PlannerWrapper : public PlannerInterface {
private:
    std::unique_ptr<PlannerType> planner_;

public:
    PlannerWrapper(gco::WorldPtr world, double goal_tolerance, const std::map<std::string, std::string>& heuristic_params, bool disregard_orientation = true, double goal_check_tolerance = -1.0) 
        : planner_(std::make_unique<PlannerType>(world, goal_tolerance, heuristic_params, 42, 600.0, disregard_orientation, goal_check_tolerance)) {}

    gco::PlannerStats plan(const std::map<std::string, gco::Configuration2>& cfgs_start,
                           const std::map<std::string, gco::Configuration2>& cfgs_goal,
                           const std::map<gco::Configuration2, gco::HeuristicPtr>& goal_heuristics,
                           bool is_modify_starts_goals_locally = true) override {
        return planner_->plan(cfgs_start, cfgs_goal, goal_heuristics, is_modify_starts_goals_locally);
    }

    void setVerbose(bool verbose) override {
        planner_->setVerbose(verbose);
    }
    
    void setMaxIterations(int max_iterations) override {
        planner_->setMaxIterations(max_iterations);
    }
};

std::unique_ptr<PlannerInterface> createPlanner(const std::string& planner_name, 
                                               gco::WorldPtr world, 
                                               double goal_tolerance, 
                                               const std::map<std::string, std::string>& heuristic_params) {
    if (planner_name == "gspi" || planner_name == "g" || 
               planner_name == "gspi" || planner_name == "GSPI") {
        return std::make_unique<PlannerWrapper<gco::GSPIPlanner>>(world, goal_tolerance, heuristic_params, true, -1.0);
    } else {
        throw std::runtime_error("Unknown planner: " + planner_name + 
                                ". Available planners: gspi");
    }
}

void runPlannerExample(const std::string& planner_name, 
                      int task_type, 
                      int num_robots, 
                      const std::string& output_filename = "amrmp_solution.json") {
    std::cout << "Using planner: " << planner_name << std::endl;
    
    // Generate robot names
    std::vector<std::string> robot_names;
    for (int i = 0; i < num_robots; i++) {
        robot_names.push_back("robot_" + std::to_string(i));
    }

    // Setup task configuration
    gco_examples::TaskConfig task_config = gco_examples::setupTask(task_type, robot_names);

    // Create robots
    gco::JointRanges joint_ranges({-2.0, -2.0, 0.0}, {2.0, 2.0, 1.0}, {0.01, 0.01, 0.01});

    std::map<std::string, gco::RobotPtr> robots;
    for (const auto& robot_name : robot_names) {
        robots[robot_name] = std::make_shared<gco::RobotDisk>(robot_name, joint_ranges, task_config.radius_robot);
    }

    // Create the world
    gco::WorldPtr world = std::make_shared<gco::World>();
    for (const auto& robot : robots) {
        world->addRobot(robot.second);
    }

    // Add the prescribed obstacles
    for (const auto& [obstacle_name, obstacle_pair] : task_config.obstacles) {
        world->addObstacle(obstacle_pair.first, obstacle_pair.second);
    }

    // Print configurations
    // gco_examples::printConfigurations(task_config.cfgs_start, task_config.cfgs_goal);

    // Create goal heuristics
    std::map<gco::Configuration2, gco::HeuristicPtr> goal_heuristics;
    // for (const auto& [robot_name, cfg_goal] : task_config.cfgs_goal) {
    //     auto heuristic = std::make_shared<gco::EuclideanHeuristic>();
    //     goal_heuristics[cfg_goal] = heuristic;
    // }

    // Create planner and run planning
    auto start_time = std::chrono::high_resolution_clock::now();

    // Additional parameters for heuristics.
    std::map<std::string, std::string> heuristic_params;
    heuristic_params["heuristic_type"] = "bwd_dijkstra";
    // heuristic_params["heuristic_type"] = "euclidean";
    heuristic_params["grid_resolution"] = "0.05";
    heuristic_params["max_distance_meters"] = "10";

    if (task_config.extra_info.find("heuristic_grid_resolution") != task_config.extra_info.end()) {
        heuristic_params["grid_resolution"] = task_config.extra_info["heuristic_grid_resolution"];
    }
    if (task_config.extra_info.find("heuristic_type") != task_config.extra_info.end()) {
        heuristic_params["heuristic_type"] = task_config.extra_info["heuristic_type"];
    }
    if (task_config.extra_info.find("max_distance_meters") != task_config.extra_info.end()) {
        heuristic_params["max_distance_meters"] = task_config.extra_info["max_distance_meters"];
    }
    
    auto planner = createPlanner(planner_name, world, task_config.goal_tolerance, heuristic_params);
    planner->setVerbose(false);
    
    // Set maximum iterations to 500 for this example
    planner->setMaxIterations(500);
    
    // Call the plan function with heuristics and get statistics
    auto stats = planner->plan(task_config.cfgs_start, task_config.cfgs_goal, goal_heuristics, false);
    
    // Extract paths from stats
    gco::MultiRobotPaths paths = stats.paths;

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    if (!paths.empty()) {
        // Add post-goal configurations if any
        gco_examples::addPostGoalConfigurations(paths, task_config.cfgs_post_goal, task_config.cfgs_goal);

        // Save the solution to a JSON file
        gco_examples::saveVisualizationJSON(robots, task_config.obstacles, task_config.cfgs_start, task_config.cfgs_goal, paths, output_filename);
        std::cout << "Solution saved to " << output_filename << std::endl;

        // Call visualization script with the correct file path
        std::string command = "python3 /home/yoraish/code/gco-dev/utils/visualize_trajectories.py " + output_filename;
        system(command.c_str());
    } else {
        std::cout << "Planner did not return paths (might still be exploring). Skipping visualization." << std::endl;
    }
}
void printUsage(const std::string& program_name) {
    std::cerr << "Usage: " << program_name << " <planner> <task_type> <num_robots> [output_file]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Arguments:" << std::endl;
    std::cerr << "  planner      - Planner to use (pibt, ctswap, hybrid)" << std::endl;
    std::cerr << "  task_type    - Task type (0-8)" << std::endl;
    std::cerr << "  num_robots   - Number of robots" << std::endl;
    std::cerr << "  output_file  - Output JSON file (optional, default: amrmp_solution.json)" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Available planners:" << std::endl;
    std::cerr << "  pibt         - Priority Inheritance with Backtracking" << std::endl;
    std::cerr << "  ctswap       - Continuous-Time SWAP" << std::endl;
    std::cerr << "  hybrid       - Hybrid CTSWAP-PIBT" << std::endl;
    std::cerr << "  orca         - Theta* + ORCA decentralized" << std::endl;
    std::cerr << "  c-unav       - Continuous-Time SWAP with Unavailable Goals" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Available task types:" << std::endl;
    std::cerr << "  0 - Circle to square" << std::endl;
    std::cerr << "  1 - Square to circle" << std::endl;
    std::cerr << "  2 - Line to square" << std::endl;
    std::cerr << "  3 - Square to square (obstacles)" << std::endl;
    std::cerr << "  4 - Circle to rectangle" << std::endl;
    std::cerr << "  5 - Random starts and goals" << std::endl;
    std::cerr << "  6 - Write CMU" << std::endl;
    std::cerr << "  7 - Deadlock scenario" << std::endl;
    std::cerr << "  8 - Swapping scenario" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Examples:" << std::endl;
    std::cerr << "  " << program_name << " pibt 0 4                    - Run PIBT on task 0 with 4 robots" << std::endl;
    std::cerr << "  " << program_name << " ctswap 7 2                  - Run CTSWAP on task 7 with 2 robots" << std::endl;
    std::cerr << "  " << program_name << " hybrid 3 4 solution.json    - Run Hybrid on task 3 with 4 robots, save to solution.json" << std::endl;
    std::cerr << "  " << program_name << " orca 5 10 orca.json        - Run ORCA on task 5 with 10 robots" << std::endl;
}

bool parseArguments(int argc, char* argv[], std::string& planner_name, int& task_type, int& num_robots, std::string& output_file) {
    if (argc < 4) {
        std::cerr << "Error: Insufficient arguments" << std::endl;
        return false;
    }
    
    planner_name = argv[1];
    
    try {
        task_type = std::stoi(argv[2]);
        num_robots = std::stoi(argv[3]);
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid arguments. Task type and number of robots must be integers." << std::endl;
        return false;
    }
    
    if (task_type < 0 || task_type > 100) {
        std::cerr << "Error: Task type must be between 0 and 10" << std::endl;
        return false;
    }
    
    if (num_robots <= 0) {
        std::cerr << "Error: Number of robots must be positive" << std::endl;
        return false;
    }
    
    // Set output file (optional)
    output_file = (argc >= 5) ? argv[4] : "amrmp_solution.json";
    
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        printUsage(argv[0]);
        return 1;
    }

    std::string planner_name, output_file;
    int task_type, num_robots;
    
    if (!parseArguments(argc, argv, planner_name, task_type, num_robots, output_file)) {
        printUsage(argv[0]);
        return 1;
    }

    try {
        runPlannerExample(planner_name, task_type, num_robots, output_file);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 