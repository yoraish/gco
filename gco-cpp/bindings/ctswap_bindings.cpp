#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <memory>

#include "gco/planners/ctswap.hpp"
#include "gco/planners/pibt.hpp"
#include "gco/planners/gspi.hpp"
#include "gco/planners/object_gspi.hpp"
#include "gco/planners/object_centric_a_star.hpp"
#include "gco/planners/planner_stats.hpp"
#include "gco/world/world.hpp"
#include "gco/robot.hpp"
#include "gco/obstacles/obstacles.hpp"
#include "gco/objects/objects.hpp"
#include "gco/types.hpp"
#include "gco/spatial/transforms.hpp"

namespace py = pybind11;

gco::RobotPtr convert_python_robot_to_cpp(py::object robot_obj) {
    std::string name = robot_obj.attr("name").cast<std::string>();
    double radius = robot_obj.attr("radius").cast<double>();
    
    // Create joint ranges (we might want to make this configurable) TODO(yoraish).
    gco::Configuration2 mins(-2.0, -2.0, 0.0);
    gco::Configuration2 maxs(2.0, 2.0, 1.0);
    gco::Configuration2 discretization(0.01, 0.01, 0.01);
    gco::JointRanges joint_ranges(mins, maxs, discretization);
    return std::make_shared<gco::RobotDisk>(name, joint_ranges, radius);
}

gco::ObstaclePtr convert_python_obstacle_to_cpp(py::object obstacle_obj) {
    std::string name = obstacle_obj.attr("name").cast<std::string>();
    
    // Check if it's a circle obstacle
    if (py::hasattr(obstacle_obj, "radius")) {
        double radius = obstacle_obj.attr("radius").cast<double>();
        return gco::Obstacle::createCircle(name, radius);
    }
    // Check if it's a square obstacle
    else if (py::hasattr(obstacle_obj, "width") && !py::hasattr(obstacle_obj, "height")) {
        double width = obstacle_obj.attr("width").cast<double>();
        return gco::Obstacle::createSquare(name, width);
    }
    // Check if it's a rectangle obstacle. 
    else if (py::hasattr(obstacle_obj, "width") && py::hasattr(obstacle_obj, "height")) {
        double width = obstacle_obj.attr("width").cast<double>();
        double height = obstacle_obj.attr("height").cast<double>();
        return gco::Obstacle::createRectangle(name, width, height);
    }
    // Check if it's a polygon obstacle TODO(yoraish).
    // else if (py::hasattr(obstacle_obj, "vertices")) {
    //     py::object vertices_tensor = obstacle_obj.attr("vertices");
    //     return std::make_shared<gco::ObstaclePolygon>(name, vertices_tensor);
    // }
    else {
        throw std::runtime_error("Unknown obstacle type");
    }
}

gco::ObjectPtr convert_python_object_to_cpp(py::object object_obj) {
    std::string name = object_obj.attr("name").cast<std::string>();
    
    // Check if it's a circle object
    if (py::hasattr(object_obj, "radius")) {
        double radius = object_obj.attr("radius").cast<double>();
        return gco::Object::createCircle(name, radius);
    }
    // Check if it's a square object
    else if (py::hasattr(object_obj, "width") && !py::hasattr(object_obj, "height")) {
        double width = object_obj.attr("width").cast<double>();
        return gco::Object::createSquare(name, width);
    }
    // Check if it's a rectangle object. 
    else if (py::hasattr(object_obj, "width") && py::hasattr(object_obj, "height")) {
        double width = object_obj.attr("width").cast<double>();
        double height = object_obj.attr("height").cast<double>();
        return gco::Object::createRectangle(name, width, height);
    }
    // Check if it is a polygon.
    else if (py::hasattr(object_obj, "vertices")) {
        py::object vertices_tensor = object_obj.attr("vertices");
        std::vector<gco::Translation2> vertices;
        py::ssize_t num_vertices = py::len(vertices_tensor);
        for (py::ssize_t i = 0; i < num_vertices; i++) {
            py::object vertex = vertices_tensor.attr("__getitem__")(i);
            double x = vertex.attr("__getitem__")(0).cast<double>();
            double y = vertex.attr("__getitem__")(1).cast<double>();
            vertices.push_back(gco::Translation2(x, y));
        }
        return gco::Object::createPolygon(name, vertices);
    }
    else {
        throw std::runtime_error("Unknown object type");
    }
}

gco::Transform2 convert_python_transform_to_cpp(py::object transform_obj) {
    py::object t_tensor = transform_obj.attr("get_t")();
    py::object theta_tensor = transform_obj.attr("get_theta")();
    
    double x = t_tensor.attr("__getitem__")(0).cast<double>();
    double y = t_tensor.attr("__getitem__")(1).cast<double>();
    double theta = theta_tensor.attr("__getitem__")(0).cast<double>();
    
    return gco::Transform2(x, y, theta);
}

gco::WorldPtr convert_python_world_to_cpp(py::object world_obj) {
    auto cpp_world = std::make_shared<gco::World>();
    
    py::dict robots_dict = world_obj.attr("robots_d").cast<py::dict>();
    
    for (auto item : robots_dict) {
        std::string robot_name = item.first.cast<std::string>();
        py::object robot_obj = item.second.cast<py::object>();
        
        auto cpp_robot = convert_python_robot_to_cpp(robot_obj);
        cpp_world->addRobot(cpp_robot);
    }
    
    py::dict obstacles_dict = world_obj.attr("obstacles_d").cast<py::dict>();
    py::dict objects_dict = world_obj.attr("objects_d").cast<py::dict>();
    
    // Get obstacle poses using the world's getter methods
    py::dict obstacle_poses_dict = world_obj.attr("get_all_obstacle_poses")().cast<py::dict>();
    
    for (auto item : obstacles_dict) {
        std::string obstacle_name = item.first.cast<std::string>();
        py::object obstacle_obj = item.second.cast<py::object>();
        py::object pose_obj = obstacle_poses_dict[obstacle_name.c_str()];
        
        auto cpp_obstacle = convert_python_obstacle_to_cpp(obstacle_obj);
        auto cpp_pose = convert_python_transform_to_cpp(pose_obj);
        cpp_world->addObstacle(cpp_obstacle, cpp_pose);
    }
    
    py::dict object_poses_dict = world_obj.attr("get_all_object_poses")().cast<py::dict>();
    
    for (auto item : objects_dict) {
        std::string object_name = item.first.cast<std::string>();
        py::object object_obj = item.second.cast<py::object>();
        py::object pose_obj = object_poses_dict[object_name.c_str()];
        
        auto cpp_object = convert_python_object_to_cpp(object_obj);
        auto cpp_pose = convert_python_transform_to_cpp(pose_obj);
        cpp_world->addObject(cpp_object, cpp_pose);
    }
    
    return cpp_world;
}

py::dict convert_cpp_paths_to_python(const gco::MultiRobotPaths& paths) {
    py::dict python_paths;
    
    for (const auto& [robot_name, path] : paths) {
        py::list python_path;
        for (const auto& cfg : path) {
            py::list config = py::list();
            config.append(cfg.x);
            config.append(cfg.y);
            config.append(cfg.theta);
            python_path.append(config);
        }
        python_paths[robot_name.c_str()] = python_path;
    }
    
    return python_paths;
}

class CTSWAPWrapper {
public:
    CTSWAPWrapper(py::object world_obj, double goal_tolerance = 0.03, py::dict heuristic_params = py::dict(), unsigned int seed = 42, double timeout_seconds = 60.0, bool disregard_orientation = true, double goal_check_tolerance = -1.0) {
        cpp_world_ = convert_python_world_to_cpp(world_obj);
        
        // Convert Python dict to C++ map
        std::map<std::string, std::string> cpp_heuristic_params;
        for (auto item : heuristic_params) {
            std::string key = item.first.cast<std::string>();
            std::string value = item.second.cast<std::string>();
            cpp_heuristic_params[key] = value;
        }
        
        cpp_planner_ = std::make_unique<gco::CTSWAPPlanner>(cpp_world_, goal_tolerance, cpp_heuristic_params, seed, timeout_seconds, disregard_orientation, goal_check_tolerance);
    }
    
    py::dict plan(py::dict starts_dict, py::dict goals_dict) {
        // Convert Python dictionaries to C++ maps
        std::map<std::string, gco::Configuration2> cfgs_start;
        std::map<std::string, gco::Configuration2> cfgs_goal;
        
        for (auto item : starts_dict) {
            std::string robot_name = item.first.cast<std::string>();
            py::list config_list = item.second.cast<py::list>();
            cfgs_start[robot_name] = gco::Configuration2(
                config_list[0].cast<double>(),
                config_list[1].cast<double>(),
                config_list[2].cast<double>()
            );
        }
        
        for (auto item : goals_dict) {
            std::string robot_name = item.first.cast<std::string>();
            py::list config_list = item.second.cast<py::list>();
            cfgs_goal[robot_name] = gco::Configuration2(
                config_list[0].cast<double>(),
                config_list[1].cast<double>(),
                config_list[2].cast<double>()
            );
        }
        
        // Call the C++ planner and get statistics
        auto stats = cpp_planner_->plan(cfgs_start, cfgs_goal);
        
        // Extract paths from stats and convert back to Python format
        return convert_cpp_paths_to_python(stats.paths);
    }
    
    void set_verbose(bool verbose) {
        cpp_planner_->setVerbose(verbose);
    }
    
    // Heuristic parameter setters
    void set_grid_resolution_heuristic(double resolution) {
        cpp_planner_->setGridResolutionHeuristic(resolution);
    }
    
    void set_max_distance_heuristic(double max_distance_meters) {
        cpp_planner_->setMaxDistanceHeuristic(max_distance_meters);
    }
    
    // Heuristic parameter getters
    double get_grid_resolution_heuristic() const {
        return cpp_planner_->getGridResolutionHeuristic();
    }
    
    double get_max_distance_heuristic() const {
        return cpp_planner_->getMaxDistanceHeuristic();
    }
    
    
private:
    gco::WorldPtr cpp_world_;
    std::unique_ptr<gco::CTSWAPPlanner> cpp_planner_;
};

gco::Configuration2 convert_python_config_to_cpp(py::dict config_dict) {
    double x = config_dict["x"].cast<double>();
    double y = config_dict["y"].cast<double>();
    double theta = config_dict["theta"].cast<double>();
    return gco::Configuration2(x, y, theta);
}

class ObjectCentricAStarWrapper {
public:
    ObjectCentricAStarWrapper(py::object world_obj, double goal_tolerance = 0.05, double weight = 1.0) {
        cpp_world_ = convert_python_world_to_cpp(world_obj);
        cpp_planner_ = std::make_unique<gco::ObjectCentricAStarPlanner>(cpp_world_, goal_tolerance, weight);
    }
    
    py::dict plan(py::dict starts_dict, py::dict start_objects_dict, py::dict goal_objects_dict, double goal_tolerance = 0.05) {
        // Convert Python dictionaries to C++ maps
        std::map<std::string, gco::Configuration2> starts;
        for (auto item : starts_dict) {
            std::string name = item.first.cast<std::string>();
            py::dict config_dict = item.second.cast<py::dict>();
            starts[name] = convert_python_config_to_cpp(config_dict);
        }
        
        std::map<std::string, gco::Configuration2> start_objects;
        for (auto item : start_objects_dict) {
            std::string name = item.first.cast<std::string>();
            py::dict config_dict = item.second.cast<py::dict>();
            start_objects[name] = convert_python_config_to_cpp(config_dict);
        }
        
        std::map<std::string, gco::Configuration2> goal_objects;
        for (auto item : goal_objects_dict) {
            std::string name = item.first.cast<std::string>();
            py::dict config_dict = item.second.cast<py::dict>();
            goal_objects[name] = convert_python_config_to_cpp(config_dict);
        }
        
        // Call the C++ planner
        gco::MultiRobotPaths paths = cpp_planner_->plan(starts, start_objects, goal_objects, goal_tolerance);
        
        // Convert back to Python
        return convert_cpp_paths_to_python(paths);
    }
    
    void save_planner_output(py::list path_list, py::dict start_config, py::dict goal_config, 
                           const std::string& filename, const std::string& object_type, double object_size) {
        // Convert path list to vector of Configuration2
        std::vector<gco::Configuration2> path;
        for (auto item : path_list) {
            py::dict config_dict = item.cast<py::dict>();
            path.push_back(convert_python_config_to_cpp(config_dict));
        }
        
        // Convert start and goal configs
        gco::Configuration2 start_cfg = convert_python_config_to_cpp(start_config);
        gco::Configuration2 goal_cfg = convert_python_config_to_cpp(goal_config);
        
        // Call the C++ method
        cpp_planner_->savePlannerOutput(path, start_cfg, goal_cfg, filename, object_type, object_size);
    }
    
    void save_planner_output_rectangle(py::list path_list, py::dict start_config, py::dict goal_config, 
                                     const std::string& filename, const std::string& object_type, 
                                     double object_width, double object_height) {
        // Convert path list to vector of Configuration2
        std::vector<gco::Configuration2> path;
        for (auto item : path_list) {
            py::dict config_dict = item.cast<py::dict>();
            path.push_back(convert_python_config_to_cpp(config_dict));
        }
        
        // Convert start and goal configs
        gco::Configuration2 start_cfg = convert_python_config_to_cpp(start_config);
        gco::Configuration2 goal_cfg = convert_python_config_to_cpp(goal_config);
        
        // Call the C++ method
        cpp_planner_->savePlannerOutput(path, start_cfg, goal_cfg, filename, object_type, object_width, object_height);
    }
    
    void set_verbose(bool verbose) {
        // TODO: Add verbose setting if needed
    }
    
    void set_grid_resolution(double resolution) {
        // TODO: Add grid resolution setting if needed
    }
    
    void set_max_iterations(int max_iterations) {
        // TODO: Add max iterations setting if needed
    }
    
    double get_weight() const {
        // TODO: Add getter for weight if needed
        return 1.0;
    }
    
    double get_goal_tolerance() const {
        // TODO: Add getter for goal tolerance if needed
        return 0.05;
    }

private:
    gco::WorldPtr cpp_world_;
    std::unique_ptr<gco::ObjectCentricAStarPlanner> cpp_planner_;
};

class HybridCTSWAPPIBTWrapper {
public:
    HybridCTSWAPPIBTWrapper(py::object world_obj, double goal_tolerance = 0.05, py::dict heuristic_params = py::dict(), unsigned int seed = 42) {
        cpp_world_ = convert_python_world_to_cpp(world_obj);
        
        // Convert Python dict to C++ map
        std::map<std::string, std::string> cpp_heuristic_params;
        for (auto item : heuristic_params) {
            std::string key = item.first.cast<std::string>();
            std::string value = item.second.cast<std::string>();
            cpp_heuristic_params[key] = value;
        }
        
        cpp_planner_ = std::make_unique<gco::GSPIPlanner>(cpp_world_, goal_tolerance, cpp_heuristic_params, seed, 60.0, true);
    }

    py::dict plan(py::dict starts_dict, py::dict goals_dict) {
        // Convert Python dictionaries to C++ maps
        std::map<std::string, gco::Configuration2> cfgs_start;
        std::map<std::string, gco::Configuration2> cfgs_goal;
        
        for (auto item : starts_dict) {
            std::string robot_name = item.first.cast<std::string>();
            py::list config_list = item.second.cast<py::list>();
            cfgs_start[robot_name] = gco::Configuration2(
                config_list[0].cast<double>(),
                config_list[1].cast<double>(),
                config_list[2].cast<double>()
            );
        }
        
        for (auto item : goals_dict) {
            std::string robot_name = item.first.cast<std::string>();
            py::list config_list = item.second.cast<py::list>();
            cfgs_goal[robot_name] = gco::Configuration2(
                config_list[0].cast<double>(),
                config_list[1].cast<double>(),
                config_list[2].cast<double>()
            );
        }
        
        // Run the hybrid planner and get statistics
        auto stats = cpp_planner_->plan(cfgs_start, cfgs_goal);
        
        // Extract paths from stats and convert back to Python
        return convert_cpp_paths_to_python(stats.paths);
    }

    void set_verbose(bool verbose) {
        cpp_planner_->setVerbose(verbose);
    }

    void set_max_iterations(int max_iterations) {
        cpp_planner_->setMaxIterations(max_iterations);
    }

    int get_max_iterations() const {
        return cpp_planner_->getMaxIterations();
    }

    // Heuristic parameter setters
    void set_grid_resolution_heuristic(double resolution) {
        cpp_planner_->setGridResolutionHeuristic(resolution);
    }
    
    void set_max_distance_heuristic(double max_distance_meters) {
        cpp_planner_->setMaxDistanceHeuristic(max_distance_meters);
    }
    
    // Heuristic parameter getters
    double get_grid_resolution_heuristic() const {
        return cpp_planner_->getGridResolutionHeuristic();
    }
    
    double get_max_distance_heuristic() const {
        return cpp_planner_->getMaxDistanceHeuristic();
    }

private:
    gco::WorldPtr cpp_world_;
    std::unique_ptr<gco::GSPIPlanner> cpp_planner_;
};

PYBIND11_MODULE(gcocpp, m) {
    m.doc() = "Python bindings for GCo C++ planners";

    py::class_<ObjectCentricAStarWrapper>(m, "ObjectCentricAStarPlanner")
        .def(py::init<py::object, double, double>(), 
             py::arg("world"), 
             py::arg("goal_tolerance") = 0.05,
             py::arg("weight") = 1.0,
             "Initialize ObjectCentricAStar planner with a Python world object")
        .def("plan", &ObjectCentricAStarWrapper::plan, 
             py::arg("starts"), 
             py::arg("start_objects"),
             py::arg("goal_objects"),
             py::arg("goal_tolerance") = 0.05,
             "Plan paths for robots and objects from start to goal configurations")
        .def("save_planner_output", &ObjectCentricAStarWrapper::save_planner_output,
             py::arg("path"), py::arg("start_config"), py::arg("goal_config"), 
             py::arg("filename"), py::arg("object_type"), py::arg("object_size"),
             "Save planner output to JSON file")
        .def("save_planner_output_rectangle", &ObjectCentricAStarWrapper::save_planner_output_rectangle,
             py::arg("path"), py::arg("start_config"), py::arg("goal_config"), 
             py::arg("filename"), py::arg("object_type"), py::arg("object_width"), py::arg("object_height"),
             "Save planner output to JSON file for rectangle objects")
        .def("set_verbose", &ObjectCentricAStarWrapper::set_verbose,
             py::arg("verbose"),
             "Set verbose mode for debug output")
        .def("set_grid_resolution", &ObjectCentricAStarWrapper::set_grid_resolution,
             py::arg("resolution"),
             "Set the grid resolution for planning")
        .def("set_max_iterations", &ObjectCentricAStarWrapper::set_max_iterations,
             py::arg("max_iterations"),
             "Set the maximum number of iterations for planning")
        .def("get_weight", &ObjectCentricAStarWrapper::get_weight,
             "Get the current weight for weighted A*")
        .def("get_goal_tolerance", &ObjectCentricAStarWrapper::get_goal_tolerance,
             "Get the current goal tolerance");
    
    py::class_<HybridCTSWAPPIBTWrapper>(m, "GSPIPlanner")
        .def(py::init<py::object, double, py::dict, unsigned int>(), 
             py::arg("world"), 
             py::arg("goal_tolerance") = 0.05,
             py::arg("heuristic_params") = py::dict(),
             py::arg("seed") = 42,
             "Initialize Hybrid CTSWAP-PIBT planner with a Python world object and heuristic parameters")
        .def("plan", &HybridCTSWAPPIBTWrapper::plan, 
             py::arg("starts"), 
             py::arg("goals"),
             "Plan paths for robots from start to goal configurations using hybrid CTSWAP-PIBT algorithm")
        .def("set_verbose", &HybridCTSWAPPIBTWrapper::set_verbose,
             py::arg("verbose"),
             "Set verbose mode for debug output")
        .def("set_max_iterations", &HybridCTSWAPPIBTWrapper::set_max_iterations,
             py::arg("max_iterations"),
             "Set the maximum number of iterations for planning")
        .def("get_max_iterations", &HybridCTSWAPPIBTWrapper::get_max_iterations,
             "Get the current maximum number of iterations for planning")
        .def("set_grid_resolution_heuristic", &HybridCTSWAPPIBTWrapper::set_grid_resolution_heuristic,
             py::arg("resolution"),
             "Set the grid resolution for heuristic computation")
        .def("set_max_distance_heuristic", &HybridCTSWAPPIBTWrapper::set_max_distance_heuristic,
             py::arg("max_distance_meters"),
             "Set the maximum distance for heuristic computation")
        .def("get_grid_resolution_heuristic", &HybridCTSWAPPIBTWrapper::get_grid_resolution_heuristic,
             "Get the current grid resolution for heuristic computation")
        .def("get_max_distance_heuristic", &HybridCTSWAPPIBTWrapper::get_max_distance_heuristic,
             "Get the current maximum distance for heuristic computation");

class ObjectGSPIWrapper {
public:
    ObjectGSPIWrapper(py::object world_obj, double goal_tolerance = 0.05, py::dict heuristic_params = py::dict(), unsigned int seed = 42) {
        cpp_world_ = convert_python_world_to_cpp(world_obj);
        
        // Convert Python dict to C++ map
        std::map<std::string, std::string> cpp_heuristic_params;
        for (auto item : heuristic_params) {
            std::string key = item.first.cast<std::string>();
            std::string value = item.second.cast<std::string>();
            cpp_heuristic_params[key] = value;
        }
        
        cpp_planner_ = std::make_unique<gco::ObjectGSPIPlanner>(cpp_world_, goal_tolerance, cpp_heuristic_params, seed, 60.0, true);
    }
    
    void initialize_object_targets(py::dict object_targets_dict) {
        // Convert Python dictionary to C++ map
        std::map<std::string, gco::Configuration2> object_targets;
        
        for (auto item : object_targets_dict) {
            std::string object_name = item.first.cast<std::string>();
            py::list config_list = item.second.cast<py::list>();
            object_targets[object_name] = gco::Configuration2(
                config_list[0].cast<double>(),
                config_list[1].cast<double>(),
                config_list[2].cast<double>()
            );
        }
        
        cpp_planner_->initializeObjectTargets(object_targets);
    }
    
    py::dict get_next_object_moves(py::dict current_positions_dict, int horizon = 3) {
        // Convert Python dictionary to C++ map
        std::map<std::string, gco::Configuration2> current_positions;
        
        for (auto item : current_positions_dict) {
            std::string object_name = item.first.cast<std::string>();
            py::list config_list = item.second.cast<py::list>();
            current_positions[object_name] = gco::Configuration2(
                config_list[0].cast<double>(),
                config_list[1].cast<double>(),
                config_list[2].cast<double>()
            );
        }
        
        // Get next moves from C++ planner
        auto object_moves = cpp_planner_->getNextObjectMoves(current_positions, horizon);
        
        // Convert back to Python dictionary
        py::dict result;
        for (const auto& [object_name, moves] : object_moves) {
            py::list moves_list;
            for (const auto& move : moves) {
                moves_list.append(py::make_tuple(move.x, move.y, move.theta));
            }
            result[object_name.c_str()] = moves_list;
        }
        
        return result;
    }
    
    bool are_all_objects_at_goal(py::dict current_positions_dict) {
        // Convert Python dictionary to C++ map
        std::map<std::string, gco::Configuration2> current_positions;
        
        for (auto item : current_positions_dict) {
            std::string object_name = item.first.cast<std::string>();
            py::list config_list = item.second.cast<py::list>();
            current_positions[object_name] = gco::Configuration2(
                config_list[0].cast<double>(),
                config_list[1].cast<double>(),
                config_list[2].cast<double>()
            );
        }
        
        return cpp_planner_->areAllObjectsAtGoal(current_positions);
    }
    
    void set_grid_resolution_heuristic(double resolution) {
        cpp_planner_->setGridResolutionHeuristic(resolution);
    }
    
    void set_max_distance_heuristic(double max_distance_meters) {
        cpp_planner_->setMaxDistanceHeuristic(max_distance_meters);
    }
    
    double get_grid_resolution_heuristic() const {
        return cpp_planner_->getGridResolutionHeuristic();
    }
    
    double get_max_distance_heuristic() const {
        return cpp_planner_->getMaxDistanceHeuristic();
    }
    
    py::dict get_current_assignments() const {
        // Get current assignments from C++ planner
        auto assignments = cpp_planner_->getCurrentAssignments();
        
        // Convert to Python dictionary
        py::dict result;
        for (const auto& [object_name, config] : assignments) {
            result[object_name.c_str()] = py::make_tuple(config.x, config.y, config.theta);
        }
        
        return result;
    }
    
    void set_verbose(bool verbose) {
        cpp_planner_->setVerbose(verbose);
    }
    
    void reset_planner_state() {
        cpp_planner_->resetPlannerState();
    }

private:
    gco::WorldPtr cpp_world_;
    std::unique_ptr<gco::ObjectGSPIPlanner> cpp_planner_;
};

py::class_<ObjectGSPIWrapper>(m, "ObjectGSPIPlanner")
    .def(py::init<py::object, double, py::dict, unsigned int>(), 
         py::arg("world"), 
         py::arg("goal_tolerance") = 0.05,
         py::arg("heuristic_params") = py::dict(),
         py::arg("seed") = 42,
         "Initialize Object GSPI planner with a Python world object and heuristic parameters")
    .def("initialize_object_targets", &ObjectGSPIWrapper::initialize_object_targets,
         py::arg("object_targets"),
         "Initialize the planner with object target configurations")
    .def("get_next_object_moves", &ObjectGSPIWrapper::get_next_object_moves,
         py::arg("current_positions"), py::arg("horizon") = 3,
         "Get next moves for objects from current positions")
    .def("are_all_objects_at_goal", &ObjectGSPIWrapper::are_all_objects_at_goal,
         py::arg("current_positions"),
         "Check if all objects are at their goals")
    .def("set_grid_resolution_heuristic", &ObjectGSPIWrapper::set_grid_resolution_heuristic,
         py::arg("resolution"),
         "Set the grid resolution for heuristic computation")
    .def("set_max_distance_heuristic", &ObjectGSPIWrapper::set_max_distance_heuristic,
         py::arg("max_distance_meters"),
         "Set the maximum distance for heuristic computation")
    .def("get_grid_resolution_heuristic", &ObjectGSPIWrapper::get_grid_resolution_heuristic,
         "Get the current grid resolution for heuristic computation")
    .def("get_max_distance_heuristic", &ObjectGSPIWrapper::get_max_distance_heuristic,
         "Get the current maximum distance for heuristic computation")
    .def("get_current_assignments", &ObjectGSPIWrapper::get_current_assignments,
         "Get current object-to-goal assignments")
    .def("set_verbose", &ObjectGSPIWrapper::set_verbose,
         py::arg("verbose"),
         "Set verbose mode for debug output")
    .def("reset_planner_state", &ObjectGSPIWrapper::reset_planner_state,
         "Reset the planner state for new scenarios");
} 