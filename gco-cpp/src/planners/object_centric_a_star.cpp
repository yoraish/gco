// General imports.
#include "gco/types.hpp"
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>

// Project imports.
#include <gco/planners/object_centric_a_star.hpp>
#include <gco/obstacles/obstacles.hpp>

gco::ObjectCentricAStarPlanner::ObjectCentricAStarPlanner(const gco::WorldPtr& world, const double goal_tolerance, const double weight) 
    : world_(world), goal_tolerance_(goal_tolerance), weight_(weight) {
    if (weight_ < 1.0) {
        throw std::invalid_argument("Weight must be >= 1.0 for weighted A*");
    }
}

gco::ObjectCentricAStarPlanner::~ObjectCentricAStarPlanner() {
}

void gco::ObjectCentricAStarPlanner::checkStartAndGoalConfigurationsOrDie(
    const std::map<std::string, Configuration2>& cfgs_start_robots, 
    const std::map<std::string, Configuration2>& cfg_start_objects,
    const std::map<std::string, Configuration2>& cfg_goal_objects) const {

    // Check that the start and goal objects are the same.
    if (cfg_start_objects.begin()->first != cfg_goal_objects.begin()->first) {
        throw std::invalid_argument("Not Implemented: The start and goal objects must have the same name");
    }

    // Check that the start and goals are valid.
    auto cfg_start_multi = std::make_shared<MultiObjectMultiRobotConfiguration>(cfgs_start_robots, cfg_start_objects);
    auto cfg_goal_multi = std::make_shared<MultiObjectMultiRobotConfiguration>(std::map<std::string, Configuration2>(), cfg_goal_objects);

    if (!isConfigurationValid(cfg_start_multi, true)) {
        throw std::invalid_argument("The start configuration must be valid");
    }
    if (!isConfigurationValid(cfg_goal_multi, true)) {
        throw std::invalid_argument("The goal configuration must be valid");
    }
}

gco::MultiRobotPaths gco::ObjectCentricAStarPlanner::plan(const std::map<std::string, Configuration2>& cfgs_start_robots, 
    const std::map<std::string, Configuration2>& cfg_start_objects,
    const std::map<std::string, Configuration2>& cfg_goal_objects,
    const double goal_tolerance) {

    // For now: check that there is only one object.
    if (cfg_start_objects.size() != 1 || cfg_goal_objects.size() != 1) {
        throw std::invalid_argument("Not Implemented: There must be exactly one object in the start and goal configurations");
    }

    // Check that all objects are in the world.
    for (const auto& [obj_name, obj_cfg] : cfg_start_objects) {
        if (world_->getObject(obj_name) == nullptr) {
            throw std::invalid_argument("Object " + obj_name + " is not in the world. Start objects: " + cfg_start_objects.begin()->first);
        }
    }
    for (const auto& [obj_name, obj_cfg] : cfg_goal_objects) {
        if (world_->getObject(obj_name) == nullptr) {
            throw std::invalid_argument("Object " + obj_name + " is not in the world. Goal objects: " + cfg_goal_objects.begin()->first);
        }
    }

    auto time_start = std::chrono::high_resolution_clock::now(); 
    
    // Clear previous search data.
    search_states_.clear();
    closed_set_.clear();
    
    // Clear the priority queue by creating a new one.
    open_list_ = std::priority_queue<SearchStatePtr, std::vector<SearchStatePtr>, CompareSearchState>();
    
    // Get the object name (we know there's exactly one object).
    std::string object_name = cfg_start_objects.begin()->first;  // TODO(yoraish): remove this and create a single object planner function.
    
    // Create ObjectRobots for each object and add them to the world for collision checking.
    std::vector<RobotPtr> added_object_robots;
    for (const auto& [obj_name, obj_cfg] : cfg_start_objects) {
        // Create ObjectRobot based on object type.
        // Get the object from the world.
        auto object = world_->getObject(obj_name);
        gco::JointRanges object_joint_ranges({-10.0, -10.0, -M_PI}, {10.0, 10.0, M_PI}, {0.01, 0.01, 0.01});
        
        // Create ObjectRobot based on the actual object type and parameters
        RobotPtr object_robot;
        if (auto circle_obj = std::dynamic_pointer_cast<gco::CircleShape>(object->getShape())) {
            object_robot = gco::ObjectRobot::createCircle(obj_name, object_joint_ranges, circle_obj->getRadius() + inflation_radius_);
        } else if (auto square_obj = std::dynamic_pointer_cast<gco::SquareShape>(object->getShape())) {
            object_robot = gco::ObjectRobot::createSquare(obj_name, object_joint_ranges, square_obj->getWidth() + 2*inflation_radius_);
        } else if (auto rect_obj = std::dynamic_pointer_cast<gco::RectangleShape>(object->getShape())) {
            object_robot = gco::ObjectRobot::createRectangle(obj_name, object_joint_ranges, rect_obj->getWidth() + 2*inflation_radius_, rect_obj->getHeight() + 2*inflation_radius_);
        } else if (auto polygon_obj = std::dynamic_pointer_cast<gco::PolygonShape>(object->getShape())) {
            // Inflate the polygon by adding a buffer to the vertices in the direction away from the center (0,0).
            std::vector<gco::Translation2> inflated_vertices = polygon_obj->getVertices();
            double center_x = 0.0;
            double center_y = 0.0;
            for (const auto& vertex : inflated_vertices) {
                center_x += vertex.x;
                center_y += vertex.y;
            }
            center_x /= inflated_vertices.size();
            center_y /= inflated_vertices.size();
            
            for (auto& vertex : inflated_vertices) {
                double x_direction = vertex.x - center_x;
                double y_direction = vertex.y - center_y;
                double distance = std::sqrt(x_direction * x_direction + y_direction * y_direction);
                double delta_x = inflation_radius_ * x_direction / distance;
                double delta_y = inflation_radius_ * y_direction / distance;
                vertex.x += delta_x;
                vertex.y += delta_y;
            }

            object_robot = gco::ObjectRobot::createPolygon(obj_name, object_joint_ranges, inflated_vertices);
        }
        else {
            throw std::runtime_error("Unknown object type for object: " + obj_name);
        }
        
        world_->addRobot(object_robot);
        added_object_robots.push_back(object_robot);
    }

    // checkStartAndGoalConfigurationsOrDie(cfgs_start_robots, cfg_start_objects, cfg_goal_objects);
    
    // Create goal configuration. This is for the object only, so it is partially defined.
    cfg_goal_multi_ = std::make_shared<MultiObjectMultiRobotConfiguration>(std::map<std::string, Configuration2>(), cfg_goal_objects);
    
    // Initialize the start state.
    SearchStatePtr start_search_state = std::make_shared<SearchState>();
    start_search_state->cfg_multi_object_multi_robot = std::make_shared<MultiObjectMultiRobotConfiguration>(cfgs_start_robots, cfg_start_objects);
    // start_search_state->edge_from_parent = nullptr;
    start_search_state->parent_search_state = nullptr;
    start_search_state->g = 0.0;
    start_search_state->h = getHeuristic(start_search_state->cfg_multi_object_multi_robot, cfg_goal_multi_, object_name);
    start_search_state->f = start_search_state->g + weight_ * start_search_state->h;
    start_search_state->is_open = true;
    start_search_state->is_closed = false;

    // Add start state to search states and open list.
    search_states_[start_search_state->cfg_multi_object_multi_robot] = start_search_state;
    open_list_.push(start_search_state);
    
    SearchStatePtr goal_search_state = nullptr;
    int iteration_count = 0;
    
    while (!open_list_.empty() && iteration_count < max_iterations_) {
        // Get the state with lowest f-value.
        SearchStatePtr current_state = open_list_.top();
        open_list_.pop();
        
        // Check if we've reached the goal.
        if (isObjectAtGoal(current_state->cfg_multi_object_multi_robot, cfg_goal_multi_, object_name, goal_tolerance)) {
            goal_search_state = current_state;
            break;
        }
        
        // Skip if already closed.
        if (current_state->is_closed) {
            continue;
        }
        
        // Mark as closed.
        current_state->is_closed = true;
        closed_set_.insert(current_state->cfg_multi_object_multi_robot);
        
        // Get successor configurations.
        std::vector<MultiObjectMultiRobotConfigurationPtr> successors = getSuccessorConfigurations(current_state->cfg_multi_object_multi_robot);
        
        for (const auto& successor_cfg : successors) {
            // Skip if already in closed set.
            if (closed_set_.find(successor_cfg) != closed_set_.end()) {
                continue;
            }
            
            // Calculate cost to reach this successor.
            double edge_cost = configurationDistance(current_state->cfg_multi_object_multi_robot->objects.at(object_name), 
                                                   successor_cfg->objects.at(object_name));
            double tentative_g = current_state->g + edge_cost;
            
            // Check if we've seen this state before.
            auto it = search_states_.find(successor_cfg);
            if (it == search_states_.end()) {
                // New state.
                SearchStatePtr new_state = std::make_shared<SearchState>();
                new_state->cfg_multi_object_multi_robot = successor_cfg;
                // new_state->edge_from_parent = nullptr; // TODO: Add edge information if needed
                new_state->parent_search_state = current_state;
                new_state->g = tentative_g;
                new_state->h = getHeuristic(successor_cfg, cfg_goal_multi_, object_name);
                new_state->f = new_state->g + weight_ * new_state->h;
                new_state->is_open = true;
                new_state->is_closed = false;
                
                search_states_[successor_cfg] = new_state;
                open_list_.push(new_state);
            } else {
                // Existing state - check if we found a better path.
                SearchStatePtr existing_state = it->second;
                if (tentative_g < existing_state->g) {
                    // Update with better path.
                    existing_state->parent_search_state = current_state;
                    existing_state->g = tentative_g;
                    existing_state->f = tentative_g + weight_ * existing_state->h;
                    
                    // Re-add to open list if it was closed.
                    if (existing_state->is_closed) {
                        existing_state->is_closed = false;
                        closed_set_.erase(successor_cfg);
                        open_list_.push(existing_state); 
                    }
                }
            }
        }
        
        iteration_count++;
    }
    
    auto time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = time_end - time_start;
    
    // Clean up: Remove ObjectRobots from the world
    for (const auto& object_robot : added_object_robots) {
        // Note: We can't easily remove robots from the world, so we'll leave them
        // The world will be reset for the next planning call anyway
    }
    
    if (goal_search_state != nullptr) {
        return reconstructObjectPath(goal_search_state);
    } else {
        return MultiRobotPaths();
    }
}

gco::MultiRobotPaths gco::ObjectCentricAStarPlanner::reconstructObjectPath(const SearchStatePtr& goal_search_state) const {
    MultiRobotPaths paths;
    SearchStatePtr current_state = goal_search_state;
    
    // Get the object name (we know there's exactly one object)
    std::string object_name = current_state->cfg_multi_object_multi_robot->objects.begin()->first;
    
    // Initialize the object path
    paths[object_name] = std::vector<Configuration2>();
    
    // Collect configurations in reverse order
    while (current_state != nullptr) {
        paths[object_name].push_back(current_state->cfg_multi_object_multi_robot->objects.at(object_name));
        current_state = current_state->parent_search_state;
    }
    
    // Reverse to get chronological order
    std::reverse(paths[object_name].begin(), paths[object_name].end());

    // Add the goal configuration to the path.
    paths[object_name].push_back(cfg_goal_multi_->objects.at(object_name));
    
    return paths;
}

gco::MultiRobotPaths gco::ObjectCentricAStarPlanner::reconstructPath(const SearchStatePtr& goal_search_state) const {
    // TODO: Implement for multi-robot planning
    std::cout << RED << "Not implemented: reconstructPath for multi-robot planning" << RESET << std::endl;
    return MultiRobotPaths();
}

gco::ObjectCentricAStarPlanner::SearchStatePtr gco::ObjectCentricAStarPlanner::getSearchStateFromConfiguration(const MultiObjectMultiRobotConfigurationPtr& cfg) const {
    auto it = search_states_.find(cfg);
    if (it != search_states_.end()) {
        return it->second;
    }
    return nullptr;
}

gco::MultiObjectMultiRobotConfigurationPtr gco::ObjectCentricAStarPlanner::roundToGrid(const MultiObjectMultiRobotConfigurationPtr& cfg) const {
    auto rounded_cfg = std::make_shared<MultiObjectMultiRobotConfiguration>();
    
    // Round robot configurations
    for (const auto& [name, config] : cfg->robots) {
        Configuration2 rounded_config;
        rounded_config.x = std::round(config.x / grid_resolution_) * grid_resolution_;
        rounded_config.y = std::round(config.y / grid_resolution_) * grid_resolution_;
        rounded_config.theta = std::round(config.theta / grid_resolution_) * grid_resolution_;
        rounded_cfg->robots.insert({name, rounded_config});
    }
    
    // Round object configurations
    for (const auto& [name, config] : cfg->objects) {
        Configuration2 rounded_config;
        rounded_config.x = std::round(config.x / grid_resolution_) * grid_resolution_;
        rounded_config.y = std::round(config.y / grid_resolution_) * grid_resolution_;
        rounded_config.theta = std::round(config.theta / grid_resolution_) * grid_resolution_;
        rounded_cfg->objects.insert({name, rounded_config});
    }
    
    return rounded_cfg;
}

double gco::ObjectCentricAStarPlanner::getHeuristic(const MultiObjectMultiRobotConfigurationPtr& cfg_current, const MultiObjectMultiRobotConfigurationPtr& cfg_goal, const std::string& object_name) const {
    // Use Euclidean distance as heuristic for the specified object
    return configurationDistance(cfg_current->objects.at(object_name), cfg_goal->objects.at(object_name));
}

std::vector<gco::MultiObjectMultiRobotConfigurationPtr> gco::ObjectCentricAStarPlanner::getSuccessorConfigurations(const MultiObjectMultiRobotConfigurationPtr& cfg_current) const {
    std::vector<MultiObjectMultiRobotConfigurationPtr> successors;
    
    // Get the current object configuration (we know there's exactly one object)
    const Configuration2& cfg_current_object = cfg_current->objects.begin()->second;
     
    // Define motion primitives (8-connected grid with rotation)
    std::vector<std::pair<double, double>> translations = {
        {0.0, 0.0},      // Stay in place
        {grid_resolution_, 0.0},      // Right
        {-grid_resolution_, 0.0},     // Left
        {0.0, grid_resolution_},      // Up
        {0.0, -grid_resolution_},     // Down
        {grid_resolution_, grid_resolution_},      // Up-right
        {grid_resolution_, -grid_resolution_},     // Down-right
        {-grid_resolution_, grid_resolution_},     // Up-left
        {-grid_resolution_, -grid_resolution_}     // Down-left
    };
    
    std::vector<double> rotations = {
        0.0,                    // No rotation
        grid_resolution_,       // Clockwise
        -grid_resolution_       // Counter-clockwise
    };
    
    // Generate successor configurations
    for (const auto& translation : translations) {
        for (const auto& rotation : rotations) {
            // Do not allow both stay in place and no rotation.
            if (translation.first == 0.0 && translation.second == 0.0 && rotation == 0.0) {
                continue;
            }

            Configuration2 successor_cfg;
            successor_cfg.x = cfg_current_object.x + translation.first;
            successor_cfg.y = cfg_current_object.y + translation.second;
            successor_cfg.theta = cfg_current_object.theta + rotation;
            
            // Normalize theta to [-π, π]
            while (successor_cfg.theta > M_PI) {
                successor_cfg.theta -= 2 * M_PI;
            }
            while (successor_cfg.theta < -M_PI) {
                successor_cfg.theta += 2 * M_PI;
            }
            
            // Create multi-object multi-robot configuration for successor
            auto successor_multi_cfg = std::make_shared<MultiObjectMultiRobotConfiguration>();
            successor_multi_cfg->robots = cfg_current->robots;  // Keep robots in same position
            successor_multi_cfg->objects = cfg_current->objects;  // Copy all objects
            successor_multi_cfg->objects.begin()->second = successor_cfg;  // Update the object position

            // Round to grid. This is here in preparation for more flexible motion primitives.
            successor_multi_cfg = roundToGrid(successor_multi_cfg);
            
            // Check if the configuration is valid
            if (isConfigurationValid(successor_multi_cfg)) {
                successors.push_back(successor_multi_cfg);
            }
        }
    }

    if (successors.size() == 0) {
        std::cout << "No successors found!" << std::endl;
    }

  
    return successors;
}

bool gco::ObjectCentricAStarPlanner::isConfigurationValid(const MultiObjectMultiRobotConfigurationPtr& cfg, bool verbose) const {
    // Check if configuration is within reasonable bounds
    const double max_x = 10.0;
    const double max_y = 10.0;
    
    // Check object bounds
    for (const auto& [object_name, object_cfg] : cfg->objects) {
        if (std::abs(object_cfg.x) > max_x || std::abs(object_cfg.y) > max_y) {
            if (verbose) {
                std::cout << "Object " << object_name << " is out of bounds: (" << object_cfg.x << ", " << object_cfg.y << ")" << std::endl;
            }
            return false;
        }
    }
    
    // Create a combined configuration map for collision checking
    std::map<std::string, Configuration2> combined_cfg;
    
    // // Do not add robot configurations as we only care for the object here. If/when motion primitives would include robots, we should add them back.
    // for (const auto& [robot_name, robot_cfg] : cfg->robots) {
    //     combined_cfg[robot_name] = robot_cfg;
    // }
    
    // Add object configurations
    for (const auto& [object_name, object_cfg] : cfg->objects) {
        combined_cfg[object_name] = object_cfg;
    }
    
    // Use world's collision checking (objects will be treated as robots for collision checking)
    CollisionResult collision_result;
    world_->checkCollision(combined_cfg, collision_result);
    if (verbose && !collision_result.collisions.empty()) {
        std::cout << "Collision result: " << std::endl;
        for (const auto& [name, collisions] : collision_result.collisions) {
            std::cout << "  Collision " << name << " between " << std::endl;
            for (const auto& collision : collisions) {
                std::cout << "    " << collision.name_entity1 << " and " << collision.name_entity2 << std::endl;
            }
        }
    }
    return collision_result.collisions.empty();
}

bool gco::ObjectCentricAStarPlanner::isObjectAtGoal(const MultiObjectMultiRobotConfigurationPtr& cfg_current, const MultiObjectMultiRobotConfigurationPtr& cfg_goal, const std::string& object_name, const double tolerance) const {
    return configurationDistance(cfg_current->objects.at(object_name), cfg_goal->objects.at(object_name)) <= tolerance;
}

void gco::ObjectCentricAStarPlanner::savePlannerOutput(const std::vector<Configuration2>& path,
                                                      const Configuration2& cfg_start_object,
                                                      const Configuration2& cfg_goal_object,
                                                      const std::string& filename,
                                                      const std::string& object_type,
                                                      const double object_size) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    // Write JSON format
    file << "{" << std::endl;
    file << "  \"start_configuration\": {" << std::endl;
    file << "    \"x\": " << std::fixed << std::setprecision(6) << cfg_start_object.x << "," << std::endl;
    file << "    \"y\": " << std::fixed << std::setprecision(6) << cfg_start_object.y << "," << std::endl;
    file << "    \"theta\": " << std::fixed << std::setprecision(6) << cfg_start_object.theta << std::endl;
    file << "  }," << std::endl;
    file << "  \"goal_configuration\": {" << std::endl;
    file << "    \"x\": " << std::fixed << std::setprecision(6) << cfg_goal_object.x << "," << std::endl;
    file << "    \"y\": " << std::fixed << std::setprecision(6) << cfg_goal_object.y << "," << std::endl;
    file << "    \"theta\": " << std::fixed << std::setprecision(6) << cfg_goal_object.theta << std::endl;
    file << "  }," << std::endl;
    file << "  \"object_info\": {" << std::endl;
    file << "    \"type\": \"" << object_type << "\"," << std::endl;
    file << "    \"size\": " << std::fixed << std::setprecision(6) << object_size << std::endl;
    file << "  }," << std::endl;
    
    // Add obstacle information
    file << "  \"obstacle_positions\": {" << std::endl;
    const auto& obstacles = world_->getObstacles();
    bool first_obstacle = true;
    for (const auto& [obstacle_name, obstacle_with_pose] : obstacles) {
        if (!first_obstacle) {
            file << "," << std::endl;
        }
        file << "    \"" << obstacle_name << "\": [" << std::fixed << std::setprecision(6) 
             << obstacle_with_pose.second.x << ", " << obstacle_with_pose.second.y << "]";
        first_obstacle = false;
    }
    file << std::endl << "  }," << std::endl;
    
    file << "  \"obstacle_radii\": {" << std::endl;
    first_obstacle = true;
    for (const auto& [obstacle_name, obstacle_with_pose] : obstacles) {
        if (!first_obstacle) {
            file << "," << std::endl;
        }
        const auto& obstacle = obstacle_with_pose.first;
        
        // Determine obstacle type and size
        if (auto circle_obs = std::dynamic_pointer_cast<gco::CircleShape>(obstacle->getShape())) {
            file << "    \"" << obstacle_name << "\": " << std::fixed << std::setprecision(6) << circle_obs->getRadius();
        } else if (auto square_obs = std::dynamic_pointer_cast<gco::SquareShape>(obstacle->getShape())) {
            file << "    \"" << obstacle_name << "\": " << std::fixed << std::setprecision(6) << square_obs->getWidth();
        } else if (auto rect_obs = std::dynamic_pointer_cast<gco::RectangleShape>(obstacle->getShape())) {
            file << "    \"" << obstacle_name << "\": [" << std::fixed << std::setprecision(6) 
                 << rect_obs->getWidth() << ", " << rect_obs->getHeight() << "]";
        } else {
            // Default to circle with radius 0.1 for unknown types
            file << "    \"" << obstacle_name << "\": 0.1";
        }
        first_obstacle = false;
    }
    file << std::endl << "  }," << std::endl;
    
    file << "  \"path\": [" << std::endl;
    
    for (size_t i = 0; i < path.size(); ++i) {
        file << "    {" << std::endl;
        file << "      \"step\": " << i << "," << std::endl;
        file << "      \"x\": " << std::fixed << std::setprecision(6) << path[i].x << "," << std::endl;
        file << "      \"y\": " << std::fixed << std::setprecision(6) << path[i].y << "," << std::endl;
        file << "      \"theta\": " << std::fixed << std::setprecision(6) << path[i].theta << std::endl;
        file << "    }";
        if (i < path.size() - 1) {
            file << ",";
        }
        file << std::endl;
    }
    
    file << "  ]," << std::endl;
    file << "  \"path_length\": " << path.size() << "," << std::endl;
    file << "  \"total_distance\": " << std::fixed << std::setprecision(6);
    
    // Calculate total path distance
    double total_distance = 0.0;
    for (size_t i = 1; i < path.size(); ++i) {
        total_distance += configurationDistance(path[i-1], path[i]);
    }
    file << total_distance << "," << std::endl;
    
    file << "  \"planner_settings\": {" << std::endl;
    file << "    \"weight\": " << std::fixed << std::setprecision(2) << weight_ << "," << std::endl;
    file << "    \"grid_resolution\": " << std::fixed << std::setprecision(6) << grid_resolution_ << "," << std::endl;
    file << "    \"goal_tolerance\": " << std::fixed << std::setprecision(6) << goal_tolerance_ << std::endl;
    file << "  }" << std::endl;
    file << "}" << std::endl;
    
    file.close();
    std::cout << "Planner output saved to " << filename << std::endl;
}

void gco::ObjectCentricAStarPlanner::savePlannerOutput(const std::vector<Configuration2>& path,
                                                      const Configuration2& cfg_start_object,
                                                      const Configuration2& cfg_goal_object,
                                                      const std::string& filename,
                                                      const std::string& object_type,
                                                      const double object_width,
                                                      const double object_height) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    // Write JSON format
    file << "{" << std::endl;
    file << "  \"start_configuration\": {" << std::endl;
    file << "    \"x\": " << std::fixed << std::setprecision(6) << cfg_start_object.x << "," << std::endl;
    file << "    \"y\": " << std::fixed << std::setprecision(6) << cfg_start_object.y << "," << std::endl;
    file << "    \"theta\": " << std::fixed << std::setprecision(6) << cfg_start_object.theta << std::endl;
    file << "  }," << std::endl;
    file << "  \"goal_configuration\": {" << std::endl;
    file << "    \"x\": " << std::fixed << std::setprecision(6) << cfg_goal_object.x << "," << std::endl;
    file << "    \"y\": " << std::fixed << std::setprecision(6) << cfg_goal_object.y << "," << std::endl;
    file << "    \"theta\": " << std::fixed << std::setprecision(6) << cfg_goal_object.theta << std::endl;
    file << "  }," << std::endl;
    file << "  \"object_info\": {" << std::endl;
    file << "    \"type\": \"" << object_type << "\"," << std::endl;
    file << "    \"width\": " << std::fixed << std::setprecision(6) << object_width << "," << std::endl;
    file << "    \"height\": " << std::fixed << std::setprecision(6) << object_height << std::endl;
    file << "  }," << std::endl;
    
    // Add obstacle information
    file << "  \"obstacle_positions\": {" << std::endl;
    const auto& obstacles = world_->getObstacles();
    bool first_obstacle = true;
    for (const auto& [obstacle_name, obstacle_with_pose] : obstacles) {
        if (!first_obstacle) {
            file << "," << std::endl;
        }
        file << "    \"" << obstacle_name << "\": [" << std::fixed << std::setprecision(6) 
             << obstacle_with_pose.second.x << ", " << obstacle_with_pose.second.y << "]";
        first_obstacle = false;
    }
    file << std::endl << "  }," << std::endl;
    
    file << "  \"obstacle_radii\": {" << std::endl;
    first_obstacle = true;
    for (const auto& [obstacle_name, obstacle_with_pose] : obstacles) {
        if (!first_obstacle) {
            file << "," << std::endl;
        }
        const auto& obstacle = obstacle_with_pose.first;
        
        // Determine obstacle type and size
        if (auto circle_obs = std::dynamic_pointer_cast<gco::CircleShape>(obstacle->getShape())) {
            file << "    \"" << obstacle_name << "\": " << std::fixed << std::setprecision(6) << circle_obs->getRadius();
        } else if (auto square_obs = std::dynamic_pointer_cast<gco::SquareShape>(obstacle->getShape())) {
            file << "    \"" << obstacle_name << "\": " << std::fixed << std::setprecision(6) << square_obs->getWidth();
        } else if (auto rect_obs = std::dynamic_pointer_cast<gco::RectangleShape>(obstacle->getShape())) {
            file << "    \"" << obstacle_name << "\": [" << std::fixed << std::setprecision(6) 
                 << rect_obs->getWidth() << ", " << rect_obs->getHeight() << "]";
        } else {
            // Default to circle with radius 0.1 for unknown types
            file << "    \"" << obstacle_name << "\": 0.1";
        }
        first_obstacle = false;
    }
    file << std::endl << "  }," << std::endl;
    
    file << "  \"path\": [" << std::endl;
    
    for (size_t i = 0; i < path.size(); ++i) {
        file << "    {" << std::endl;
        file << "      \"step\": " << i << "," << std::endl;
        file << "      \"x\": " << std::fixed << std::setprecision(6) << path[i].x << "," << std::endl;
        file << "      \"y\": " << std::fixed << std::setprecision(6) << path[i].y << "," << std::endl;
        file << "      \"theta\": " << std::fixed << std::setprecision(6) << path[i].theta << std::endl;
        file << "    }";
        if (i < path.size() - 1) {
            file << ",";
        }
        file << std::endl;
    }
    
    file << "  ]," << std::endl;
    file << "  \"path_length\": " << path.size() << "," << std::endl;
    file << "  \"total_distance\": " << std::fixed << std::setprecision(6);
    
    // Calculate total path distance
    double total_distance = 0.0;
    for (size_t i = 1; i < path.size(); ++i) {
        total_distance += configurationDistance(path[i-1], path[i]);
    }
    file << total_distance << "," << std::endl;
    
    file << "  \"planner_settings\": {" << std::endl;
    file << "    \"weight\": " << std::fixed << std::setprecision(2) << weight_ << "," << std::endl;
    file << "    \"grid_resolution\": " << std::fixed << std::setprecision(6) << grid_resolution_ << "," << std::endl;
    file << "    \"goal_tolerance\": " << std::fixed << std::setprecision(6) << goal_tolerance_ << std::endl;
    file << "  }" << std::endl;
    file << "}" << std::endl;
    
    file.close();
    std::cout << "Planner output saved to " << filename << std::endl;
}