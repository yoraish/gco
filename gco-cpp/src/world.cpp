// General includes.
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>

// Project includes.
#include "gco/world/world.hpp"
#include "gco/collisions/collisions.hpp"
#include "gco/utils.hpp"

namespace gco {

// World constructor
World::World() {
    // Nothing to do.
}

// World destructor
World::~World() {
    // Nothing to do.
}

// Add a robot to the world
void World::addRobot(const RobotPtr& robot) {
    robots_[robot->getName()] = robot;
}

// Get a robot by name.
RobotPtr World::getRobot(const std::string& name) const {
    if (robots_.find(name) == robots_.end()) {
        std::cout << RED << "Robot " << name << " not found in world" << RESET << std::endl;
        return nullptr;
    }
    return robots_.at(name);
}

// Get an object by name.
ObjectPtr World::getObject(const std::string& name) const {
    if (objects_.find(name) == objects_.end()) {
        std::cout << RED << "Object " << name << " not found in world" << RESET << std::endl;
        return nullptr;
    }
    return objects_.at(name).first;
}

// Transform a point from robot local frame to world frame
std::pair<double, double> World::transformPointToWorld(const Transform2& x_robot_point, const Configuration2& x_world_robot) const {
    // Apply rotation first, then translation
    double cos_theta = std::cos(x_world_robot.theta);
    double sin_theta = std::sin(x_world_robot.theta);
    
    double world_x = x_robot_point.x * cos_theta - x_robot_point.y * sin_theta + x_world_robot.x;
    double world_y = x_robot_point.x * sin_theta + x_robot_point.y * cos_theta + x_world_robot.y;
    
    return {world_x, world_y};
}



// Check collision between two circles
bool World::checkCircleCollision(double x1, double y1, double r1, double x2, double y2, double r2) const {
    double dx = x2 - x1;
    double dy = y2 - y1;
    double distance_squared = dx * dx + dy * dy;
    double radius_sum = r1 + r2;
    return distance_squared < radius_sum * radius_sum;
}

// Check collision between a circle and a square
bool World::checkCircleSquareCollision(double circle_x, double circle_y, double circle_radius, 
                                      double square_x, double square_y, double square_width, double square_theta) const {
    // Create a polygon from the square.
    std::vector<gco::Translation2> square_vertices;
    square_vertices.push_back(gco::Translation2(- square_width/2, - square_width/2));
    square_vertices.push_back(gco::Translation2( square_width/2, - square_width/2));
    square_vertices.push_back(gco::Translation2( square_width/2,  square_width/2));
    square_vertices.push_back(gco::Translation2(- square_width/2,  square_width/2));
    
    // Check collision between the circle and the polygon.
    return checkPolygonCircleCollision(square_x, 
                                       square_y, 
                                       square_theta, 
                                       square_vertices, 
                                       circle_x, 
                                       circle_y, 
                                       circle_radius);
}

// Check collision between a polygon and a circle
bool World::checkPolygonCircleCollision(double polygon_x, double polygon_y, double polygon_theta, const std::vector<gco::Translation2>& polygon_vertices, 
                                       double circle_x, double circle_y, double circle_radius) const {
    // The circle and polygon vertices are already in the world frame.
    double local_circle_x = circle_x;
    double local_circle_y = circle_y;

    // First, check if the circle center is inside the polygon using ray casting algorithm
    bool inside_polygon = false;
    int j = polygon_vertices.size() - 1;
    for (int i = 0; i < polygon_vertices.size(); i++) {
        if (((polygon_vertices[i].y > local_circle_y) != (polygon_vertices[j].y > local_circle_y)) &&
            (local_circle_x < (polygon_vertices[j].x - polygon_vertices[i].x) * (local_circle_y - polygon_vertices[i].y) / 
                              (polygon_vertices[j].y - polygon_vertices[i].y) + polygon_vertices[i].x)) {
            inside_polygon = !inside_polygon;
        }
        j = i;
    }
    
    // If the circle center is inside the polygon, there's definitely a collision
    if (inside_polygon) {
        return true;
    }

    // If not inside, check if the circle intersects with the polygon boundary
    double min_distance_squared = std::numeric_limits<double>::max();
    
    for (size_t i = 0; i < polygon_vertices.size(); ++i) {
        const auto& v1 = polygon_vertices[i];
        const auto& v2 = polygon_vertices[(i + 1) % polygon_vertices.size()];
        
        // Check distance to line segment
        double dx = v2.x - v1.x;
        double dy = v2.y - v1.y;
        double length_squared = dx * dx + dy * dy;
        
        if (length_squared == 0) {
            // Point to point distance
            double dist_squared = (local_circle_x - v1.x) * (local_circle_x - v1.x) + 
                                 (local_circle_y - v1.y) * (local_circle_y - v1.y);
            min_distance_squared = std::min(min_distance_squared, dist_squared);
        } else {
            // Project circle center onto line segment
            double t = std::max(0.0, std::min(1.0, 
                ((local_circle_x - v1.x) * dx + (local_circle_y - v1.y) * dy) / length_squared));
            
            double closest_x = v1.x + t * dx;
            double closest_y = v1.y + t * dy;
            
            double dist_squared = (local_circle_x - closest_x) * (local_circle_x - closest_x) + 
                                 (local_circle_y - closest_y) * (local_circle_y - closest_y);
            min_distance_squared = std::min(min_distance_squared, dist_squared);
        }
    }
    
    return min_distance_squared <= circle_radius * circle_radius;
}



// Check collision between two primitives
bool World::checkPrimitiveCollision(const std::string& name1, const CollisionPrimitivePtr& prim1,
                                   const std::string& name2, const CollisionPrimitivePtr& prim2,
                                   CollisionType collision_type, CollisionResult& collision_result) const {
    
    bool collision_detected = false;
    
    // Check collision based on primitive types
    if (prim1->type == CollisionPrimitiveType::CIRCLE && prim2->type == CollisionPrimitiveType::CIRCLE) {
        // Circle-circle collision
        auto circle1 = std::dynamic_pointer_cast<CircleCollisionPrimitive>(prim1);
        auto circle2 = std::dynamic_pointer_cast<CircleCollisionPrimitive>(prim2);
        collision_detected = checkCircleCollision(circle1->local_cfg.x, circle1->local_cfg.y, circle1->radius, 
                                                 circle2->local_cfg.x, circle2->local_cfg.y, circle2->radius);
    } else if (prim1->type == CollisionPrimitiveType::CIRCLE && prim2->type == CollisionPrimitiveType::SQUARE) {
        // Circle-square collision
        auto circle = std::dynamic_pointer_cast<CircleCollisionPrimitive>(prim1);
        auto square = std::dynamic_pointer_cast<SquareCollisionPrimitive>(prim2);
        collision_detected = checkCircleSquareCollision(circle->local_cfg.x, circle->local_cfg.y, circle->radius, 
                                                       square->local_cfg.x, square->local_cfg.y, square->width, square->local_cfg.theta);
    } else if (prim1->type == CollisionPrimitiveType::SQUARE && prim2->type == CollisionPrimitiveType::CIRCLE) {
        // Square-circle collision (commutative)
        auto square = std::dynamic_pointer_cast<SquareCollisionPrimitive>(prim1);
        auto circle = std::dynamic_pointer_cast<CircleCollisionPrimitive>(prim2);
        collision_detected = checkCircleSquareCollision(circle->local_cfg.x, circle->local_cfg.y, circle->radius, 
                                                       square->local_cfg.x, square->local_cfg.y, square->width, square->local_cfg.theta);
    } else if (prim1->type == CollisionPrimitiveType::SQUARE && prim2->type == CollisionPrimitiveType::SQUARE) {
        // Square-square collision (not implemented yet)
        // TODO: Implement square-square collision detection
        throw std::runtime_error("Square-square collision not implemented");
        collision_detected = false;
    } else if (prim1->type == CollisionPrimitiveType::POLYGON && prim2->type == CollisionPrimitiveType::CIRCLE) {
        // Polygon-circle collision
        auto polygon = std::dynamic_pointer_cast<PolygonCollisionPrimitive>(prim1);
        auto circle = std::dynamic_pointer_cast<CircleCollisionPrimitive>(prim2);
        collision_detected = checkPolygonCircleCollision(polygon->local_cfg.x, polygon->local_cfg.y, polygon->local_cfg.theta, 
                                                        polygon->getVertices(), circle->local_cfg.x, circle->local_cfg.y, circle->radius);
    } else if (prim1->type == CollisionPrimitiveType::CIRCLE && prim2->type == CollisionPrimitiveType::POLYGON) {
        // Circle-polygon collision (commutative)
        auto circle = std::dynamic_pointer_cast<CircleCollisionPrimitive>(prim1);
        auto polygon = std::dynamic_pointer_cast<PolygonCollisionPrimitive>(prim2);
        collision_detected = checkPolygonCircleCollision(polygon->local_cfg.x, polygon->local_cfg.y, polygon->local_cfg.theta, 
                                                        polygon->getVertices(), circle->local_cfg.x, circle->local_cfg.y, circle->radius);
    } else if (prim1->type == CollisionPrimitiveType::POLYGON && prim2->type == CollisionPrimitiveType::POLYGON) {
        // Polygon-polygon collision (not implemented yet)
        throw std::runtime_error("Polygon-polygon collision not implemented");
    }
    
    if (collision_detected) {
        std::string collision_key = name1 + "_" + name2;
        Collision collision(name1, name2, 
                           prim1->local_cfg, 
                           prim2->local_cfg, 
                           collision_type);
        // Add the collision to the collision result.
        // If the collision key is not in the collision result, then add it.
        if (collision_result.collisions.find(collision_key) == collision_result.collisions.end()) {
            collision_result.collisions[collision_key] = std::vector<Collision>();
        }
        collision_result.collisions[collision_key].push_back(collision);
        return true;
    }
    return false;
}

// Check for collisions between a robot and an obstacle.
void World::checkCollision(const std::map<std::string, Configuration2>& cfgs, CollisionResult& collision_result, bool check_robot_robot, bool check_robot_obstacle) const {
    // Clear previous collision results
    collision_result.collisions.clear();
    
    // Collect all robot collision primitives in world frame.
    std::vector<std::vector<std::pair<std::string, CollisionPrimitivePtr>>> robot_primitives;
    std::vector<std::string> robot_names;
    
    for (const auto& [robot_name, robot] : robots_) {
        auto cfg_it = cfgs.find(robot_name);
        if (cfg_it == cfgs.end()) {
            continue; // Skip robots without configuration
        }
        
        const Configuration2& robot_cfg = cfg_it->second;
        const auto& local_primitives = robot->getLocalCollisionPrimitives();
        
        std::vector<std::pair<std::string, CollisionPrimitivePtr>> world_primitives;
        for (const auto& primitive : local_primitives) {
            // Transform primitive to world coordinates
            auto world_primitive = primitive->transformToWorld(robot_cfg);
            world_primitives.emplace_back(robot_name, world_primitive);
        }
        
        robot_primitives.push_back(world_primitives);
        robot_names.push_back(robot_name);
    }
    
    // Check robot-robot collisions.
    if (check_robot_robot) {
        for (size_t i = 0; i < robot_primitives.size(); ++i) {
            for (size_t j = i + 1; j < robot_primitives.size(); ++j) {
                const auto& primitives_i = robot_primitives[i];
                const auto& primitives_j = robot_primitives[j];
                
                for (const auto& [name_i, prim_i] : primitives_i) {
                    for (const auto& [name_j, prim_j] : primitives_j) {
                        if (checkPrimitiveCollision(name_i, prim_i, name_j, prim_j, CollisionType::ROBOT_ROBOT, collision_result)) {
                            goto next_robot_pair; // Break out of nested loops
                        }
                    }
                }
                next_robot_pair:;
            }
        }
    }
    
    if (check_robot_obstacle) {
    // Check robot-obstacle collisions
        for (const auto& [obstacle_name, obstacle_with_pose] : obstacles_) {
            const ObstaclePtr& obstacle = obstacle_with_pose.first;
            const Transform2& obstacle_pose = obstacle_with_pose.second;
            const auto& collision_primitives = obstacle->getLocalCollisionPrimitives();
            
            for (const auto& primitive : collision_primitives) {
                // Transform obstacle primitive to world coordinates
                auto world_obstacle_primitive = primitive->transformToWorld(obstacle_pose);
                
                // Check collision with all robot primitives
                for (const auto& robot_primitive_list : robot_primitives) {
                    for (const auto& [robot_name, robot_primitive] : robot_primitive_list) {
                        if (checkPrimitiveCollision(robot_name, robot_primitive, obstacle_name, world_obstacle_primitive, CollisionType::ROBOT_OBSTACLE, collision_result)) {
                            goto next_obstacle_primitive; // Break out of nested loops
                        }
                    }
                }
                next_obstacle_primitive:;
            }
        }
    }
}



void World::getSuccessorEdges(const std::string& robot_name, const Configuration2& cfg_current, std::vector<ActionSequencePtr>& successor_edges, std::vector<std::string>& successor_edges_primitive_names) const {
    // Get the motion primitives for the robot. This is in the local frame of the robot. Of form [origin, x_origin_cfg1, x_origin_cfg2, ..., x_origin_cfg_n].
    RobotPtr robot = getRobot(robot_name);
    std::vector<gco::ActionSequencePtr> motion_primitives_local;
    std::vector<std::string> motion_primitives_local_names;
    robot->getActionSequences(cfg_current, motion_primitives_local, motion_primitives_local_names);

    // Check if the motion primitives are valid.
    std::vector<ActionSequencePtr> valid_motion_primitives;
    for (size_t i = 0; i < motion_primitives_local.size(); i++) {
        const auto& motion_primitive_local = motion_primitives_local[i];
        // Transform the motion primitives to the world frame. This gives successor states as well as the action sequence (edge) to each one.
        auto motion_primitive_world = transformActionSequenceToWorld(motion_primitive_local, cfg_current);

        // Discretize and normalize the motion primitive.
        normalizeDiscretizePath(motion_primitive_world, robot->getJointRanges());

        // Check if the motion primitive is valid.
        if (isPathValid(robot_name, motion_primitive_world)) {
            successor_edges.push_back(motion_primitive_world);
            successor_edges_primitive_names.push_back(motion_primitives_local_names[i]);
        }
    }
}

bool World::isPathValid(const std::string& robot_name, const PathPtr& path, bool verbose) const {
    // Check if the path is valid.
    bool is_first_cfg = true;
    for (int i = 1; i < path->size() - 1; i++){
        const auto& cfg = path->at(i);
        // Check if each configuration in the path is valid.
        if (!isConfigurationValid(robot_name, cfg)) {
            if (verbose){
                std::cout << "[isPathValid] cfg i " << i << "/" << path->size() << " is invalid, returning false" << std::endl;
            }
            return false;
        }
    }
    return true;
}

bool World::isConfigurationValid(const std::string& robot_name, const Configuration2& cfg) const {
    // Check validity w.r.t. joint limits.
    RobotPtr robot = getRobot(robot_name);
    // if (cfg.theta > robot->getJointRanges().maxs.theta || cfg.theta < robot->getJointRanges().mins.theta ||
    //     cfg.x > robot->getJointRanges().maxs.x || cfg.x < robot->getJointRanges().mins.x ||
    //     cfg.y > robot->getJointRanges().maxs.y || cfg.y < robot->getJointRanges().mins.y) {
    //     return false;
    // }

    // Check if the configuration is valid.
    CollisionResult collision_result;
    checkCollision({{robot_name, cfg}}, collision_result);
    return collision_result.collisions.size() == 0;
}



// Get all obstacles (including objects converted to obstacles)
std::map<std::string, std::pair<ObstaclePtr, Transform2>> World::getAllObstacles() const {
    std::map<std::string, std::pair<ObstaclePtr, Transform2>> all_obstacles = obstacles_;
    
    // Convert objects to obstacles and add them
    for (const auto& [object_name, object_with_pose] : objects_) {
        const ObjectPtr& object = object_with_pose.first;
        const Transform2& pose = object_with_pose.second;
        
        // Convert object to obstacle by creating a new obstacle with the same properties
        ObstaclePtr obstacle;
        if (auto circle_obj = std::dynamic_pointer_cast<CircleShape>(object->getShape())) {
            obstacle = gco::Obstacle::createCircle(object_name, circle_obj->getRadius());
        } else if (auto square_obj = std::dynamic_pointer_cast<SquareShape>(object->getShape())) {
            obstacle = gco::Obstacle::createSquare(object_name, square_obj->getWidth());
        } else if (auto rect_obj = std::dynamic_pointer_cast<RectangleShape>(object->getShape())) {
            obstacle = gco::Obstacle::createRectangle(object_name, rect_obj->getWidth(), rect_obj->getHeight());
        } else {
            throw std::runtime_error("Unknown object type: " + object_name);
        }
        all_obstacles[object_name] = std::make_pair(obstacle, pose);
    }
    
    return all_obstacles;
}

// Create a copy of this world with objects converted to obstacles
WorldPtr World::createCopyWithObjectsAsObstacles() const {
    auto world_copy = std::make_shared<World>();
    
    // Copy robots
    for (const auto& [robot_name, robot] : robots_) {
        world_copy->addRobot(robot);
    }
    
    // Copy obstacles
    for (const auto& [obstacle_name, obstacle_with_pose] : obstacles_) {
        const ObstaclePtr& obstacle = obstacle_with_pose.first;
        const Transform2& pose = obstacle_with_pose.second;
        world_copy->addObstacle(obstacle, pose);
    }
    
    // Convert objects to obstacles and add them
    for (const auto& [object_name, object_with_pose] : objects_) {
        const ObjectPtr& object = object_with_pose.first;
        const Transform2& pose = object_with_pose.second;
        
        // Convert object to obstacle by creating a new obstacle with the same properties
        ObstaclePtr obstacle;
        if (auto circle_obj = std::dynamic_pointer_cast<CircleShape>(object->getShape())) {
            obstacle = gco::Obstacle::createCircle(object_name, circle_obj->getRadius());
        } else if (auto square_obj = std::dynamic_pointer_cast<SquareShape>(object->getShape())) {
            obstacle = gco::Obstacle::createSquare(object_name, square_obj->getWidth());
        } else if (auto rect_obj = std::dynamic_pointer_cast<RectangleShape>(object->getShape())) {
            obstacle = gco::Obstacle::createRectangle(object_name, rect_obj->getWidth(), rect_obj->getHeight());
        } else if (auto polygon_obj = std::dynamic_pointer_cast<PolygonShape>(object->getShape())) {
            obstacle = gco::Obstacle::createPolygon(object_name, polygon_obj->getVertices());
        }
        else {
            throw std::runtime_error("Unknown object type: " + object_name);
        }
        world_copy->addObstacle(obstacle, pose);
    }
    
    return world_copy;
}

} // namespace gco 