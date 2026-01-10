#pragma once

// General includes.
#include <map>

// Local includes.
#include "gco/obstacles/obstacles.hpp"
#include "gco/objects/objects.hpp"
#include "gco/robot.hpp"
#include "gco/collisions/collisions.hpp"
#include "gco/types.hpp"

namespace gco {

// Forward declaration
class World;
using WorldPtr = std::shared_ptr<World>;

class World {
    public:
    // ====================
    // Public methods.
    // ====================
    World();
    ~World();

    // Add a robot to the world.
    void addRobot(const RobotPtr& robot);

    // Get a robot by name.
    RobotPtr getRobot(const std::string& name) const;

    // Get all robots.
    const std::map<std::string, RobotPtr>& getAllRobots() const {
        return robots_;
    }

    // Get an object by name.
    ObjectPtr getObject(const std::string& name) const;

    // Add an obstacle to the world.
    void addObstacle(const ObstaclePtr& obstacle, const Transform2& pose) {
        obstacles_[obstacle->getName()] = std::make_pair(obstacle, pose);
    }

    // Add an object to the world.
    void addObject(const ObjectPtr& object, const Transform2& pose) {
        objects_[object->getName()] = std::make_pair(object, pose);
    }

    // Get all obstacles (including objects converted to obstacles)
    std::map<std::string, std::pair<ObstaclePtr, Transform2>> getAllObstacles() const;

    // Get only the original obstacles (excluding objects)
    const std::map<std::string, std::pair<ObstaclePtr, Transform2>>& getObstacles() const {
        return obstacles_;
    }

    // Get only the objects
    const std::map<std::string, std::pair<ObjectPtr, Transform2>>& getObjects() const {
        return objects_;
    }

    // Create a copy of this world with objects converted to obstacles
    WorldPtr createCopyWithObjectsAsObstacles() const;

    // Check for collisions between a robot and an obstacle.
    void checkCollision(const std::map<std::string, Configuration2>& cfgs, CollisionResult& collision_result, bool check_robot_robot = true, bool check_robot_obstacle = true) const;

    // Check if a path is valid.
    /**
     * @brief Check if a path is valid.
     * 
     * @param robot_name The name of the robot.
     * @param path The path to check. A sequence of configurations in the world frame.
     * @return True if the path is valid, false otherwise.
     */
    bool isPathValid(const std::string& robot_name, const PathPtr& path, bool verbose = false) const;

    // Check if a configuration is valid.
    /**
     * @brief Check if a configuration is valid.
     * 
     * @param robot_name The name of the robot.
     * @param cfg The configuration to check. In the world frame.
     * @return True if the configuration is valid, false otherwise.
     */
    bool isConfigurationValid(const std::string& robot_name, const Configuration2& cfg) const;

    // Get the successor edges for a robot.
    /**
     * @brief Get the successor edges for a robot.
     * 
     * @param robot_name The name of the robot.
     * @param cfg_current The current configuration of the robot.
     * @param successor_edges The vector of edge sequences. Each edge sequence is a list of local configurations. Of form [cfg_parent, cfg_1, cfg_2, ..., cfg_this].
     * The edge sequences are valid successors of the current configuration.
     * @param successor_edges_primitive_names The names of the primitive actions that correspond to the edge sequences.
     */
    void getSuccessorEdges(const std::string& robot_name, 
                           const Configuration2& cfg_current,
                           std::vector<ActionSequencePtr>& successor_edges, 
                           std::vector<std::string>& successor_edges_primitive_names) const;

    // ====================
    // Public variables.
    // ====================

    private:
    // ====================
    // Private variables and methods.
    // ====================
    
    std::pair<double, double> transformPointToWorld(const Transform2& x_robot_point, const Configuration2& x_world_robot) const;
    
    // Check collision between two circles
    bool checkCircleCollision(double x1, double y1, double r1, double x2, double y2, double r2) const;
    
    // Check collision between a circle and a square
    bool checkCircleSquareCollision(double circle_x, double circle_y, double circle_radius, 
                                   double square_x, double square_y, double square_width, double square_theta) const;

    // Check collision between a polygon and a circle
    bool checkPolygonCircleCollision(double polygon_x, double polygon_y, double polygon_theta, const std::vector<gco::Translation2>& polygon_vertices, 
                                    double circle_x, double circle_y, double circle_radius) const;
    
    // Check collision between two primitives
    bool checkPrimitiveCollision(const std::string& name1, const CollisionPrimitivePtr& prim1,
                                const std::string& name2, const CollisionPrimitivePtr& prim2,
                                CollisionType collision_type, CollisionResult& collision_result) const;

    // ====================
    // Private variables.
    // ====================
    std::map<std::string, RobotPtr> robots_;
    std::map<std::string, std::pair<ObstaclePtr, Transform2>> obstacles_;
    std::map<std::string, std::pair<ObjectPtr, Transform2>> objects_;
};

} // namespace gco