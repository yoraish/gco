#pragma once

// General includes.
#include <map>
#include <string>
#include <memory>
#include <cmath>
#include <vector>

// Project includes.
#include "gco/types.hpp"
#include "gco/spatial/transforms.hpp"

// Define M_PI if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace gco {

    // Collision type.
    enum class CollisionType {
        UNSET,
        ROBOT_ROBOT,
        ROBOT_OBSTACLE,
    };

    struct Collision {
        std::string name_entity1;
        std::string name_entity2;
        gco::Transform2 cfg_entity1;
        gco::Transform2 cfg_entity2;
        CollisionType collision_type = CollisionType::UNSET;

        // Constructor.
        Collision() : name_entity1(""), 
                      name_entity2(""), 
                      cfg_entity1(0.0, 0.0, 0.0), 
                      cfg_entity2(0.0, 0.0, 0.0), 
                      collision_type(CollisionType::UNSET) {}
        Collision(const std::string& name_entity1, 
                  const std::string& name_entity2, 
                  const gco::Transform2& cfg_entity1, 
                  const gco::Transform2& cfg_entity2, 
                  const CollisionType& collision_type = CollisionType::UNSET) : name_entity1(name_entity1), 
                                                                                name_entity2(name_entity2), 
                                                                                cfg_entity1(cfg_entity1), 
                                                                                cfg_entity2(cfg_entity2), 
                                                                                collision_type(collision_type) {}

        // Print function stream.
        std::ostream& print(std::ostream& os) const {
            os << "Collision between " << name_entity1 << " and " << name_entity2 << " at " << cfg_entity1 << " and " << cfg_entity2;
            // Print center distances.
            double dist_x = std::abs(cfg_entity1.x - cfg_entity2.x);
            double dist_y = std::abs(cfg_entity1.y - cfg_entity2.y);
            double dist = std::sqrt(dist_x * dist_x + dist_y * dist_y); 
            os << " Center distance: " << dist << std::endl;
            return os;
        }
    };

    // Collision result.
    struct CollisionResult {
        std::map<std::string, std::vector<Collision>> collisions;
    };
    
    // ====================
    // Collision primitives.
    // ====================
    enum class CollisionPrimitiveType {
        UNSET,
        CIRCLE,
        SQUARE,
        POLYGON,
    };

    // Forward declaration
    using CollisionPrimitivePtr = std::shared_ptr<struct CollisionPrimitive>;

    // Collision primitive. A geometric primitive. Union of these is used to represent the collision geometry of an entity.
    struct CollisionPrimitive {
        // Configuration of the primitive relative to the entity's frame.
        gco::Transform2 local_cfg;
        CollisionPrimitiveType type = CollisionPrimitiveType::UNSET;

        // Constructor.
        CollisionPrimitive(const gco::Transform2& local_cfg, const CollisionPrimitiveType& type = CollisionPrimitiveType::UNSET) : local_cfg(local_cfg), type(type) {}
        
        // Virtual destructor to make the class polymorphic
        virtual ~CollisionPrimitive() = default;

        // Transform the primitive to world coordinates
        virtual CollisionPrimitivePtr transformToWorld(const gco::Transform2& entity_pose) const = 0;
    };

    // Circle collision primitive.
    struct CircleCollisionPrimitive : public CollisionPrimitive {
        double radius;

        // Constructor.
        CircleCollisionPrimitive(const gco::Transform2& local_cfg, const double radius) : CollisionPrimitive(local_cfg, CollisionPrimitiveType::CIRCLE), radius(radius) {}

        // Transform to world coordinates
        CollisionPrimitivePtr transformToWorld(const gco::Transform2& entity_pose) const override {
            // Apply entity pose transformation to primitive
            // Robot frame: X=forward, Y=left
            // World frame: X=right, Y=up
            // Need to rotate robot frame by π/2 to align with world frame
            double cos_theta = std::cos(entity_pose.theta);
            double sin_theta = std::sin(entity_pose.theta);
            
            double world_x = local_cfg.x * cos_theta - local_cfg.y * sin_theta + entity_pose.x;
            double world_y = local_cfg.x * sin_theta + local_cfg.y * cos_theta + entity_pose.y;
            double world_theta = local_cfg.theta + entity_pose.theta;
            
            return std::make_shared<CircleCollisionPrimitive>(gco::Transform2(world_x, world_y, world_theta), radius);
        }
    };
    using CircleCollisionPrimitivePtr = std::shared_ptr<CircleCollisionPrimitive>;


    // Square collision primitive.
    struct SquareCollisionPrimitive : public CollisionPrimitive {
        double width;

        // Constructor.
        SquareCollisionPrimitive(const gco::Transform2& local_cfg, const double width) : CollisionPrimitive(local_cfg, CollisionPrimitiveType::SQUARE), width(width) {}

        // Transform to world coordinates
        CollisionPrimitivePtr transformToWorld(const gco::Transform2& entity_pose) const override {
            // Apply entity pose transformation to primitive
            // Robot frame: X=forward, Y=left
            // World frame: X=right, Y=up
            // Need to rotate robot frame by π/2 to align with world frame
            double cos_theta = std::cos(entity_pose.theta);
            double sin_theta = std::sin(entity_pose.theta);
            
            double world_x = local_cfg.x * cos_theta - local_cfg.y * sin_theta + entity_pose.x;
            double world_y = local_cfg.x * sin_theta + local_cfg.y * cos_theta + entity_pose.y;
            double world_theta = local_cfg.theta + entity_pose.theta;
            
            return std::make_shared<SquareCollisionPrimitive>(gco::Transform2(world_x, world_y, world_theta), width);
        }
    };
    using SquareCollisionPrimitivePtr = std::shared_ptr<SquareCollisionPrimitive>;

    // Polygon collision primitive.
    struct PolygonCollisionPrimitive : public CollisionPrimitive {
        std::vector<gco::Translation2> vertices;

        // Constructor.
        PolygonCollisionPrimitive(const gco::Transform2& local_cfg, 
                                  const std::vector<gco::Translation2>& vertices) : 
                                        CollisionPrimitive(local_cfg, CollisionPrimitiveType::POLYGON), 
                                        vertices(vertices) {}

        // Get the polygon vertices.
        const std::vector<gco::Translation2>& getVertices() const {
            return vertices;
        }

        // Transform to world coordinates
        CollisionPrimitivePtr transformToWorld(const gco::Transform2& entity_pose) const override {
            // Apply entity pose transformation to primitive
            // Robot frame: X=forward, Y=left
            // World frame: X=right, Y=up
            // Need to rotate robot frame by π/2 to align with world frame
            double cos_theta = std::cos(entity_pose.theta);
            double sin_theta = std::sin(entity_pose.theta);
            
            double world_x = local_cfg.x * cos_theta - local_cfg.y * sin_theta + entity_pose.x;
            double world_y = local_cfg.x * sin_theta + local_cfg.y * cos_theta + entity_pose.y;
            double world_theta = local_cfg.theta + entity_pose.theta;
            
            // Transform vertices to world coordinates
            std::vector<gco::Translation2> world_vertices;
            world_vertices.reserve(vertices.size());
            
            for (const auto& vertex : vertices) {
                double vx = vertex.x * cos_theta - vertex.y * sin_theta + world_x;
                double vy = vertex.x * sin_theta + vertex.y * cos_theta + world_y;
                world_vertices.emplace_back(vx, vy);
            }
            
            return std::make_shared<PolygonCollisionPrimitive>(gco::Transform2(world_x, world_y, world_theta), world_vertices);
        }
    };
} // namespace gco