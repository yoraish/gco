#pragma once

// General includes.
#include <memory>
#include <string>

// Local includes.
#include "gco/collisions/collisions.hpp"
#include "gco/geometric_shapes/geometric_shapes.hpp"

namespace gco {

// Base obstacle class that uses composition with GeometricShape.
class Obstacle {
    public:
    Obstacle(const std::string& name, const GeometricShapePtr& shape);
    virtual ~Obstacle() = default;

    // Get the obstacle name.
    const std::string& getName() const {
        return name_;
    }

    // Get the geometric shape.
    const GeometricShapePtr& getShape() const {
        return shape_;
    }

    // Get the collision primitives (delegates to the shape).
    const std::vector<CollisionPrimitivePtr> getLocalCollisionPrimitives() const {
        return shape_->getLocalCollisionPrimitives();
    }

    // Factory methods for creating common obstacle types.
    static std::shared_ptr<Obstacle> createCircle(const std::string& name, const double radius);
    static std::shared_ptr<Obstacle> createSquare(const std::string& name, const double width);
    static std::shared_ptr<Obstacle> createRectangle(const std::string& name, const double width, const double height);
    static std::shared_ptr<Obstacle> createPolygon(const std::string& name, const std::vector<gco::Translation2>& vertices);
    
    protected:
    // ====================
    // Protected variables.
    // ====================
    std::string name_;
    GeometricShapePtr shape_;
};

using ObstaclePtr = std::shared_ptr<Obstacle>;

} // namespace gco