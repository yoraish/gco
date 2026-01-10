#pragma once

// General includes.
#include <memory>
#include <string>

// Local includes.
#include "gco/collisions/collisions.hpp"
#include "gco/geometric_shapes/geometric_shapes.hpp"

namespace gco {

// Base object class that uses composition with GeometricShape.
class Object {
    public:
    Object(const std::string& name, const GeometricShapePtr& shape);
    virtual ~Object() = default;

    // Get the object name.
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

    // Factory methods for creating common object types.
    static std::shared_ptr<Object> createCircle(const std::string& name, const double radius);
    static std::shared_ptr<Object> createSquare(const std::string& name, const double width);
    static std::shared_ptr<Object> createRectangle(const std::string& name, const double width, const double height);
    static std::shared_ptr<Object> createPolygon(const std::string& name, const std::vector<gco::Translation2>& vertices);
    
    protected:
    // ====================
    // Protected variables.
    // ====================
    std::string name_;
    GeometricShapePtr shape_;
};

using ObjectPtr = std::shared_ptr<Object>;

} // namespace gco 