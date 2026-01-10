#pragma once

// General includes.
#include <memory>
#include <string>

// Local includes.
#include "gco/collisions/collisions.hpp"

namespace gco {

// Base geometric shape class that provides common functionality
class GeometricShape {
    public:
    GeometricShape(const std::string& name);
    virtual ~GeometricShape() = 0;

    // Get the shape name.
    const std::string& getName() const {
        return name_;
    }

    // Get the collision primitives for this shape.
    virtual const std::vector<CollisionPrimitivePtr> getLocalCollisionPrimitives() const = 0;

    protected:
    // ====================
    // Protected variables.
    // ====================
    std::string name_;
};

// Circle shape
class CircleShape : public GeometricShape {
    public:
    CircleShape(const std::string& name, const double radius);

    // Get the collision primitives.
    const std::vector<CollisionPrimitivePtr> getLocalCollisionPrimitives() const override;

    // Get the radius.
    double getRadius() const {
        return radius_;
    }

    private:
    // ====================
    // Private variables.
    // ====================
    double radius_;
};

// Square shape
class SquareShape : public GeometricShape {
    public:
    SquareShape(const std::string& name, const double width);

    // Get the collision primitives.
    const std::vector<CollisionPrimitivePtr> getLocalCollisionPrimitives() const override;

    // Get the width.
    double getWidth() const {
        return width_;
    }

    private:
    // ====================
    // Private variables.
    // ====================
    double width_;
};

// Rectangle shape
class RectangleShape : public GeometricShape {
    public:
    RectangleShape(const std::string& name, const double width, const double height);

    // Get the collision primitives.
    const std::vector<CollisionPrimitivePtr> getLocalCollisionPrimitives() const override;

    // Get the width.
    double getWidth() const {
        return width_;
    }

    // Get the height.
    double getHeight() const {
        return height_;
    }

    private:
    // ====================
    // Private variables.
    // ====================
    double width_;
    double height_;
};

// Polygon shape
class PolygonShape : public GeometricShape {
    public:
    PolygonShape(const std::string& name, const std::vector<gco::Translation2>& vertices);

    // Get the collision primitives.
    const std::vector<CollisionPrimitivePtr> getLocalCollisionPrimitives() const override;

    // Get the vertices.
    const std::vector<gco::Translation2>& getVertices() const {
        return vertices_;
    }

    private:
    // ====================
    // Private variables.
    // ====================
    std::vector<gco::Translation2> vertices_;
};

using GeometricShapePtr = std::shared_ptr<GeometricShape>;

} // namespace gco 