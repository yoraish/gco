#include "gco/geometric_shapes/geometric_shapes.hpp"

namespace gco {

// Base geometric shape constructor.
GeometricShape::GeometricShape(const std::string& name) : name_(name) {}

// Base geometric shape destructor.
GeometricShape::~GeometricShape() = default;

// Circle shape constructor.
CircleShape::CircleShape(const std::string& name, const double radius) 
    : GeometricShape(name), radius_(radius) {}

// Circle shape collision primitives.
const std::vector<CollisionPrimitivePtr> CircleShape::getLocalCollisionPrimitives() const {
    // Circle shape has one circular collision primitive at the origin
    auto circle_primitive = std::make_shared<CircleCollisionPrimitive>(gco::Transform2(0.0, 0.0, 0.0), radius_);
    return {circle_primitive};
}

// Square shape constructor.
SquareShape::SquareShape(const std::string& name, const double width) 
    : GeometricShape(name), width_(width) {}

// Square shape collision primitives.
const std::vector<CollisionPrimitivePtr> SquareShape::getLocalCollisionPrimitives() const {
    // Square shape has one square collision primitive at the origin
    auto square_primitive = std::make_shared<SquareCollisionPrimitive>(gco::Transform2(0.0, 0.0, 0.0), width_);
    return {square_primitive};
}

// Rectangle shape constructor.
RectangleShape::RectangleShape(const std::string& name, const double width, const double height) 
    : GeometricShape(name), width_(width), height_(height) {}

// Rectangle shape collision primitives.
const std::vector<CollisionPrimitivePtr> RectangleShape::getLocalCollisionPrimitives() const {
    // Rectangle shape has an associated polygon collision primitive.
    std::vector<gco::Translation2> vertices = {  // These are all in the local frame of the shape (x forward, y left). Counter-clockwise order.
        gco::Translation2(-height_/2.0, -width_/2.0),
        gco::Translation2( height_/2.0, -width_/2.0),
        gco::Translation2( height_/2.0,  width_/2.0),
        gco::Translation2(-height_/2.0,  width_/2.0)
    };
    auto rectangle_primitive = std::make_shared<PolygonCollisionPrimitive>(gco::Transform2(0.0, 0.0, 0.0), vertices);
    return {rectangle_primitive};
}

// Polygon shape constructor.
PolygonShape::PolygonShape(const std::string& name, const std::vector<gco::Translation2>& vertices) 
    : GeometricShape(name), vertices_(vertices) {}

// Polygon shape collision primitives.
const std::vector<CollisionPrimitivePtr> PolygonShape::getLocalCollisionPrimitives() const {
    auto polygon_primitive = std::make_shared<PolygonCollisionPrimitive>(gco::Transform2(0.0, 0.0, 0.0), vertices_);
    return {polygon_primitive};
}

} // namespace gco 