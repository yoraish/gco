#include "gco/objects/objects.hpp"

namespace gco {

// Base object constructor.
Object::Object(const std::string& name, const GeometricShapePtr& shape) 
    : name_(name), shape_(shape) {}

// Factory method for creating a circle object.
std::shared_ptr<Object> Object::createCircle(const std::string& name, const double radius) {
    auto shape = std::make_shared<CircleShape>(name, radius);
    return std::make_shared<Object>(name, shape);
}

// Factory method for creating a square object.
std::shared_ptr<Object> Object::createSquare(const std::string& name, const double width) {
    auto shape = std::make_shared<SquareShape>(name, width);
    return std::make_shared<Object>(name, shape);
}

// Factory method for creating a rectangle object.
std::shared_ptr<Object> Object::createRectangle(const std::string& name, const double width, const double height) {
    auto shape = std::make_shared<RectangleShape>(name, width, height);
    return std::make_shared<Object>(name, shape);
}

// Factory method for creating a polygon object.
std::shared_ptr<Object> Object::createPolygon(const std::string& name, const std::vector<gco::Translation2>& vertices) {
    auto shape = std::make_shared<PolygonShape>(name, vertices);
    return std::make_shared<Object>(name, shape);
}

} // namespace gco 