#include "gco/obstacles/obstacles.hpp"

namespace gco {

// Base obstacle constructor.
Obstacle::Obstacle(const std::string& name, const GeometricShapePtr& shape) 
    : name_(name), shape_(shape) {}

// Factory method for creating a circle obstacle.
std::shared_ptr<Obstacle> Obstacle::createCircle(const std::string& name, const double radius) {
    auto shape = std::make_shared<CircleShape>(name, radius);
    return std::make_shared<Obstacle>(name, shape);
}

// Factory method for creating a square obstacle.
std::shared_ptr<Obstacle> Obstacle::createSquare(const std::string& name, const double width) {
    auto shape = std::make_shared<SquareShape>(name, width);
    return std::make_shared<Obstacle>(name, shape);
}

// Factory method for creating a rectangle obstacle.
std::shared_ptr<Obstacle> Obstacle::createRectangle(const std::string& name, const double width, const double height) {
    auto shape = std::make_shared<RectangleShape>(name, width, height);
    return std::make_shared<Obstacle>(name, shape);
}

// Factory method for creating a polygon obstacle.
std::shared_ptr<Obstacle> Obstacle::createPolygon(const std::string& name, const std::vector<gco::Translation2>& vertices) {
    auto shape = std::make_shared<PolygonShape>(name, vertices);
    return std::make_shared<Obstacle>(name, shape);
}

} // namespace gco
