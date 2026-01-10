// Implementation of the Robot class.

// Local includes.
#include "gco/robot.hpp"
#include "gco/types.hpp"
#include "gco/spatial/transforms.hpp"

// General includes.
#include <iostream>
#include <iomanip>

// ====================
// Implementation of the Robot class.
// ====================
gco::Robot::Robot(const std::string& name, const JointRanges& joint_ranges) : joint_ranges_(joint_ranges), name_(name) {
    // Nothing to do.
}

// ====================
// Implementation of the RobotDisk class.
// ====================
gco::RobotDisk::RobotDisk(const std::string& name, const JointRanges& joint_ranges, const double radius) : Robot(name, joint_ranges), radius_(radius) {
    // Nothing to do.
}

void gco::RobotDisk::getActionSequences(const Configuration2& cfg, 
                                        std::vector<ActionSequencePtr>& action_sequences, 
                                        std::vector<std::string>& action_sequences_names) const {
    // Return up down left right plus diagonals at various angles.
    action_sequences.clear();
    action_sequences_names.clear();

    // Cardinal directions
    action_sequences.push_back(std::make_shared<ActionSequence>(std::vector<Configuration2>{Transform2(0.0, 0.0, 0.0),
                                                                                               Transform2(0.0, 0.01, 0.0),
                                                                                               Transform2(0.0, 0.02, 0.0),
                                                                                               Transform2(0.0, 0.03, 0.0),
                                                                                               Transform2(0.0, 0.04, 0.0),
                                                                                               Transform2(0.0, 0.05, 0.0)}));
    action_sequences_names.push_back("up");

    action_sequences.push_back(std::make_shared<ActionSequence>(std::vector<Configuration2>{Transform2(0.0, 0.0, 0.0),
                                                                                               Transform2(0.0, -0.01, 0.0),
                                                                                               Transform2(0.0, -0.02, 0.0),
                                                                                               Transform2(0.0, -0.03, 0.0),
                                                                                               Transform2(0.0, -0.04, 0.0),
                                                                                               Transform2(0.0, -0.05, 0.0)}));
    action_sequences_names.push_back("down");
    action_sequences.push_back(std::make_shared<ActionSequence>(std::vector<Configuration2>{Transform2(0.0, 0.0, 0.0),
                                                                                               Transform2(0.01, 0.0, 0.0),
                                                                                               Transform2(0.02, 0.0, 0.0),
                                                                                               Transform2(0.03, 0.0, 0.0),
                                                                                               Transform2(0.04, 0.0, 0.0),
                                                                                               Transform2(0.05, 0.0, 0.0)}));

    action_sequences_names.push_back("right");
    action_sequences.push_back(std::make_shared<ActionSequence>(std::vector<Configuration2>{Transform2(0.0, 0.0, 0.0),
                                                                                               Transform2(-0.01, 0.0, 0.0),
                                                                                               Transform2(-0.02, 0.0, 0.0),
                                                                                               Transform2(-0.03, 0.0, 0.0),
                                                                                               Transform2(-0.04, 0.0, 0.0),
                                                                                               Transform2(-0.05, 0.0, 0.0)}));
    action_sequences_names.push_back("left");
    // Diagonal movements at various angles
    // 45 degrees (original diagonals)
    action_sequences.push_back(std::make_shared<ActionSequence>(std::vector<Configuration2>{Transform2(0.0, 0.0, 0.0),
                                                                                               Transform2(0.01, 0.01, 0.0),
                                                                                               Transform2(0.02, 0.02, 0.0),
                                                                                               Transform2(0.03, 0.03, 0.0),
                                                                                               Transform2(0.04, 0.04, 0.0),
                                                                                               Transform2(0.05, 0.05, 0.0)}));
    action_sequences_names.push_back("up-right");
    action_sequences.push_back(std::make_shared<ActionSequence>(std::vector<Configuration2>{Transform2(0.0, 0.0, 0.0),
                                                                                               Transform2(-0.01, -0.01, 0.0),
                                                                                               Transform2(-0.02, -0.02, 0.0),
                                                                                               Transform2(-0.03, -0.03, 0.0),
                                                                                               Transform2(-0.04, -0.04, 0.0),
                                                                                               Transform2(-0.05, -0.05, 0.0)}));
    action_sequences_names.push_back("down-right");
    action_sequences.push_back(std::make_shared<ActionSequence>(std::vector<Configuration2>{Transform2(0.0, 0.0, 0.0),
                                                                                               Transform2(-0.01, 0.01, 0.0),
                                                                                               Transform2(-0.02, 0.02, 0.0),
                                                                                               Transform2(-0.03, 0.03, 0.0),
                                                                                               Transform2(-0.04, 0.04, 0.0),
                                                                                               Transform2(-0.05, 0.05, 0.0)}));
    action_sequences_names.push_back("up-left");
    action_sequences.push_back(std::make_shared<ActionSequence>(std::vector<Configuration2>{Transform2(0.0, 0.0, 0.0),
                                                                                               Transform2(0.01, -0.01, 0.0),
                                                                                               Transform2(0.02, -0.02, 0.0),
                                                                                               Transform2(0.03, -0.03, 0.0),
                                                                                               Transform2(0.04, -0.04, 0.0),
                                                                                               Transform2(0.05, -0.05, 0.0)}));
    action_sequences_names.push_back("down-left");
    // 30 degrees (shallow diagonals)
    action_sequences.push_back(std::make_shared<ActionSequence>(std::vector<Configuration2>{Transform2(0.0, 0.0, 0.0),
                                                                                               Transform2(0.01, 0.0058, 0.0),
                                                                                               Transform2(0.02, 0.0116, 0.0),
                                                                                               Transform2(0.03, 0.0174, 0.0),
                                                                                               Transform2(0.04, 0.0232, 0.0),
                                                                                               Transform2(0.05, 0.03, 0.0)}));
    action_sequences_names.push_back("up-right-30");
    action_sequences.push_back(std::make_shared<ActionSequence>(std::vector<Configuration2>{Transform2(0.0, 0.0, 0.0),
                                                                                               Transform2(0.01, -0.0058, 0.0),
                                                                                               Transform2(0.02, -0.0116, 0.0),
                                                                                               Transform2(0.03, -0.0174, 0.0),
                                                                                               Transform2(0.04, -0.0232, 0.0),
                                                                                               Transform2(0.05, -0.03, 0.0)}));
    action_sequences_names.push_back("down-right-30");
    action_sequences.push_back(std::make_shared<ActionSequence>(std::vector<Configuration2>{Transform2(0.0, 0.0, 0.0),
                                                                                               Transform2(-0.01, 0.0058, 0.0),
                                                                                               Transform2(-0.02, 0.0116, 0.0),
                                                                                               Transform2(-0.03, 0.0174, 0.0),
                                                                                               Transform2(-0.04, 0.0232, 0.0),
                                                                                               Transform2(-0.05, 0.03, 0.0)}));
    action_sequences_names.push_back("up-left-30");
    action_sequences.push_back(std::make_shared<ActionSequence>(std::vector<Configuration2>{Transform2(0.0, 0.0, 0.0),
                                                                                               Transform2(-0.01, -0.0058, 0.0),
                                                                                               Transform2(-0.02, -0.0116, 0.0),
                                                                                               Transform2(-0.03, -0.0174, 0.0),
                                                                                               Transform2(-0.04, -0.0232, 0.0),
                                                                                               Transform2(-0.05, -0.03, 0.0)}));
    action_sequences_names.push_back("down-left-30");
    // 60 degrees (steep diagonals)
    action_sequences.push_back(std::make_shared<ActionSequence>(std::vector<Configuration2>{Transform2(0.0, 0.0, 0.0),
                                                                                               Transform2(0.0058, 0.01, 0.0),
                                                                                               Transform2(0.0116, 0.02, 0.0),
                                                                                               Transform2(0.0174, 0.03, 0.0),
                                                                                               Transform2(0.0232, 0.04, 0.0),
                                                                                               Transform2(0.03, 0.05, 0.0)}));
    action_sequences_names.push_back("up-right-60");
    action_sequences.push_back(std::make_shared<ActionSequence>(std::vector<Configuration2>{Transform2(0.0, 0.0, 0.0),
                                                                                               Transform2(0.0058, -0.01, 0.0),
                                                                                               Transform2(0.0116, -0.02, 0.0),
                                                                                               Transform2(0.0174, -0.03, 0.0),
                                                                                               Transform2(0.0232, -0.04, 0.0),
                                                                                               Transform2(0.03, -0.05, 0.0)}));
    action_sequences_names.push_back("down-right-60");
    action_sequences.push_back(std::make_shared<ActionSequence>(std::vector<Configuration2>{Transform2(0.0, 0.0, 0.0),
                                                                                               Transform2(-0.0058, 0.01, 0.0),
                                                                                               Transform2(-0.0116, 0.02, 0.0),
                                                                                               Transform2(-0.0174, 0.03, 0.0),
                                                                                               Transform2(-0.0232, 0.04, 0.0),
                                                                                               Transform2(-0.03, 0.05, 0.0)}));
    action_sequences_names.push_back("up-left-60");
    action_sequences.push_back(std::make_shared<ActionSequence>(std::vector<Configuration2>{Transform2(0.0, 0.0, 0.0),
                                                                                               Transform2(-0.0058, -0.01, 0.0),
                                                                                               Transform2(-0.0116, -0.02, 0.0),
                                                                                               Transform2(-0.0174, -0.03, 0.0),
                                                                                               Transform2(-0.0232, -0.04, 0.0),
                                                                                               Transform2(-0.03, -0.05, 0.0)}));
    action_sequences_names.push_back("down-left-60");
    // Scale all by a factor.
    double scale_factor = 1.0;
    for (auto& action_sequence : action_sequences) {
        for (auto& cfg : *action_sequence) {
            cfg.x *= scale_factor;
            cfg.y *= scale_factor;
            // cfg.theta *= scale_factor;
        }
    }
}

const std::vector<gco::CollisionPrimitivePtr> gco::RobotDisk::getLocalCollisionPrimitives() const {
    // Only one collision circle. Center is at the origin and radius is the robot's radius.
    auto circle_primitive = std::make_shared<CircleCollisionPrimitive>(gco::Transform2(0.0, 0.0, 0.0), radius_);
    return {circle_primitive};
}

// ====================
// ObjectRobot implementation.
// ====================
gco::ObjectRobot::ObjectRobot(const std::string& name, const JointRanges& joint_ranges, 
                              const GeometricShapePtr& shape) 
    : Robot(name, joint_ranges), shape_(shape) {
}

void gco::ObjectRobot::getActionSequences(const Configuration2& cfg, 
                                          std::vector<ActionSequencePtr>& action_sequences, 
                                          std::vector<std::string>& action_sequences_names) const {
    // Objects don't have action sequences - they are moved by robots
    action_sequences.clear();
    action_sequences_names.clear();
}

const std::vector<gco::CollisionPrimitivePtr> gco::ObjectRobot::getLocalCollisionPrimitives() const {
    // Delegate to the shape
    return shape_->getLocalCollisionPrimitives();
}

// ====================
// ObjectRobot factory methods.
// ====================
std::shared_ptr<gco::ObjectRobot> gco::ObjectRobot::createCircle(const std::string& name, const JointRanges& joint_ranges, const double radius) {
    auto shape = std::make_shared<CircleShape>(name, radius);
    return std::make_shared<ObjectRobot>(name, joint_ranges, shape);
}

std::shared_ptr<gco::ObjectRobot> gco::ObjectRobot::createSquare(const std::string& name, const JointRanges& joint_ranges, const double side_length) {
    auto shape = std::make_shared<SquareShape>(name, side_length);
    return std::make_shared<ObjectRobot>(name, joint_ranges, shape);
}

std::shared_ptr<gco::ObjectRobot> gco::ObjectRobot::createRectangle(const std::string& name, const JointRanges& joint_ranges, const double width, const double height) {
    auto shape = std::make_shared<RectangleShape>(name, width, height);
    return std::make_shared<ObjectRobot>(name, joint_ranges, shape);
}

std::shared_ptr<gco::ObjectRobot> gco::ObjectRobot::createPolygon(const std::string& name, const JointRanges& joint_ranges, const std::vector<gco::Translation2>& vertices) {
    auto shape = std::make_shared<PolygonShape>(name, vertices);
    return std::make_shared<ObjectRobot>(name, joint_ranges, shape);
}


