#pragma once

// General includes.
#include <memory>

// Project includes.
#include "gco/types.hpp"
#include "gco/collisions/collisions.hpp"
#include "gco/geometric_shapes/geometric_shapes.hpp"

namespace gco {

struct JointRanges {
    Configuration2 mins;
    Configuration2 maxs;
    Configuration2 discretization;

    // Constructor.
    JointRanges(const Configuration2& mins, const Configuration2& maxs, const Configuration2& discretization) : mins(mins), maxs(maxs), discretization(discretization) {}
};

// A robot is a pair of a configuration and a cost.
class Robot {
    public:
    // ====================
    // Public methods.
    // ====================
    Robot(const std::string& name, const JointRanges& joint_ranges);
    ~Robot() = default;

    // Get the joint range.
    const JointRanges& getJointRanges() const {
        return joint_ranges_;
    }

    // Get all actions that are relevant to the current state. Each action sequence is a list of local configurations. Of form [origin, x_origin_1, x_origin_2, ..., x_origin_n].
    virtual void getActionSequences(const Configuration2& cfg, 
                                    std::vector<ActionSequencePtr>& action_sequences, 
                                    std::vector<std::string>& action_sequences_names) const = 0;

    // Get all collision primitives. Those are defined as a pair of a Translation2 and a radius, in the robot's frame (X-front, Y-left).
    virtual const std::vector<CollisionPrimitivePtr> getLocalCollisionPrimitives() const = 0;

    // Get the robot name.
    const std::string& getName() const {
        return name_;
    }

    // ====================
    // Public variables.
    // ====================

    protected:
    // ====================
    // Protected variables (accessible by derived classes).
    // ====================

    // Robot configuration discretization. A vector of dimension equal to dim(q) specifying the discretization of the configuration space.
    JointRanges joint_ranges_;

    // Robot name.
    std::string name_;

    private:
    // ====================
    // Private variables.
    // ====================
};


// ====================
// RobotDisk class.
// ====================
class RobotDisk : public Robot {
    public:
    RobotDisk(const std::string& name, const JointRanges& joint_ranges, const double radius);

    // Get the action sequences for the robot.
    void getActionSequences(const Configuration2& cfg, 
                            std::vector<ActionSequencePtr>& action_sequences, 
                            std::vector<std::string>& action_sequences_names) const override;

    // Get all collision primitives.
    const std::vector<CollisionPrimitivePtr> getLocalCollisionPrimitives() const override;

    // Get the robot radius.
    double getRadius() const {
        return radius_;
    }

    private:
    // Robot radius.
    double radius_;
};

using RobotPtr = std::shared_ptr<Robot>;

// ====================
// ObjectRobot base class.
// ====================
class ObjectRobot : public Robot {
    public:
    ObjectRobot(const std::string& name, const JointRanges& joint_ranges, const GeometricShapePtr& shape);

    // Get the action sequences for the object (empty for objects).
    void getActionSequences(const Configuration2& cfg, 
                            std::vector<ActionSequencePtr>& action_sequences, 
                            std::vector<std::string>& action_sequences_names) const override;

    // Get all collision primitives for the object (delegates to the shape).
    const std::vector<CollisionPrimitivePtr> getLocalCollisionPrimitives() const override;

    // Get the geometric shape.
    const GeometricShapePtr& getShape() const {
        return shape_;
    }

    // Factory methods for creating common object robot types.
    static std::shared_ptr<ObjectRobot> createCircle(const std::string& name, const JointRanges& joint_ranges, const double radius);
    static std::shared_ptr<ObjectRobot> createSquare(const std::string& name, const JointRanges& joint_ranges, const double side_length);
    static std::shared_ptr<ObjectRobot> createRectangle(const std::string& name, const JointRanges& joint_ranges, const double width, const double height);
    static std::shared_ptr<ObjectRobot> createPolygon(const std::string& name, const JointRanges& joint_ranges, const std::vector<gco::Translation2>& vertices);

    protected:
    // Geometric shape of the object robot.
    GeometricShapePtr shape_;
};

} // namespace gco