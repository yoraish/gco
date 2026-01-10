#pragma once

// General includes.
#include <cmath>
#include <cmath>
#include <functional>
#include <ostream>
#include <iomanip>

namespace gco {
// Pose.

struct Transform2
{
    double x;
    double y;
    double theta;   // radians in [-pi, pi]

    // Default constructor
    constexpr Transform2() : x(0.0), y(0.0), theta(0.0) {}

    // Value constructor
    constexpr Transform2(double x_, double y_, double theta_)
        : x(x_), y(y_), theta(theta_) {}

    static constexpr double wrapAngle(double a)
    {
        // atan2 keeps result in (-pi, pi]
        return std::atan2(std::sin(a), std::cos(a));
    }

    // Compose (this ∘ other)
    constexpr Transform2 operator*(const Transform2& other) const
    {
        return Transform2(
            x + std::cos(theta) * other.x - std::sin(theta) * other.y,
            y + std::sin(theta) * other.x + std::cos(theta) * other.y,
            wrapAngle(theta + other.theta)
        );
    }

    // Comparison operators for use in containers
    bool operator==(const Transform2& other) const {
        const double epsilon = 1e-9; // Small tolerance for floating-point comparison
        return std::abs(x - other.x) < epsilon &&
               std::abs(y - other.y) < epsilon &&
               std::abs(wrapAngle(theta - other.theta)) < epsilon;
    }

    bool operator!=(const Transform2& other) const {
        return !(*this == other);
    }

    bool operator<(const Transform2& other) const {
        // Lexicographic ordering: x, then y, then theta
        if (std::abs(x - other.x) >= 1e-9) {
            return x < other.x;
        }
        if (std::abs(y - other.y) >= 1e-9) {
            return y < other.y;
        }
        return wrapAngle(theta) < wrapAngle(other.theta);
    }

    bool operator<=(const Transform2& other) const {
        return *this < other || *this == other;
    }

    bool operator>(const Transform2& other) const {
        return !(*this <= other);
    }

    bool operator>=(const Transform2& other) const {
        return !(*this < other);
    }

    // Operator << for printing.
    friend std::ostream& operator<<(std::ostream& os, const Transform2& t) {
        os << "(" << std::fixed << std::setprecision(3) << t.x << ", " << std::fixed << std::setprecision(3) << t.y << ", " << std::fixed << std::setprecision(3) << t.theta << ")";
        return os;
    }
};

// Hash function for Transform2 (useful for unordered containers)
struct Transform2Hash {
    std::size_t operator()(const Transform2& t) const {
        // Use a simple hash combining the three components
        std::size_t h1 = std::hash<double>{}(t.x);
        std::size_t h2 = std::hash<double>{}(t.y);
        std::size_t h3 = std::hash<double>{}(t.theta);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

// Translation.
struct Translation2 : public Transform2 {
    // Constructor.
    Translation2(const double x, const double y) : Transform2(x, y, 0.0) {}

    // Multiply by another translation.
    Translation2 operator*(const Translation2& other) const {
        return Translation2(x + other.x, y + other.y);
    }
};

} // namespace gco

// Specialization of std::hash for Transform2
namespace std {
    template<>
    struct hash<gco::Transform2> {
        std::size_t operator()(const gco::Transform2& t) const {
            gco::Transform2Hash hasher;
            return hasher(t);
        }
    };
}