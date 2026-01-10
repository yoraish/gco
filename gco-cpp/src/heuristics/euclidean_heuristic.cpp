// Project includes.
#include "gco/heuristics/euclidean_heuristic.hpp"
#include "gco/utils.hpp"
#include <cmath>

namespace gco {

double EuclideanHeuristic::getHeuristic(const Configuration2& cfg_current, const Configuration2& cfg_goal) const {
    return configurationDistance(cfg_current, cfg_goal);
}

double EuclideanHeuristic::getHeuristic(const Configuration2& cfg_current, 
                                       const Configuration2& cfg_neighbor, 
                                       const Configuration2& cfg_goal) const {
    // For Euclidean heuristic, the directional version is the same as the regular version
    return getHeuristic(cfg_current, cfg_goal);
}

} // namespace gco 