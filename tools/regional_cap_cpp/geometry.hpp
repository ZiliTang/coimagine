#pragma once

// Native counterparts of the geometry helpers in the ten frozen regional CAPs.
// IDs are s004,s008,s011,s013,s019,n_s000,n_s001,n_s002,n_s005,n_s009.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace rc {
namespace py = pybind11;
using Point = std::array<double, 2>;
using Points = std::vector<Point>;
constexpr double geometry_pi = 3.141592653589793238462643383279502884;

inline py::object geometry_get(py::dict d, const char* key, py::object fallback = py::none()) {
    return d.contains(key) ? py::reinterpret_borrow<py::object>(d[key]) : fallback;
}
inline double geometry_number(py::dict d, const char* key, double fallback = 0.) {
    return d.contains(key) ? py::cast<double>(d[key]) : fallback;
}
inline bool geometry_truth(py::handle value) { return PyObject_IsTrue(value.ptr()) == 1; }
inline bool geometry_equal(py::handle a, py::handle b) {
    int result = PyObject_RichCompareBool(a.ptr(), b.ptr(), Py_EQ);
    if (result < 0) throw py::error_already_set();
    return result == 1;
}
inline int geometry_round(double x) {
    // Python round(x), without std::round's away-from-zero half ties.
    const double lo = std::floor(x), remainder = x - lo;
    return static_cast<int>(remainder < .5 ? lo : remainder > .5 ? lo + 1. :
                            std::fmod(lo, 2.) == 0. ? lo : lo + 1.);
}
inline int geometry_direction(double x, double y) {
    int d = geometry_round(std::atan2(y, x) / (geometry_pi / 4.));
    return (d % 8 + 8) % 8;
}
inline double geometry_hypot(double x, double y) {
    // Two-coordinate specialization of CPython 3.10/3.11 mathmodule.c's
    // compensated vector_norm (PSF license). This preserves math.hypot's
    // rounding; the platform's hypot is sometimes one ULP different.
    const Point values{std::abs(x), std::abs(y)};
    double maximum = std::max(values[0], values[1]);
    if (std::isinf(values[0]) || std::isinf(values[1])) return std::numeric_limits<double>::infinity();
    if (std::isnan(values[0]) || std::isnan(values[1])) return std::numeric_limits<double>::quiet_NaN();
    if (maximum == 0.) return maximum;
    int exponent;
    std::frexp(maximum, &exponent);
    double sum = 1., fraction1 = 0., fraction2 = 0., fraction3 = 0.;
    if (exponent < -1023) {
        for (double value : values) {
            value /= maximum; value *= value;
            double old = sum; sum += value; fraction1 += (old - sum) + value;
        }
        return maximum * std::sqrt(sum - 1. + fraction1);
    }
    double scale = std::ldexp(1., -exponent);
    constexpr double splitter = 134217729.;
    for (double value : values) {
        value *= scale;
        double temporary = value * splitter;
        double hi = temporary - (temporary - value), lo = value - hi;
        double square = hi * hi, old = sum;
        sum += square; fraction1 += (old - sum) + square;
        double cross = 2. * hi * lo; old = sum;
        sum += cross; fraction2 += (old - sum) + cross;
        fraction3 += lo * lo;
    }
    double root = std::sqrt(sum - 1. + (fraction1 + fraction2 + fraction3));
    double temporary = root * splitter;
    double hi = temporary - (temporary - root), lo = root - hi;
    double square = -hi * hi, old = sum;
    sum += square; fraction1 += (old - sum) + square;
    double cross = -2. * hi * lo; old = sum;
    sum += cross; fraction2 += (old - sum) + cross;
    square = -lo * lo; old = sum;
    sum += square; fraction3 += (old - sum) + square;
    double residual = sum - 1. + (fraction1 + fraction2 + fraction3);
    return (root + residual / (2. * root)) / scale;
}
inline Point norm(double x, double y) {
    double denominator = std::max(geometry_hypot(x, y), 1e-9);
    return {x / denominator, y / denominator};
}

inline py::object estimate(int id, py::dict memory, py::dict observation) {
    auto target = py::reinterpret_borrow<py::object>(py::cast<py::dict>(observation["goal"])["target_id"]);
    Points points;
    for (auto item : observation["entities"]) {
        auto entity = py::cast<py::dict>(item);
        if (geometry_equal(entity["id"], target)) points = py::cast<Points>(entity["points"]);
    }
    if (points.empty()) return geometry_get(memory, "c");
    std::vector<py::dict> patches;
    if (geometry_equal(geometry_get(observation, "active_target"), target)) {
        for (auto item : observation["robots"]) {
            auto robot = py::cast<py::dict>(item);
            for (auto patch_item : geometry_get(robot, "patches", py::list())) {
                auto patch = py::cast<py::dict>(patch_item);
                if (geometry_truth(geometry_get(patch, "valid")) && geometry_number(patch, "span_m") > .08)
                    patches.push_back(patch);
            }
        }
    }
    double old_angle = py::cast<double>(memory["ang"]), angle = old_angle;
    if (!patches.empty()) {
        auto best = patches.begin();
        for (auto p = patches.begin() + 1; p != patches.end(); ++p)
            if (geometry_number(*p, "span_m") > geometry_number(*best, "span_m")) best = p;
        Point tangent = py::cast<Point>((*best)["tangent_world"]);
        angle = std::fmod(std::atan2(tangent[1], tangent[0]), geometry_pi / 2.);
        if (angle < 0.) angle += geometry_pi / 2.;
        if (angle == 0.) angle = 0.;
    }
    Point u{std::cos(angle), std::sin(angle)}, v{-u[1], u[0]};
    Point lo{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    Point hi{-lo[0], -lo[1]};
    for (auto p : points) {
        Point projection{p[0] * u[0] + p[1] * u[1], p[0] * v[0] + p[1] * v[1]};
        for (int k = 0; k < 2; ++k) { lo[k] = std::min(lo[k], projection[k]); hi[k] = std::max(hi[k], projection[k]); }
    }
    Point old_dims = py::cast<Point>(memory["dims"]);
    if (std::abs(angle - old_angle) > (id < 4 ? .6 : geometry_pi / 4.)) std::swap(old_dims[0], old_dims[1]);
    Point spans{hi[0] - lo[0], hi[1] - lo[1]}, dims;
    if (id < 4) {
        dims = {std::max(old_dims[0], spans[0]), std::max(old_dims[1], spans[1])};
    } else {
        double longest = std::max(geometry_number(memory, "long_size"), std::max(spans[0], spans[1]));
        double shortest = geometry_number(memory, "short_size");
        if (std::max(spans[0], spans[1]) > .6 * longest && std::min(spans[0], spans[1]) < .6 * longest)
            shortest = std::max(shortest, std::min(spans[0], spans[1]));
        int long_axis = std::max(spans[0], spans[1]) > .6 * longest ?
            (spans[0] >= spans[1] ? 0 : 1) : (old_dims[0] >= old_dims[1] ? 0 : 1);
        dims = {shortest, shortest}; dims[long_axis] = longest;
        memory["long_size"] = longest; memory["short_size"] = shortest;
    }
    Point center{(lo[0] + hi[0]) / 2., (lo[1] + hi[1]) / 2.};
    for (int k = 0; k < 2; ++k) {
        Point axis = k == 0 ? u : v;
        std::vector<double> signs;
        for (auto patch : patches) {
            Point normal = py::cast<Point>(patch["normal_world"]);
            double s = normal[0] * axis[0] + normal[1] * axis[1];
            if (std::abs(s) > .9) signs.push_back(s);
        }
        if (!signs.empty() && *std::min_element(signs.begin(), signs.end()) * *std::max_element(signs.begin(), signs.end()) > 0.)
            center[k] += (dims[k] - spans[k]) * .5 * (signs[0] > 0. ? 1. : -1.);
    }
    py::list result; result.append(center[0] * u[0] + center[1] * v[0]); result.append(center[0] * u[1] + center[1] * v[1]);
    py::list dim_list; dim_list.append(dims[0]); dim_list.append(dims[1]);
    memory["dims"] = dim_list; memory["ang"] = angle; memory["c"] = result;
    return result;
}

inline Points obstacles(int id, py::dict observation, py::object exclude) {
    Points result;
    for (auto item : observation["robots"]) {
        auto scan = py::cast<py::dict>(py::cast<py::dict>(item)["scan"]);
        Point origin = py::cast<Point>(scan["origin"]);
        auto angles = py::cast<py::sequence>(scan["ray_angles"]), distances = py::cast<py::sequence>(scan["distances"]);
        auto valid = py::cast<py::sequence>(scan["valid"]), category = py::cast<py::sequence>(scan["category"]), ids = py::cast<py::sequence>(scan["entity_ids"]);
        auto size = std::min({py::len(angles), py::len(distances), py::len(valid), py::len(category), py::len(ids)});
        for (py::size_t i = 0; i < size; ++i) {
            if (!geometry_truth(valid[i]) || geometry_equal(category[i], py::int_(3))) continue;
            if (!exclude.is_none() && ((id != 0 && !geometry_equal(category[i], py::int_(2))) || geometry_equal(ids[i], exclude))) continue;
            double d = py::cast<double>(distances[i]);
            if (d < 4.95) { double a = py::cast<double>(angles[i]); result.push_back({origin[0] + d * std::cos(a), origin[1] + d * std::sin(a)}); }
        }
    }
    return result;
}

inline Point route(int id, Point start, Point goal, const Points& points, double radius, Point arena) {
    if (id != 0) {
        double dx = goal[0] - start[0], dy = goal[1] - start[1], dd = dx * dx + dy * dy;
        bool blocked = false;
        if (dd > .001) for (auto p : points) {
            double t = ((p[0] - start[0]) * dx + (p[1] - start[1]) * dy) / dd;
            double ex = p[0] - start[0] - t * dx, ey = p[1] - start[1] - t * dy;
            if (.08 < t && t < .98 && ex * ex + ey * ey < radius * radius) { blocked = true; break; }
        }
        if (!blocked) return goal;
    }
    constexpr double step = .18;
    int nx = static_cast<int>(arena[0] / step) + 1, ny = static_cast<int>(arena[1] / step) + 1;
    using Cell = std::pair<int, int>;
    auto cell = [&](Point p) -> Cell { return {std::max(1, std::min(nx - 2, geometry_round(p[0] / step))), std::max(1, std::min(ny - 2, geometry_round(p[1] / step)))}; };
    Cell startcell = cell(start), dest = cell(goal);
    const auto original_dest = dest;
    std::set<Cell> occupied;
    int n = static_cast<int>(std::ceil(radius / step));
    for (auto p : points) {
        int ix = geometry_round(p[0] / step), iy = geometry_round(p[1] / step);
        for (int dx = -n; dx <= n; ++dx) for (int dy = -n; dy <= n; ++dy) {
            double ex = (ix + dx) * step - p[0], ey = (iy + dy) * step - p[1];
            if (ex * ex + ey * ey < radius * radius) occupied.emplace(ix + dx, iy + dy);
        }
    }
    for (int dx = -2; dx <= 2; ++dx) for (int dy = -2; dy <= 2; ++dy)
        if (dx * dx + dy * dy <= 4) occupied.erase({startcell.first + dx, startcell.second + dy});
    if (occupied.count(dest)) {
        std::tuple<double, Cell> best{std::numeric_limits<double>::infinity(), dest};
        for (int x = std::max(1, dest.first - 4); x < std::min(nx - 1, dest.first + 5); ++x)
            for (int y = std::max(1, dest.second - 4); y < std::min(ny - 1, dest.second + 5); ++y)
                if (!occupied.count({x, y})) best = std::min(best, std::make_tuple(geometry_hypot(x - original_dest.first, y - original_dest.second), Cell{x, y}));
        if (std::isfinite(std::get<0>(best))) dest = std::get<1>(best);
    }
    using Entry = std::tuple<double, double, Cell>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
    queue.emplace(0., 0., startcell);
    std::map<Cell, double> cost{{startcell, 0.}};
    std::map<Cell, Cell> previous;
    Cell last = startcell;
    constexpr std::array<Cell, 8> directions{{{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1},{0,-1},{1,-1}}};
    for (int iteration = 0; iteration < nx * ny; ++iteration) {
        if (queue.empty()) break;
        auto [f, g, node] = queue.top(); queue.pop();
        auto found = cost.find(node);
        if (g > (found == cost.end() ? 1e9 : found->second)) continue;
        last = node;
        if (node == dest) break;
        for (auto [dx, dy] : directions) {
            Cell next{node.first + dx, node.second + dy};
            if (next.first < 1 || next.first >= nx - 1 || next.second < 1 || next.second >= ny - 1 || occupied.count(next)) continue;
            if (dx && dy && (occupied.count({node.first + dx, node.second}) || occupied.count({node.first, node.second + dy}))) continue;
            double ng = g + geometry_hypot(dx, dy);
            auto old = cost.find(next);
            if (ng < (old == cost.end() ? 1e9 : old->second)) {
                cost[next] = ng; previous[next] = node;
                queue.emplace(ng + geometry_hypot(next.first - dest.first, next.second - dest.second), ng, next);
            }
        }
    }
    if (last != dest) return goal;
    std::vector<Cell> path{last};
    while (path.back() != startcell && previous.count(path.back())) path.push_back(previous.at(path.back()));
    std::reverse(path.begin(), path.end());
    if (path.size() <= 2) return goal;
    auto p = path[std::min<std::size_t>(3, path.size() - 1)];
    return {p.first * step, p.second * step};
}

inline int move_action(int id, py::dict robot, Point goal, const Points& points, Point arena) {
    auto own = py::cast<py::dict>(robot["own"]);
    Point p{py::cast<double>(own["x"]), py::cast<double>(own["y"])};
    double distance = geometry_hypot(goal[0] - p[0], goal[1] - p[1]);
    if (distance < .12) return 44;
    auto way = route(id, p, goal, points, .30, arena);
    auto velocity = norm(way[0] - p[0], way[1] - p[1]);
    int direction = geometry_direction(velocity[0], velocity[1]);
    auto scan = py::cast<py::dict>(robot["scan"]);
    auto angles = py::cast<py::sequence>(scan["ray_angles"]), distances = py::cast<py::sequence>(scan["distances"]), valid = py::cast<py::sequence>(scan["valid"]);
    auto size = std::min({py::len(angles), py::len(distances), py::len(valid)});
    using Clearance = std::tuple<double, int, double>;
    std::array<Clearance, 8> clear;
    for (int j = 0; j < 8; ++j) {
        double angle = j * geometry_pi / 4., clearance = 5.; bool any = false;
        for (py::size_t k = 0; k < size; ++k) if (geometry_truth(valid[k]) && std::cos(py::cast<double>(angles[k]) - angle) > .9) {
            double d = py::cast<double>(distances[k]); clearance = any ? std::min(clearance, d) : d; any = true;
        }
        double cx = std::cos(angle), sy = std::sin(angle);
        if (id == 5) {
            for (auto point : points) {
                double rx = point[0] - p[0], ry = point[1] - p[1];
                double along = rx * cx + ry * sy, across = std::abs(-rx * sy + ry * cx);
                if (along > .025 && across < .29)
                    clearance = std::min(clearance, along - std::sqrt(std::max(0., .29 * .29 - across * across)));
            }
            if (cx > .001) clearance = std::min(clearance, (arena[0] - .26 - p[0]) / cx);
            if (cx < -.001) clearance = std::min(clearance, (.26 - p[0]) / cx);
            if (sy > .001) clearance = std::min(clearance, (arena[1] - .26 - p[1]) / sy);
            if (sy < -.001) clearance = std::min(clearance, (.26 - p[1]) / sy);
        }
        double score = cx * velocity[0] + sy * velocity[1];
        score -= std::max(0., .50 - clearance) * 10.;
        clear[j] = {score, j, clearance};
    }
    auto best = *std::max_element(clear.begin(), clear.end());
    if (std::get<2>(clear[direction]) < .60) direction = std::get<1>(best);
    if (geometry_equal(geometry_get(py::cast<py::dict>(robot["execution"]), "stage"), py::str("move_blocked"))) {
        std::pair<double, int> choice{-std::numeric_limits<double>::infinity(), -1};
        for (auto [score, j, clearance] : clear) if (id == 5 || j != direction)
            choice = std::max(choice, std::make_pair(std::min(clearance, 1.1) + (id == 5 ? .12 : .20) * score, j));
        direction = choice.second;
    }
    return 47 + 4 * direction + (distance > .4 ? 3 : 1);
}
} // namespace rc
