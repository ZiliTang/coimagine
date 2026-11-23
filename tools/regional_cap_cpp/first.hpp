#pragma once
#include "geometry.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace rc {
namespace first_detail {
constexpr double pi = 3.14159265358979323846;
inline bool truth(py::handle x) {
    int r = PyObject_IsTrue(x.ptr());
    if (r < 0)
        throw py::error_already_set();
    return r != 0;
}
inline py::object get(py::dict d, const char *key, py::object fallback = py::none()) {
    return d.contains(key) ? py::reinterpret_borrow<py::object>(d[key]) : fallback;
}
inline double num(py::dict d, const char *key, double fallback = 0.) {
    return d.contains(key) ? py::cast<double>(d[key]) : fallback;
}
inline int integer(py::dict d, const char *key, int fallback = 0) {
    return d.contains(key) ? py::cast<int>(d[key]) : fallback;
}
inline Point point(py::handle p) {
    auto s = py::reinterpret_borrow<py::sequence>(p);
    return {py::cast<double>(s[0]), py::cast<double>(s[1])};
}
inline py::list pair_list(int a, int b) {
    py::list q;
    q.append(a);
    q.append(b);
    return q;
}
inline bool legal_pair(py::handle legal, int a, int b) {
    auto q = pair_list(a, b);
    int r = PySequence_Contains(legal.ptr(), q.ptr());
    if (r < 0)
        throw py::error_already_set();
    return r;
}
inline bool stage(py::dict execution, std::initializer_list<const char *> stages) {
    auto s = get(execution, "stage");
    for (auto item : stages)
        if (s.equal(py::str(item)))
            return true;
    return false;
}
inline bool poor(py::dict robot, bool include_wait) {
    auto e = py::cast<py::dict>(robot["execution"]);
    return stage(e, {"find", "view_insufficient", "blocked", "peer_wait"}) ||
           (include_wait && stage(e, {"wait"})) || truth(get(e, "exit_reason"));
}
inline double mod(double a, double b) {
    double v = std::fmod(a, b);
    return v < 0 ? v + b : v;
}
inline int direction(double y, double x) {
    int d = static_cast<int>(std::nearbyint(std::atan2(y, x) / (pi / 4)));
    return (d % 8 + 8) % 8;
}
inline py::dict proposal(py::object target, int a0, int a1, int ticks) {
    py::dict d;
    d["target_id"] = target;
    d["a0"] = a0;
    d["a1"] = a1;
    d["ticks"] = ticks;
    return d;
}
inline py::dict result(py::dict p, py::list state) {
    py::dict d;
    py::list v;
    v.append(py::float_(py::cast<double>(p["ticks"])));
    py::list ps;
    ps.append(p);
    d["values"] = v;
    d["state"] = state;
    d["proposals"] = ps;
    return d;
}
inline py::dict compute(int id, py::dict m, py::dict o) {
    auto robots = py::cast<py::list>(o["robots"]);
    auto r0 = py::cast<py::dict>(robots[0]);
    Point arena = point(r0["arena"]);
    auto goal_dict = py::cast<py::dict>(o["goal"]);
    auto target = py::reinterpret_borrow<py::object>(goal_dict["target_id"]);
    Point goal = point(goal_dict["xy"]);
    py::object center = estimate(id, m, o);
    if (center.is_none()) {
        py::list s;
        s.append(py::none());
        s.append(0.);
        s.append(0.);
        for (int i = 0; i < 5; ++i)
            s.append(py::none());
        return result(proposal(py::none(), 50, 66, 40), s);
    }
    Point c = point(center);
    double dist = geometry_hypot(goal[0] - c[0], goal[1] - c[1]);
    int contacts = 0;
    for (auto rh : robots)
        contacts += truth(py::cast<py::dict>(rh)["contacting_object"]);
    bool stalled = false;
    auto last = get(m, "last");
    if (truth(last)) {
        auto ctx = py::cast<py::dict>(get(m, "context", py::dict()));
        py::list default_ps;
        default_ps.append(py::dict());
        auto ps = py::cast<py::list>(get(ctx, "proposals", default_ps));
        auto p = py::cast<py::dict>(ps[0]);
        if (get(p, "target_id").equal(target)) {
            auto lp = point(last);
            stalled = geometry_hypot(c[0] - lp[0], c[1] - lp[1]) < .015;
        }
    }
    m["stall"] = stalled ? integer(m, "stall") + 1 : 0;
    py::list last_copy;
    last_copy.append(c[0]);
    last_copy.append(c[1]);
    m["last"] = last_copy;
    Points stat = obstacles(id, o, target);
    Point way = id == 0 ? goal : route(id, c, goal, stat, .30, arena);
    Point vel = norm(way[0] - c[0], way[1] - c[1]);
    double vx = vel[0], vy = vel[1];
    int desired = direction(vy, vx);
    if (dist < .55)
        desired = direction(goal[1] - c[1], goal[0] - c[0]);
    const int tick = integer(m, "tick");
    if (id == 4 && integer(m, "stall") >= 3 && tick >= integer(m, "avoid_until")) {
        bool unavailable = false;
        for (auto rh : robots) {
            auto e = py::cast<py::dict>(py::cast<py::dict>(rh)["execution"]);
            unavailable = unavailable || stage(e, {"find", "view_insufficient"}) ||
                          get(e, "direction_unavailable").is(py::bool_(true));
        }
        if (unavailable) {
            m["avoid_dir"] = m["dir"];
            m["avoid_until"] = tick + 120;
        }
    }
    double ang = num(m, "ang");
    Point dims = point(m["dims"]);
    bool feasible = false;
    double best = -std::numeric_limits<double>::infinity();
    for (int d = 0; d < 8; ++d) {
        if (id == 4 && d == integer(m, "avoid_dir", -1) && tick < integer(m, "avoid_until"))
            continue;
        double aa = d * pi / 4 - ang;
        if (std::max(std::abs(std::cos(aa)), std::abs(std::sin(aa))) >= std::cos(.60)) {
            double score = std::cos(d * pi / 4) * vx + std::sin(d * pi / 4) * vy;
            if (d == integer(m, "dir"))
                score += .12;
            if (!feasible || score >= best) {
                best = score;
                desired = d;
                feasible = true;
            }
        }
    }
    m["dir"] = desired;
    int speed = dist > .8 ? 3 : (dist > .4 ? 1 : 0);
    int a = 4 * desired + speed;
    auto legal_dict = py::cast<py::dict>(o["entity_legal_pairs"]);
    auto legal = legal_dict.contains(target)
                     ? py::reinterpret_borrow<py::object>(legal_dict[target])
                     : py::object(py::list());
    int a0 = a, a1 = a;
    double ux = std::cos(desired * pi / 4), uy = std::sin(desired * pi / 4);
    double transverse = std::abs(-uy * std::cos(ang) + ux * std::sin(ang)) * dims[0] +
                        std::abs(uy * std::sin(ang) + ux * std::cos(ang)) * dims[1];
    bool weak = false;
    for (auto rh : robots)
        weak = weak || poor(py::cast<py::dict>(rh), true);
    weak = weak && integer(m, "stall") >= 2;
    if (transverse < .48 || !legal_pair(legal, a0, a1) || weak) {
        bool found = false;
        double score_best = std::numeric_limits<double>::infinity();
        for (int i = 0; i < 2; ++i) {
            int q0 = i == 0 ? a : 46, q1 = i == 1 ? a : 46;
            if (legal_pair(legal, q0, q1)) {
                auto r = py::cast<py::dict>(robots[i]);
                auto own = py::cast<py::dict>(r["own"]);
                double rear = (num(own, "x") - c[0]) * ux + (num(own, "y") - c[1]) * uy;
                double score =
                    rear - (truth(r["contacting_object"]) ? .7 : 0.) + (poor(r, false) ? 4. : 0.);
                if (!found || score < score_best) {
                    found = true;
                    score_best = score;
                    a0 = q0;
                    a1 = q1;
                }
            }
        }
    }
    bool done = false;
    for (int i = 0; i < 2; ++i)
        if ((i == 0 ? a0 : a1) != 46)
            done = done || truth(get(py::cast<py::dict>(py::cast<py::dict>(robots[i])["execution"]),
                                     "done"));
    py::dict p;
    if (legal_pair(legal, a0, a1) && !(done || integer(m, "stall") > 6))
        p = proposal(target, a0, a1, dist > .8 ? 40 : 15);
    else {
        double tx = -uy, ty = ux;
        double extent =
            std::max(.2, .5 * (std::abs(ux * std::cos(ang) + uy * std::sin(ang)) * dims[0] +
                               std::abs(-ux * std::sin(ang) + uy * std::cos(ang)) * dims[1]));
        double behind = extent + .36;
        Point goals[2];
        Point positions[2];
        for (int i = 0; i < 2; ++i) {
            double s = i == 0 ? -.33 : .33;
            goals[i] = {std::max(.30, std::min(arena[0] - .30, c[0] - ux * behind + tx * s)),
                        std::max(.30, std::min(arena[1] - .30, c[1] - uy * behind + ty * s))};
            auto own = py::cast<py::dict>(py::cast<py::dict>(robots[i])["own"]);
            positions[i] = {num(own, "x"), num(own, "y")};
        }
        auto distance = [](Point a, Point b) { return geometry_hypot(a[0] - b[0], a[1] - b[1]); };
        if (distance(positions[0], goals[1]) + distance(positions[1], goals[0]) <
            distance(positions[0], goals[0]) + distance(positions[1], goals[1]))
            std::swap(goals[0], goals[1]);
        Points allpts = obstacles(id, o, py::none());
        int acts[2];
        for (int i = 0; i < 2; ++i)
            acts[i] = move_action(id, py::cast<py::dict>(robots[i]), goals[i], allpts, arena);
        if (acts[0] == 44 && acts[1] == 44) {
            acts[0] = 47 + 4 * ((desired + 2) % 8) + 1;
            acts[1] = 47 + 4 * ((desired + 6) % 8) + 1;
            m["stall"] = 0;
        }
        p = proposal(py::none(), acts[0], acts[1], 25);
    }
    bool any_done = false;
    for (auto rh : robots)
        any_done =
            any_done || truth(get(py::cast<py::dict>(py::cast<py::dict>(rh)["execution"]), "done"));
    if (id == 4) {
        bool found = false;
        double bs = -10.;
        int bp0 = 0, bp1 = 0;
        for (auto pair_h : py::reinterpret_borrow<py::iterable>(legal)) {
            auto pair = py::reinterpret_borrow<py::sequence>(pair_h);
            int p0 = py::cast<int>(pair[0]), p1 = py::cast<int>(pair[1]);
            if (p0 < 0 || p1 < 0 || p0 >= 32 || p1 >= 32 || p0 % 4 != speed || p1 % 4 != speed)
                continue;
            int d0 = p0 / 4, d1 = p1 / 4;
            if (d0 == d1)
                continue;
            double x0 = std::cos(d0 * pi / 4), y0 = std::sin(d0 * pi / 4),
                   x1 = std::cos(d1 * pi / 4), y1 = std::sin(d1 * pi / 4);
            if (x0 * x1 + y0 * y1 < -.05 || x0 * vx + y0 * vy < .20 || x1 * vx + y1 * vy < .20)
                continue;
            auto n = norm(x0 + x1, y0 + y1);
            double score = n[0] * vx + n[1] * vy - .04;
            if (score > bs) {
                bs = score;
                bp0 = p0;
                bp1 = p1;
                found = true;
            }
        }
        double current = std::cos(desired * pi / 4) * vx + std::sin(desired * pi / 4) * vy;
        if (found && bs > current + .03 && !any_done)
            p = proposal(target, bp0, bp1, dist > .8 ? 40 : 15);
    }
    if (id >= 2) {
        if (p["target_id"].is_none() && integer(m, "nav_ticks") > 100) {
            int actions[2];
            for (int i = 0; i < 2; ++i) {
                auto own = py::cast<py::dict>(py::cast<py::dict>(robots[i])["own"]);
                double robot_ang = std::atan2(num(own, "y") - c[1], num(own, "x") - c[0]);
                double rear = std::atan2(-uy, -ux);
                double delta = mod(rear - robot_ang + pi, 2 * pi) - pi;
                actions[i] = delta > 0 ? 40 : 43;
            }
            if (legal_pair(legal, actions[0], actions[1]) && !any_done)
                p = proposal(target, actions[0], actions[1], 40);
            if (integer(m, "nav_ticks") > 300)
                m["nav_ticks"] = 0;
        }
        if (dist < .23 && tick - integer(m, "goal_brake", -100) > 55) {
            m["goal_brake"] = m["tick"];
            p = proposal(py::none(), 44, 44, 30);
        }
    }
    if (id == 3) {
        if (!m.contains("progress_dist")) {
            m["progress_dist"] = dist;
            m["progress_tick"] = m["tick"];
            m["rotate_until"] = 0;
            m["rotate_cooldown"] = 0;
        }
        if (dist < num(m, "progress_dist") - .07) {
            m["progress_dist"] = dist;
            m["progress_tick"] = m["tick"];
        }
        if (tick - integer(m, "progress_tick") > 200 && tick >= integer(m, "rotate_cooldown") &&
            dist > .35) {
            double long_angle = ang + (dims[1] > dims[0] ? pi / 2 : 0.);
            double desired_angle = std::atan2(goal[1] - c[1], goal[0] - c[0]) + pi / 2;
            double err = mod(desired_angle - long_angle + pi / 2, pi) - pi / 2;
            m["rotate_action"] = err >= 0 ? 34 : 37;
            m["rotate_until"] = tick + 200;
            m["rotate_cooldown"] = tick + 440;
            m["progress_dist"] = dist;
            m["progress_tick"] = m["tick"];
        }
        int ra = integer(m, "rotate_action", 34);
        if (tick < integer(m, "rotate_until") && legal_pair(legal, ra, ra))
            p = proposal(target, ra, ra, 40);
    }
    py::list state;
    state.append(dist);
    state.append(double(contacts));
    state.append(double(integer(m, "stall")));
    state.append(c[0]);
    state.append(c[1]);
    state.append(double(desired));
    state.append(dims[0]);
    state.append(dims[1]);
    return result(p, state);
}
} // namespace first_detail
inline py::dict first_initial(int id) {
    if (id < 0 || id > 4)
        throw py::value_error("first policy id out of range");
    py::dict m;
    m["tick"] = 0;
    m["c"] = py::none();
    py::list dims;
    dims.append(0.);
    dims.append(0.);
    m["dims"] = dims;
    m["ang"] = 0.;
    m["stall"] = 0;
    m["last"] = py::none();
    m["dir"] = -1;
    m["nav"] = 0;
    return m;
}
inline py::dict first_update(int id, py::dict m, py::dict e) {
    using namespace first_detail;
    auto ticks = e["executed_ticks"];
    int dt = truth(ticks) ? py::cast<int>(ticks) : 0;
    m["tick"] = integer(m, "tick") + dt;
    if (id >= 2) {
        auto last = get(e, "action");
        bool has = truth(last);
        bool edge = false;
        py::dict action;
        if (has) {
            action = py::cast<py::dict>(last);
            auto a0 = get(action, "a0"), a1 = get(action, "a1");
            edge = a0.equal(py::int_(40)) || a0.equal(py::int_(43)) || a1.equal(py::int_(40)) ||
                   a1.equal(py::int_(43));
        }
        if (has && (get(action, "target_id").is_none() || (id == 4 && edge)))
            m["nav_ticks"] = integer(m, "nav_ticks") + dt;
        else if (!has || !edge)
            m["nav_ticks"] = 0;
    }
    m["context"] = compute(id, m, py::cast<py::dict>(e["observation"]));
    return m;
}
} // namespace rc
