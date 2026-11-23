#pragma once
#include "geometry.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>
namespace rc {
namespace next_detail {
inline py::object get(py::dict d, const char *key, py::object def = py::none()) {
    return d.contains(key) ? py::reinterpret_borrow<py::object>(d[key]) : def;
}
inline double num(py::dict d, const char *key, double def = 0) {
    auto x = get(d, key);
    return x.is_none() ? def : py::cast<double>(x);
}
inline bool truth(py::handle x) {
    return PyObject_IsTrue(x.ptr()) == 1;
}
inline bool eq(py::handle a, py::handle b) {
    return a.equal(b);
}
inline py::dict dict_or_empty(py::object x) {
    return truth(x) ? py::cast<py::dict>(x) : py::dict();
}
inline Point point(py::handle x) {
    auto s = py::reinterpret_borrow<py::sequence>(x);
    return {py::cast<double>(s[0]), py::cast<double>(s[1])};
}
inline py::list plist(Point p) {
    py::list l;
    l.append(p[0]);
    l.append(p[1]);
    return l;
}
inline py::list pairlist(std::array<int, 2> p) {
    py::list l;
    l.append(p[0]);
    l.append(p[1]);
    return l;
}
inline bool legal_has(py::object menu, std::array<int, 2> p) {
    return PySequence_Contains(menu.ptr(), pairlist(p).ptr()) == 1;
}
inline bool stage_is(py::dict r, std::initializer_list<const char *> names) {
    auto e = py::cast<py::dict>(r["execution"]);
    auto s = get(e, "stage");
    for (auto name : names)
        if (eq(s, py::str(name)))
            return true;
    return false;
}
inline bool exited(py::dict r) {
    return truth(get(py::cast<py::dict>(r["execution"]), "exit_reason"));
}
inline bool done(py::dict r) {
    return truth(get(py::cast<py::dict>(r["execution"]), "done"));
}
inline int direction(double y, double x) {
    int d = geometry_round(std::atan2(y, x) / (M_PI / 4));
    return (d % 8 + 8) % 8;
}
inline py::dict proposal(py::object target, int a0, int a1, int ticks) {
    py::dict p;
    p["target_id"] = target;
    p["a0"] = a0;
    p["a1"] = a1;
    p["ticks"] = ticks;
    return p;
}
inline double hypot(Point a, Point b) {
    return geometry_hypot(a[0] - b[0], a[1] - b[1]);
}
inline py::object secondary_clear(py::dict m, py::dict o, Point c, Point goal) {
    py::dict gd = py::cast<py::dict>(o["goal"]);
    py::object target = gd["target_id"];
    double dist = hypot(goal, c);
    Point g = norm(goal[0] - c[0], goal[1] - c[1]);
    py::dict visible;
    for (auto h : py::cast<py::list>(o["entities"])) {
        auto e = py::cast<py::dict>(h);
        auto pts = py::cast<py::list>(e["points"]);
        if (!eq(e["id"], target) && pts.size()) {
            Point lo = point(pts[0]), hi = lo;
            for (auto p : pts) {
                auto q = point(p);
                for (int k = 0; k < 2; k++) {
                    lo[k] = std::min(lo[k], q[k]);
                    hi[k] = std::max(hi[k], q[k]);
                }
            }
            visible[e["id"]] = plist({.5 * (lo[0] + hi[0]), .5 * (lo[1] + hi[1])});
        }
    }
    double secondary = 9.;
    bool first = true;
    for (auto item : visible) {
        double d = hypot(point(item.second), c);
        if (first || d < secondary)
            secondary = d;
        first = false;
    }
    m["secondary_distance"] = secondary;
    if (!m.contains("clear_progress")) {
        m["clear_progress"] = dist;
        m["clear_progress_tick"] = m["tick"];
        m["clear_until"] = 0;
        m["clear_after"] = 0;
    }
    if (dist < num(m, "clear_progress") - .08) {
        m["clear_progress"] = dist;
        m["clear_progress_tick"] = m["tick"];
    }
    double tick = num(m, "tick");
    if (tick >= num(m, "clear_after") && tick - num(m, "clear_progress_tick") > 180 && dist > .65) {
        py::object best = py::none();
        double bestdist = 0, bestside = 0;
        for (auto item : visible) {
            Point p = point(item.second);
            double dx = p[0] - c[0], dy = p[1] - c[1], along = dx * g[0] + dy * g[1],
                   side = -dx * g[1] + dy * g[0], d = geometry_hypot(dx, dy);
            if (along > -.8 && along < std::min(dist, 1.4) && std::abs(side) < .65 && d < 1.5) {
                bool smaller = best.is_none() || d < bestdist;
                if (!smaller && d == bestdist) {
                    int cmp = PyObject_RichCompareBool(item.first.ptr(), best.ptr(), Py_LT);
                    smaller = cmp == 1 || (eq(item.first, best) && side < bestside);
                }
                if (smaller) {
                    best = py::reinterpret_borrow<py::object>(item.first);
                    bestdist = d;
                    bestside = side;
                }
            }
        }
        if (!best.is_none()) {
            m["clear_entity"] = best;
            m["clear_side"] = bestside >= 0 ? 1 : -1;
            m["clear_until"] = py::int_(static_cast<long long>(tick + 240));
            m["clear_after"] = py::int_(static_cast<long long>(tick + 600));
            m["clear_progress"] = dist;
            m["clear_progress_tick"] = m["tick"];
            m["clear_failed"] = pairlist({0, 0});
        }
    }
    if (dist < .35) {
        m["clear_until"] = 0;
        return py::none();
    }
    if (tick >= num(m, "clear_until"))
        return py::none();
    auto eid = get(m, "clear_entity");
    if (!visible.contains(eid))
        return py::none();
    auto failed = py::cast<py::list>(get(m, "clear_failed", pairlist({0, 0})));
    auto last = dict_or_empty(get(m, "last_request"));
    auto robots = py::cast<py::list>(o["robots"]);
    if (eq(get(o, "active_target"), eid) && eq(get(last, "target_id"), eid))
        for (int i = 0; i < 2; i++) {
            std::string key = "a" + std::to_string(i);
            double a = num(last, key.c_str(), -1);
            auto r = py::cast<py::dict>(robots[i]);
            if (a >= 0 && a < 32 && (done(r) || exited(r)))
                failed[i] = 1;
        }
    m["clear_failed"] = failed;
    Point p = point(visible[eid]);
    double sx = -g[1] * num(m, "clear_side"), sy = g[0] * num(m, "clear_side");
    Point arena = point(py::cast<py::dict>(robots[0])["arena"]);
    if (p[0] + .7 * sx < .4 || p[0] + .7 * sx > arena[0] - .4 || p[1] + .7 * sy < .4 ||
        p[1] + .7 * sy > arena[1] - .4) {
        sx = -sx;
        sy = -sy;
    }
    auto menus = py::cast<py::dict>(o["entity_legal_pairs"]);
    py::object menu =
        menus.contains(eid) ? py::reinterpret_borrow<py::object>(menus[eid]) : py::list();
    bool found = false;
    double bestscore = 0;
    std::array<int, 2> bestpair{};
    for (int i = 0; i < 2; i++) {
        if (truth(failed[i]))
            continue;
        auto own = py::cast<py::dict>(py::cast<py::dict>(robots[i])["own"]);
        double reach = geometry_hypot(num(own, "x") - p[0], num(own, "y") - p[1]);
        for (int d = 0; d < 8; d++) {
            std::array<int, 2> q{46, 46};
            q[i] = 4 * d + 3;
            if (legal_has(menu, q)) {
                double align = std::cos(d * M_PI / 4) * sx + std::sin(d * M_PI / 4) * sy;
                if (align > .30) {
                    double score = align - .08 * reach;
                    if (!found || score > bestscore || (score == bestscore && q > bestpair)) {
                        found = true;
                        bestscore = score;
                        bestpair = q;
                    }
                }
            }
        }
    }
    if (!found) {
        m["clear_until"] = 0;
        return py::none();
    }
    return proposal(eid, bestpair[0], bestpair[1], 40);
}
inline py::dict compute(int id, py::dict m, py::dict o) {
    auto robots = py::cast<py::list>(o["robots"]);
    auto r0 = py::cast<py::dict>(robots[0]);
    auto r1 = py::cast<py::dict>(robots[1]);
    std::array<py::dict, 2> rs{r0, r1};
    Point arena = point(r0["arena"]);
    auto gd = py::cast<py::dict>(o["goal"]);
    py::object target = gd["target_id"];
    Point goal = point(gd["xy"]);
    auto co = estimate(id, m, o);
    py::dict result;
    py::list values, state, proposals;
    if (co.is_none()) {
        values.append(40.);
        state.append(py::none());
        state.append(0.);
        state.append(0.);
        for (int k = 0; k < 5; k++)
            state.append(py::none());
        if (id == 8) {
            state.append(py::none());
            state.append(0.);
        }
        proposals.append(proposal(py::none(), 50, 66, 40));
        result["values"] = values;
        result["state"] = state;
        result["proposals"] = proposals;
        return result;
    }
    Point c = point(co);
    double dist = hypot(goal, c),
           contacts = int(truth(r0["contacting_object"])) + int(truth(r1["contacting_object"]));
    bool prior_target = false;
    auto context = get(m, "context", py::dict());
    auto ctx = py::cast<py::dict>(context);
    auto pp = get(ctx, "proposals");
    if (!pp.is_none()) {
        auto ps = py::cast<py::list>(pp);
        if (ps.size())
            prior_target = eq(get(py::cast<py::dict>(ps[0]), "target_id"), target);
    }
    auto last = get(m, "last");
    if (truth(last) && prior_target && hypot(c, point(last)) < .015)
        m["stall"] = py::int_(static_cast<long long>(num(m, "stall") + 1));
    else
        m["stall"] = 0;
    m["last"] = plist(c);
    auto staticpts = obstacles(id, o, target);
    Point way = route(id, c, goal, staticpts, id == 9 ? .56 : .30, arena);
    Point v = norm(way[0] - c[0], way[1] - c[1]);
    int desired = direction(v[1], v[0]);
    if (dist < .55) {
        if (id == 9)
            v = norm(goal[0] - c[0], goal[1] - c[1]);
        desired = direction(goal[1] - c[1], goal[0] - c[0]);
    }
    double tick = num(m, "tick"), stall = num(m, "stall"), ang = num(m, "ang");
    bool unavailable = false;
    for (auto r : rs) {
        auto e = py::cast<py::dict>(r["execution"]);
        if (stage_is(r, {"find", "view_insufficient"}) ||
            get(e, "direction_unavailable").is(py::bool_(true)))
            unavailable = true;
    }
    if (stall >= 3 && tick >= num(m, "avoid_until") && unavailable) {
        m["avoid_dir"] = m["dir"];
        m["avoid_until"] = py::int_(static_cast<long long>(tick + 120));
    }
    bool has_feasible = false;
    double best = -1e99;
    for (int d = 0; d < 8; d++) {
        if (d == num(m, "avoid_dir", -1) && tick < num(m, "avoid_until"))
            continue;
        double aa = d * M_PI / 4 - ang;
        if (std::max(std::abs(std::cos(aa)), std::abs(std::sin(aa))) >= std::cos(.60)) {
            double score = std::cos(d * M_PI / 4) * v[0] + std::sin(d * M_PI / 4) * v[1];
            if (d == num(m, "dir"))
                score += .12;
            if (!has_feasible || score > best || (score == best && d > desired)) {
                desired = d;
                best = score;
                has_feasible = true;
            }
        }
    }
    m["dir"] = desired;
    int speed = dist > .8 ? 3 : (dist > .4 ? 1 : 0), a = 4 * desired + speed;
    auto menus = py::cast<py::dict>(o["entity_legal_pairs"]);
    py::object legal =
        menus.contains(target) ? py::reinterpret_borrow<py::object>(menus[target]) : py::list();
    std::array<int, 2> pair{a, a};
    double ux = std::cos(desired * M_PI / 4), uy = std::sin(desired * M_PI / 4);
    Point dims = point(m["dims"]);
    double transverse = std::abs(-uy * std::cos(ang) + ux * std::sin(ang)) * dims[0] +
                        std::abs(uy * std::sin(ang) + ux * std::cos(ang)) * dims[1];
    bool weak = false;
    for (auto r : rs)
        if (stage_is(r, {"find", "view_insufficient", "blocked", "wait", "peer_wait"}) || exited(r))
            weak = true;
    weak = weak && stall >= 2;
    bool solo = id == 7 || id == 9;
    if (solo || transverse < .48 || !legal_has(legal, pair) || weak) {
        bool found = false;
        double bestscore = 0;
        for (int i = 0; i < 2; i++) {
            std::array<int, 2> q{46, 46};
            q[i] = a;
            if (!legal_has(legal, q))
                continue;
            auto own = py::cast<py::dict>(rs[i]["own"]);
            double ox = num(own, "x"), oy = num(own, "y"),
                   rear = (ox - c[0]) * ux + (oy - c[1]) * uy;
            double score = rear - (truth(rs[i]["contacting_object"]) ? .7 : 0.);
            if (solo) {
                auto prior = dict_or_empty(get(m, "last_request"));
                std::string key = "a" + std::to_string(i);
                double pa = num(prior, key.c_str(), -1);
                bool active = eq(get(prior, "target_id"), target) && pa >= 0 && pa < 32;
                score = geometry_hypot(ox - c[0], oy - c[1]) + .15 * rear -
                        (truth(rs[i]["contacting_object"]) ? 1.2 : 0.) - (active ? .40 : 0.);
            }
            if (stage_is(rs[i], {"find", "view_insufficient", "blocked", "peer_wait"}) ||
                exited(rs[i]))
                score += 4.;
            if (!found || score < bestscore) {
                bestscore = score;
                pair = q;
                found = true;
            }
        }
    }
    bool active_done = false;
    for (int i = 0; i < 2; i++)
        if (pair[i] != 46 && done(rs[i]))
            active_done = true;
    py::dict prop;
    if (legal_has(legal, pair) && !(active_done || stall > 6))
        prop = proposal(target, pair[0], pair[1], dist > .8 ? 40 : 15);
    else {
        double tx = -uy, ty = ux,
               extent = std::max(
                   .2, .5 * (std::abs(ux * std::cos(ang) + uy * std::sin(ang)) * dims[0] +
                             std::abs(-ux * std::sin(ang) + uy * std::cos(ang)) * dims[1])),
               behind = extent + .36;
        std::array<Point, 2> goals;
        std::array<Point, 2> positions;
        for (int i = 0; i < 2; i++) {
            double s = i == 0 ? -.33 : .33;
            goals[i] = {std::max(.30, std::min(arena[0] - .30, c[0] - ux * behind + tx * s)),
                        std::max(.30, std::min(arena[1] - .30, c[1] - uy * behind + ty * s))};
            auto own = py::cast<py::dict>(rs[i]["own"]);
            positions[i] = {num(own, "x"), num(own, "y")};
        }
        if (hypot(positions[0], goals[1]) + hypot(positions[1], goals[0]) <
            hypot(positions[0], goals[0]) + hypot(positions[1], goals[1]))
            std::swap(goals[0], goals[1]);
        auto allpts = obstacles(id, o, py::none());
        std::array<int, 2> acts{move_action(id, r0, goals[0], allpts, arena),
                                move_action(id, r1, goals[1], allpts, arena)};
        if (acts[0] == 44 && acts[1] == 44) {
            acts = {47 + 4 * ((desired + 2) % 8) + 1, 47 + 4 * ((desired + 6) % 8) + 1};
            m["stall"] = 0;
        }
        prop = proposal(py::none(), acts[0], acts[1], 25);
    }
    bool any_done = done(r0) || done(r1);
    if (!solo) {
        bool found = false;
        double bestscore = -10.;
        std::array<int, 2> bestpair;
        for (auto ph : py::reinterpret_borrow<py::iterable>(legal)) {
            auto p = py::reinterpret_borrow<py::sequence>(ph);
            int p0 = py::cast<int>(p[0]), p1 = py::cast<int>(p[1]);
            if (p0 < 0 || p1 < 0 || p0 >= 32 || p1 >= 32 || p0 % 4 != speed || p1 % 4 != speed)
                continue;
            int d0 = p0 / 4, d1 = p1 / 4;
            if (d0 == d1)
                continue;
            double x0 = std::cos(d0 * M_PI / 4), y0 = std::sin(d0 * M_PI / 4),
                   x1 = std::cos(d1 * M_PI / 4), y1 = std::sin(d1 * M_PI / 4);
            if (x0 * x1 + y0 * y1 < -.05 || x0 * v[0] + y0 * v[1] < .20 ||
                x1 * v[0] + y1 * v[1] < .20)
                continue;
            Point n = norm(x0 + x1, y0 + y1);
            double score = n[0] * v[0] + n[1] * v[1] - .04;
            if (score > bestscore) {
                found = true;
                bestscore = score;
                bestpair = {p0, p1};
            }
        }
        double current = std::cos(desired * M_PI / 4) * v[0] + std::sin(desired * M_PI / 4) * v[1];
        if (found && bestscore > current + .03 && !any_done)
            prop = proposal(target, bestpair[0], bestpair[1], dist > .8 ? 40 : 15);
    }
    if (py::reinterpret_borrow<py::object>(prop["target_id"]).is_none() &&
        num(m, "nav_ticks") > 100) {
        std::array<int, 2> actions;
        for (int i = 0; i < 2; i++) {
            auto own = py::cast<py::dict>(rs[i]["own"]);
            double angle = std::atan2(num(own, "y") - c[1], num(own, "x") - c[0]),
                   rear = std::atan2(-uy, -ux), raw = rear - angle + M_PI,
                   delta = raw - std::floor(raw / (2 * M_PI)) * (2 * M_PI) - M_PI;
            actions[i] = delta > 0 ? 40 : 43;
        }
        if (legal_has(legal, actions) && !any_done)
            prop = proposal(target, actions[0], actions[1], 40);
        if (num(m, "nav_ticks") > 300)
            m["nav_ticks"] = 0;
    }
    if (dist < .23 && tick - num(m, "goal_brake", -100) > 55) {
        m["goal_brake"] = m["tick"];
        prop = proposal(py::none(), 44, 44, 30);
    }
    if (id == 6 || id == 9) {
        double ga = std::atan2(goal[1] - c[1], goal[0] - c[0]), raw = ga - ang + M_PI / 4,
               err = raw - std::floor(raw / (M_PI / 2)) * (M_PI / 2) - M_PI / 4;
        if (!m.contains("align_until")) {
            m["align_until"] = 0;
            m["align_after"] = -1;
            m["align_sign"] = 0;
        }
        if (tick >= num(m, "align_after") && tick >= num(m, "align_until") && dist > 1.0 &&
            std::abs(err) > .18) {
            m["align_until"] = py::int_(static_cast<long long>(tick + 240));
            m["align_after"] = py::int_(static_cast<long long>(tick + 600));
            m["align_sign"] = err > 0 ? 1 : -1;
        }
        if (tick < num(m, "align_until")) {
            if (std::abs(err) < .10 || err * num(m, "align_sign") < 0 || any_done) {
                m["align_until"] = 0;
                prop = proposal(py::none(), 44, 44, 20);
            } else {
                int ra = err > 0 ? 34 : 37;
                if (legal_has(legal, {ra, ra}))
                    prop = proposal(target, ra, ra, 20);
            }
        }
    }
    py::object clear = py::none();
    if (id == 8) {
        clear = secondary_clear(m, o, c, goal);
        if (!clear.is_none())
            prop = py::cast<py::dict>(clear);
    }
    values.append(num(prop, "ticks"));
    state.append(dist);
    state.append(contacts);
    state.append(num(m, "stall"));
    state.append(c[0]);
    state.append(c[1]);
    state.append(double(desired));
    state.append(dims[0]);
    state.append(dims[1]);
    if (id == 8) {
        state.append(get(m, "secondary_distance"));
        state.append(clear.is_none() ? 0. : 1.);
    }
    proposals.append(prop);
    result["values"] = values;
    result["state"] = state;
    result["proposals"] = proposals;
    return result;
}
} // namespace next_detail
inline py::dict next_initial(int id) {
    (void)id;
    py::dict m;
    m["tick"] = 0;
    m["c"] = py::none();
    m["dims"] = next_detail::plist({0., 0.});
    m["ang"] = 0.;
    m["stall"] = 0;
    m["last"] = py::none();
    m["dir"] = -1;
    m["nav"] = 0;
    return m;
}
inline py::dict next_update(int id, py::dict m, py::dict event) {
    using namespace next_detail;
    auto ticks = get(event, "executed_ticks");
    double dt = truth(ticks) ? py::cast<double>(ticks) : 0.;
    m["tick"] = py::int_(static_cast<long long>(num(m, "tick") + dt));
    auto last = get(event, "action");
    if (id == 7 || id == 8 || id == 9)
        m["last_request"] = last;
    bool navigation = false;
    if (truth(last)) {
        auto action = py::cast<py::dict>(last);
        double a0 = num(action, "a0", -1), a1 = num(action, "a1", -1);
        navigation =
            get(action, "target_id").is_none() || a0 == 40 || a0 == 43 || a1 == 40 || a1 == 43;
    }
    if (navigation)
        m["nav_ticks"] = py::int_(static_cast<long long>(num(m, "nav_ticks") + dt));
    else
        m["nav_ticks"] = 0;
    m["context"] = compute(id, m, py::cast<py::dict>(event["observation"]));
    return m;
}
} // namespace rc
