#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include "geometry.hpp"
#include "first.hpp"
#include "next.hpp"

namespace py = pybind11;

static void check_id(int id) {
    if (id < 0 || id >= 10) throw py::index_error("unknown frozen CAP index");
}

static py::dict initial(int id) {
    check_id(id);
    return id < 5 ? rc::first_initial(id) : rc::next_initial(id);
}

static py::dict update(int id, py::dict memory, py::dict event) {
    check_id(id);
    return id < 5 ? rc::first_update(id, memory, event) : rc::next_update(id, memory, event);
}

static py::list advance_all(py::list memories, py::dict event) {
    if (memories.size() != 10) throw py::value_error("expected ten independent memories");
    py::list result;
    for (int id = 0; id < 10; ++id) {
        py::dict memory = update(id, memories[id].cast<py::dict>(), event);
        memories[id] = memory;
        py::list actions = memory["context"]["proposals"].cast<py::list>();
        result.append(actions.size() ? py::object(actions[0]) : py::none());
    }
    return result;
}

PYBIND11_MODULE(_regional_cap_cpp, m) {
    m.def("initial_memory", &initial);
    m.def("update_memory", &update);
    m.def("prepare_context", [](py::dict memory, py::dict) { return py::object(memory["context"]); });
    m.def("advance_all", &advance_all);
    m.def("evaluate_candidates", [](py::array_t<double,0> context, py::array_t<double,0> candidates,
                                    py::array_t<double,0> output) {
        auto c = context.unchecked<1>(); auto a = candidates.unchecked<2>(); auto y = output.mutable_unchecked<2>();
        if (a.shape(0) && (c.shape(0)<1 || a.shape(1)<4 || y.shape(0)<a.shape(0) || y.shape(1)<1))
            throw py::index_error("invalid candidate-score buffer");
        for (py::ssize_t i = 0; i < a.shape(0); ++i) y(i,0) = -std::abs(a(i,3) - c(0))/40.;
    }, py::arg("context").noconvert(), py::arg("candidates").noconvert(), py::arg("output").noconvert());
}
