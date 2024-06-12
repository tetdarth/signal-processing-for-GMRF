#include <iostream>
#include <string>
#include <pybind11/pybind11.h>

#include "IVODGMRF.hpp"
#include "IVHODGMRF.hpp"
#include "DVODGMRF.hpp"
#include "DVHODGMRF.hpp"
#include "include.hpp"

using namespace std;
namespace py = pybind11;

class Cat {
public:
    Cat(const std::string& _name) : name(_name) {}
    void say() const { std::cout << name << " said meow." << std::endl; }
    void poop() const { std::cout << name << " is pooping right now..." << std::endl; }
    void eat(const std::string& food) const {
        std::cout << name << " is eating a " << food << std::endl;
    }
private:
    std::string name;
};


PYBIND11_MODULE(gmrf, m) {
    py::class_<Cat>(m, "Cat")
        .def(py::init<const std::string&>())
        .def("say", &Cat::say)
        .def("poop", &Cat::poop)
        .def("eat", &Cat::eat, py::arg("food") = "fish")
        ;
}
