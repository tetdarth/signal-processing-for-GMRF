#include <iostream>
#include <string>
#include <pybind11/pybind11.h>
namespace py = pybind11;

class Cat {
public:
    Cat(const std::string& name) : name_{ name } {}
    void say() const { std::cout << name_ << " said meow." << std::endl; }
    void poop() const { std::cout << name_ << " pooping now..." << std::endl; }
private:
    std::string name_;
};

PYBIND11_MODULE(gmrf, m) {
    py::class_<Cat>(m, "Cat")
        .def(py::init<const std::string&>())
        .def("say", &Cat::say)
        .def("poop", &Cat::poop)
        ;
}