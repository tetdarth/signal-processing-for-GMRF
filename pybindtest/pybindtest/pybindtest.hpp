#pragma once

#include <vector>
#include <utility>
#include <string>
using namespace std;

int add(int x, int y);

class POINT {
private:
    int x;
    int y;
public:
    int sum;
    POINT(pair<int, int> xy) { this->x = xy.first; this->y = xy.second; this->sum = this->x + this->y; }
    POINT(int x, int y) { this->x = x; this->y = y; this->sum = x + y; }
    int X() { return x; }
    int Y() { return y; }
    POINT operator+(const POINT& v) const { return POINT(this->x + v.x, this->y + v.y); }
    std::string toString() const {
        return "(" + std::to_string(this->x) + ", " + std::to_string(this->y) + ")";
    }
};
