// Per-check reporting shared by quant_test.cpp and gpu_test.cu.
// Every check prints one line, PASS <name> or FAIL <name>: <expected> vs <actual>,
// and a failing (or throwing) check never stops the remaining checks of a stage.
#pragma once
#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <sstream>
#include <string>

namespace chk {
inline int &failures() {
    static int count = 0;
    return count;
}
inline int &passes() {
    static int count = 0;
    return count;
}
inline std::string num(double v) {
    std::ostringstream out;
    out.precision(6);
    out << v;
    return out.str();
}
inline bool check(bool ok, const std::string &name, const std::string &detail = "") {
    if (ok) {
        ++passes();
        std::printf("PASS %s\n", name.c_str());
    } else {
        ++failures();
        std::printf("FAIL %s: %s\n", name.c_str(), detail.c_str());
    }
    std::fflush(stdout);
    return ok;
}
// Run one block of checks; an exception (e.g. "TODO A1") becomes one FAIL line.
inline void guard(const std::string &name, const std::function<void()> &body) {
    try {
        body();
    } catch (const std::exception &e) {
        check(false, name, std::string("threw: ") + e.what());
    }
}
// True when the callable throws std::invalid_argument (the contract for bad inputs).
template <class F> bool rejects(F &&f) {
    try {
        f();
    } catch (const std::invalid_argument &) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}
inline double rel(double a, double b) {
    return std::abs(a - b) / (1e-12 + std::abs(b));
}
inline int summary(const std::string &stage) {
    std::printf("SUMMARY %s: %d passed, %d failed\n", stage.c_str(), passes(), failures());
    std::fflush(stdout);
    return failures() ? 1 : 0;
}
} // namespace chk
