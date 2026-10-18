// Minimal per-check reporting: every check prints PASS/FAIL and the run continues.
#pragma once
#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

namespace chk {

inline std::string num(double x) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.10g", x);
    return buf;
}

struct Suite {
    std::string part;
    int passed = 0, failed = 0;
    explicit Suite(std::string p) : part(std::move(p)) {}

    void expect(const std::string &name, bool ok, const std::string &detail = "") {
        if (ok) {
            ++passed;
            std::printf("PASS %s\n", name.c_str());
        } else {
            ++failed;
            std::printf("FAIL %s%s%s\n", name.c_str(), detail.empty() ? "" : ": ", detail.c_str());
        }
        std::fflush(stdout);
    }
    // |actual - expected| <= abs_tol + rel_tol * |expected|
    void near(const std::string &name, double expected, double actual, double rel_tol,
              double abs_tol = 0) {
        bool ok = std::isfinite(actual) &&
                  std::abs(actual - expected) <= abs_tol + rel_tol * std::abs(expected);
        expect(name, ok, "expected " + num(expected) + " vs actual " + num(actual));
    }
    // Run one group of checks; an exception fails the group and the run continues.
    template <class F> void group(const std::string &name, F &&body) {
        try {
            body();
        } catch (const std::exception &e) {
            expect(name, false, std::string("threw: ") + e.what());
        }
    }
    template <class F> void throws(const std::string &name, F &&body) {
        bool threw = false;
        std::string what;
        try {
            body();
        } catch (const std::invalid_argument &) {
            threw = true;
        } catch (const std::exception &e) {
            what = e.what();
        }
        expect(name, threw,
               what.empty() ? "expected std::invalid_argument" : "wrong exception: " + what);
    }
    int finish() const {
        std::printf("SUMMARY part %s: %d passed, %d failed\n", part.c_str(), passed, failed);
        std::fflush(stdout);
        return failed ? 1 : 0;
    }
};

} // namespace chk
