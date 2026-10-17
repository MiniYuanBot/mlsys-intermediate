#pragma once
#include "src/moe_student.hpp"
#include <random>
struct Fixture {
    int tokens, width, hidden;
    std::vector<moe::Expert> experts;
    std::vector<float> x;
    moe::Dispatch dispatch;
    Fixture(int t, int w, int h, int e, int seed) : tokens(t), width(w), hidden(h) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-.2f, .2f);
        x.resize(t * w);
        for (auto &v : x) v = dist(rng);
        for (int i = 0; i < e; ++i) {
            std::vector<float> up(h * w), down(w * h);
            for (auto &v : up) v = dist(rng);
            for (auto &v : down) v = dist(rng);
            experts.push_back({moe::quantize(up, h, w), moe::quantize(down, w, h)});
        }
        std::vector<moe::Route> routes;
        for (int i = 0; i < t; ++i) {
            routes.push_back({i, i % e, .75f});
            routes.push_back({i, (i + 2) % e, .25f});
        }
        dispatch = moe::dispatch(routes, t, e);
    }
};
