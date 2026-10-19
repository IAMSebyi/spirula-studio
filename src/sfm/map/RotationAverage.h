// Robust rotation averaging on relative-view edges, with one gauge per component.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <vector>

#include "sfm/core/Pose.h"
#include "sfm/map/RelativeViewGraph.h"

namespace sfm {

struct RotationAverageOptions {
    int max_iterations = 500;
    double huber_radians = 0.10;
    double max_error_radians = 0.20;
    double tolerance = 1e-9;
};

struct RotationAverageResult {
    std::vector<Mat3> rotations;
    // Isolated images have no component; all others use their smallest image index.
    std::vector<uint32_t> component;
    std::vector<uint8_t> accepted;
    bool converged = false;
};

inline RotationAverageResult averageViewRotations(const MatchesDatabase& db,
                                                   const std::vector<RelativeViewEdge>& views,
                                                   const RotationAverageOptions& opt = {}) {
    if (views.size() != db.pairs.size() || db.images.size() >= UINT32_MAX ||
        opt.max_iterations < 1 || !(opt.huber_radians > 0) ||
        !(opt.max_error_radians > 0) || !(opt.tolerance > 0))
        throw std::invalid_argument("rotation averaging: invalid inputs");
    const uint32_t n = (uint32_t)db.images.size();
    RotationAverageResult out;
    out.rotations.assign(n, mat3Identity());
    out.component.assign(n, UINT32_MAX);
    out.accepted.assign(views.size(), 0);
    std::vector<size_t> order;
    for (size_t i = 0; i < views.size(); i++) {
        if (views[i].status != RelativeViewStatus::Accepted) continue;
        const auto& p = db.pairs[i];
        if (p.image1 >= n || p.image2 >= n || p.image1 == p.image2)
            throw std::invalid_argument("rotation averaging: invalid accepted pair");
        for (double v : views[i].pose.R)
            if (!std::isfinite(v))
                throw std::invalid_argument("rotation averaging: non-finite rotation");
        out.accepted[i] = 1;
        order.push_back(i);
    }
    struct Adjacent { uint32_t other; size_t edge; };
    std::vector<std::vector<Adjacent>> adjacency(n);
    for (size_t e : order) {
        const auto& p = db.pairs[e];
        adjacency[p.image1].push_back({p.image2, e});
        adjacency[p.image2].push_back({p.image1, e});
    }
    for (auto& list : adjacency)
        std::sort(list.begin(), list.end(), [](const Adjacent& a, const Adjacent& b) {
            return a.other != b.other ? a.other < b.other : a.edge < b.edge;
        });
    auto fromTo = [&](uint32_t from, size_t e) {
        return db.pairs[e].image1 == from ? views[e].pose.R : transpose(views[e].pose.R);
    };
    std::vector<double> consistency(views.size(), 0.5);
    for (size_t e : order) {
        const auto& p = db.pairs[e];
        const auto& a = adjacency[p.image1];
        const auto& b = adjacency[p.image2];
        size_t ia = 0, ib = 0;
        int checked = 0, agreed = 0;
        while (ia < a.size() && ib < b.size() && checked < 32) {
            if (a[ia].other < b[ib].other) { ia++; continue; }
            if (b[ib].other < a[ia].other) { ib++; continue; }
            const uint32_t common = a[ia].other;
            const Mat3 implied = mul(transpose(fromTo(p.image2, b[ib].edge)),
                                     fromTo(p.image1, a[ia].edge));
            const double error = rotationToAngleAxis(mul(implied, transpose(views[e].pose.R))).norm();
            agreed += error <= opt.max_error_radians;
            checked++;
            while (ia < a.size() && a[ia].other == common) ia++;
            while (ib < b.size() && b[ib].other == common) ib++;
        }
        if (checked) consistency[e] = (double)agreed / checked;
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return consistency[a] != consistency[b] ? consistency[a] > consistency[b]
                                               : views[a].inliers > views[b].inliers;
    });
    auto solve = [&]() {
        std::vector<uint32_t> parent(n);
        std::iota(parent.begin(), parent.end(), 0);
        auto root = [&](uint32_t x) {
            while (parent[x] != x) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        };
        std::vector<std::vector<size_t>> tree(n);
        for (size_t e : order) {
            if (!out.accepted[e]) continue;
            const auto& p = db.pairs[e];
            uint32_t a = root(p.image1), b = root(p.image2);
            if (a == b) continue;
            parent[b] = a;
            tree[p.image1].push_back(e);
            tree[p.image2].push_back(e);
        }
        std::fill(out.component.begin(), out.component.end(), UINT32_MAX);
        std::fill(out.rotations.begin(), out.rotations.end(), mat3Identity());
        for (uint32_t start = 0; start < n; start++) {
            if (tree[start].empty() || out.component[start] != UINT32_MAX) continue;
            std::queue<uint32_t> q;
            out.component[start] = start;
            q.push(start);
            while (!q.empty()) {
                const uint32_t u = q.front();
                q.pop();
                for (size_t e : tree[u]) {
                    const auto& p = db.pairs[e];
                    const bool forward = p.image1 == u;
                    const uint32_t v = forward ? p.image2 : p.image1;
                    if (out.component[v] != UINT32_MAX) continue;
                    out.rotations[v] = mul(forward ? views[e].pose.R : transpose(views[e].pose.R),
                                           out.rotations[u]);
                    out.component[v] = start;
                    q.push(v);
                }
            }
        }
        std::vector<Vec3> sums(n);
        std::vector<double> weights(n);
        for (int iter = 0; iter < opt.max_iterations; iter++) {
            cancel::check();
            std::fill(sums.begin(), sums.end(), Vec3{});
            std::fill(weights.begin(), weights.end(), 0.0);
            for (size_t e : order) {
                if (!out.accepted[e]) continue;
                const auto& p = db.pairs[e];
                const Mat3& rel = views[e].pose.R;
                const Vec3 ra = rotationToAngleAxis(mul(mul(transpose(rel), out.rotations[p.image2]),
                                                        transpose(out.rotations[p.image1])));
                const Vec3 rb = rotationToAngleAxis(mul(mul(rel, out.rotations[p.image1]),
                                                        transpose(out.rotations[p.image2])));
                const double error = rb.norm();
                const double w = error > opt.huber_radians ? opt.huber_radians / error : 1.0;
                sums[p.image1] = sums[p.image1] + ra * w;
                sums[p.image2] = sums[p.image2] + rb * w;
                weights[p.image1] += w;
                weights[p.image2] += w;
            }
            double largest = 0;
            for (uint32_t i = 0; i < n; i++) {
                if (out.component[i] == UINT32_MAX || out.component[i] == i || weights[i] == 0)
                    continue;
                const Vec3 step = sums[i] * (0.5 / weights[i]);
                largest = std::max(largest, step.norm());
                out.rotations[i] = mul(angleAxisToRotation(step), out.rotations[i]);
            }
            if (largest < opt.tolerance) return true;
        }
        return false;
    };
    const bool first_converged = solve();
    bool removed = false;
    for (size_t e : order) {
        const auto& p = db.pairs[e];
        const Mat3 predicted = mul(out.rotations[p.image2], transpose(out.rotations[p.image1]));
        const double error = rotationToAngleAxis(mul(predicted, transpose(views[e].pose.R))).norm();
        if (error > opt.max_error_radians) {
            out.accepted[e] = 0;
            removed = true;
        }
    }
    out.converged = removed ? solve() : first_converged;
    return out;
}

}  // namespace sfm
