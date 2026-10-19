#include <cmath>
#include <cstdio>
#include <vector>

#include "sfm/map/RotationAverage.h"
#include "sfm/tests/TestMain.h"

using namespace sfm;

static int testRotationAverage(int, char**) {
    MatchesDatabase db;
    db.images.resize(8);
    std::vector<Mat3> truth;
    for (int i = 0; i < 8; i++)
        truth.push_back(angleAxisToRotation({0.04 * i, -0.07 * i, 0.025 * i}));
    std::vector<RelativeViewEdge> edges;
    auto add = [&](uint32_t a, uint32_t b, const Mat3& rotation, uint32_t weight) {
        TwoViewMatches p;
        p.image1 = a;
        p.image2 = b;
        db.pairs.push_back(p);
        RelativeViewEdge e;
        e.pose.R = rotation;
        e.inliers = weight;
        e.status = RelativeViewStatus::Accepted;
        edges.push_back(e);
    };
    for (uint32_t i = 0; i < 5; i++)
        for (uint32_t j = i + 1; j < 5; j++) {
            const Mat3 relative = mul(truth[j], transpose(truth[i]));
            const double d = 0.002 * (int(j - i) - 2);
            add(i, j, mul(angleAxisToRotation({d, -d * 0.5, d * 0.3}), relative), 100);
        }
    const size_t bad_edge = edges.size();
    add(0, 4, mul(angleAxisToRotation({0.5, 0, 0}),
                  mul(truth[4], transpose(truth[0]))), 1000);
    add(6, 5, mul(truth[5], transpose(truth[6])), 100);
    RotationAverageOptions opt;
    opt.huber_radians = 0.025;
    opt.max_error_radians = 0.10;
    const RotationAverageResult result = averageViewRotations(db, edges, opt);
    int failures = 0;
    auto check = [&](bool ok, const char* name) {
        std::printf("%s: %s\n", name, ok ? "PASS" : "FAIL");
        failures += !ok;
    };
    check(result.converged, "rotation solver converged");
    double worst = 0;
    for (int i = 0; i < 5; i++) {
        const Mat3 expected = mul(truth[i], transpose(truth[0]));
        worst = std::max(worst,
                         rotationToAngleAxis(mul(result.rotations[i], transpose(expected))).norm());
    }
    check(worst < 0.01, "noisy rotations recovered up to gauge");
    check(!result.accepted[bad_edge], "inconsistent relative rotation removed");
    check(result.component[0] == 0 && result.component[4] == 0 &&
          result.component[5] == 5 && result.component[6] == 5 &&
          result.component[7] == UINT32_MAX,
          "disconnected components and isolated image");
    const Mat3 expected = mul(truth[6], transpose(truth[5]));
    check(rotationToAngleAxis(mul(result.rotations[6], transpose(expected))).norm() < 1e-8,
          "reversed edge convention");
    return failures ? 1 : 0;
}

int main(int argc, char** argv) { return sfmTestMain(argc, argv, testRotationAverage); }
