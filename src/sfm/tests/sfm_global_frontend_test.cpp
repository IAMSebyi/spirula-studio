#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "sfm/map/GlobalTracks.h"
#include "sfm/map/GlobalReconstruction.h"
#include "sfm/map/RotationAverage.h"
#include "sfm/tests/TestMain.h"

using namespace sfm;

static int testGlobalFrontend(int, char**) {
    constexpr uint32_t nimages = 4, npoints = 120;
    MatchesDatabase db;
    db.images.resize(nimages);
    std::vector<FeatureSet> features(nimages);
    std::vector<Camera> cameras(nimages);
    std::vector<Pose> poses(nimages);
    for (uint32_t i = 0; i < nimages; i++) {
        cameras[i].model = CamModel::Pinhole;
        cameras[i].fx = cameras[i].fy = 1000;
        cameras[i].cx = 640;
        cameras[i].cy = 480;
        features[i].width = 1280;
        features[i].height = 960;
        poses[i].R = angleAxisToRotation({0.02 * i, 0.045 * i, -0.01 * i});
        poses[i].t = {0.7 * i, 0.12 * i, 0.04 * i};
    }
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> xy(-2, 2), z(5, 10);
    std::normal_distribution<double> noise(0, 0.08);
    for (uint32_t k = 0; k < npoints; k++) {
        const Vec3 X{xy(rng), xy(rng), z(rng)};
        for (uint32_t i = 0; i < nimages; i++) {
            const Vec3 p = mul(poses[i].R, X) + poses[i].t;
            Keypoint kp;
            kp.x = (float)(1000 * p.x / p.z + 640 + noise(rng));
            kp.y = (float)(1000 * p.y / p.z + 480 + noise(rng));
            features[i].keypoints.push_back(kp);
        }
    }
    for (uint32_t i = 0; i < nimages; i++)
        for (uint32_t j = i + 1; j < nimages; j++) {
            TwoViewMatches pair;
            pair.image1 = i;
            pair.image2 = j;
            pair.config = (int)TwoViewConfig::Uncalibrated;
            for (uint32_t k = 0; k < npoints; k++)
                pair.matches.push_back({k, i == 0 && j == 1 && k < 15 ? (k + 40) % npoints : k});
            db.pairs.push_back(std::move(pair));
        }
    RelativeViewOptions options;
    options.threads = 2;
    options.geometry.ransac.min_num_trials = 50;
    options.geometry.ransac.max_num_trials = 300;
    const auto edges = buildRelativeViewGraph(db, features, cameras, options);
    const auto rotations = averageViewRotations(db, edges);
    const auto tracks = buildGlobalTracks(db, features, edges, 3, &rotations.accepted);
    int failures = 0;
    auto check = [&](bool ok, const char* name) {
        std::printf("%s: %s\n", name, ok ? "PASS" : "FAIL");
        failures += !ok;
    };
    size_t accepted = 0;
    for (uint8_t a : rotations.accepted) accepted += a;
    check(accepted == db.pairs.size(), "relative views survive rotation averaging");
    double worst = 0;
    for (uint32_t i = 0; i < nimages; i++) {
        const Mat3 expected = mul(poses[i].R, transpose(poses[0].R));
        worst = std::max(worst,
                         rotationToAngleAxis(mul(rotations.rotations[i], transpose(expected))).norm());
    }
    check(rotations.converged && worst < 0.03, "end-to-end orientations");
    size_t correct = 0;
    for (size_t t = 0; t < tracks.size(); t++) {
        bool same = true;
        for (size_t k = tracks.offsets[t] + 1; k < tracks.offsets[t + 1]; k++)
            same = same && tracks.observations[k].point2D_idx ==
                           tracks.observations[tracks.offsets[t]].point2D_idx;
        correct += same;
    }
    check(correct >= 100 && correct == tracks.size(), "end-to-end tracks reject outliers");
    GlobalReconstructionOptions reconstruction_opt;
    reconstruction_opt.relative = options;
    GlobalReconstructionStats stats;
    const auto models = buildGlobalReconstructions(db, features, std::vector<uint32_t>(nimages, 1),
                                                    {{1, cameras[0]}}, {1}, reconstruction_opt, stats);
    check(models.size() == 1 && models[0].numRegistered() == nimages &&
          models[0].points3D.size() >= 90, "global reconstruction from matches");
    if (!models.empty()) {
        double numerator = 0, denominator = 0;
        std::vector<Vec3> estimated, truth;
        for (uint32_t i = 0; i < nimages; i++) {
            const Pose& pose = models[0].images.at(i).pose;
            estimated.push_back(mul(transpose(pose.R), pose.t) * -1.0);
            truth.push_back(mul(transpose(poses[i].R), poses[i].t) * -1.0);
        }
        for (uint32_t i = 1; i < nimages; i++) {
            numerator += (estimated[i] - estimated[0]).dot(truth[i] - truth[0]);
            denominator += (estimated[i] - estimated[0]).dot(estimated[i] - estimated[0]);
        }
        const double scale = numerator / denominator;
        double error = 0;
        for (uint32_t i = 1; i < nimages; i++)
            error = std::max(error, ((estimated[i] - estimated[0]) * scale -
                                     (truth[i] - truth[0])).norm());
        std::printf("camera position error: %.5g\n", error);
        check(std::isfinite(error) && error < 0.15, "global camera positions");
    }
    MatchesDatabase separate = db;
    separate.images.insert(separate.images.end(), db.images.begin(), db.images.end());
    for (const TwoViewMatches& source : db.pairs) {
        TwoViewMatches pair = source;
        pair.image1 += nimages;
        pair.image2 += nimages;
        separate.pairs.push_back(std::move(pair));
    }
    std::vector<FeatureSet> separate_features = features;
    separate_features.insert(separate_features.end(), features.begin(), features.end());
    const auto parts = buildGlobalReconstructions(separate, separate_features,
        std::vector<uint32_t>(2 * nimages, 1), {{1, cameras[0]}}, {1}, reconstruction_opt, stats);
    check(parts.size() == 2 && parts[0].numRegistered() == nimages &&
          parts[1].numRegistered() == nimages, "separate global components");
    return failures ? 1 : 0;
}

int main(int argc, char** argv) { return sfmTestMain(argc, argv, testGlobalFrontend); }
