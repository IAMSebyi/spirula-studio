#include <cmath>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <random>
#include <vector>

#include "sfm/Pipeline.h"
#include "sfm/map/Mapper.h"
#include "sfm/tests/TestMain.h"

using namespace sfm;

static int testGlobalMapper(int argc, char** argv) {
    constexpr uint32_t nimages = 5, npoints = 160;
    MatchesDatabase db;
    db.images.resize(nimages);
    std::vector<FeatureSet> features(nimages);
    const Camera camera = Camera::defaultFor(1, 1280, 960, 1000);
    const std::vector<Vec3> centres = {{0, 0, 0}, {0.8, 0, 0}, {0.1, 0.7, 0.1},
                                       {0.8, 0.7, 0.1}, {0.4, 0.3, 0.4}};
    std::vector<Pose> poses(nimages);
    for (uint32_t i = 0; i < nimages; i++) {
        db.images[i] = {"global_" + std::to_string(i) + ".jpg", npoints};
        features[i].width = 1280;
        features[i].height = 960;
        poses[i].R = angleAxisToRotation({0.01 * i, 0.025 * i, -0.005 * i});
        poses[i].t = mul(poses[i].R, centres[i]) * -1.0;
    }
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> xy(-1.5, 1.5), z(4, 8);
    std::normal_distribution<double> noise(0, 0.12);
    for (uint32_t p = 0; p < npoints; p++) {
        const Vec3 x{xy(rng), xy(rng), z(rng)};
        for (uint32_t i = 0; i < nimages; i++) {
            const Vec2 pixel = camera.project(mul(poses[i].R, x) + poses[i].t);
            features[i].keypoints.push_back({(float)(pixel.x + noise(rng)),
                                             (float)(pixel.y + noise(rng)), 2, 0, 0});
        }
    }
    for (uint32_t i = 0; i < nimages; i++)
        for (uint32_t j = i + 1; j < nimages; j++) {
            TwoViewMatches pair;
            pair.image1 = i;
            pair.image2 = j;
            pair.config = (int)TwoViewConfig::Uncalibrated;
            for (uint32_t p = 0; p < npoints; p++) pair.matches.push_back({p, p});
            db.pairs.push_back(std::move(pair));
        }
    SfmConfig cfg;
    cfg.mapper_mode = "global";
    cfg.quiet = true;
    cfg.threads = 2;
    cfg.mapper.verbose = false;
    cfg.mapper.initial_cameras = {{1, camera}};
    cfg.mapper.known_focal_cameras = {1};
    cfg.mapper.given_focal_cameras = {1};
    cfg.mapper.ba_real = argc > 1 && std::string(argv[1]) == "--vulkan" ? "float" : "cpu";
    cfg.mapper.ba_real_coarse = cfg.mapper.ba_real;
    cfg.manager.do_grow = false;
    cfg.manager.do_reseed = false;
    cfg.manager.do_audit = false;
    cfg.assemble.reseed_trials = 0;
    Mapper mapper(db, features, cfg.mapper, std::vector<uint32_t>(nimages, 1));
    AssembleStats stats;
    auto models = runMapper(mapper, db, features, cfg, stats);
    if (models.size() != 1 || models[0].numRegistered() != nimages ||
        models[0].points3D.size() < 80) {
        std::printf("global mapping and BA: FAIL (%zu models)\n", models.size());
        return 1;
    }
    double mean = 0, median = 0;
    size_t observations = 0;
    reprojStats(models[0], features, mean, median, observations);
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
                      ("spirula_global_mapper_test_" + std::to_string(nonce));
    if (!std::filesystem::create_directory(path)) return 1;
    models[0].writeBinary(path.string());
    const Reconstruction loaded = Reconstruction::readBinary(path.string());
    std::filesystem::remove_all(path);
    const bool ok = std::isfinite(mean) && mean < 2 && observations >= 300 &&
                    loaded.numRegistered() == nimages &&
                    loaded.points3D.size() == models[0].points3D.size();
    std::printf("global mapping, BA and export: %s (%zu points, %zu observations, %.3f px)\n",
                ok ? "PASS" : "FAIL", models[0].points3D.size(), observations, mean);
    return ok ? 0 : 1;
}

int main(int argc, char** argv) { return sfmTestMain(argc, argv, testGlobalMapper); }
