#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "sfm/map/GlobalPositionGpu.h"
#include "sfm/tests/TestMain.h"

using namespace sfm;

static int testGlobalPositionGpu(int, char**) {
    const std::vector<Vec3> truth = {{0, 0, 0}, {1, 0.1, 0.05}, {0.1, 1, 0.1},
                                     {0.8, 0.9, 0.2}, {0.4, -0.3, 0.3}};
    GlobalPositionProblem input;
    input.cameras.resize(truth.size());
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> xy(-2, 2), depth(3, 8);
    std::normal_distribution<double> noise(0, 0.0005);
    for (uint32_t j = 0; j < 80; j++) {
        const Vec3 point{xy(rng), xy(rng), depth(rng)};
        input.points.push_back(point);
        for (uint32_t i = 0; i < truth.size(); i++) {
            Vec3 ray = (point - truth[i]).normalized();
            ray = (ray + Vec3{noise(rng), noise(rng), noise(rng)}).normalized();
            if ((j * truth.size() + i) % 23 == 0)
                ray = Vec3{xy(rng), xy(rng), depth(rng)}.normalized();
            input.observations.push_back({i, j, ray, 1});
        }
    }
    initializeGlobalPosition(input, 17);
    GlobalPositionProblem cpu = input, gpu = input;
    GlobalPositionOptions opt;
    opt.max_iterations = 150;
    opt.huber = 0.03;
    const auto ref = solveGlobalPosition(cpu, opt);
    const auto actual = solveGlobalPositionGpu(gpu, opt);
    double numer = 0, denom = 0;
    for (size_t i = 1; i < truth.size(); i++) {
        numer += gpu.cameras[i].dot(truth[i]);
        denom += gpu.cameras[i].dot(gpu.cameras[i]);
    }
    const double scale = numer / denom;
    double error = 0;
    for (size_t i = 1; i < truth.size(); i++)
        error = std::max(error, (gpu.cameras[i] * scale - truth[i]).norm());
    bool invalid_weight = false, invalid_option = false;
    GlobalPositionProblem invalid = input;
    invalid.observations[0].weight = std::numeric_limits<double>::denorm_min();
    try { solveGlobalPositionGpu(invalid, opt); }
    catch (const std::invalid_argument&) { invalid_weight = true; }
    GlobalPositionOptions invalid_opt = opt;
    invalid_opt.huber = std::numeric_limits<double>::infinity();
    try { solveGlobalPositionGpu(invalid, invalid_opt); }
    catch (const std::invalid_argument&) { invalid_option = true; }
    const bool gauge = (gpu.cameras[0] - input.cameras[0]).norm() == 0 &&
                       gpu.log_scales[0] == input.log_scales[0];
    const bool ok = std::isfinite(actual.final_cost) && actual.accepted > 0 &&
                    actual.final_cost < actual.initial_cost &&
                    actual.final_cost < ref.final_cost * 1.05 && error < 0.15 &&
                    gauge && invalid_weight && invalid_option;
    std::printf("global position CPU %.6g, GPU %.6g, camera error %.5g, iterations %d: %s\n",
                ref.final_cost, actual.final_cost, error, actual.iterations, ok ? "PASS" : "FAIL");
    if (!ok) std::printf("gauge %d, invalid weight %d, invalid option %d\n",
                         gauge, invalid_weight, invalid_option);
    return ok ? 0 : 1;
}

int main(int argc, char** argv) { return sfmTestMain(argc, argv, testGlobalPositionGpu); }
