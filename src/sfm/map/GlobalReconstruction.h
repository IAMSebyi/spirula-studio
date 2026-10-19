// Build initial reconstructions from globally averaged rotations and tracks.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

#include "sfm/map/GlobalPosition.h"
#include "sfm/map/GlobalTracks.h"
#include "sfm/map/RotationAverage.h"

namespace sfm {

struct GlobalReconstructionOptions {
    RelativeViewOptions relative;
    RotationAverageOptions rotation;
    GlobalPositionOptions position;
    std::function<GlobalPositionResult(GlobalPositionProblem&, const GlobalPositionOptions&)>
        position_solver;
    double max_initial_angle = 0.10;
    double min_triangulation_angle = 0.02617993877991494;
};

struct GlobalReconstructionStats {
    size_t edges = 0, tracks = 0, conflicting = 0, models = 0;
    bool rotation_converged = false;
};

inline std::vector<Reconstruction> buildGlobalReconstructions(
    const MatchesDatabase& db, const std::vector<FeatureSet>& features,
    const std::vector<uint32_t>& camera_ids, const std::map<uint32_t, Camera>& camera_groups,
    const std::set<uint32_t>& known_focals, const GlobalReconstructionOptions& opt,
    GlobalReconstructionStats& stats) {
    stats = {};
    const size_t n = db.images.size();
    if (features.size() != n || camera_ids.size() != n ||
        !(opt.max_initial_angle > 0) || !(opt.min_triangulation_angle >= 0))
        throw std::invalid_argument("global reconstruction: invalid inputs");
    std::vector<Camera> cameras(n);
    for (size_t i = 0; i < n; i++) {
        auto it = camera_groups.find(camera_ids[i]);
        if (it == camera_groups.end())
            throw std::invalid_argument("global reconstruction: missing camera group");
        cameras[i] = it->second;
        cameras[i].pixel_scale = features[i].pixelScale();
    }
    const auto edges = buildRelativeViewGraph(db, features, cameras, opt.relative);
    const auto rotations = averageViewRotations(db, edges, opt.rotation);
    stats.rotation_converged = rotations.converged;
    for (uint8_t b : rotations.accepted) stats.edges += b;
    const auto tracks = buildGlobalTracks(db, features, edges, 3, &rotations.accepted);
    stats.tracks = tracks.size();
    stats.conflicting = tracks.conflicting;
    std::vector<size_t> parent(n + tracks.size());
    std::iota(parent.begin(), parent.end(), 0);
    auto root = [&](size_t x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    for (size_t t = 0; t < tracks.size(); t++)
        for (size_t j = tracks.offsets[t]; j < tracks.offsets[t + 1]; j++) {
            const size_t a = root(n + t), b = root(tracks.observations[j].image_id);
            if (a != b) parent[b] = a;
        }
    std::map<size_t, std::vector<size_t>> groups;
    for (size_t t = 0; t < tracks.size(); t++) groups[root(n + t)].push_back(t);
    std::vector<Reconstruction> models;
    for (const auto& group : groups) {
        cancel::check();
        std::set<uint32_t> image_set;
        for (size_t t : group.second)
            for (size_t j = tracks.offsets[t]; j < tracks.offsets[t + 1]; j++)
                image_set.insert(tracks.observations[j].image_id);
        if (image_set.size() < 2) continue;
        std::vector<uint32_t> images(image_set.begin(), image_set.end());
        std::vector<uint32_t> local(n, UINT32_MAX);
        for (size_t i = 0; i < images.size(); i++) local[images[i]] = (uint32_t)i;
        GlobalPositionProblem problem;
        problem.cameras.resize(images.size());
        problem.points.resize(group.second.size());
        for (size_t p = 0; p < group.second.size(); p++) {
            const size_t t = group.second[p];
            for (size_t j = tracks.offsets[t]; j < tracks.offsets[t + 1]; j++) {
                const TrackElement& obs = tracks.observations[j];
                const Keypoint& kp = features[obs.image_id].keypoints[obs.point2D_idx];
                const Vec3 ray = cameras[obs.image_id].bearing({kp.x, kp.y}).normalized();
                const Vec3 world = mul(transpose(rotations.rotations[obs.image_id]), ray);
                problem.observations.push_back({local[obs.image_id], (uint32_t)p, world,
                    known_focals.count(camera_ids[obs.image_id]) ? 1.0 : 0.5});
            }
        }
        initializeGlobalPosition(problem, opt.position.seed);
        const GlobalPositionResult solved = opt.position_solver
            ? opt.position_solver(problem, opt.position)
            : solveGlobalPosition(problem, opt.position);
        if (!std::isfinite(solved.final_cost) || solved.accepted == 0) continue;
        Reconstruction model;
        for (size_t i = 0; i < images.size(); i++) {
            const uint32_t id = images[i], cid = camera_ids[id];
            model.cameras[cid] = camera_groups.at(cid);
            Image image;
            image.id = id;
            image.camera_id = cid;
            image.name = db.images[id].name;
            image.exif_orientation = features[id].exif_orientation;
            image.registered = true;
            image.pose.R = rotations.rotations[id];
            image.pose.t = mul(image.pose.R, problem.cameras[i]) * -1.0;
            image.points2D.reserve(features[id].count());
            for (const Keypoint& kp : features[id].keypoints) image.points2D.push_back({kp.x, kp.y});
            image.point3D_ids.assign(features[id].count(), kInvalidPoint3D);
            model.images.emplace(id, std::move(image));
        }
        for (size_t p = 0; p < group.second.size(); p++) {
            const size_t t = group.second[p];
            std::vector<TrackElement> keep;
            std::vector<Vec3> rays;
            for (size_t j = tracks.offsets[t]; j < tracks.offsets[t + 1]; j++) {
                const TrackElement obs = tracks.observations[j];
                const Vec3 delta = problem.points[p] - problem.cameras[local[obs.image_id]];
                if (delta.norm() < 1e-12) continue;
                const Keypoint& kp = features[obs.image_id].keypoints[obs.point2D_idx];
                const Vec3 observed = mul(transpose(rotations.rotations[obs.image_id]),
                    cameras[obs.image_id].bearing({kp.x, kp.y}).normalized());
                const Vec3 actual = delta.normalized();
                if (observed.dot(actual) < std::cos(opt.max_initial_angle)) continue;
                keep.push_back(obs);
                rays.push_back(actual);
            }
            bool parallax = false;
            const double max_cos = std::cos(opt.min_triangulation_angle);
            for (size_t a = 0; a < rays.size() && !parallax; a++)
                for (size_t b = a + 1; b < rays.size(); b++)
                    if (rays[a].dot(rays[b]) < max_cos) { parallax = true; break; }
            if (keep.size() < 2 || !parallax) continue;
            const uint64_t id = model.addPoint3D(problem.points[p], keep);
            uint64_t colors[3] = {0, 0, 0};
            size_t count = 0;
            for (const TrackElement& obs : keep) {
                const auto& f = features[obs.image_id];
                if (!f.hasColors()) continue;
                for (int c = 0; c < 3; c++) colors[c] += f.colors[3 * (size_t)obs.point2D_idx + c];
                count++;
            }
            if (count)
                for (int c = 0; c < 3; c++) model.points3D[id].rgb[c] = (uint8_t)(colors[c] / count);
        }
        if (model.points3D.size() >= 10) models.push_back(std::move(model));
    }
    stats.models = models.size();
    std::sort(models.begin(), models.end(), [](const Reconstruction& a, const Reconstruction& b) {
        return a.numRegistered() > b.numRegistered();
    });
    return models;
}

}  // namespace sfm
