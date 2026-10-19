// Tracks from the inliers of accepted relative-view edges.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "sfm/core/Model.h"
#include "sfm/map/RelativeViewGraph.h"

namespace sfm {

struct GlobalTracks {
    std::vector<TrackElement> observations;
    std::vector<size_t> offsets{0};
    size_t conflicting = 0;
    size_t too_short = 0;

    size_t size() const { return offsets.size() - 1; }
};

inline GlobalTracks buildGlobalTracks(const MatchesDatabase& db,
                                      const std::vector<FeatureSet>& features,
                                      const std::vector<RelativeViewEdge>& edges,
                                      size_t min_views = 2,
                                      const std::vector<uint8_t>* accepted = nullptr) {
    if (features.size() != db.images.size() || edges.size() != db.pairs.size() ||
        (accepted && accepted->size() != edges.size()))
        throw std::invalid_argument("global tracks: input counts differ");
    std::vector<size_t> offsets(features.size() + 1, 0);
    for (size_t i = 0; i < features.size(); i++) {
        offsets[i + 1] = offsets[i] + features[i].keypoints.size();
        if (offsets[i + 1] >= std::numeric_limits<uint32_t>::max())
            throw std::length_error("global tracks: too many features");
    }
    constexpr uint32_t unused = std::numeric_limits<uint32_t>::max();
    std::vector<uint32_t> parent(offsets.back(), unused);
    std::vector<uint32_t> support(offsets.back(), 0);
    std::vector<uint8_t> rank(offsets.back(), 0);
    auto root = [&](uint32_t v) {
        uint32_t r = v;
        while (parent[r] != r) r = parent[r];
        while (parent[v] != v) {
            const uint32_t next = parent[v];
            parent[v] = r;
            v = next;
        }
        return r;
    };
    for (size_t p = 0; p < db.pairs.size(); p++) {
        cancel::check();
        const RelativeViewEdge& edge = edges[p];
        if (edge.status != RelativeViewStatus::Accepted || (accepted && !(*accepted)[p])) continue;
        const TwoViewMatches& pair = db.pairs[p];
        if (pair.image1 >= features.size() || pair.image2 >= features.size() ||
            pair.image1 == pair.image2)
            throw std::invalid_argument("global tracks: invalid accepted pair");
        for (size_t k = 0; k < pair.matches.size(); k++) {
            if (!edge.isInlier(k)) continue;
            const FeatureMatch& m = pair.matches[k];
            if (m.idx1 >= features[pair.image1].keypoints.size() ||
                m.idx2 >= features[pair.image2].keypoints.size())
                throw std::invalid_argument("global tracks: invalid feature index");
            uint32_t a = (uint32_t)(offsets[pair.image1] + m.idx1);
            uint32_t b = (uint32_t)(offsets[pair.image2] + m.idx2);
            if (support[a] != unused) support[a]++;
            if (support[b] != unused) support[b]++;
            if (parent[a] == unused) parent[a] = a;
            if (parent[b] == unused) parent[b] = b;
            a = root(a);
            b = root(b);
            if (a == b) continue;
            if (rank[a] < rank[b]) std::swap(a, b);
            parent[b] = a;
            if (rank[a] == rank[b]) rank[a]++;
        }
    }
    struct Item { uint32_t root, image, feature; };
    std::vector<Item> items;
    for (size_t image = 0; image < features.size(); image++)
        for (size_t k = offsets[image]; k < offsets[image + 1]; k++)
            if (parent[k] != unused)
                items.push_back({root((uint32_t)k), (uint32_t)image,
                                 (uint32_t)(k - offsets[image])});
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        if (a.root != b.root) return a.root < b.root;
        if (a.image != b.image) return a.image < b.image;
        return a.feature < b.feature;
    });
    GlobalTracks out;
    for (size_t begin = 0; begin < items.size();) {
        size_t end = begin + 1;
        while (end < items.size() && items[end].root == items[begin].root) end++;
        bool conflict = false;
        std::vector<TrackElement> selected;
        for (size_t i = begin; i < end;) {
            size_t next = i + 1, best = i;
            while (next < end && items[next].image == items[i].image) next++;
            for (size_t a = i; a < next; a++) {
                const Keypoint& ka = features[items[a].image].keypoints[items[a].feature];
                if (support[offsets[items[a].image] + items[a].feature] >
                    support[offsets[items[best].image] + items[best].feature]) best = a;
                for (size_t b = i; b < a; b++) {
                    const Keypoint& kb = features[items[b].image].keypoints[items[b].feature];
                    const double dx = ka.x - kb.x, dy = ka.y - kb.y;
                    if (dx * dx + dy * dy > 100.0) conflict = true;
                }
            }
            selected.push_back({items[best].image, items[best].feature});
            i = next;
        }
        if (conflict) {
            out.conflicting++;
        } else if (selected.size() < std::max<size_t>(2, min_views)) {
            out.too_short++;
        } else {
            out.observations.insert(out.observations.end(), selected.begin(), selected.end());
            out.offsets.push_back(out.observations.size());
        }
        begin = end;
    }
    return out;
}

}  // namespace sfm
