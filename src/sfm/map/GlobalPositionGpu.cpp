#include "sfm/map/GlobalPositionGpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "sfm/vk/EmbeddedSpirv.h"
#include "sfm/vk/VkContext.h"

namespace sfm {
namespace {

struct UInt2 { uint32_t x, y; };
struct Float4 { float x, y, z, w; };

class PositionGpu {
public:
    PositionGpu(const GlobalPositionProblem& p, int device, const std::string& selector)
        : nc_((uint32_t)p.cameras.size()), np_((uint32_t)p.points.size()),
          no_((uint32_t)p.observations.size()), nodes_(nc_ + np_),
          dim_(3 * nodes_), var_dim_(dim_ + no_), metric_scale_(p.metric_scale) {
        ctx_.init(deviceOnlyOpt(device, selector));
        std::vector<UInt2> obs(no_);
        std::vector<Float4> rays((size_t)no_ * 2);
        std::vector<uint32_t> ranges((size_t)nodes_ + 1, 0);
        std::vector<float> vars(var_dim_);
        for (uint32_t i = 0; i < nc_; i++) put(vars, 3 * i, p.cameras[i]);
        for (uint32_t i = 0; i < np_; i++) put(vars, 3 * (nc_ + i), p.points[i]);
        for (uint32_t i = 0; i < no_; i++) {
            const auto& o = p.observations[i];
            obs[i] = {o.camera, o.point};
            rays[2 * (size_t)i] = {(float)o.ray.x, (float)o.ray.y, (float)o.ray.z, (float)o.weight};
            rays[2 * (size_t)i + 1] = {(float)o.camera_offset.x, (float)o.camera_offset.y,
                                       (float)o.camera_offset.z, 0};
            if (!(rays[2 * (size_t)i].w > 0) || !std::isfinite(rays[2 * (size_t)i].w))
                throw std::invalid_argument("global positioning: observation weight exceeds GPU precision");
            const Float4 offset = rays[2 * (size_t)i + 1];
            if (!std::isfinite(offset.x) || !std::isfinite(offset.y) || !std::isfinite(offset.z))
                throw std::invalid_argument("global positioning: rig offset exceeds GPU precision");
            vars[3 * nodes_ + i] = (float)p.log_scales[i];
            ranges[o.camera + 1]++;
            ranges[nc_ + o.point + 1]++;
        }
        for (float value : vars)
            if (!std::isfinite(value))
                throw std::invalid_argument("global positioning: parameter exceeds GPU precision");
        std::partial_sum(ranges.begin(), ranges.end(), ranges.begin());
        std::vector<uint32_t> cursor = ranges, adjacency((size_t)no_ * 2);
        for (uint32_t i = 0; i < no_; i++) {
            adjacency[cursor[obs[i].x]++] = i;
            adjacency[cursor[nc_ + obs[i].y]++] = i;
        }
        const uint32_t reductions = groups(std::max(dim_, no_));
        std::vector<VkDeviceSize> bytes = {
            (VkDeviceSize)no_ * sizeof(UInt2), (VkDeviceSize)no_ * 2 * sizeof(Float4),
            (VkDeviceSize)var_dim_ * 4, (VkDeviceSize)no_ * sizeof(Float4),
            (VkDeviceSize)no_ * sizeof(Float4), (VkDeviceSize)(nodes_ + 1) * 4,
            (VkDeviceSize)no_ * 8, (VkDeviceSize)dim_ * 8 * 4,
            (VkDeviceSize)reductions * sizeof(Float4)};
        std::vector<VkBuffer> handles;
        for (VkDeviceSize size : bytes) {
            if (size > ctx_.maxBufferBytes())
                throw std::length_error("global positioning: GPU buffer exceeds device limit");
            buffers_.push_back(ctx_.createBuffer(size));
            handles.push_back(buffers_.back().buf);
        }
        ctx_.createDescriptors(handles);
        size_t words = 0;
        const uint32_t* shader = findSpirv("global_position", &words);
        if (!shader) throw std::runtime_error("global positioning: shader is unavailable");
        ctx_.loadPipelines(shader, words * 4,
            {"gp_linearize", "gp_nodes", "gp_points", "gp_scale_update", "gp_vector", "gp_reduce"});
        const VkContext::UploadItem uploads[] = {
            {&buffers_[0], obs.data(), bytes[0]}, {&buffers_[1], rays.data(), bytes[1]},
            {&buffers_[2], vars.data(), bytes[2]}, {&buffers_[5], ranges.data(), bytes[5]},
            {&buffers_[6], adjacency.data(), bytes[6]}};
        ctx_.uploadMany(uploads, sizeof(uploads) / sizeof(uploads[0]));
    }

    GlobalPositionResult solve(GlobalPositionProblem& p, const GlobalPositionOptions& opt) {
        GlobalPositionResult result;
        double cost = reduce(2, no_, (float)opt.huber).x;
        if (!std::isfinite(cost)) throw std::runtime_error("global positioning: non-finite GPU cost");
        result.initial_cost = cost;
        double damping = opt.initial_damping;
        int small_steps = 0;
        for (int iteration = 0; iteration < opt.max_iterations; iteration++) {
            cancel::check();
            VkCommandBuffer cb = ctx_.begin();
            Push push = base(); push.f0 = (float)opt.huber;
            dispatch(cb, "gp_linearize", groups(no_), push);
            ctx_.barrier(cb);
            push.u3 = 0; push.f0 = (float)damping;
            dispatch(cb, "gp_nodes", nc_, push);
            dispatch(cb, "gp_points", groups(np_), push);
            ctx_.barrier(cb);
            push.f0 = (float)damping;
            dispatch(cb, "gp_vector", groups(dim_), push);
            ctx_.barrier(cb);
            const Float4 initial = reduce(cb, 0, dim_, 0);
            double rz = initial.x, residual0 = std::sqrt(std::max(0.0, (double)initial.y));
            if (!(residual0 > 0) || !std::isfinite(residual0)) {
                result.converged = residual0 == 0;
                break;
            }
            float beta = 0;
            for (int cg = 0; cg < opt.cg_iterations; cg++) {
                if (cg % 10 == 0) cancel::check();
                cb = ctx_.begin();
                if (cg > 0) {
                    push = base(); push.u3 = 2; push.f0 = beta;
                    dispatch(cb, "gp_vector", groups(dim_), push);
                    ctx_.barrier(cb);
                }
                push = base(); push.u3 = 1; push.f0 = (float)damping;
                dispatch(cb, "gp_nodes", nc_, push);
                dispatch(cb, "gp_points", groups(np_), push);
                ctx_.barrier(cb);
                const double denom = reduce(cb, 1, dim_, 0).x;
                if (!(denom > 0) || !std::isfinite(denom) || !(rz > 0)) break;
                cb = ctx_.begin();
                push = base(); push.u3 = 1; push.f0 = (float)(rz / denom);
                push.f1 = (float)damping;
                dispatch(cb, "gp_vector", groups(dim_), push);
                ctx_.barrier(cb);
                const Float4 next = reduce(cb, 0, dim_, 0);
                if (std::sqrt(std::max(0.0, (double)next.y)) <= opt.cg_tolerance * residual0) break;
                if (!(next.x > 0) || !std::isfinite(next.x)) break;
                beta = (float)(next.x / rz);
                rz = next.x;
            }
            cb = ctx_.begin();
            push = base(); push.u3 = 3;
            dispatch(cb, "gp_vector", groups(dim_), push);
            push.u3 = 0; push.f0 = (float)damping;
            dispatch(cb, "gp_scale_update", groups(no_), push);
            ctx_.barrier(cb);
            const double next_cost = reduce(cb, 3, no_, (float)opt.huber).x;
            result.iterations = iteration + 1;
            if (std::isfinite(next_cost) && next_cost < cost) {
                const double change = (cost - next_cost) / std::max(cost, 1e-30);
                cb = ctx_.begin();
                push = base(); push.u3 = 4;
                dispatch(cb, "gp_vector", groups(dim_), push);
                push.u3 = 1;
                dispatch(cb, "gp_scale_update", groups(no_), push);
                ctx_.barrier(cb);
                ctx_.submit(cb);
                cost = next_cost;
                result.accepted++;
                damping = std::max(1e-12, damping * 0.5);
                small_steps = change < std::max(opt.relative_tolerance, 1e-6)
                                  ? small_steps + 1 : 0;
                if (small_steps >= 3) {
                    result.converged = true;
                    break;
                }
            } else {
                small_steps = 0;
                damping = std::min(1e12, damping * 4.0);
            }
        }
        std::vector<float> vars(var_dim_);
        ctx_.download(buffers_[2], vars.data(), (VkDeviceSize)var_dim_ * 4);
        for (uint32_t i = 0; i < nc_; i++) p.cameras[i] = get(vars, 3 * i);
        for (uint32_t i = 0; i < np_; i++) p.points[i] = get(vars, 3 * (nc_ + i));
        for (uint32_t i = 0; i < no_; i++) p.log_scales[i] = vars[3 * nodes_ + i];
        result.final_cost = globalPositionCost(p, opt.huber);
        return result;
    }

private:
    static uint32_t groups(uint32_t n) { return (uint32_t)(((uint64_t)n + 63) / 64); }
    static void put(std::vector<float>& v, uint32_t i, const Vec3& x) {
        v[i] = (float)x.x; v[i + 1] = (float)x.y; v[i + 2] = (float)x.z;
    }
    static Vec3 get(const std::vector<float>& v, uint32_t i) {
        return {v[i], v[i + 1], v[i + 2]};
    }
    Push base() const { Push p; p.u0 = nc_; p.u1 = np_; p.u2 = no_; p.u4 = metric_scale_ ? 1 : 0; return p; }
    void dispatch(VkCommandBuffer cb, const char* entry, uint32_t count, const Push& p) {
        const uint32_t x = std::min(count, 65535u);
        ctx_.dispatch(cb, entry, x, p, (count + 65534u) / 65535u);
    }
    Float4 reduce(uint32_t mode, uint32_t count, float parameter) {
        return reduce(ctx_.begin(), mode, count, parameter);
    }
    Float4 reduce(VkCommandBuffer cb, uint32_t mode, uint32_t count, float parameter) {
        Push p = base(); p.u3 = mode; p.f0 = parameter;
        const uint32_t n = groups(count);
        dispatch(cb, "gp_reduce", n, p);
        ctx_.barrier(cb);
        const VkDeviceSize bytes = (VkDeviceSize)n * sizeof(Float4);
        std::vector<Float4> parts(n);
        if (bytes <= VkContext::stagingCapacity()) {
            ctx_.recordDownload(cb, buffers_[8], bytes);
            ctx_.submit(cb);
            const auto* src = static_cast<const Float4*>(ctx_.stagingDownloadPtr());
            std::copy(src, src + n, parts.begin());
        } else {
            ctx_.submit(cb);
            ctx_.download(buffers_[8], parts.data(), bytes);
        }
        double x = 0, y = 0, z = 0, w = 0;
        for (const auto& part : parts) { x += part.x; y += part.y; z += part.z; w += part.w; }
        return {(float)x, (float)y, (float)z, (float)w};
    }
    VkContext ctx_;
    std::vector<GpuBuffer> buffers_;
    uint32_t nc_, np_, no_, nodes_, dim_, var_dim_;
    bool metric_scale_;
};

}  // namespace

GlobalPositionResult solveGlobalPositionGpu(GlobalPositionProblem& p,
                                            const GlobalPositionOptions& options,
                                            int device, const std::string& selector) {
    validateGlobalPosition(p, options);
    const float huber = (float)options.huber, damping = (float)options.initial_damping;
    if (!(huber > 0) || !std::isfinite(huber) || !(damping > 0) || !std::isfinite(damping))
        throw std::invalid_argument("global positioning: options exceed GPU precision");
    const uint64_t nodes = p.cameras.size() + p.points.size();
    const uint64_t dim = 3 * nodes;
    if (p.observations.size() > UINT32_MAX / 2 || nodes > UINT32_MAX ||
        dim > UINT32_MAX / 8 || dim + p.observations.size() > UINT32_MAX)
        throw std::length_error("global positioning: GPU problem exceeds indexing limit");
    PositionGpu solver(p, device, selector);
    return solver.solve(p, options);
}

}  // namespace sfm
