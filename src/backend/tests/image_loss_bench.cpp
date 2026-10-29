// Fixed-input timing for SSIM and the bilagrid PPISP image kernels.
// image_loss_bench [ssim|ssim_bt|bilagrid] [width] [height] [iters] [dump]

#include <kernels/loss/PerPixelLoss.cuh>
#include <kernels/bilagrid/BilagridBindings.h>

#include <chrono>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

using backend::MemcpyKind;

template <typename T>
T* upload(const std::vector<T>& v) {
    auto* p = static_cast<T*>(backend::device_malloc(v.size() * sizeof(T)));
    backend::memcpy_sync(p, v.data(), v.size() * sizeof(T), MemcpyKind::HostToDevice);
    return p;
}

TorchTensorView view(const void* p, int h, int w, int c) {
    return {(uint64_t)p, 4u, {1, h, w, c}};
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "ssim_bt";
    const int w = argc > 2 ? std::atoi(argv[2]) : 4946;
    const int h = argc > 3 ? std::atoi(argv[3]) : 3286;
    const int iters = argc > 4 ? std::atoi(argv[4]) : 20;
    const bool grid = std::strcmp(mode, "bilagrid") == 0;
    const bool bt = std::strcmp(mode, "ssim_bt") == 0;
    if ((!grid && !bt && std::strcmp(mode, "ssim") != 0) || w < 2 || h < 2 || iters < 1)
        return 2;

    const size_t pixels = size_t(w) * h;
    std::mt19937 rng(20261029u);
    auto values = [&](size_t n, float lo, float hi) {
        std::vector<float> v(n);
        for (float& x : v) x = lo + (hi - lo) * float(rng() & 0xffffffu) / 16777215.f;
        return v;
    };
    float* a = upload(values(3 * pixels, 0.f, 1.f));
    float* b = upload(values(3 * pixels, 0.f, 1.f));
    float* grad = upload(values(3 * pixels, -0.001f, 0.001f));
    float* out = upload(std::vector<float>(3 * pixels));
    float* map = upload(std::vector<float>(pixels));
    constexpr int cells = 8 * 16 * 16 * 9;
    float* g = upload(values(cells, -0.02f, 0.02f));
    float* vg = upload(std::vector<float>(cells));
    float* transmittance = upload(std::vector<float>(pixels, 0.5f));
    float* losses = upload(std::vector<float>((int)LossIndex::length));
    std::array<float, (int)LossWeightIndex::length> weights{};
    weights[(int)LossWeightIndex::RgbSupL1] = 0.8f;
    std::vector<bool> needs(13, false);
    needs[0] = true;
    PerPixelGrads grads{};
    grads.v_render_rgb = view(out, h, w, 3);
    auto one = [&]() {
        if (grid) {
            backend::memset_async(vg, 0, cells * sizeof(float));
            bilagrid_ppisp_uniform_sample_forward(BilagridReader(g), PixelPtr(a), PixelOut(out),
                1, 8, 16, 16, h, w, backend::kDefaultStream);
            bilagrid_ppisp_uniform_sample_backward_v1(BilagridReader(g), PixelPtr(a), grad, vg, out,
                1, 8, 16, 16, h, w, 4, backend::kDefaultStream);
        } else {
            backend::memset_async(out, 0, 3 * pixels * sizeof(float));
            backend::memset_async(map, 0, pixels * sizeof(float));
            compute_multi_scale_per_pixel_losses(1,
                view(a, h, w, 3), view(b, h, w, 3), {}, {}, {}, {}, {},
                view(transmittance, h, w, 1), {}, {}, {}, {}, {}, {},
                false, weights, 0.2f,
                (int)(bt ? PerceptualLoss::SsimBlurTolerant : PerceptualLoss::Ssim),
                0.f, view(losses, 1, 1, (int)LossIndex::length), needs, 1, {},
                view(map, h, w, 1), (int)DensifyLossMapMode::SsimContrastStruct,
                0.9f, 0.f, grads);
        }
    };
    for (int i = 0; i < 3; ++i) one();
    backend::device_synchronize();
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) one();
    backend::device_synchronize();
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count() / iters;
    std::printf("%s: %.3f ms/call (%dx%d)\n", mode, ms, w, h);
    if (argc > 5) {
        std::ofstream f(argv[5], std::ios::binary);
        auto dump = [&](const float* p, size_t n) {
            std::vector<float> v(n);
            backend::memcpy_sync(v.data(), p, n * sizeof(float), MemcpyKind::DeviceToHost);
            f.write(reinterpret_cast<const char*>(v.data()), n * sizeof(float));
        };
        dump(out, 3 * pixels);
        if (grid) dump(vg, cells);
        else dump(map, pixels);
        if (!f) return 1;
    }
    return 0;
}
