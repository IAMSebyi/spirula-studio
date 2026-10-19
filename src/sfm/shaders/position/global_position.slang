[[vk::push_constant]] cbuffer PC {
    uint nc; uint np; uint no; uint mode;
    float f0; float f1; float f2; float f3;
    uint u4;
};

[[vk::binding(0)]] StructuredBuffer<uint2> obs;
[[vk::binding(1)]] StructuredBuffer<float4> ray;
[[vk::binding(2)]] RWStructuredBuffer<float> vars;
[[vk::binding(3)]] RWStructuredBuffer<float4> delta_scale;
[[vk::binding(4)]] RWStructuredBuffer<float4> residual_weight;
[[vk::binding(5)]] StructuredBuffer<uint> ranges;
[[vk::binding(6)]] StructuredBuffer<uint> adjacency;
[[vk::binding(7)]] RWStructuredBuffer<float> vec;
[[vk::binding(8)]] RWStructuredBuffer<float4> partial;

groupshared float4 sum64[64];

uint groupIndex(uint3 g) { return g.x + 65535 * g.y; }
uint dim() { return 3 * (nc + np) + no; }
uint scaleBase() { return 3 * (nc + np); }
uint offset(uint slot, uint i) { return slot * dim() + i; }
float3 read3(uint i) { return float3(vars[i], vars[i + 1], vars[i + 2]); }
float3 vec3(uint slot, uint i) {
    return float3(vec[offset(slot, i)], vec[offset(slot, i + 1)], vec[offset(slot, i + 2)]);
}
void put3(uint slot, uint i, float3 v) {
    vec[offset(slot, i)] = v.x; vec[offset(slot, i + 1)] = v.y; vec[offset(slot, i + 2)] = v.z;
}
float4 reduce64(float4 v, uint lane) {
    sum64[lane] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) sum64[lane] += sum64[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    return sum64[0];
}

[shader("compute")] [numthreads(64, 1, 1)]
void gp_linearize(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint i = groupIndex(gid) * 64 + lid.x;
    if (i >= no) return;
    uint2 o = obs[i];
    float3 d = read3(3 * (nc + o.y)) - read3(3 * o.x);
    float s = exp(vars[scaleBase() + i]);
    float3 r = ray[i].xyz - s * d;
    float length_r = length(r);
    float w = ray[i].w * (length_r > f0 ? f0 / length_r : 1.0);
    delta_scale[i] = float4(d, s);
    residual_weight[i] = float4(r, w);
    uint si = scaleBase() + i;
    // Fix one observation scale to remove the global scale gauge.
    vec[offset(0, si)] = i == 0 ? 0.0 : w * s * dot(d, r);
    vec[offset(1, si)] = i == 0 ? 1.0 : w * s * s * dot(d, d);
}

[shader("compute")] [numthreads(64, 1, 1)]
void gp_nodes(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint node = groupIndex(gid);
    if (node >= nc + np) return;
    bool camera = node < nc;
    float4 acc = float4(0, 0, 0, 0);
    for (uint j = ranges[node] + lid.x; j < ranges[node + 1]; j += 64) {
        uint i = adjacency[j];
        uint2 o = obs[i];
        float4 ds = delta_scale[i], rw = residual_weight[i];
        float ws = rw.w * ds.w;
        if (mode == 0) {
            acc.xyz += (camera ? -1.0 : 1.0) * ws * rw.xyz;
            acc.w += ws * ds.w;
        } else {
            float3 c = vec3(5, 3 * o.x);
            float3 p = vec3(5, 3 * (nc + o.y));
            float s = vec[offset(5, scaleBase() + i)];
            float3 jt = rw.w * ds.w * ds.w * (c - p - ds.xyz * s);
            acc.xyz += (camera ? 1.0 : -1.0) * jt;
        }
    }
    acc = reduce64(acc, lid.x);
    if (lid.x != 0) return;
    uint base = 3 * node;
    if (mode == 0) {
        put3(0, base, node == 0 ? float3(0, 0, 0) : acc.xyz);
        put3(1, base, node == 0 ? float3(1, 1, 1) : float3(acc.w, acc.w, acc.w));
    } else {
        float3 diag = vec3(1, base);
        float3 x = vec3(5, base);
        float3 result = (node == 0 ? float3(0, 0, 0) : acc.xyz) + f0 * max(diag, 1e-8) * x;
        put3(6, base, result);
    }
}

[shader("compute")] [numthreads(64, 1, 1)]
void gp_scale_product(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint i = groupIndex(gid) * 64 + lid.x;
    if (i >= no) return;
    uint2 o = obs[i];
    float4 ds = delta_scale[i], rw = residual_weight[i];
    uint si = scaleBase() + i;
    float3 diff = vec3(5, 3 * o.x) - vec3(5, 3 * (nc + o.y)) - ds.xyz * vec[offset(5, si)];
    float value = -rw.w * ds.w * ds.w * dot(ds.xyz, diff);
    vec[offset(6, si)] = (i == 0 ? 0.0 : value) +
        f0 * max(vec[offset(1, si)], 1e-8) * vec[offset(5, si)];
}

[shader("compute")] [numthreads(64, 1, 1)]
void gp_vector(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint i = groupIndex(gid) * 64 + lid.x;
    if (i >= dim()) return;
    float d = vec[offset(1, i)];
    if (mode == 0) {
        float r = vec[offset(0, i)];
        float z = r / (d + f0 * max(d, 1e-8));
        vec[offset(2, i)] = 0; vec[offset(3, i)] = r;
        vec[offset(4, i)] = z; vec[offset(5, i)] = z;
    } else if (mode == 1) {
        float r = vec[offset(3, i)] - f0 * vec[offset(6, i)];
        vec[offset(2, i)] += f0 * vec[offset(5, i)];
        vec[offset(3, i)] = r;
        vec[offset(4, i)] = r / (d + f1 * max(d, 1e-8));
    } else if (mode == 2) {
        vec[offset(5, i)] = vec[offset(4, i)] + f0 * vec[offset(5, i)];
    } else if (mode == 3) {
        float step = vec[offset(2, i)];
        if (i >= scaleBase()) step = clamp(step, -2.0, 2.0);
        float x = vars[i] + step;
        vec[offset(7, i)] = i >= scaleBase() ? clamp(x, -20.0, 20.0) : x;
    } else {
        vars[i] = vec[offset(7, i)];
    }
}

[shader("compute")] [numthreads(64, 1, 1)]
void gp_reduce(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint group = groupIndex(gid), i = group * 64 + lid.x;
    float4 v = float4(0, 0, 0, 0);
    if (mode == 0 && i < dim()) {
        float r = vec[offset(3, i)];
        v.xy = float2(r * vec[offset(4, i)], r * r);
    } else if (mode == 1 && i < dim()) {
        v.x = vec[offset(5, i)] * vec[offset(6, i)];
    } else if (mode >= 2 && i < no) {
        uint2 o = obs[i];
        uint cb = 3 * o.x, pb = 3 * (nc + o.y), si = scaleBase() + i;
        float3 c, p;
        float s;
        if (mode == 2) {
            c = read3(cb); p = read3(pb); s = exp(vars[si]);
        } else {
            c = vec3(7, cb); p = vec3(7, pb); s = exp(vec[offset(7, si)]);
        }
        float n = length(ray[i].xyz - s * (p - c));
        v.x = 0.5 * ray[i].w * (n <= f0 ? n * n : 2 * f0 * n - f0 * f0);
    }
    v = reduce64(v, lid.x);
    if (lid.x == 0) partial[group] = v;
}
