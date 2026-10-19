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
uint dim() { return 3 * (nc + np); }
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
float scaleDiagonal(float4 ds, float4 rw) {
    float h = rw.w * ds.w * ds.w * dot(ds.xyz, ds.xyz);
    return h + f0 * max(h, 1e-8);
}
float3 scaleRhsCorrection(uint i, float4 ds, float4 rw) {
    if (i == 0 && u4 == 0) return float3(0, 0, 0);
    float ws = rw.w * ds.w;
    return ds.xyz * (ws * ds.w * ws * dot(ds.xyz, rw.xyz) / scaleDiagonal(ds, rw));
}
float3 reducedProduct(uint i, float4 ds, float4 rw, float3 diff) {
    float wss = rw.w * ds.w * ds.w;
    if (i != 0 || u4 != 0) diff -= ds.xyz * (wss * dot(ds.xyz, diff) / scaleDiagonal(ds, rw));
    return wss * diff;
}

[shader("compute")] [numthreads(64, 1, 1)]
void gp_linearize(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint i = groupIndex(gid) * 64 + lid.x;
    if (i >= no) return;
    uint2 o = obs[i];
    float3 d = read3(3 * (nc + o.y)) - read3(3 * o.x) - ray[2 * i + 1].xyz;
    float s = exp(vars[scaleBase() + i]);
    float3 r = ray[2 * i].xyz - s * d;
    float length_r = length(r);
    float w = ray[2 * i].w * (length_r > f0 ? f0 / length_r : 1.0);
    delta_scale[i] = float4(d, s);
    residual_weight[i] = float4(r, w);
}

[shader("compute")] [numthreads(64, 1, 1)]
void gp_nodes(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint node = groupIndex(gid);
    if (node >= nc) return;
    float4 acc = float4(0, 0, 0, 0);
    for (uint j = ranges[node] + lid.x; j < ranges[node + 1]; j += 64) {
        uint i = adjacency[j];
        uint2 o = obs[i];
        float4 ds = delta_scale[i], rw = residual_weight[i];
        float ws = rw.w * ds.w;
        if (mode == 0) {
            acc.xyz += -ws * rw.xyz + scaleRhsCorrection(i, ds, rw);
            acc.w += ws * ds.w;
        } else {
            float3 c = vec3(5, 3 * o.x);
            float3 p = vec3(5, 3 * (nc + o.y));
            float3 jt = reducedProduct(i, ds, rw, c - p);
            acc.xyz += jt;
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
void gp_points(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint point = groupIndex(gid) * 64 + lid.x;
    if (point >= np) return;
    uint node = nc + point;
    float4 acc = float4(0, 0, 0, 0);
    for (uint j = ranges[node]; j < ranges[node + 1]; j++) {
        uint i = adjacency[j];
        uint2 o = obs[i];
        float4 ds = delta_scale[i], rw = residual_weight[i];
        float ws = rw.w * ds.w;
        if (mode == 0) {
            acc.xyz += ws * rw.xyz - scaleRhsCorrection(i, ds, rw);
            acc.w += ws * ds.w;
        } else {
            float3 c = vec3(5, 3 * o.x);
            float3 p = vec3(5, 3 * node);
            acc.xyz -= reducedProduct(i, ds, rw, c - p);
        }
    }
    uint base = 3 * node;
    if (mode == 0) {
        put3(0, base, acc.xyz);
        put3(1, base, float3(acc.w, acc.w, acc.w));
    } else {
        float3 diag = vec3(1, base);
        put3(6, base, acc.xyz + f0 * max(diag, 1e-8) * vec3(5, base));
    }
}

[shader("compute")] [numthreads(64, 1, 1)]
void gp_scale_update(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint i = groupIndex(gid) * 64 + lid.x;
    if (i >= no) return;
    uint si = scaleBase() + i;
    if (mode == 0) {
        uint2 o = obs[i];
        float4 ds = delta_scale[i], rw = residual_weight[i];
        float3 diff = vec3(2, 3 * o.x) - vec3(2, 3 * (nc + o.y));
        float wss = rw.w * ds.w * ds.w;
        float rhs = rw.w * ds.w * dot(ds.xyz, rw.xyz);
        float step = i == 0 && u4 == 0 ? 0.0 : (rhs + wss * dot(ds.xyz, diff)) / scaleDiagonal(ds, rw);
        residual_weight[i].x = clamp(vars[si] + clamp(step, -2.0, 2.0), -20.0, 20.0);
    } else {
        vars[si] = residual_weight[i].x;
    }
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
        vec[offset(7, i)] = vars[i] + vec[offset(2, i)];
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
            c = vec3(7, cb); p = vec3(7, pb); s = exp(residual_weight[i].x);
        }
        float n = length(ray[2 * i].xyz - s * (p - c - ray[2 * i + 1].xyz));
        v.x = 0.5 * ray[2 * i].w * (n <= f0 ? n * n : 2 * f0 * n - f0 * f0);
    }
    v = reduce64(v, lid.x);
    if (lid.x == 0) partial[group] = v;
}
