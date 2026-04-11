struct StreamlineUniforms {
    time: f32,
    step_size: f32,
    n_steps: u32,
    n_seeds: u32,
    bbox_half: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct StreamVert {
    position: vec3<f32>,
    t: f32,
}

@group(0) @binding(0) var<uniform> uniforms: StreamlineUniforms;
@group(0) @binding(1) var<storage, read> seeds: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<StreamVert>;

fn field(p: vec3<f32>, t: f32) -> vec3<f32> {
    let s = p * 0.15;

    let vx = sin(s.y + t * 0.31) + 0.4 * cos(s.z + t * 0.17);
    let vy = cos(s.x + t * 0.23) + 0.4 * sin(s.z - t * 0.11);
    let vz = sin(s.x - s.y * 0.5 + t * 0.13) * 0.8;

    return vec3<f32>(vx, vy, vz);
}

fn rk4(p: vec3<f32>, t: f32, dt: f32) -> vec3<f32> {
    let k1 = field(p, t);
    let k2 = field(p + k1 * (dt * 0.5), t);
    let k3 = field(p + k2 * (dt * 0.5), t);
    let k4 = field(p + k3 *  dt, t);
    return p + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (dt / 6.0);
}

@compute @workgroup_size(64)
fn cs_streamline(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seed_idx = gid.x;
    if seed_idx >= uniforms.n_seeds {
        return;
    }

    var pos  = seeds[seed_idx].xyz;
    let base = seed_idx * uniforms.n_steps;
    let bbox = uniforms.bbox_half;

    for (var i = 0u; i < uniforms.n_steps; i++) {
        output[base + i] = StreamVert(
            pos,
            f32(i) / f32(uniforms.n_steps - 1u),
        );

        let next = rk4(pos, uniforms.time, uniforms.step_size);

        if any(abs(next) > vec3<f32>(bbox)) {
            for (var j = i + 1u; j < uniforms.n_steps; j++) {
                output[base + j] = StreamVert(pos, 1.0);
            }
            return;
        }

        pos = next;
    }
}