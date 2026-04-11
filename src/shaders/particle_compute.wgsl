struct Particle {
    position: vec3<f32>,
    age: f32,
    velocity: vec3<f32>,
    _pad: f32,
}

struct SimUniforms {
    dt: f32,
    time: f32,
    max_age: f32,
    bbox_half: f32,
    seed: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read> src: array<Particle>;
@group(0) @binding(2) var<storage, read_write> dst: array<Particle>;

fn field(p: vec3<f32>, t: f32) -> vec3<f32> {
    let s = p * 0.15;
    
    let vx = sin(s.y + t * 0.31) + 0.4 * cos(s.z + t * 0.17);
    let vy = cos(s.x + t * 0.23) + 0.4 * sin(s.z - t * 0.11);
    let vz = sin(s.x - s.y * 0.5 + t * 0.13) * 0.8;
    
    return vec3<f32>(vx, vy, vz);
}

fn rk4(p: vec3<f32>, t: f32, dt: f32) -> vec3<f32> {
    let k1 = field(p, t);
    let k2 = field(p + k1 * (dt * 0.5), t + dt * 0.5);
    let k3 = field(p + k2 * (dt * 0.5), t + dt * 0.5);
    let k4 = field(p + k3 * dt, t + dt);
    
    return p + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (dt / 6.0);
}

fn pcg(v: u32) -> u32 {
    var s : u32 = v * 747796405u + 2891336453u;
    s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (s >> 22u) ^ s;
}

fn rand_f32(seed: u32) -> f32 {
    return f32(pcg(seed)) / 4294967295.0;  
}

fn random_position(idx: u32, frame_seed: u32, bbox: f32) -> vec3<f32> {
    let s0 = pcg(idx ^ frame_seed);
    let s1 = pcg(s0 + 1u);
    let s2 = pcg(s1 + 1u);
    return vec3<f32>(
        (rand_f32(s0) * 2.0 - 1.0) * bbox,
        (rand_f32(s1) * 2.0 - 1.0) * bbox,
        (rand_f32(s2) * 2.0 - 1.0) * bbox,
    );
}

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    
    if idx >= arrayLength(&src) {
        return;
    }
    
    var p = src[idx];
    
    let oob = any(abs(p.position) > vec3<f32>(uniforms.bbox_half));
    let old = p.age >= uniforms.max_age;
    
    if oob || old {
        p.position = random_position(idx, uniforms.seed, uniforms.bbox_half);
        p.age = 0.0;
        p.velocity = field(p.position, uniforms.time);
    } else {
        let new_pos = rk4(p.position, uniforms.time, uniforms.dt);
        p.velocity = field(new_pos, uniforms.time);
        p.position = new_pos;
        p.age += uniforms.dt;
    }
    
    dst[idx] = p;
}