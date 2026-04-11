struct CameraUniforms {
    view_proj: mat4x4<f32>,
}

struct Particle {
    position: vec3<f32>,
    age: f32,
    velocity: vec3<f32>,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<storage, read> particles: array<Particle>;

struct VertOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

fn speed_color(t: f32) -> vec3<f32> {
    let blue = vec3<f32>(0.05, 0.10, 0.80);
    let cyan = vec3<f32>(0.00, 0.80, 0.70);
    let yellow = vec3<f32>(1.00, 0.85, 0.00);
    let red = vec3<f32>(1.00, 0.15, 0.05);
    
    if t < 0.33 {
        return mix(blue, cyan, t / 0.33);
    } else if t < 0.66 {
        return mix(cyan, yellow, (t - 0.33) / 0.33);
    } else {
        return mix(yellow, red, (t - 0.66) / 0.34);
    }
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertOut {
    let p = particles[vid];
    let speed = length(p.velocity);
    
    let speed_t = clamp(speed / 2.0, 0.0, 1.0);
    
    let age_t = p.age / 10.0;
    let alpha = clamp(1.0 - age_t * age_t, 0.1, 1.0);
    
    var out: VertOut;
    out.clip_pos = camera.view_proj * vec4<f32>(p.position, 1.0);
    out.color = vec4<f32>(speed_color(speed_t), alpha);
    
    return out;
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    return in.color;
}