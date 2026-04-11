struct CameraUniforms {
    view_proj: mat4x4<f32>,
}

struct StreamVert {
    position: vec3<f32>,
    t: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<storage, read> verts: array<StreamVert>;

struct VertOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

fn streamline_color(t: f32) -> vec4<f32> {
    let cool = vec3<f32>(0.20, 0.10, 0.60);
    let mid = vec3<f32>(0.00, 0.70, 0.60);
    let warm = vec3<f32>(1.00, 0.60, 0.10);
    
    var rgb: vec3<f32>;
    if t < 0.5 {
        rgb = mix(cool, mid, t * 2.0);
    } else {
        rgb = mix(mid, warm, (t - 0.5) * 2.0);
    }
    
    let alpha = mix(0.9, 0.15, t * t);
    
    return vec4<f32>(rgb, alpha);
}

@vertex
fn vs_streamline(@builtin(vertex_index) vid: u32) -> VertOut {
    let sv = verts[vid];
    
    var out: VertOut;
    out.clip_pos = camera.view_proj * vec4<f32>(sv.position, 1.0);
    out.color = streamline_color(sv.t);
    
    return out;
}

@fragment
fn fs_streamline(in: VertOut) -> @location(0) vec4<f32> {
    return in.color;
}