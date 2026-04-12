# GPU Patricle Advection in a Vector Field

A real-time GPU particle simulation built in Rust and wgpu. 100,000 particles are advected through a time-varying 3D vector field every fram entirely on GPU, with streamlines computed on the GPU as well

## How to Run

```bash
cargo run --release
```

## Controls

|Input|Action|
|-----|------|
|Left mouse drag| Orbit camera|
|Scroll wheel| Zoom in/out|
|S|Toggle streamlines on/off|
|T|Toggle between analytical field and 3D texture lookup|
|Escape|Exit|

## What you're looking at

Particles are colored points. There are 100,000 of them. Each one is colored by how fast its moving through the field (blue means slow, cyan/yellow means medium, red means fast). They fade out as they get older. When a particles hits the boundary of the simulation volume or reaches its maximum age, it respawns at a random position.

Streamlines are the glowing lines threading through the particle cloud. Each one starts at a fixed seed point and traces forward through the field for 256 steps. They're colored purple at the seed end and fade throguh teal toward orange at the tail. Press `S` to hide them if you want to see just the particles.

Both layers use additive blending, so areas where many particles or lines overlap glow brighter. This makes the regions of high flow density naturally stand out.

## The vector field

The field is defined analytically, no external data is loaded. The formula is:
```wgsl
fn field_analytical(p: vec3<f32>, t: f32) -> vec3<f32> {
    let s = p * 0.15;
    
    let vx = sin(s.y + t * 0.31) + 0.4 * cos(s.z + t * 0.17);
    let vy = cos(s.x + t * 0.23) + 0.4 * sin(s.z - t * 0.11);
    let vz = sin(s.x - s.y * 0.5 + t * 0.13) * 0.8;
    
    return vec3<f32>(vx, vy, vz);
}
```
The time coefficients (`0.31`, `0.23`, `0.13`, etc.) are chosen to be irrational relative to each other so the field never exactly repeats. The cross-axis terms (`sin(s.y)` driving the X-component, etc.) create swirling behaviour rather than straight flow lines.

## Integration

Each particle is adevcted using a 4rth-order Runge-Kutta (RK4) step every frame. RK4 evaluated field at four points per step and combines them with Simpson's-rule weighting:
```wgsl
fn rk4(p: vec3<f32>, t: f32, dt: f32) -> vec3<f32> {
    let k1 = field(p, t);
    let k2 = field(p + k1 * (dt * 0.5), t + dt * 0.5);
    let k3 = field(p + k2 * (dt * 0.5), t + dt * 0.5);
    let k4 = field(p + k3 * dt, t + dt);
    
    return p + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (dt / 6.0);
}
```
This is more expensive than Euler integration (4 field evaluations instead of 1) but significantly more accurate, which matters when particles need to follow curved field lines faithfully over many steps without drifting.
Streamlines use the same RK4 integrator but hold time fixed during tracing, they show the field frozen at the current instant rather than following a particle through time.

## Ping-pong buffer architecture

This was one of the trickier things to get right. The core problem is that a compute shader cannot safely read and write the same buffer in one dispatch. If the GPU processes particles 8000 before particle 7999, and particle 8000's new position was written before particle 7999 read it, the result depends on the execution order, which the GPU does not guarantee.
The solution is two identically-sized buffers, A and B, that alternate roles each frame:
```
Frame 0: read A -> compute -> write B -> render B
Frame 1: read B -> compute -> write A -> render A
Frame 2: read A -> compute -> write B -> render B
```
Every particle always reads from last frame's clean state and writes to a separate buffer. There is no aliasing possible.
The implementation builds two bind groups at startup, one for each direction, and selects between them with a single XOR each frame:
```rust
self.ping_pong.current ^= 1;
cpass.set_bind_group(0, &self.compute_bgs[self.pin_pong.current], &[]);
```
No per-frame allocation. The render pass always reads from whichever buffer was just written, so it always sees the freshest positions.

## 3D texture acceleration structure

By default the field i sevaluated analytically in the shader. Press `T` to switch to texture-based lookup.

The idea is straightforward: instead of computing 16 trig functions per particle per frame, pre-sample the field into a 64x64x64 `Rgba16Float` 3D texture and let the GPU's texture hardware do the application.

Each lookup becomes a single `textureSampleLevel` call. The GPU performs trilinear interpolation across the 8 surrounding texels automatically, no manual lerp code needed. The texture takes 2MB of VRAM.

One thing that caught me during implementation: `textureSample` is not legal in a compute shader. It relies on implicit scree-space derivatives (`dpdx`/`dpdy`) to select a mip leel, and those derivatives only exist inside fragment shader invocations where adjacent pixels are being processed in parallel. In a compute shader there is no such context. The fix is `textureSampleLevel` which takes an explicit mip level argument (passing `0.0` since the texture only has one mip anyway).

The texture is re-uploaded from the CPU every 0.5 seconds to stay roughly in sync with the animated field. This cadence is a deliberate tradeoff, uploadinf every frame would cost 2MB of CPU->GPU bandwidth at 60fps (120MB/s just for the field), which is wasteful when the field changes slowly. At 0.5s intervals the texture is slightly behind the analytical field, which you can actually see if you switch between modes with `T` while watching closely.

The `f16` format introduces about 0.1% relative error compared tot eh `f32` analytical evaluation. In practice this is invisible, the particles follow the same flow patterns in both modes.