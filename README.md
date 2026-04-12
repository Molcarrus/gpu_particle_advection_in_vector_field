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

There are 100,000 colored particles moving through the field. The color shows their speed — blue is slow, cyan and yellow are medium, red is fast. They fade out as they get older, and when they hit the edge of the volume or reach their max age they just respawn at a random spot.

The glowing lines that weave through the particle cloud are streamlines. Each one starts at a fixed seed point and traces forward for 256 steps. They go from purple at the start to teal and then orange at the end. Hit S if they feel too busy and you just want to see the particles.

I used additive blending for both the particles and the streamlines, so places where a lot of them overlap glow brighter. It makes the high-flow regions stand out naturally.

## The vector field

I defined the field analytically in the shader.
```wgsl
fn field_analytical(p: vec3<f32>, t: f32) -> vec3<f32> {
    let s = p * 0.15;
    
    let vx = sin(s.y + t * 0.31) + 0.4 * cos(s.z + t * 0.17);
    let vy = cos(s.x + t * 0.23) + 0.4 * sin(s.z - t * 0.11);
    let vz = sin(s.x - s.y * 0.5 + t * 0.13) * 0.8;
    
    return vec3<f32>(vx, vy, vz);
}
```
I picked the time coefficients so they’re irrational relative to each other and the pattern never repeats exactly. The cross terms (like `sin(s.y`) affecting the `x` velocity) create the nice swirling motion instead of straight lines.

## Integration

I advect each particle every frame using fourth-order Runge-Kutta (RK4):
```wgsl
fn rk4(p: vec3<f32>, t: f32, dt: f32) -> vec3<f32> {
    let k1 = field(p, t);
    let k2 = field(p + k1 * (dt * 0.5), t + dt * 0.5);
    let k3 = field(p + k2 * (dt * 0.5), t + dt * 0.5);
    let k4 = field(p + k3 * dt, t + dt);
    
    return p + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (dt / 6.0);
}
```
It’s more work than simple Euler (four field lookups instead of one), but the extra accuracy really matters so the particles follow the curved flow lines properly over time without drifting. The streamlines use the same RK4 code but with time frozen so they show the field at the current moment.

## Ping-pong buffer architecture

This part was one of the trickier things to get right. A compute shader can’t safely read from and write to the same buffer in one dispatch because the GPU doesn’t guarantee execution order. If particle 8000 writes its new position before particle 7999 reads the old one, you get random glitches.

I solved it with two identical buffers, A and B, that swap every frame:
```
Frame 0: read A -> compute -> write B -> render B
Frame 1: read B -> compute -> write A -> render A
Frame 2: read A -> compute -> write B -> render B
```
Each particle always reads the clean previous-frame data and writes to the other buffer. I create the two bind groups once at startup and just flip between them with a single XOR:
```rust
self.ping_pong.current ^= 1;
cpass.set_bind_group(0, &self.compute_bgs[self.pin_pong.current], &[]);
```
No per-frame allocations, and the render pass always sees the freshest positions.

## 3D texture acceleration structure

By default the field is evaluated analytically. Press `T` to switch to a 64×64×64 `Rgba16Float` 3D texture lookup instead.

It turns 16 trig calls per particle into one `textureSampleLevel` and lets the GPU do the trilinear interpolation for free. The texture is only 2 MB.

One thing I ran into: `textureSample` doesn’t work in compute shaders because it needs screen-space derivatives that only exist in fragment shaders. So I use `textureSampleLevel(..., 0.0)` with an explicit mip level.

I refresh the texture from the CPU every 0.5 seconds. Updating every frame would be 120 MB/s of bandwidth for basically no gain, so this felt like the right tradeoff. The tiny lag is only noticeable when you toggle `T` back and forth, and the `f16` format adds about 0.1 % error which you can’t see.

That’s the project! I had a lot of fun building it and learning these GPU patterns. 