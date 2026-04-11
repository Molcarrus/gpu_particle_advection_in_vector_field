use std::{sync::Arc, time::Instant};

use wgpu::{CommandEncoderDescriptor, Instance, InstanceDescriptor};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use crate::{
    camera::OrbitCamera,
    renderer::{
        context::GpuContext,
        particle_system::{CameraUniforms, ParticleSystem},
        streamlines::StreamlineSystem,
    },
};

pub struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    ctx: GpuContext,
    particle_system: ParticleSystem,
    streamlines: StreamlineSystem,
    camera: OrbitCamera,
    last_frame: Instant,
    elapsed: f32,
    mouse_pressed: bool,
    last_mouse: Option<(f32, f32)>,
}

impl App {
    pub fn new_pending() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("GPU Particle Advection")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720)),
                )
                .unwrap(),
        );
        let ctx = pollster::block_on(GpuContext::new(window.clone()));

        let particle_system = ParticleSystem::new(&ctx.device, ctx.surface_config.format);

        let streamlines = StreamlineSystem::new(
            &ctx.device,
            &ctx.queue,
            ctx.surface_config.format,
            particle_system.camera_buf(),
        );

        let size = window.inner_size();
        let camera = OrbitCamera::new(size.width as f32 / size.height.max(1) as f32);

        self.state = Some(AppState {
            window,
            ctx,
            particle_system,
            streamlines,
            camera,
            last_frame: Instant::now(),
            elapsed: 0.0,
            mouse_pressed: false,
            last_mouse: None,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let s = match self.state.as_mut() {
            Some(s) => s,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                s.ctx.resize(size);
                s.camera.resize(size.width, size.height);
                s.window.request_redraw();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match key {
                KeyCode::Escape => event_loop.exit(),

                KeyCode::KeyS => {
                    s.streamlines.visible = !s.streamlines.visible;
                    let label = if s.streamlines.visible { "ON" } else { "OFF" };
                    log::info!("Streamlines: {label}");
                }

                _ => {}
            },
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                s.mouse_pressed = state == ElementState::Pressed;
                if !s.mouse_pressed {
                    s.last_mouse = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let pos = (position.x as f32, position.y as f32);
                if s.mouse_pressed {
                    if let Some((lx, ly)) = s.last_mouse {
                        s.camera.orbit(pos.0 - lx, pos.1 - ly);
                    }
                }
                s.last_mouse = Some(pos);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.05,
                };
                s.camera.zoom(scroll);
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(s.last_frame).as_secs_f32().min(0.05);
                s.last_frame = now;
                s.elapsed += dt;

                let frame = match s.ctx.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(texture) => texture,
                    wgpu::CurrentSurfaceTexture::Occluded
                    | wgpu::CurrentSurfaceTexture::Timeout => return,
                    wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
                        drop(texture);
                        s.ctx
                            .surface
                            .configure(&s.ctx.device, &s.ctx.surface_config);
                        return;
                    }
                    wgpu::CurrentSurfaceTexture::Outdated => {
                        s.ctx
                            .surface
                            .configure(&s.ctx.device, &s.ctx.surface_config);
                        return;
                    }
                    wgpu::CurrentSurfaceTexture::Validation => {
                        unreachable!("No error scope registered, so validation errors will panic")
                    }
                    wgpu::CurrentSurfaceTexture::Lost => {
                        let instance =
                            Instance::new(InstanceDescriptor::new_without_display_handle());
                        s.ctx.surface = instance.create_surface(s.window.clone()).unwrap();
                        s.ctx
                            .surface
                            .configure(&s.ctx.device, &s.ctx.surface_config);
                        return;
                    }
                };

                let frame_view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(s.ctx.surface_config.format.add_srgb_suffix()),
                    ..Default::default()
                });

                let camera_uniforms = CameraUniforms::from_mat4(s.camera.view_proj());

                let mut encoder = s
                    .ctx
                    .device
                    .create_command_encoder(&CommandEncoderDescriptor {
                        label: Some("Frame Encoder"),
                    });

                s.particle_system.update(
                    &s.ctx.queue,
                    &mut encoder,
                    dt,
                    s.elapsed,
                    camera_uniforms,
                );

                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Main Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &frame_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.02,
                                    g: 0.02,
                                    b: 0.05,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &s.ctx.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Discard,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });

                    s.particle_system.render(&mut rpass);
                }

                s.ctx.queue.submit(std::iter::once(encoder.finish()));
                frame.present();

                s.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
    }
}
