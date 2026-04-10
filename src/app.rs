use std::sync::Arc;

use winit::{application::ApplicationHandler, event::WindowEvent, window::Window};

use crate::renderer::context::GpuContext;

pub struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    ctx: GpuContext,
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
                .create_window(Window::default_attributes().with_title("Particle Advection"))
                .unwrap(),
        );
        let ctx = pollster::block_on(GpuContext::new(window.clone()));
        self.state = Some(AppState { window, ctx });
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let state = match self.state.as_mut() {
            Some(s) => s,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(sz) => state.ctx.resize(sz),
            WindowEvent::RedrawRequested => {
                state.window.request_redraw();
            }
            _ => {}
        }
    }
}
