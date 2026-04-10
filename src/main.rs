use winit::event_loop::EventLoop;

pub mod app;
pub mod camera;
pub mod renderer;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = app::App::new_pending();
    event_loop.run_app(&mut app).unwrap();
}
