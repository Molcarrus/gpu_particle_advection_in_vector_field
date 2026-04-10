use std::sync::Arc;

use wgpu::{
    wgt::{DeviceDescriptor, TextureDescriptor, TextureViewDescriptor},
    *,
};
use winit::window::Window;

pub struct GpuContext {
    pub surface: Surface<'static>,
    pub device: Device,
    pub queue: Queue,
    pub surface_config: SurfaceConfiguration,
    pub depth_texture: Texture,
    pub depth_view: TextureView,
}

impl GpuContext {
    pub async fn new(window: Arc<Window>) -> Self {
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default())
            .await
            .unwrap();

        let size = window.inner_size();
        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let (depth_texture, depth_view) = Self::create_depth(&device, &surface_config);

        Self {
            surface,
            device,
            queue,
            surface_config,
            depth_texture,
            depth_view,
        }
    }

    pub fn create_depth(device: &Device, config: &SurfaceConfiguration) -> (Texture, TextureView) {
        let tex = device.create_texture(&TextureDescriptor {
            label: Some("Depth"),
            size: Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&TextureViewDescriptor::default());

        (tex, view)
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);
        (self.depth_texture, self.depth_view) =
            Self::create_depth(&self.device, &self.surface_config);
    }
}
