use std::mem;

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use rand::RngExt;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BlendComponent, BlendFactor, BlendState, Buffer,
    BufferBindingType, BufferUsages, ColorTargetState, ColorWrites, CommandEncoder,
    CompareFunction, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    DepthBiasState, DepthStencilState, Device, FragmentState, MultisampleState,
    PipelineLayoutDescriptor, PrimitiveState, Queue, RenderPass, RenderPipeline,
    RenderPipelineDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    StencilState, TextureFormat, VertexState, wgt::BufferDescriptor,
};

use crate::renderer::ping_pong::PingPongBuffers;

pub const NUM_PARTICLES: usize = 100_000;
pub const MAX_AGE: f32 = 10.0;
pub const BBOX_HALF: f32 = 15.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub struct Particle {
    pub position: [f32; 3],
    pub age: f32,
    pub velocity: [f32; 3],
    pub _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SimUniforms {
    pub dt: f32,
    pub time: f32,
    pub max_age: f32,
    pub bbox_half: f32,
    pub seed: u32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraUniforms {
    pub view_proj: [[f32; 4]; 4],
}

impl CameraUniforms {
    pub fn from_mat4(m: Mat4) -> Self {
        Self {
            view_proj: m.to_cols_array_2d(),
        }
    }
}

pub struct ParticleSystem {
    pub n: usize,
    ping_pong: PingPongBuffers,
    sim_uniform_buf: Buffer,
    camera_uniform_buf: Buffer,
    compute_pipeline: ComputePipeline,
    compute_bgs: [BindGroup; 2],
    render_pipeline: RenderPipeline,
    camera_bg: BindGroup,
    render_bgs: [BindGroup; 2],
}

impl ParticleSystem {
    pub fn new(device: &Device, surface_format: TextureFormat) -> Self {
        let mut rng = rand::rng();
        let initial = (0..NUM_PARTICLES)
            .map(|_| Particle {
                position: [
                    rng.random_range(-BBOX_HALF..BBOX_HALF),
                    rng.random_range(-BBOX_HALF..BBOX_HALF),
                    rng.random_range(-BBOX_HALF..BBOX_HALF),
                ],
                age: rng.random_range(0.0..MAX_AGE),
                velocity: [0.0, 0.0, 0.0],
                _pad: 0.0,
            })
            .collect::<Vec<_>>();

        let initial_bytes: &[u8] = bytemuck::cast_slice(&initial);

        let ping_pong = PingPongBuffers::new(device, initial_bytes, BufferUsages::VERTEX);

        let sim_uniform_buf = device.create_buffer(&BufferDescriptor {
            label: Some("SimUniforms"),
            size: mem::size_of::<SimUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_uniform_buf = device.create_buffer(&BufferDescriptor {
            label: Some("CameraUniforms"),
            size: mem::size_of::<CameraUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let compute_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Particle Compute"),
            source: ShaderSource::Wgsl(include_str!("../shaders/particle_compute.wgsl").into()),
        });

        let render_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Particle Render"),
            source: ShaderSource::Wgsl(include_str!("../shaders/particle_render.wgsl").into()),
        });

        let compute_bgl = Self::make_compute_bgl(device);
        let camera_bgl = Self::make_camera_bgl(device);
        let storage_bgl = Self::make_storage_bgl(device);

        let compute_pipeline = Self::make_compute_pipeline(device, &compute_bgl, &compute_shader);
        let render_pipeline = Self::make_render_pipeline(
            device,
            surface_format,
            &camera_bgl,
            &storage_bgl,
            &render_shader,
        );

        let compute_bgs =
            Self::make_compute_bind_groups(device, &compute_bgl, &sim_uniform_buf, &ping_pong);

        let camera_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Camera BGL"),
            layout: &camera_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: camera_uniform_buf.as_entire_binding(),
            }],
        });

        let render_bgs = Self::make_render_bind_groups(device, &storage_bgl, &ping_pong);

        Self {
            n: NUM_PARTICLES,
            ping_pong,
            sim_uniform_buf,
            camera_uniform_buf,
            compute_pipeline,
            compute_bgs,
            render_pipeline,
            camera_bg,
            render_bgs,
        }
    }

    pub fn update(
        &mut self,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        dt: f32,
        time: f32,
        camera: CameraUniforms,
    ) {
        self.ping_pong.swap();

        queue.write_buffer(&self.camera_uniform_buf, 0, bytemuck::bytes_of(&camera));

        let sim = SimUniforms {
            dt,
            time,
            max_age: MAX_AGE,
            bbox_half: BBOX_HALF,
            seed: (time * 1_000.0) as u32 ^ 0xDEAD_BEEF,
            _pad: [0; 3],
        };
        queue.write_buffer(&self.sim_uniform_buf, 0, bytemuck::bytes_of(&sim));

        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Particle Advection"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.compute_pipeline);
        cpass.set_bind_group(0, &self.compute_bgs[self.ping_pong.current], &[]);
        let workgtoups = (self.n as u32 + 255) / 255;
        cpass.dispatch_workgroups(workgtoups, 1, 1);
    }

    pub fn render<'rp>(&'rp self, rpass: &mut RenderPass<'rp>) {
        rpass.set_pipeline(&self.render_pipeline);
        rpass.set_bind_group(0, &self.camera_bg, &[]);
        rpass.set_bind_group(1, &self.render_bgs[self.ping_pong.current], &[]);
        rpass.draw(0..self.n as u32, 0..1);
    }

    fn make_compute_bgl(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Compute BGL"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn make_camera_bgl(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Camera BGL"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    fn make_storage_bgl(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Storage BGL"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    fn make_compute_pipeline(
        device: &Device,
        bgl: &BindGroupLayout,
        shader: &ShaderModule,
    ) -> ComputePipeline {
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Compute PL"),
            bind_group_layouts: &[Some(bgl)],
            ..Default::default()
        });
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&layout),
            module: shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn make_render_pipeline(
        device: &Device,
        surface_format: TextureFormat,
        camera_bgl: &BindGroupLayout,
        storage_bgl: &BindGroupLayout,
        shader: &ShaderModule,
    ) -> RenderPipeline {
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Render PL"),
            bind_group_layouts: &[Some(camera_bgl), Some(storage_bgl)],
            ..Default::default()
        });
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Less),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: surface_format,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::SrcAlpha,
                            dst_factor: BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: BlendComponent::OVER,
                    }),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        })
    }

    fn make_compute_bind_groups(
        device: &Device,
        layout: &BindGroupLayout,
        uniforms: &Buffer,
        pp: &PingPongBuffers,
    ) -> [BindGroup; 2] {
        let make = |write_idx: usize| {
            let read_idx = 1 - write_idx;
            device.create_bind_group(&BindGroupDescriptor {
                label: Some("Compute BG"),
                layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: uniforms.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: pp.buffers[read_idx].as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: pp.buffers[write_idx].as_entire_binding(),
                    },
                ],
            })
        };

        [make(0), make(1)]
    }

    fn make_render_bind_groups(
        device: &Device,
        layout: &BindGroupLayout,
        pp: &PingPongBuffers,
    ) -> [BindGroup; 2] {
        let make = |idx: usize| {
            device.create_bind_group(&BindGroupDescriptor {
                label: Some("Render BG"),
                layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: pp.buffers[idx].as_entire_binding(),
                }],
            })
        };

        [make(0), make(1)]
    }
}
