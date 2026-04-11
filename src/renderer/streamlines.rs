use bytemuck::{Pod, Zeroable};
use rand::RngExt;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BlendComponent, BlendState, Buffer, BufferBindingType,
    BufferUsages, ColorTargetState, ColorWrites, CommandEncoder, CompareFunction,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, DepthBiasState,
    DepthStencilState, Device, FragmentState, MultisampleState, PipelineLayoutDescriptor,
    PrimitiveState, Queue, RenderPass, RenderPipeline, RenderPipelineDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StencilState, TextureFormat, VertexState,
    util::{BufferInitDescriptor, DeviceExt},
    wgt::BufferDescriptor,
};

pub const N_SEEDS: u32 = 512;
pub const N_STEPS: u32 = 256;
pub const STEP_SIZE: f32 = 0.12;
const TOTAL_VERTS: u32 = N_SEEDS * N_STEPS;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct StreamlineUniforms {
    pub time: f32,
    pub step_size: f32,
    pub n_steps: u32,
    pub n_seeds: u32,
    pub bbox_half: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct StreamVert {
    pub position: [f32; 3],
    pub t: f32,
}

pub struct StreamlineSystem {
    uniform_buf: Buffer,
    seed_buf: Buffer,
    vertex_buf: Buffer,
    compute_pipeline: ComputePipeline,
    compute_bg: BindGroup,
    index_buf: Buffer,
    index_count: u32,
    render_pipeline: RenderPipeline,
    camera_bg: BindGroup,
    render_bg: BindGroup,
    pub visible: bool,
}

impl StreamlineSystem {
    pub fn new(
        device: &Device,
        queue: &Queue,
        surface_format: TextureFormat,
        camera_buf: &Buffer,
    ) -> Self {
        let seeds = Self::generate_seeds(N_SEEDS, 15.0);
        let seed_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Seed Buffer"),
            contents: bytemuck::cast_slice(&seeds),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let uniform_buf = device.create_buffer(&BufferDescriptor {
            label: Some("Streamline Uniforms"),
            size: std::mem::size_of::<StreamlineUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vertex_buf = device.create_buffer(&BufferDescriptor {
            label: Some("Streamline Vertices"),
            size: (TOTAL_VERTS as usize * std::mem::size_of::<StreamVert>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indices = Self::build_index_buffer(N_SEEDS, N_STEPS);
        let index_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Streamline Indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: BufferUsages::INDEX,
        });
        let index_count = indices.len() as u32;

        let compute_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Streamline Compute"),
            source: ShaderSource::Wgsl(include_str!("../shaders/streamline_compute.wgsl").into()),
        });
        let render_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Streamline Render"),
            source: ShaderSource::Wgsl(include_str!("../shaders/streamline_render.wgsl").into()),
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

        let compute_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Streamline Compute BG"),
            layout: &compute_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: seed_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: vertex_buf.as_entire_binding(),
                },
            ],
        });

        let camera_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Streamline Camera BG"),
            layout: &camera_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        let render_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Streamline Render BG"),
            layout: &storage_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: vertex_buf.as_entire_binding(),
            }],
        });

        let initial_unifroms = StreamlineUniforms {
            time: 0.0,
            step_size: STEP_SIZE,
            n_steps: N_STEPS,
            n_seeds: N_SEEDS,
            bbox_half: 15.0,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&initial_unifroms));

        Self {
            uniform_buf,
            seed_buf,
            vertex_buf,
            compute_pipeline,
            compute_bg,
            index_buf,
            index_count,
            render_pipeline,
            camera_bg,
            render_bg,
            visible: true,
        }
    }

    pub fn update(&self, queue: &Queue, encoder: &mut CommandEncoder, time: f32) {
        let uniforms = StreamlineUniforms {
            time,
            step_size: STEP_SIZE,
            n_steps: N_STEPS,
            n_seeds: N_SEEDS,
            bbox_half: 15.0,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Streamline Compute"),
            ..Default::default()
        });
        cpass.set_pipeline(&self.compute_pipeline);
        cpass.set_bind_group(0, &self.compute_bg, &[]);

        let workgroups = (N_SEEDS + 63) / 64;
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    pub fn render<'rp>(&'rp self, rpass: &mut RenderPass<'rp>) {
        if !self.visible {
            return;
        }
        rpass.set_pipeline(&self.render_pipeline);
        rpass.set_bind_group(0, &self.camera_bg, &[]);
        rpass.set_bind_group(1, &self.render_bg, &[]);
        rpass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..self.index_count, 0, 0..1);
    }

    fn generate_seeds(n: u32, bbox: f32) -> Vec<[f32; 4]> {
        let mut rng = rand::rng();
        let cbrt = (n as f32).cbrt().ceil() as u32;
        let mut out = Vec::with_capacity(n as usize);

        'outer: for xi in 0..cbrt {
            for yi in 0..cbrt {
                for zi in 0..cbrt {
                    if out.len() >= n as usize {
                        break 'outer;
                    }

                    let gx = (xi as f32 / cbrt as f32) * 2.0 * bbox - bbox;
                    let gy = (yi as f32 / cbrt as f32) * 2.0 * bbox - bbox;
                    let gz = (zi as f32 / cbrt as f32) * 2.0 * bbox - bbox;

                    let jitter = bbox * 0.15;
                    let jx = rng.random_range(-jitter..jitter);
                    let jy = rng.random_range(-jitter..jitter);
                    let jz = rng.random_range(-jitter..jitter);

                    out.push([gx + jx, gy + jy, gz + jz, 0.0]);
                }
            }
        }

        while out.len() < n as usize {
            out.push([
                rng.random_range(-bbox..bbox),
                rng.random_range(-bbox..bbox),
                rng.random_range(-bbox..bbox),
                0.0,
            ]);
        }

        out
    }

    fn build_index_buffer(n_seeds: u32, n_steps: u32) -> Vec<u32> {
        let segments = (n_steps - 1) as usize;
        let mut indices = Vec::with_capacity(n_seeds as usize * segments * 2);

        for s in 0..n_seeds {
            let base = s * n_steps;
            for i in 0..(n_steps - 1) {
                indices.push(base + i);
                indices.push(base + i + 1);
            }
        }

        indices
    }

    fn make_compute_bgl(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Streamline Compute BGL"),
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
            label: Some("Streamline Camera BGL"),
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
            label: Some("Streamline Storage BGL"),
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
            label: Some("Streamline Compute PL"),
            bind_group_layouts: &[Some(bgl)],
            ..Default::default()
        });
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Streamline Compute Pipeline"),
            layout: Some(&layout),
            module: shader,
            entry_point: Some("cs_streamline"),
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
            label: Some("Streamline Render PL"),
            bind_group_layouts: &[Some(camera_bgl), Some(storage_bgl)],
            ..Default::default()
        });
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Streamline Render Pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: shader,
                entry_point: Some("vs_streamline"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
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
                entry_point: Some("fs_streamline"),
                targets: &[Some(ColorTargetState {
                    format: surface_format,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
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
}
