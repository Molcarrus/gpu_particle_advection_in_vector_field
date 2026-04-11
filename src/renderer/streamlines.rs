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
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::Always),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streamline_uniforms_size_is_32_bytes() {
        assert_eq!(std::mem::size_of::<StreamlineUniforms>(), 32);
    }

    #[test]
    fn stream_vert_size_is_16_bytes() {
        assert_eq!(std::mem::size_of::<StreamVert>(), 16);
    }

    #[test]
    fn stream_vert_is_pod() {
        let sv = StreamVert {
            position: [1.0, 2.0, 3.0],
            t: 0.5,
        };
        let bytes = bytemuck::bytes_of(&sv);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn index_buffer_length_is_correct() {
        let indices = StreamlineSystem::build_index_buffer(N_SEEDS, N_STEPS);
        let expected = N_SEEDS as usize * (N_STEPS as usize - 1) * 2;
        assert_eq!(indices.len(), expected);
    }

    #[test]
    fn index_buffer_first_streamline_is_sequential() {
        let indices = StreamlineSystem::build_index_buffer(4, 4);
        assert_eq!(&indices[..6], &[0, 1, 1, 2, 2, 3]);
    }

    #[test]
    fn index_buffer_second_streamline_starts_at_n_steps() {
        let indices = StreamlineSystem::build_index_buffer(4, 4);
        assert_eq!(&indices[6..12], &[4, 5, 5, 6, 6, 7]);
    }

    #[test]
    fn index_buffer_no_index_exceeds_total_verts() {
        let indices = StreamlineSystem::build_index_buffer(N_SEEDS, N_STEPS);
        let max_valid = (N_SEEDS * N_STEPS - 1) as u32;
        for &idx in &indices {
            assert!(
                idx <= max_valid,
                "index {idx} exceeds total vertex count {max_valid}"
            );
        }
    }

    #[test]
    fn index_buffer_no_cross_streamline_segments() {
        let n_seeds = 8u32;
        let n_steps = 16u32;
        let indices = StreamlineSystem::build_index_buffer(n_seeds, n_steps);

        for s in 0..n_seeds {
            let last_valid_in_streamline = (s + 1) * n_steps - 1;
            let seg_start = s as usize * (n_steps as usize - 1) * 2;
            let seg_end = seg_start + (n_steps as usize - 1) * 2;

            for &idx in &indices[seg_start..seg_end] {
                assert!(
                    idx >= s * n_steps && idx <= last_valid_in_streamline,
                    "index {idx} crosses streamline boundary for streamline {s}"
                );
            }
        }
    }

    #[test]
    fn index_count_matches_draw_call_expectation() {
        let indices = StreamlineSystem::build_index_buffer(N_SEEDS, N_STEPS);
        assert_eq!(indices.len() % 2, 0, "LineList needs even index count");
    }

    #[test]
    fn seeds_count_is_exactly_n() {
        let seeds = StreamlineSystem::generate_seeds(N_SEEDS, 15.0);
        assert_eq!(seeds.len(), N_SEEDS as usize);
    }

    #[test]
    fn seeds_within_bounding_box() {
        let bbox = 15.0f32;
        let seeds = StreamlineSystem::generate_seeds(N_SEEDS, bbox);
        let limit = bbox * 1.2;
        for (i, seed) in seeds.iter().enumerate() {
            for &coord in &seed[..3] {
                assert!(
                    coord.abs() <= limit,
                    "seed {i} coordinate {coord} exceeds limit {limit}"
                );
            }
        }
    }

    #[test]
    fn seeds_w_component_is_zero() {
        let seeds = StreamlineSystem::generate_seeds(32, 15.0);
        for (i, seed) in seeds.iter().enumerate() {
            assert_eq!(seed[3], 0.0, "seed {i} w component should be 0.0");
        }
    }

    #[test]
    fn seeds_are_not_all_identical() {
        let seeds = StreamlineSystem::generate_seeds(N_SEEDS, 15.0);
        // At least two seeds should differ — rules out a broken RNG
        let first = seeds[0];
        let all_same = seeds.iter().all(|s| s == &first);
        assert!(!all_same, "all seeds are identical — RNG may be broken");
    }

    #[test]
    fn total_verts_fits_in_u32() {
        let total = N_SEEDS as u64 * N_STEPS as u64;
        assert!(
            total <= u32::MAX as u64,
            "total vertex count {total} overflows u32 index"
        );
    }

    #[test]
    fn workgroup_count_covers_all_seeds() {
        let workgroups = (N_SEEDS + 63) / 64;
        assert!(workgroups * 64 >= N_SEEDS);
    }

    #[test]
    fn workgroup_count_is_minimal() {
        let workgroups = (N_SEEDS + 63) / 64;
        assert!((workgroups - 1) * 64 < N_SEEDS);
    }

    #[test]
    fn step_size_is_positive() {
        assert!(STEP_SIZE > 0.0);
    }
}
