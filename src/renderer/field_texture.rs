use half::f16;
use wgpu::{
    Device, Extent3d, Origin3d, Queue, Sampler, SamplerDescriptor, Texture, TextureDescriptor,
    TextureUsages, TextureView, TextureViewDescriptor,
};

pub const FIELD_RES: u32 = 64;
pub const FIELD_BBOX: f32 = 15.0;

pub struct FieldTexture {
    pub texture: Texture,
    pub view: TextureView,
    pub sampler: Sampler,
}

impl FieldTexture {
    pub fn new(device: &Device, queue: &Queue, time: f32) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("Field Texture 3D"),
            size: Extent3d {
                width: FIELD_RES,
                height: FIELD_RES,
                depth_or_array_layers: FIELD_RES,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = texture.create_view(&TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("Field Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let mut ft = Self {
            texture,
            view,
            sampler,
        };
        ft.upload(queue, time);

        ft
    }

    pub fn upload(&mut self, queue: &Queue, time: f32) {
        let data = Self::sample_field(FIELD_RES, FIELD_BBOX, time);

        queue.write_texture(
            wgpu::TexelCopyTextureInfoBase {
                texture: &self.texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(FIELD_RES * 4 * 2),
                rows_per_image: Some(FIELD_RES),
            },
            Extent3d {
                width: FIELD_RES,
                height: FIELD_RES,
                depth_or_array_layers: FIELD_RES,
            },
        );
    }

    fn sample_field(res: u32, bbox: f32, time: f32) -> Vec<u16> {
        let mut data = Vec::with_capacity((res * res * res * 4) as usize);

        for z in 0..res {
            for y in 0..res {
                for x in 0..res {
                    let px = (x as f32 / res as f32) * 2.0 * bbox - bbox;
                    let py = (y as f32 / res as f32) * 2.0 * bbox - bbox;
                    let pz = (z as f32 / res as f32) * 2.0 * bbox - bbox;

                    let sx = px * 0.15;
                    let sy = py * 0.15;
                    let sz = pz * 0.15;

                    let vx = (sy * time * 0.31).sin() + 0.4 * (sz + time * 0.17).cos();
                    let vy = (sx + time * 0.23).cos() + 0.4 * (sz - time * 0.17).sin();
                    let vz = (sx - sy * 0.5 + time * 0.13).sin() * 0.8;

                    data.push(f16::from_f32(vx).to_bits());
                    data.push(f16::from_f32(vy).to_bits());
                    data.push(f16::from_f32(vz).to_bits());
                    data.push(f16::from_f32(0.0).to_bits());
                }
            }
        }

        data
    }
}
