use wgpu::{
    Buffer, BufferUsages, Device,
    util::{BufferInitDescriptor, DeviceExt},
};

pub struct PingPongBuffers {
    pub buffers: [Buffer; 2],
    pub current: usize,
}

impl PingPongBuffers {
    pub fn new(device: &Device, initial_bytes: &[u8], extra_usages: BufferUsages) -> Self {
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | extra_usages;

        let make = |label: &'static str| {
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some(label),
                contents: initial_bytes,
                usage,
            })
        };

        Self {
            buffers: [make("PingPong A"), make("PingPong B")],
            current: 0,
        }
    }

    #[inline]
    pub fn read_buf(&self) -> &Buffer {
        &self.buffers[1 - self.current]
    }

    #[inline]
    pub fn write_buf(&self) -> &Buffer {
        &self.buffers[self.current]
    }

    #[inline]
    pub fn swap(&mut self) {
        self.current ^= 1;
    }
}
