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

#[cfg(test)]
mod tests {
    struct FakePP {
        current: usize,
    }

    impl FakePP {
        fn new() -> Self {
            Self { current: 0 }
        }
        fn read_idx(&self) -> usize {
            1 - self.current
        }
        fn write_idx(&self) -> usize {
            self.current
        }
        fn swap(&mut self) {
            self.current ^= 1;
        }
    }

    #[test]
    fn initial_state_read_is1_write_is_0() {
        let pp = FakePP::new();
        assert_eq!(pp.write_idx(), 0);
        assert_eq!(pp.read_idx(), 1);
    }

    #[test]
    fn after_one_swap_directions_reverse() {
        let mut pp = FakePP::new();
        pp.swap();
        assert_eq!(pp.write_idx(), 1);
        assert_eq!(pp.read_idx(), 0);
    }

    #[test]
    fn read_and_write_never_same_index() {
        let mut pp = FakePP::new();
        for _ in 0..10 {
            assert_ne!(
                pp.read_idx(),
                pp.write_idx(),
                "read and write must always point to different buffers"
            );
            pp.swap();
        }
    }

    #[test]
    fn swap_us_idempotent_after_even_calls() {
        let mut pp = FakePP::new();
        let w0 = pp.write_idx();
        pp.swap();
        pp.swap();
        pp.swap();
        pp.swap();
        assert_eq!(pp.write_idx(), w0);
    }
}
