use std::io::Write;

use crate::bytes::LeBytes;

/// Decodes distance and length previously encoded with [`encode_distance_length`].
fn decode_offset_length(value: u16) -> (u16, u16) {
    let kind = value >> 14;
    let bits = match kind {
        0b00 => 5,
        0b01 => 7,
        0b10 => 9,
        0b11 => 11,
        _ => unreachable!(),
    };

    let value = value & 0b0011111111111111;
    let length = value >> bits;
    let distance = value & ((1 << bits) - 1);

    (distance, length)
}

struct Encoder<T> {
    window: Vec<T>,
    head: u16,
    distance: u16,
    length: u16,
}

fn window_index(length: u16, head: u16, distance: u16, index: u16) -> u16 {
    let index = length - distance - 1 + (index % (distance + 1));
    (head + index) % length
}

impl<T> Encoder<T>
where
    T: Copy + Eq + LeBytes,
{
    pub fn new(init: T, length: u16) -> Self {
        Encoder {
            window: vec![init; length as usize],
            head: 0,
            distance: 0,
            length: 0,
        }
    }

    pub fn encode(&mut self, input: T, writer: impl Write) -> std::io::Result<()> {
        let check = self.window[usize::from(window_index(
            self.window.len() as u16,
            self.head,
            self.distance,
            self.length,
        ))];
        if check != input {
            self.flush(writer)?;
        }

        Ok(())
    }

    pub fn finish(self, writer: impl Write) -> std::io::Result<()> {}

    fn flush(&mut self, writer: impl Write) -> std::io::Result<()> {
        self.shift_insert();
        Ok(())
    }

    fn shift_insert(&mut self) {
        for i in 0..self.length {
            let elem = self.window[usize::from(window_index(
                self.window.len() as u16,
                self.head,
                self.distance,
                i,
            ))];
            self.window[usize::from(self.head + i)] = elem;
        }

        let new_head = (self.head + self.length) % (self.window.len() as u16);
        self.head = new_head;
        self.length = 0;
    }

    fn push(&mut self, elem: T) {
        self.window[usize::from(self.head)] = elem;

        let new_head = (self.head + 1) % (self.window.len() as u16);
        self.head = new_head;
    }
}
