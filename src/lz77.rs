use std::io::Write;

use crate::bytes::LeBytes;

const WINDOW_LENGTH: u16 = 8192;

fn is_distance_length_limit(distance: u16, length: u16) -> bool {
    debug_assert!(distance < WINDOW_LENGTH);

    let bits = match distance {
        0..32 => 5,
        32..128 => 7,
        128..512 => 9,
        512.. => 11,
    };

    let max_length = (1 << (14 - bits)) - 1;
    debug_assert!(length <= max_length);
    length == max_length
}

/// Encodes distance and length into single u16.
///
/// Caveat - length might my clamped.
///
/// Offset must be less than 8192, otherwise result won't be decodable correctly.
fn encode_distance_length(distance: u16, length: u16) -> u16 {
    debug_assert!(distance < WINDOW_LENGTH);

    let (kind, bits) = match distance {
        0..32 => (0b00, 5),
        32..128 => (0b01, 7),
        128..512 => (0b10, 9),
        512.. => (0b11, 11),
    };

    let max_length = (1 << (14 - bits)) - 1;
    debug_assert!(length <= max_length);

    (kind << 14) | (length << bits) | distance
}

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
    window: [T; WINDOW_LENGTH as usize],
    head: u16,
    distance: u16,
    length: u16,
}

fn window_index(head: u16, distance: u16, index: u16) -> u16 {
    let index = WINDOW_LENGTH - distance - 1 + (index % (distance + 1));
    (head + index) % WINDOW_LENGTH
}

impl<T> Encoder<T>
where
    T: Copy + Eq + LeBytes,
{
    pub fn new(init: T) -> Self {
        Encoder {
            window: [init; WINDOW_LENGTH as usize],
            head: 0,
            distance: 0,
            length: 0,
        }
    }

    pub fn encode(&mut self, input: T, writer: impl Write) -> std::io::Result<()> {
        let check = self.window[usize::from(window_index(self.head, self.distance, self.length))];
        if check != input {
            self.flush(writer)?;
        }

        Ok(())
    }

    pub fn finish(self, writer: impl Write) -> std::io::Result<()> {}

    fn flush(&mut self, writer: impl Write) -> std::io::Result<()> {
        encode_distance_length(self.distance, &mut self.length);

        self.shift_insert();
        Ok(())
    }

    fn shift_insert(&mut self) {
        for i in 0..self.length {
            let elem = self.window[usize::from(window_index(self.head, self.distance, i))];
            self.window[usize::from(self.head + i)] = elem;
        }

        let new_head = (self.head + self.length) % WINDOW_LENGTH;
        self.head = new_head;
        self.length = 0;
    }

    fn push(&mut self, elem: T) {
        self.window[usize::from(self.head)] = elem;

        let new_head = (self.head + 1) % WINDOW_LENGTH;
        self.head = new_head;
    }
}
