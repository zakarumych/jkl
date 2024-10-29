pub fn even_odd_split_squash(index: u32) -> (u16, u16) {
    // Mask even and odd bits
    let mut even_bits: u32 = index & 0x55555555; // Mask for even bits (0x5 = 0101)
    let mut odd_bits: u32 = index & 0xAAAAAAAA; // Mask for odd bits (0xA = 1010)

    // Compact the even bits
    even_bits = (even_bits | (even_bits >> 1)) & 0x33333333;
    even_bits = (even_bits | (even_bits >> 2)) & 0x0F0F0F0F;
    even_bits = (even_bits | (even_bits >> 4)) & 0x00FF00FF;
    even_bits = (even_bits | (even_bits >> 8)) & 0x0000FFFF;

    // Compact the odd bits
    odd_bits >>= 1; // Align odd bits to LSBs
    odd_bits = (odd_bits | (odd_bits >> 1)) & 0x33333333;
    odd_bits = (odd_bits | (odd_bits >> 2)) & 0x0F0F0F0F;
    odd_bits = (odd_bits | (odd_bits >> 4)) & 0x00FF00FF;
    odd_bits = (odd_bits | (odd_bits >> 8)) & 0x0000FFFF;

    (even_bits as u16, odd_bits as u16)
}

#[test]
fn test_even_odd_split_squash() {
    assert_eq!(even_odd_split_squash(0b00), (0b0, 0b0,));
    assert_eq!(even_odd_split_squash(0b10), (0b0, 0b1,));
    assert_eq!(even_odd_split_squash(0b11), (0b1, 0b1,));

    assert_eq!(even_odd_split_squash(0b10101010), (0b0000, 0b1111));
    assert_eq!(even_odd_split_squash(0b01010101), (0b1111, 0b0000));
    assert_eq!(even_odd_split_squash(0b11011000), (0b1100, 0b1010));

    assert_eq!(
        even_odd_split_squash(0b10101010101010101010101010101010),
        (0b0000000000000000, 0b1111111111111111)
    );
}

pub struct BoundZCurve {
    width: u16,
    height: u16,
    next_index: u32,
}

impl BoundZCurve {
    pub fn new(width: u16, height: u16) -> Self {
        BoundZCurve {
            width,
            height,
            next_index: 0,
        }
    }
}

impl Iterator for BoundZCurve {
    type Item = (u16, u16);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let index = self.next_index;

            let (x, y) = even_odd_split_squash(index);

            let x_out = x >= self.width;
            let y_out = y >= self.height;

            if x_out && y_out {
                return None;
            }

            if x_out {
                if x > (self.width + 1).next_power_of_two() || self.width == 0 {
                    return None;
                }

                let tz = x.trailing_zeros();
                let msb = 1u32 << (tz * 2 + 1);
                let mask = msb - 1;
                let until_next = msb - (index & mask);

                // Skip until next x < width
                self.next_index += until_next as u32;
                continue;
            }

            if y_out {
                if y > (self.height + 1).next_power_of_two() || self.height == 0 {
                    return None;
                }

                let tz = y.trailing_zeros();
                let msb = 1u32 << (tz * 2 + 2);
                let mask = msb - 1;
                let until_next = msb - (index & mask);

                // Skip until next y = 0
                self.next_index += until_next as u32;
                continue;
            }

            self.next_index += 1;
            return Some((x, y));
        }
    }
}

#[test]
fn test_rect_z_order() {
    let rzo = BoundZCurve::new(4, 4);

    assert_eq!(
        rzo.collect::<Vec<_>>(),
        [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (2, 0),
            (3, 0),
            (2, 1),
            (3, 1),
            (0, 2),
            (1, 2),
            (0, 3),
            (1, 3),
            (2, 2),
            (3, 2),
            (2, 3),
            (3, 3)
        ]
    );

    let rzo = BoundZCurve::new(3, 2);

    assert_eq!(
        rzo.collect::<Vec<_>>(),
        [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1),]
    );

    let rzo = BoundZCurve::new(2, 3);

    assert_eq!(
        rzo.collect::<Vec<_>>(),
        [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2),]
    );

    let rzo = BoundZCurve::new(3, 3);

    assert_eq!(
        rzo.collect::<Vec<_>>(),
        [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (2, 0),
            (2, 1),
            (0, 2),
            (1, 2),
            (2, 2),
        ]
    );
}
