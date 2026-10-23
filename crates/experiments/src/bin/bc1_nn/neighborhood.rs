//! Encoded RGB565 neighborhood reference with two explicit index policies.
use super::{Texels, error, pack};
use jkl::{image::block::bc1::Block, math::Rgb565};

pub fn neighbors(color: Rgb565) -> Vec<Rgb565> {
    let mut result = Vec::with_capacity(26);
    for dr in -1i32..=1 {
        for dg in -1i32..=1 {
            for db in -1i32..=1 {
                if dr == 0 && dg == 0 && db == 0 {
                    continue;
                }
                let r = color.r() as i32 + dr;
                let g = color.g() as i32 + dg;
                let b = color.b() as i32 + db;
                if (0..=31).contains(&r) && (0..=63).contains(&g) && (0..=31).contains(&b) {
                    result.push(Rgb565::new(r as u8, g as u8, b as u8));
                }
            }
        }
    }
    result
}
fn has_three(indices: [u8; 4]) -> bool {
    indices.iter().any(|&b| b & (b >> 1) & 0x55 != 0)
}
pub fn candidate(
    t: &Texels,
    mut a: Rgb565,
    mut b: Rgb565,
    indices: Option<[u8; 4]>,
) -> Option<Block> {
    if let Some(mut indices) = indices {
        if a == b && has_three(indices) {
            return None;
        }
        if a.bits() < b.bits() {
            std::mem::swap(&mut a, &mut b);
            for byte in &mut indices {
                *byte ^= 0x55;
            }
        }
        Some(Block {
            color0: a,
            color1: b,
            indices,
        })
    } else {
        let a = a.into_f32().0;
        let b = b.into_f32().0;
        Some(pack(t, [a[0], a[1], a[2], b[0], b[1], b[2]]))
    }
}
pub fn proposal_count(seed: Block, reuse: bool) -> usize {
    neighbors(seed.color0)
        .into_iter()
        .filter(|&n| !(reuse && n == seed.color1 && has_three(seed.indices)))
        .count()
        + neighbors(seed.color1)
            .into_iter()
            .filter(|&n| !(reuse && n == seed.color0 && has_three(seed.indices)))
            .count()
}
pub fn refine(t: &Texels, initial: Block, rounds: u32, reuse: bool) -> Block {
    let mut best = initial;
    let mut best_error = error(t, best);
    for _ in 0..rounds {
        let seed = best;
        for endpoint in 0..2 {
            let current = if endpoint == 0 {
                seed.color0
            } else {
                seed.color1
            };
            for neighbor in neighbors(current) {
                let a = if endpoint == 0 { neighbor } else { seed.color0 };
                let b = if endpoint == 1 { neighbor } else { seed.color1 };
                if let Some(candidate) =
                    candidate(t, a, b, if reuse { Some(seed.indices) } else { None })
                {
                    let candidate_error = error(t, candidate);
                    if candidate_error < best_error {
                        best = candidate;
                        best_error = candidate_error;
                    }
                }
            }
        }
        if best == seed {
            break;
        }
    }
    best
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn encoded_neighbors_are_unique_and_valid() {
        for (color, count) in [
            (Rgb565::new(10, 25, 12), 26),
            (Rgb565::BLACK, 7),
            (Rgb565::WHITE, 7),
            (Rgb565::new(0, 25, 12), 17),
        ] {
            let list = neighbors(color);
            assert_eq!(list.len(), count);
            assert_eq!(
                list.iter()
                    .map(|c| c.bits())
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
                count
            );
            for c in list {
                let differences = [
                    c.r() as i32 - color.r() as i32,
                    c.g() as i32 - color.g() as i32,
                    c.b() as i32 - color.b() as i32,
                ];
                assert!(differences.iter().all(|d| (-1..=1).contains(d)));
                assert_ne!(differences, [0; 3]);
            }
        }
        let seed = Block {
            color0: Rgb565::new(20, 38, 22),
            color1: Rgb565::new(10, 25, 12),
            indices: [0xe4; 4],
        };
        assert_eq!(proposal_count(seed, false), 52);
        assert_eq!(proposal_count(seed, true), 52);
        let corner = Block {
            color0: Rgb565::WHITE,
            color1: Rgb565::BLACK,
            ..seed
        };
        assert_eq!(proposal_count(corner, false), 14);
        let black = neighbors(Rgb565::BLACK);
        assert!(black.contains(&Rgb565::from_bits(1)));
        assert!(black.contains(&Rgb565::from_bits(32)));
        assert!(black.contains(&Rgb565::from_bits(2048)));
    }
    #[test]
    fn reused_indices_remap_swaps_and_reject_transparency() {
        let t = [[0; 3]; 16];
        let a = Rgb565::new(10, 25, 12);
        let b = Rgb565::new(20, 38, 22);
        let result = candidate(&t, a, b, Some([0xe4; 4])).unwrap();
        assert_eq!(result.indices, [0xb1; 4]);
        assert_eq!(result.decode()[0][0], a.into_f32());
        assert_eq!(result.decode()[0][1], b.into_f32());
        assert!(candidate(&t, a, a, Some([0xff; 4])).is_none());
        let opaque = candidate(&t, a, a, Some([0xa4; 4])).unwrap();
        assert!(
            opaque
                .decode_with_alpha()
                .iter()
                .flatten()
                .all(|c| c.a() == 1.0)
        );
    }
    #[test]
    fn both_neighbor_policies_never_worsen_ls_and_remain_opaque() {
        let mut rng = super::super::Rng(991);
        for i in 0..18 {
            let t = super::super::synthetic(&mut rng, i);
            let start = pack(&t, super::super::baseline(&super::super::canonical(&t)));
            let ls = super::super::refine(&t, start);
            for rounds in [1, 2] {
                for reuse in [false, true] {
                    let result = refine(&t, ls, rounds, reuse);
                    assert!(error(&t, result) <= error(&t, ls));
                    assert!(result.color0.bits() >= result.color1.bits());
                    assert_eq!(result, Block::from_bytes(result.bytes()));
                    assert!(
                        result
                            .decode_with_alpha()
                            .iter()
                            .flatten()
                            .all(|c| c.a() == 1.0)
                    );
                }
            }
        }
    }
}
