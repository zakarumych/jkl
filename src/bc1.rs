//! BC1 implementation.
//!

use core::f32;

use crate::math::{Region3, Rgb32F, Rgb565, Rgba32F, Vec3, Yiq32F};

/// A block of 4x4 texels compressed with BC1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Block {
    color0: Rgb565,
    color1: Rgb565,
    texels: u32,
}

impl Block {
    /// Decodes single BC1 block.
    pub fn decode(self) -> [[Rgba32F; 4]; 4] {
        // Decode endpoints.
        let color0 = self.color0.into_f32();
        let color1 = self.color1.into_f32();

        // Prepare local variables.
        let mut colors = [[Rgba32F::TRANSPARENT; 4]; 4];
        let mut texels = self.texels;

        // Check mode.
        let intermediate = if self.color0.bits() < self.color1.bits() {
            // Interpolate two intermediate colors.
            [
                Rgb32F::lerp(color0, color1, 1.0 / 3.0).with_alpha(1.0),
                Rgb32F::lerp(color0, color1, 2.0 / 3.0).with_alpha(1.0),
            ]
        } else {
            // Interpolate one intermediate color.
            [
                Rgb32F::lerp(color0, color1, 1.0 / 2.0).with_alpha(1.0),
                Rgba32F::TRANSPARENT,
            ]
        };

        let color0 = color0.with_alpha(1.0);
        let color1 = color1.with_alpha(1.0);

        // Decode texels.
        for i in 0..4 {
            for j in 0..4 {
                let index = texels & 0b11;
                texels >>= 2;

                colors[3 - i][3 - j] = match index {
                    0 => color0,
                    1 => intermediate[0],
                    2 => intermediate[1],
                    3 => color1,
                    _ => unreachable!(),
                };
            }
        }

        colors
    }

    pub fn encode(colors: [[Rgb32F; 4]; 4], opt: usize) -> Self {
        let region = Region3::new((0..16).map(|i| colors[i / 4][i % 4].into()));

        let mut best = (Rgb565::BLACK, Rgb565::BLACK);
        let mut best_errors = (Vec3::ZERO, Vec3::ZERO);
        let mut best_errors_weight = f32::INFINITY;

        for diagonal in region.diagonals() {
            // Convert diagonal to colors, loose precision.
            let color0_565 = Rgb565::from_f32(Rgb32F::from(diagonal.0));
            let color1_565 = Rgb565::from_f32(Rgb32F::from(diagonal.1));

            let (errors, errors_weight) = block_errors(colors, (color0_565, color1_565));

            if errors_weight < best_errors_weight {
                best = (color0_565, color1_565);
                best_errors = errors;
                best_errors_weight = errors_weight;
            }
        }

        let mut endpoints = best;

        if opt > 0 {
            optimize_endpoints(&mut endpoints, best_errors, best_errors_weight * 3.0);
        }

        for i in 1..opt {
            let (errors, errors_weight) = block_errors(colors, endpoints);
            optimize_endpoints(&mut endpoints, errors, errors_weight * 3.0 * (i + 1) as f32);
        }

        if endpoints.0.bits() > endpoints.1.bits() {
            core::mem::swap(&mut endpoints.0, &mut endpoints.1);
        } else if endpoints.0 == endpoints.1 {
            endpoints.1 = Rgb565::WHITE;
        }

        let texels = block_encoding(colors, endpoints);

        Block {
            color0: endpoints.0,
            color1: endpoints.1,
            texels,
        }
    }
}

fn optimize_endpoints(endpoints: &mut (Rgb565, Rgb565), errors: (Vec3, Vec3), errors_weight: f32) {
    adjust_endpoint(&mut endpoints.0, errors.0 / errors_weight);
    adjust_endpoint(&mut endpoints.1, errors.1 / errors_weight);
}

fn adjust_endpoint(endpoint: &mut Rgb565, correction: Vec3) {
    let r = endpoint.r() as i16;
    let g = endpoint.g() as i16;
    let b = endpoint.b() as i16;
    let x = (correction.x() * 31.0).clamp(0.0, 31.0) as i16;
    let y = (correction.y() * 63.0).clamp(0.0, 63.0) as i16;
    let z = (correction.z() * 31.0).clamp(0.0, 31.0) as i16;

    *endpoint = Rgb565::new(
        (r + x).clamp(0, 31) as u8,
        (g + y).clamp(0, 63) as u8,
        (b + z).clamp(0, 31) as u8,
    );
}

fn block_errors(colors: [[Rgb32F; 4]; 4], endpoints: (Rgb565, Rgb565)) -> ((Vec3, Vec3), f32) {
    let mut errors = (Vec3::ZERO, Vec3::ZERO);
    let mut errors_weight = 0.0;

    let color0 = endpoints.0.into_f32();
    let color1 = endpoints.1.into_f32();

    let color13 = Rgb32F::lerp(color0, color1, 1.0 / 3.0);
    let color23 = Rgb32F::lerp(color0, color1, 2.0 / 3.0);

    let yiq0 = Yiq32F::from(color0);
    let yiq13 = Yiq32F::from(color13);
    let yiq23 = Yiq32F::from(color23);
    let yiq1 = Yiq32F::from(color1);

    for i in 0..4 {
        for j in 0..4 {
            let rgb = colors[i][j];
            let yiq = Yiq32F::from(rgb);

            let e0 = Yiq32F::perceptual_distance(yiq, yiq0);
            let e13 = Yiq32F::perceptual_distance(yiq, yiq13);
            // let e0 = Rgb32F::distance_squared(rgb, color0);
            // let e13 = Rgb32F::distance_squared(rgb, color13);

            if e0 < e13 {
                // Closest to color0.
                let error = Rgb32F::diff(rgb, color0);
                errors.0 += error * e0;
                errors_weight += e0;
                continue;
            }

            let e23 = Yiq32F::perceptual_distance(yiq, yiq23);
            // let e23 = Rgb32F::distance_squared(rgb, color23);

            if e13 < e23 {
                // Closest to color13.
                let error = Rgb32F::diff(rgb, color13);
                errors.0 += error * e13 * (2.0 / 3.0);
                errors.1 += error * e13 * (1.0 / 3.0);
                errors_weight += e13;
                continue;
            }

            let e1 = Yiq32F::perceptual_distance(yiq, yiq1);
            // let e1 = Rgb32F::distance_squared(rgb, color1);

            if e23 < e1 {
                // Closest to color23.
                let error = Rgb32F::diff(rgb, color23);
                errors.0 += error * e23 * (1.0 / 3.0);
                errors.1 += error * e23 * (2.0 / 3.0);
                errors_weight += e23;
            } else {
                // Closest to color1.
                let error = Rgb32F::diff(rgb, color0);
                errors.1 += error * e1;
                errors_weight += e1;
            }
        }
    }

    (errors, errors_weight)
}

fn block_encoding(colors: [[Rgb32F; 4]; 4], endpoints: (Rgb565, Rgb565)) -> u32 {
    let color0 = endpoints.0.into_f32();
    let color1 = endpoints.1.into_f32();

    let color13 = Rgb32F::lerp(color0, color1, 1.0 / 3.0);
    let color23 = Rgb32F::lerp(color0, color1, 2.0 / 3.0);

    let yiq0 = Yiq32F::from(color0);
    let yiq13 = Yiq32F::from(color13);
    let yiq23 = Yiq32F::from(color23);
    let yiq1 = Yiq32F::from(color1);

    let mut texels = 0;

    for i in 0..4 {
        for j in 0..4 {
            let rgb = colors[i][j];
            let yiq = Yiq32F::from(rgb);

            let e0 = Yiq32F::perceptual_distance(yiq, yiq0);
            let e13 = Yiq32F::perceptual_distance(yiq, yiq13);
            // let e0 = Rgb32F::distance_squared(rgb, color0);
            // let e13 = Rgb32F::distance_squared(rgb, color13);

            if e0 <= e13 {
                // Closest to color0.
                texels <<= 2;
                texels |= 0b00;
                continue;
            }

            let e23 = Yiq32F::perceptual_distance(yiq, yiq23);
            // let e23 = Rgb32F::distance_squared(rgb, color23);

            if e13 <= e23 {
                // Closest to color13.
                texels <<= 2;
                texels |= 0b01;
                continue;
            }

            let e1 = Yiq32F::perceptual_distance(yiq, yiq1);
            // let e1 = Rgb32F::distance_squared(rgb, color1);

            if e23 <= e1 {
                // Closest to color23.
                texels <<= 2;
                texels |= 0b10;
            } else {
                // Closest to color1.
                texels <<= 2;
                texels |= 0b11;
            }
        }
    }

    texels
}
