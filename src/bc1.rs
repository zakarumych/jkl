//! BC1 implementation.
//!

use core::f32;

use crate::math::{Region3, Rgb32F, Rgb565, Rgba32F, Vec3, Yiq32F};

/// A block of 4x4 texels compressed with BC1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Block {
    pub color0: Rgb565,
    pub color1: Rgb565,
    pub texels: [u8; 4],
}

impl Block {
    pub const BLACK: Block = Block {
        color0: Rgb565::BLACK,
        color1: Rgb565::BLACK,
        texels: [0; 4],
    };

    pub const WHITE: Block = Block {
        color1: Rgb565::BLACK,
        color0: Rgb565::WHITE,
        texels: [0x11; 4],
    };

    pub const TRANSPARENT: Block = Block {
        color1: Rgb565::BLACK,
        color0: Rgb565::BLACK,
        texels: [0x11; 4],
    };

    pub fn bytes(&self) -> [u8; 8] {
        let color0 = self.color0.bytes();
        let color1 = self.color1.bytes();
        let texels = self.texels;

        [
            color0[0], color0[1], color1[0], color1[1], texels[0], texels[1], texels[2], texels[3],
        ]
    }

    pub fn from_bytes(bytes: [u8; 8]) -> Block {
        let color0 = Rgb565::from_bytes([bytes[0], bytes[1]]);
        let color1 = Rgb565::from_bytes([bytes[2], bytes[3]]);
        let texels = [bytes[4], bytes[5], bytes[6], bytes[7]];

        Block {
            color0,
            color1,
            texels,
        }
    }

    /// Decodes single BC1 block.
    pub fn decode(self) -> [[Rgb32F; 4]; 4] {
        // Decode endpoints.
        let color0 = self.color0.into_f32();
        let color1 = self.color1.into_f32();

        // Prepare local variables.
        let mut colors = [[Rgb32F::BLACK; 4]; 4];
        let texels = self.texels;

        // Check mode and build palette.
        let palette = if self.color0.bits() < self.color1.bits() {
            // Interpolate two intermediate colors.
            [
                color0,
                Rgb32F::lerp(color0, color1, 1.0 / 3.0),
                Rgb32F::lerp(color0, color1, 2.0 / 3.0),
                color1,
            ]
        } else {
            // Interpolate one intermediate color.
            [
                color0,
                Rgb32F::lerp(color0, color1, 1.0 / 2.0),
                color1,
                Rgb32F::BLACK,
            ]
        };

        // Decode texels.
        for i in 0..4 {
            for j in 0..4 {
                let index = (texels[i] >> 2 * j) & 0b11;

                colors[i][j] = palette[index as usize];
            }
        }

        colors
    }

    /// Decodes single BC1 block.
    pub fn decode_with_alpha(self) -> [[Rgba32F; 4]; 4] {
        // Decode endpoints.
        let color0 = self.color0.into_f32();
        let color1 = self.color1.into_f32();

        // Prepare local variables.
        let mut colors = [[Rgba32F::TRANSPARENT; 4]; 4];
        let texels = self.texels;

        // Check mode and build palette.
        let palette = if self.color0.bits() < self.color1.bits() {
            // Interpolate two intermediate colors.
            [
                color0.with_alpha(1.0),
                Rgb32F::lerp(color0, color1, 1.0 / 3.0).with_alpha(1.0),
                Rgb32F::lerp(color0, color1, 2.0 / 3.0).with_alpha(1.0),
                color1.with_alpha(1.0),
            ]
        } else {
            // Interpolate one intermediate color.
            [
                color0.with_alpha(1.0),
                Rgb32F::lerp(color0, color1, 1.0 / 2.0).with_alpha(1.0),
                color1.with_alpha(1.0),
                Rgba32F::TRANSPARENT,
            ]
        };

        // Decode texels.
        for i in 0..4 {
            for j in 0..4 {
                let index = (texels[i] >> 2 * j) & 0b11;

                colors[i][j] = palette[index as usize];
            }
        }

        colors
    }

    fn encode_optimize(
        colors: [[Rgb32F; 4]; 4],
        region: Region3,
        optimize: usize,
        block_errors: impl Fn([[Rgb32F; 4]; 4], (Rgb32F, Rgb32F)) -> ((Vec3, Vec3), f32),
    ) -> ((Rgb32F, Rgb32F), f32) {
        let mut best_endpoints = (Rgb32F::BLACK, Rgb32F::BLACK);
        let mut best_errors = (Vec3::ZERO, Vec3::ZERO);
        let mut best_errors_weight = f32::INFINITY;

        for diagonal in region.diagonals() {
            // Convert diagonal to colors, loose precision.
            let color0 = Rgb32F::from(diagonal.0);
            let color1 = Rgb32F::from(diagonal.1);

            let (errors, errors_weight) = block_errors(colors, (color0, color1));

            if errors_weight < best_errors_weight {
                best_endpoints = (color0, color1);
                best_errors = errors;
                best_errors_weight = errors_weight;
            }
        }

        for i in 0..optimize {
            if best_errors_weight < 1e-6 {
                break;
            }

            let endpoints = optimize_endpoints(
                best_endpoints,
                best_errors,
                best_errors_weight,
                0.01 / (i + 1) as f32,
            );

            let (errors, errors_weight) = block_errors(colors, best_endpoints);

            best_endpoints = endpoints;
            best_errors = errors;
            best_errors_weight = errors_weight;
        }

        (best_endpoints, best_errors_weight)
    }

    pub fn encode(colors: [[Rgb32F; 4]; 4], optimize: usize) -> Self {
        let region = Region3::new((0..16).map(|i| colors[i / 4][i % 4].into()));

        if region.is_singular() {
            // All colors are the same.
            let color = colors[0][0];
            return Block {
                color0: Rgb565::from_f32(color),
                color1: Rgb565::from_f32(color),
                texels: [0; 4],
            };
        }

        let (endpoints_3, errors_weight_3) =
            Self::encode_optimize(colors, region, optimize, block_errors_3);
        let (endpoints_4, errors_weight_4) =
            Self::encode_optimize(colors, region, optimize, block_errors_4);

        let (best_565, texels) = if errors_weight_3 < errors_weight_4 {
            let mut best_565 = (
                Rgb565::from_f32(endpoints_3.0),
                Rgb565::from_f32(endpoints_3.1),
            );

            if best_565.0 == best_565.1 {
                return Block {
                    color0: best_565.0,
                    color1: best_565.0,
                    texels: [0; 4],
                };
            }

            if best_565.0.bits() < best_565.1.bits() {
                core::mem::swap(&mut best_565.0, &mut best_565.1);
            }

            let texels = block_encoding_3(colors, best_565);

            (best_565, texels)
        } else {
            let mut best_565 = (
                Rgb565::from_f32(endpoints_4.0),
                Rgb565::from_f32(endpoints_4.1),
            );

            if best_565.0 == best_565.1 {
                return Block {
                    color0: best_565.0,
                    color1: best_565.0,
                    texels: [0; 4],
                };
            }

            if best_565.0.bits() > best_565.1.bits() {
                core::mem::swap(&mut best_565.0, &mut best_565.1);
            }

            let texels = block_encoding_4(colors, best_565);

            (best_565, texels)
        };

        Block {
            color0: best_565.0,
            color1: best_565.1,
            texels,
        }
    }

    pub fn encode_with_alpha(colors: [[Rgba32F; 4]; 4], threshold: f32, optimize: usize) -> Self {
        let mut transparent_pixels = 0;

        for i in 0..4 {
            for j in 0..4 {
                if colors[i][j].a() < threshold {
                    transparent_pixels += 1;
                }
            }
        }

        if transparent_pixels == 16 {
            // All pixels are transparent.
            return Block {
                color0: Rgb565::from_f32(Rgb32F::BLACK),
                color1: Rgb565::from_f32(Rgb32F::BLACK),
                texels: [0xff; 4],
            };
        }

        if transparent_pixels == 0 {
            return Self::encode(colors.map(|row| row.map(|c| c.rgb())), optimize);
        }

        let region = Region3::new((0..16).filter_map(|i| {
            let c = colors[i / 4][i % 4];
            if c.a() < threshold {
                None
            } else {
                Some(c.rgb().into())
            }
        }));

        let mut best_endpoints = (Rgb32F::BLACK, Rgb32F::BLACK);
        let mut best_errors = (Vec3::ZERO, Vec3::ZERO);
        let mut best_errors_weight = f32::INFINITY;

        for diagonal in region.diagonals() {
            // Convert diagonal to colors, loose precision.
            let color0 = Rgb32F::from(diagonal.0);
            let color1 = Rgb32F::from(diagonal.1);

            let (errors, errors_weight) =
                block_errors_with_alpha_3(colors, (color0, color1), threshold);

            if errors_weight < best_errors_weight {
                best_endpoints = (color0, color1);
                best_errors = errors;
                best_errors_weight = errors_weight;
            }
        }

        for i in 0..optimize {
            if best_errors_weight < 1e-6 {
                break;
            }

            let endpoints = optimize_endpoints(
                best_endpoints,
                best_errors,
                best_errors_weight,
                0.01 / (i + 1) as f32,
            );

            let (errors, errors_weight) =
                block_errors_with_alpha_3(colors, best_endpoints, threshold);

            best_endpoints = endpoints;
            best_errors = errors;
            best_errors_weight = errors_weight;
        }

        let mut best_565 = (
            Rgb565::from_f32(best_endpoints.0),
            Rgb565::from_f32(best_endpoints.1),
        );

        if best_565.0 == best_565.1 {
            let texels = block_encoding_singular_with_alpha_3(colors, threshold);
            return Block {
                color0: best_565.0,
                color1: best_565.0,
                texels,
            };
        }

        if best_565.0.bits() < best_565.1.bits() {
            core::mem::swap(&mut best_565.0, &mut best_565.1);
        }

        let texels = block_encoding_with_alpha_3(colors, best_565, threshold);

        Block {
            color0: best_565.0,
            color1: best_565.1,
            texels,
        }
    }

    // pub fn encode(colors: [[Rgb32F; 4]; 4], optimize: usize) -> Self {
    //     let region = Region3::new((0..16).map(|i| Rgb32F::from(colors[i / 4][i % 4]).into()));

    //     let mut best_endpoints = (Rgb32F::BLACK, Rgb32F::BLACK);
    //     let mut best_errors = (Vec3::ZERO, Vec3::ZERO);
    //     let mut best_errors_weight = f32::INFINITY;

    //     for diagonal in region.diagonals() {
    //         // Convert diagonal to colors, loose precision.
    //         let color0 = Rgb32F::from(diagonal.0);
    //         let color1 = Rgb32F::from(diagonal.1);

    //         let (errors, errors_weight) = block_errors_4(colors, (color0, color1));

    //         if errors_weight < best_errors_weight {
    //             best_endpoints = (color0, color1);
    //             best_errors = errors;
    //             best_errors_weight = errors_weight;
    //         }
    //     }

    //     for i in 0..optimize {
    //         if best_errors_weight < 1e-6 {
    //             break;
    //         }

    //         let endpoints = optimize_endpoints(
    //             best_endpoints,
    //             best_errors,
    //             best_errors_weight,
    //             0.01 / (i + 1) as f32,
    //         );

    //         let (errors, errors_weight) = block_errors_4(colors, best_endpoints);
    //         // if errors_weight < best_errors_weight {
    //         best_endpoints = endpoints;
    //         best_errors = errors;
    //         best_errors_weight = errors_weight;

    //         //     eprintln!("!!!OPTIMIZED!!!");
    //         // } else {
    //         //     break;
    //         // }
    //     }

    //     let mut best_565 = (
    //         Rgb565::from_f32(best_endpoints.0),
    //         Rgb565::from_f32(best_endpoints.1),
    //     );

    //     if best_565.0.bits() > best_565.1.bits() {
    //         core::mem::swap(&mut best_565.0, &mut best_565.1);
    //     }

    //     let texels = block_encoding_4(colors, best_565);

    //     Block {
    //         color0: best_565.0,
    //         color1: best_565.1,
    //         texels,
    //     }
    // }
}

fn optimize_endpoints(
    endpoints: (Rgb32F, Rgb32F),
    errors: (Vec3, Vec3),
    errors_weight: f32,
    rate: f32,
) -> (Rgb32F, Rgb32F) {
    (
        endpoints.0.offset(errors.0 / errors_weight * rate),
        endpoints.1.offset(errors.1 / errors_weight * rate),
    )
}

fn distance(v: Vec3) -> f32 {
    // v.length()
    Yiq32F::perceptual_distance(Yiq32F::from_rgb(Rgb32F::from(v)), Yiq32F::BLACK)
}

fn block_errors_3(colors: [[Rgb32F; 4]; 4], endpoints: (Rgb32F, Rgb32F)) -> ((Vec3, Vec3), f32) {
    let mut errors = (Vec3::ZERO, Vec3::ZERO);
    let mut errors_weight = 0.0;

    let color0 = endpoints.0;
    let color1 = endpoints.1;

    let color12 = Rgb32F::lerp(color0, color1, 1.0 / 2.0);

    for i in 0..4 {
        for j in 0..4 {
            let texel = colors[i][j];

            let error0 = Rgb32F::diff(texel, color0);
            let error12 = Rgb32F::diff(texel, color12);

            let e0 = distance(error0);
            let e12 = distance(error12);

            if e0 <= e12 {
                // Closest to color0.
                errors.0 += error0;
                errors_weight += e0;
                continue;
            }

            let error1 = Rgb32F::diff(texel, color1);
            let e1 = distance(error1);

            if e12 <= e1 {
                // Closest to color12.
                errors.0 += error12 * (1.0 / 2.0);
                errors.1 += error12 * (1.0 / 2.0);
                errors_weight += e12;
            } else {
                // Closest to color1.
                errors.1 += error1;
                errors_weight += e1;
            }
        }
    }

    (errors, errors_weight)
}

fn block_errors_with_alpha_3(
    colors: [[Rgba32F; 4]; 4],
    endpoints: (Rgb32F, Rgb32F),
    threshold: f32,
) -> ((Vec3, Vec3), f32) {
    let mut errors = (Vec3::ZERO, Vec3::ZERO);
    let mut errors_weight = 0.0;

    let color0 = endpoints.0;
    let color1 = endpoints.1;

    let color12 = Rgb32F::lerp(color0, color1, 1.0 / 2.0);

    for i in 0..4 {
        for j in 0..4 {
            if colors[i][j].a() < threshold {
                continue;
            }

            let texel = colors[i][j].rgb();

            let error0 = Rgb32F::diff(texel, color0);
            let error12 = Rgb32F::diff(texel, color12);

            let e0 = distance(error0);
            let e12 = distance(error12);

            if e0 <= e12 {
                // Closest to color0.
                errors.0 += error0;
                errors_weight += e0;
                continue;
            }

            let error1 = Rgb32F::diff(texel, color1);
            let e1 = distance(error1);

            if e12 <= e1 {
                // Closest to color12.
                errors.0 += error12 * (1.0 / 2.0);
                errors.1 += error12 * (1.0 / 2.0);
                errors_weight += e12;
            } else {
                // Closest to color1.
                errors.1 += error1;
                errors_weight += e1;
            }
        }
    }

    (errors, errors_weight)
}

fn block_errors_4(colors: [[Rgb32F; 4]; 4], endpoints: (Rgb32F, Rgb32F)) -> ((Vec3, Vec3), f32) {
    let mut errors = (Vec3::ZERO, Vec3::ZERO);
    let mut errors_weight = 0.0;

    let color0 = endpoints.0;
    let color1 = endpoints.1;

    let color13 = Rgb32F::lerp(color0, color1, 1.0 / 3.0);
    let color23 = Rgb32F::lerp(color0, color1, 2.0 / 3.0);

    for i in 0..4 {
        for j in 0..4 {
            let texel = colors[i][j];

            let error0 = Rgb32F::diff(texel, color0);
            let error13 = Rgb32F::diff(texel, color13);

            let e0 = distance(error0);
            let e13 = distance(error13);

            if e0 <= e13 {
                // Closest to color0.
                errors.0 += error0;
                errors_weight += e0;
                continue;
            }

            let error23 = Rgb32F::diff(texel, color23);
            let e23 = distance(error23);

            if e13 <= e23 {
                // Closest to color13.
                errors.0 += error13 * (2.0 / 3.0);
                errors.1 += error13 * (1.0 / 3.0);
                errors_weight += e13;
                continue;
            }

            let error1 = Rgb32F::diff(texel, color1);
            let e1 = distance(error1);

            if e23 <= e1 {
                // Closest to color23.
                errors.0 += error23 * (1.0 / 3.0);
                errors.1 += error23 * (2.0 / 3.0);
                errors_weight += e23;
            } else {
                // Closest to color1.
                errors.1 += error1;
                errors_weight += e1;
            }
        }
    }

    (errors, errors_weight)
}

fn block_encoding_3(colors: [[Rgb32F; 4]; 4], endpoints: (Rgb565, Rgb565)) -> [u8; 4] {
    let color0 = endpoints.0.into_f32();
    let color1 = endpoints.1.into_f32();

    let color12 = Rgb32F::lerp(color0, color1, 1.0 / 2.0);

    let mut texels = [0; 4];

    for i in 0..4 {
        for j in 0..4 {
            let texel = colors[i][j];

            let error0 = Rgb32F::diff(texel, color0);
            let e0 = distance(error0);
            let error12 = Rgb32F::diff(texel, color12);
            let e12 = distance(error12);

            if e0 <= e12 {
                // Closest to color0.
                texels[i] |= 0b00 << (j * 2);
                continue;
            }

            let error1 = Rgb32F::diff(texel, color1);
            let e1 = distance(error1);

            if e12 <= e1 {
                // Closest to color12.
                texels[i] |= 0b01 << (j * 2);
            } else {
                // Closest to color1.
                texels[i] |= 0b10 << (j * 2);
            }
        }
    }

    texels
}

fn block_encoding_with_alpha_3(
    colors: [[Rgba32F; 4]; 4],
    endpoints: (Rgb565, Rgb565),
    threshold: f32,
) -> [u8; 4] {
    let color0 = endpoints.0.into_f32();
    let color1 = endpoints.1.into_f32();

    let color12 = Rgb32F::lerp(color0, color1, 1.0 / 2.0);

    let mut texels = [0; 4];

    for i in 0..4 {
        for j in 0..4 {
            let c = colors[i][j];
            if c.a() < threshold {
                texels[i] |= 0b11 << (j * 2);
            }

            let texel = c.rgb();

            let error0 = Rgb32F::diff(texel, color0);
            let e0 = distance(error0);
            let error12 = Rgb32F::diff(texel, color12);
            let e12 = distance(error12);

            if e0 <= e12 {
                // Closest to color0.
                texels[i] |= 0b00 << (j * 2);
                continue;
            }

            let error1 = Rgb32F::diff(texel, color1);
            let e1 = distance(error1);

            if e12 <= e1 {
                // Closest to color12.
                texels[i] |= 0b01 << (j * 2);
            } else {
                // Closest to color1.
                texels[i] |= 0b10 << (j * 2);
            }
        }
    }

    texels
}

fn block_encoding_4(colors: [[Rgb32F; 4]; 4], endpoints: (Rgb565, Rgb565)) -> [u8; 4] {
    let color0 = endpoints.0.into_f32();
    let color1 = endpoints.1.into_f32();

    let color13 = Rgb32F::lerp(color0, color1, 1.0 / 3.0);
    let color23 = Rgb32F::lerp(color0, color1, 2.0 / 3.0);

    let mut texels = [0; 4];

    for i in 0..4 {
        for j in 0..4 {
            let texel = colors[i][j];

            let error0 = Rgb32F::diff(texel, color0);
            let e0 = distance(error0);
            let error13 = Rgb32F::diff(texel, color13);
            let e13 = distance(error13);

            if e0 <= e13 {
                // Closest to color0.
                texels[i] |= 0b00 << (j * 2);
                continue;
            }

            let error23 = Rgb32F::diff(texel, color23);
            let e23 = distance(error23);

            if e13 <= e23 {
                // Closest to color13.
                texels[i] |= 0b01 << (j * 2);
                continue;
            }

            let error1 = Rgb32F::diff(texel, color1);
            let e1 = distance(error1);

            if e23 <= e1 {
                // Closest to color23.
                texels[i] |= 0b10 << (j * 2);
            } else {
                // Closest to color1.
                texels[i] |= 0b11 << (j * 2);
            }
        }
    }

    texels
}

fn block_encoding_singular_with_alpha_3(colors: [[Rgba32F; 4]; 4], threshold: f32) -> [u8; 4] {
    let mut texels = [0; 4];

    for i in 0..4 {
        for j in 0..4 {
            if colors[i][j].a() < threshold {
                texels[i] |= 0b11 << (j * 2);
            } else {
                texels[i] |= 0b00 << (j * 2);
            }
        }
    }

    texels
}
