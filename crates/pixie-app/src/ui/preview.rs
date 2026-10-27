//! Bounded CPU previews. Encoding always uses the complete source asset.
use crate::backend::{Kind, Source, TextureSource};
#[derive(Clone, Copy)]
pub(super) struct Config {
    pub width: u32,
    pub height: u32,
    pub mode: u8,
    pub zoom: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub wire: bool,
}

pub(super) fn render(
    source: &Source,
    result: Option<&Source>,
    mut c: Config,
) -> (u32, u32, Vec<u8>) {
    let scale = (c.width as f32 / 1400.).max(c.height as f32 / 900.).max(1.);
    c.width = (c.width as f32 / scale) as u32;
    c.height = (c.height as f32 / scale) as u32;
    let mut out = vec![0; c.width as usize * c.height as usize * 4];
    for y in 0..c.height {
        for x in 0..c.width {
            let v = if (x / 16 + y / 16) % 2 == 0 { 25 } else { 30 };
            put(
                &mut out,
                c.width,
                x as i32,
                y as i32,
                [v, v + 3, v + 4, 255],
            );
        }
    }
    if c.mode == 0 {
        let half = c.width / 2;
        draw_source(&mut out, c, source, 0, half, None, source);
        if let Some(r) = result {
            draw_source(&mut out, c, r, half, c.width - half, None, source);
        }
        for y in 0..c.height {
            put(
                &mut out,
                c.width,
                half as i32,
                y as i32,
                [100, 111, 104, 255],
            );
        }
    } else if c.mode == 2 {
        if let Some(r) = result {
            draw_source(&mut out, c, r, 0, c.width, None, source);
        }
    } else if c.mode != 3 || result.is_some() {
        draw_source(
            &mut out,
            c,
            source,
            0,
            c.width,
            if c.mode == 3 { result } else { None },
            source,
        );
    }
    (c.width, c.height, out)
}
fn put(out: &mut [u8], width: u32, x: i32, y: i32, rgba: [u8; 4]) {
    if x < 0 || y < 0 || x >= width as i32 {
        return;
    }
    let i = (y as usize * width as usize + x as usize) * 4;
    if i + 4 <= out.len() {
        out[i..i + 4].copy_from_slice(&rgba);
    }
}
fn draw_source(
    out: &mut [u8],
    c: Config,
    source: &Source,
    left: u32,
    width: u32,
    error: Option<&Source>,
    framing: &Source,
) {
    match source {
        Source::Texture(t) => {
            let scale = ((width.saturating_sub(48)) as f32 / t.width as f32)
                .min(c.height.saturating_sub(80) as f32 / t.height as f32)
                * c.zoom;
            let iw = t.width as f32 * scale;
            let ih = t.height as f32 * scale;
            let ox = left as f32 + (width as f32 - iw) * 0.5;
            let oy = (c.height as f32 - ih) * 0.5 + 10.;
            for y in 36..c.height {
                for x in left..left + width {
                    let sx = ((x as f32 - ox) / scale).floor();
                    let sy = ((y as f32 - oy) / scale).floor();
                    if sx < 0. || sy < 0. || sx >= t.width as f32 || sy >= t.height as f32 {
                        continue;
                    }
                    let i = sy as usize * t.width + sx as usize;
                    let mut p = t.pixels[i];
                    if let Some(Source::Texture(r)) = error {
                        if let Some(q) = r.pixels.get(i) {
                            let alpha = (p[3] - q[3]).abs();
                            p = [
                                (p[0] - q[0]).abs().max(alpha) * 8.,
                                (p[1] - q[1]).abs().max(alpha) * 8.,
                                (p[2] - q[2]).abs().max(alpha) * 8.,
                                1.,
                            ];
                        }
                    }
                    let idx = (y as usize * c.width as usize + x as usize) * 4;
                    let a = p[3].clamp(0., 1.);
                    let rgb = std::array::from_fn::<_, 3, _>(|k| {
                        (p[k].clamp(0., 1.) * 255. * a + out[idx + k] as f32 * (1. - a)).round()
                            as u8
                    });
                    put(
                        out,
                        c.width,
                        x as i32,
                        y as i32,
                        [rgb[0], rgb[1], rgb[2], 255],
                    );
                }
            }
        }
        Source::Mesh(m) => mesh(
            out,
            c,
            m,
            left,
            width,
            if let Source::Mesh(original) = framing {
                original
            } else {
                m
            },
        ),
        Source::Blob(_) => {}
    }
}
fn mesh(
    out: &mut [u8],
    c: Config,
    m: &jkl::mesh::data::MeshData,
    left: u32,
    width: u32,
    framing: &jkl::mesh::data::MeshData,
) {
    if m.positions.is_empty() {
        return;
    }
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in &framing.positions {
        for i in 0..3 {
            lo[i] = lo[i].min(p[i]);
            hi[i] = hi[i].max(p[i]);
        }
    }
    let center = std::array::from_fn::<_, 3, _>(|i| (lo[i] as f64 + hi[i] as f64) * 0.5);
    let extent = (0..3)
        .map(|i| hi[i] as f64 - lo[i] as f64)
        .fold(1e-20_f64, f64::max);
    let (cy, sy) = (c.yaw.cos(), c.yaw.sin());
    let (cp, sp) = (c.pitch.cos(), c.pitch.sin());
    let scale = width.min(c.height) as f32 * 0.70 * c.zoom;
    let points: Vec<[f32; 3]> = m
        .positions
        .iter()
        .map(|p| {
            let a = std::array::from_fn::<_, 3, _>(|i| ((p[i] as f64 - center[i]) / extent) as f32);
            let x = a[0] * cy + a[2] * sy;
            let z = -a[0] * sy + a[2] * cy;
            let y = a[1] * cp - z * sp;
            let z = a[1] * sp + z * cp;
            [
                left as f32 + width as f32 * 0.5 + x * scale,
                c.height as f32 * 0.55 - y * scale,
                z,
            ]
        })
        .collect();
    let mut depth = vec![f32::NEG_INFINITY; c.width as usize * c.height as usize];
    for tri in &m.triangles {
        let [a, b, d] = tri.map(|i| points[i as usize]);
        let area = edge(a, b, d[0], d[1]);
        if area.abs() < 0.0001 {
            continue;
        }
        let minx = a[0].min(b[0]).min(d[0]).floor().max(left as f32) as i32;
        let maxx = a[0]
            .max(b[0])
            .max(d[0])
            .ceil()
            .min((left + width - 1) as f32) as i32;
        let miny = a[1].min(b[1]).min(d[1]).floor().max(36.) as i32;
        let maxy = a[1].max(b[1]).max(d[1]).ceil().min((c.height - 1) as f32) as i32;
        let u = [b[0] - a[0], b[1] - a[1], (b[2] - a[2]) * scale];
        let v = [d[0] - a[0], d[1] - a[1], (d[2] - a[2]) * scale];
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let norm = (n.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-10);
        let light =
            (0.30 + 0.70 * ((n[0] * -0.3 + n[1] * -0.5 + n[2] * 0.8) / norm).abs()).clamp(0., 1.);
        for y in miny..=maxy {
            for x in minx..=maxx {
                let w0 = edge(b, d, x as f32 + 0.5, y as f32 + 0.5) / area;
                let w1 = edge(d, a, x as f32 + 0.5, y as f32 + 0.5) / area;
                let w2 = 1. - w0 - w1;
                if w0 < 0. || w1 < 0. || w2 < 0. {
                    continue;
                }
                let z = w0 * a[2] + w1 * b[2] + w2 * d[2];
                let i = y as usize * c.width as usize + x as usize;
                if z < depth[i] {
                    continue;
                }
                depth[i] = z;
                let edge_px = w0.min(w1).min(w2) < 0.015;
                let rgb = if c.wire && edge_px {
                    [228, 241, 176, 255]
                } else if c.wire {
                    [40, 53, 52, 255]
                } else {
                    [
                        (170. * light) as u8,
                        (205. * light) as u8,
                        (182. * light) as u8,
                        255,
                    ]
                };
                put(out, c.width, x, y, rgb);
            }
        }
    }
}
fn edge(a: [f32; 3], b: [f32; 3], x: f32, y: f32) -> f32 {
    (x - a[0]) * (b[1] - a[1]) - (y - a[1]) * (b[0] - a[0])
}
pub(super) fn fixture(kind: Kind) -> Source {
    match kind {
        Kind::Texture => {
            let (w, h) = (256, 192);
            let pixels = (0..h)
                .flat_map(|y| {
                    (0..w).map(move |x| {
                        let u = x as f32 / w as f32;
                        let v = y as f32 / h as f32;
                        let weave = ((x / 5 + y / 5) % 2) as f32 * 0.12;
                        let ring =
                            (((u - 0.5).powi(2) + (v - 0.5).powi(2)).sqrt() * 70.).sin() * 0.12;
                        [
                            0.15 + u * 0.7 + weave,
                            0.1 + v * 0.7 + ring,
                            0.65 - u * 0.4 + weave,
                            if x < 12 || y < 12 || x > w - 13 || y > h - 13 {
                                0.45
                            } else {
                                1.
                            },
                        ]
                    })
                })
                .collect();
            Source::Texture(TextureSource {
                width: w,
                height: h,
                pixels,
            })
        }
        Kind::Mesh => Source::Mesh(jkl::mesh::data::MeshData {
            positions: vec![
                [-1., -1., -1.],
                [1., -1., -1.],
                [1., 1., -1.],
                [-1., 1., -1.],
                [-1., -1., 1.],
                [1., -1., 1.],
                [1., 1., 1.],
                [-1., 1., 1.],
            ],
            triangles: vec![
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [2, 3, 7],
                [2, 7, 6],
                [0, 4, 7],
                [0, 7, 3],
                [1, 2, 6],
                [1, 6, 5],
            ],
            ..Default::default()
        }),
        Kind::Blob => Source::Blob(
            (0..4096)
                .map(|i| b"JKL ASSET WORKBENCH / verified bytes\n"[i % 37])
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounds_and_alpha_are_defined() {
        let c = Config {
            width: 320,
            height: 180,
            mode: 0,
            zoom: 1.,
            yaw: 0.4,
            pitch: 0.2,
            wire: false,
        };
        for kind in [Kind::Texture, Kind::Mesh] {
            let s = fixture(kind);
            let (w, h, p) = render(&s, Some(&s), c);
            assert_eq!(p.len(), (w * h * 4) as usize);
            assert!(p.chunks_exact(4).all(|p| p[3] == 255));
        }
    }
    #[test]
    fn missing_error_result_does_not_display_source() {
        let c = Config {
            width: 320,
            height: 180,
            mode: 3,
            zoom: 1.,
            yaw: 0.,
            pitch: 0.,
            wire: false,
        };
        let s = fixture(Kind::Texture);
        let (_, _, p) = render(&s, None, c);
        assert!(
            p.chunks_exact(4)
                .all(|p| p[1] == p[0] + 3 && p[2] == p[0] + 4)
        );
    }
    #[test]
    fn alpha_error_is_visible() {
        let c = Config {
            width: 80,
            height: 100,
            mode: 3,
            zoom: 1.,
            yaw: 0.,
            pitch: 0.,
            wire: false,
        };
        let a = Source::Texture(TextureSource {
            width: 1,
            height: 1,
            pixels: vec![[0., 0., 0., 0.]],
        });
        let b = Source::Texture(TextureSource {
            width: 1,
            height: 1,
            pixels: vec![[0., 0., 0., 1.]],
        });
        let (_, _, p) = render(&a, Some(&b), c);
        assert!(p.chunks_exact(4).any(|p| p == [255, 255, 255, 255]));
    }
}
