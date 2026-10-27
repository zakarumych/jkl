//! Deterministic patch ownership, adjacency strips, and triangle index streams.
use super::{
    data::{MeshData, MeshError},
    order::order_quantized,
    split_positions,
};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub patches: Vec<PatchLayout>,
    pub old_to_new: Vec<u32>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatchLayout {
    pub vertex_base: u32,
    pub triangle_base: u32,
    /// Input vertex IDs in attribute decode order.
    pub vertices: Vec<u32>,
    /// Patch-local triangle indices, with odd ordinal winding reversed.
    pub head: Vec<[u32; 3]>,
    /// Global triangle indices, continuing the head ordinal parity.
    pub tail: Vec<[u32; 3]>,
    pub head_face_ids: Vec<u32>,
    pub tail_face_ids: Vec<u32>,
    pub internal_vertex_count: u32,
}

fn edges(t: [u32; 3]) -> [[u32; 2]; 3] {
    [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]].map(|mut e| {
        e.sort_unstable();
        e
    })
}
fn corners(t: [u32; 3]) -> BTreeSet<u32> {
    t.into_iter().collect()
}

fn strips(faces: &[[u32; 3]], ids: &[u32]) -> Vec<Vec<u32>> {
    let mut cs: HashMap<u32, BTreeSet<u32>> = HashMap::new();
    let mut es: HashMap<[u32; 2], BTreeSet<u32>> = HashMap::new();
    for &f in ids {
        let t = faces[f as usize];
        for v in t {
            cs.entry(v).or_default().insert(f);
        }
        for e in edges(t) {
            es.entry(e).or_default().insert(f);
        }
    }
    let score = |f: u32, cs: &HashMap<u32, BTreeSet<u32>>| {
        corners(faces[f as usize])
            .iter()
            .map(|v| cs[v].len())
            .sum::<usize>()
    };
    let mut seeds = ids.to_vec();
    seeds.sort_unstable_by_key(|&f| (score(f, &cs), f));
    let mut remaining: BTreeSet<u32> = ids.iter().copied().collect();
    let mut output = Vec::new();
    for seed in seeds {
        if !remaining.contains(&seed) {
            continue;
        }
        let mut current = seed;
        let mut strip = Vec::new();
        loop {
            strip.push(current);
            remaining.remove(&current);
            let t = faces[current as usize];
            for v in t {
                cs.get_mut(&v).unwrap().remove(&current);
            }
            for e in edges(t) {
                es.get_mut(&e).unwrap().remove(&current);
            }
            let mut candidates = BTreeSet::new();
            for e in edges(t) {
                candidates.extend(es[&e].iter().copied());
            }
            if candidates.is_empty() {
                for v in t {
                    candidates.extend(cs[&v].iter().copied());
                }
            }
            match candidates.into_iter().min_by_key(|&f| (score(f, &cs), f)) {
                Some(f) => current = f,
                None => break,
            }
        }
        output.push(strip);
    }
    output
}

fn permute(mut t: [u32; 3], ordinal: usize, previous: Option<[u32; 3]>) -> [u32; 3] {
    if ordinal & 1 != 0 {
        t.swap(1, 2);
    }
    [t, [t[1], t[2], t[0]], [t[2], t[0], t[1]]]
        .into_iter()
        .min_by_key(|&a| {
            let (matches, cost) = previous.map_or((0, 0), |p| {
                (
                    a.into_iter().zip(p).filter(|(a, b)| a == b).count(),
                    a.into_iter()
                        .zip(p)
                        .map(|(a, b)| a.abs_diff(b) as u64)
                        .sum::<u64>(),
                )
            });
            (3 - matches, cost, a)
        })
        .unwrap()
}

fn loose_route(mesh: &MeshData, vertices: &[u32]) -> Result<Vec<u32>, MeshError> {
    if vertices.len() < 2 {
        return Ok(vertices.to_vec());
    }
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for &v in vertices {
        for a in 0..3 {
            let p = mesh.positions[v as usize][a] as f64;
            lo[a] = lo[a].min(p);
            hi[a] = hi[a].max(p);
        }
    }
    let quantized: Vec<_> = vertices
        .iter()
        .map(|&v| {
            std::array::from_fn(|a| {
                if hi[a] == lo[a] {
                    0
                } else {
                    ((mesh.positions[v as usize][a] as f64 - lo[a]) / (hi[a] - lo[a]) * 1048575.0)
                        .round() as u32
                }
            })
        })
        .collect();
    let mut route: Vec<_> = order_quantized(&quantized, 20)?
        .hilbert
        .into_iter()
        .map(|i| vertices[i])
        .collect();
    let distance = |a: u32, b: u32| {
        mesh.positions[a as usize]
            .into_iter()
            .zip(mesh.positions[b as usize])
            .map(|(a, b)| (a as f64 - b as f64).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    for _ in 0..4 {
        let mut changed = false;
        for i in 0..route.len() - 1 {
            for j in i + 1..route.len().min(i + 32) {
                let mut before = 0.;
                let mut after = 0.;
                if i > 0 {
                    before += distance(route[i - 1], route[i]);
                    after += distance(route[i - 1], route[j]);
                }
                if j + 1 < route.len() {
                    before += distance(route[j], route[j + 1]);
                    after += distance(route[i], route[j + 1]);
                }
                if after < before - 1e-12 {
                    route[i..=j].reverse();
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    Ok(route)
}

/// Builds disjoint balanced vertex patches and assigns every triangle once.
/// Internal faces decode attributes; cross-patch strips become index-only tails.
/// Tails belong to the patch receiving most corner votes, with patch ID breaking ties.
/// No triangle restart markers are required.
pub fn build_layout(mesh: &MeshData, max_vertices: usize) -> Result<Layout, MeshError> {
    mesh.validate()?;
    if !(1..=256).contains(&max_vertices) {
        return Err(MeshError::InvalidInput(
            "patch vertex count must be 1 through 256",
        ));
    }
    let positions: Vec<_> = mesh.positions.iter().map(|p| p.map(f64::from)).collect();
    let groups = split_positions(&positions, max_vertices)
        .map_err(|_| MeshError::InvalidInput("invalid patch positions"))?;
    let mut owner = vec![0; positions.len()];
    for (p, vs) in groups.iter().enumerate() {
        for &v in vs {
            owner[v] = p;
        }
    }
    let mut internal = vec![Vec::new(); groups.len()];
    let mut boundary = Vec::new();
    for (f, t) in mesh.triangles.iter().enumerate() {
        let p = owner[t[0] as usize];
        if t.iter().all(|&v| owner[v as usize] == p) {
            internal[p].push(f as u32);
        } else {
            boundary.push(f as u32);
        }
    }
    let mut tails = vec![Vec::new(); groups.len()];
    for strip in strips(&mesh.triangles, &boundary) {
        let mut votes: HashMap<usize, usize> = HashMap::new();
        for &f in &strip {
            for v in mesh.triangles[f as usize] {
                *votes.entry(owner[v as usize]).or_default() += 1;
            }
        }
        let p = votes
            .into_iter()
            .min_by_key(|&(p, n)| (std::cmp::Reverse(n), p))
            .unwrap()
            .0;
        tails[p].extend(strip);
    }
    let mut patches = Vec::with_capacity(groups.len());
    let mut old_to_new = vec![0; positions.len()];
    let mut vertex_base = 0;
    let mut triangle_base = 0;
    for (p, group) in groups.iter().enumerate() {
        let head_face_ids: Vec<_> = strips(&mesh.triangles, &internal[p])
            .into_iter()
            .flatten()
            .collect();
        let mut seen = BTreeSet::new();
        let mut vertices = Vec::with_capacity(group.len());
        let mut local = HashMap::<u32, u32>::new();
        let mut head = Vec::<[u32; 3]>::new();
        for (ordinal, &f) in head_face_ids.iter().enumerate() {
            let mut source = mesh.triangles[f as usize];
            if ordinal & 1 != 0 {
                source.swap(1, 2);
            }
            let rotations = [
                source,
                [source[1], source[2], source[0]],
                [source[2], source[0], source[1]],
            ];
            let (stored, source) = rotations
                .into_iter()
                .map(|source| {
                    let mut pending = Vec::new();
                    let stored = source.map(|v| {
                        *local.get(&v).unwrap_or_else(|| {
                            let i = match pending.iter().position(|&(old, _)| old == v) {
                                Some(i) => i,
                                None => {
                                    let i = pending.len();
                                    pending.push((v, vertices.len() as u32 + i as u32));
                                    i
                                }
                            };
                            &pending[i].1
                        })
                    });
                    (stored, source)
                })
                .min_by_key(|&(stored, source)| {
                    let (matches, cost) = head.last().map_or((0, 0), |&prev| {
                        (
                            stored.into_iter().zip(prev).filter(|(a, b)| a == b).count(),
                            stored
                                .into_iter()
                                .zip(prev)
                                .map(|(a, b)| a.abs_diff(b) as u64)
                                .sum::<u64>(),
                        )
                    });
                    (3 - matches, cost, stored, source)
                })
                .unwrap();
            for v in source {
                if seen.insert(v) {
                    local.insert(v, vertices.len() as u32);
                    vertices.push(v);
                }
            }
            head.push(stored);
        }
        let internal_vertex_count = vertices.len() as u32;
        let loose: Vec<_> = group
            .iter()
            .map(|&v| v as u32)
            .filter(|v| !seen.contains(v))
            .collect();
        vertices.extend(loose_route(mesh, &loose)?);
        for (i, &v) in vertices.iter().enumerate() {
            old_to_new[v as usize] = vertex_base + i as u32;
        }
        let tail_face_ids = std::mem::take(&mut tails[p]);
        let triangle_count = (head_face_ids.len() + tail_face_ids.len()) as u32;
        let vertex_count = vertices.len() as u32;
        patches.push(PatchLayout {
            vertex_base,
            triangle_base,
            vertices,
            head,
            tail: Vec::new(),
            head_face_ids,
            tail_face_ids,
            internal_vertex_count,
        });
        vertex_base += vertex_count;
        triangle_base += triangle_count;
    }
    for patch in &mut patches {
        let mut prev = patch
            .head
            .last()
            .copied()
            .map(|t| t.map(|v| v + patch.vertex_base));
        for (i, &f) in patch.tail_face_ids.iter().enumerate() {
            let t = permute(
                mesh.triangles[f as usize].map(|v| old_to_new[v as usize]),
                i + patch.head.len(),
                prev,
            );
            patch.tail.push(t);
            prev = Some(t);
        }
    }
    Ok(Layout {
        patches,
        old_to_new,
    })
}
