use jkl::mesh::data::MeshData;
use std::{collections::HashMap, fmt::Write as _};
fn obj_index(text: &str, len: usize) -> Result<usize, String> {
    let value: i64 = text.parse().map_err(|_| "Invalid OBJ index")?;
    let index = if value > 0 {
        value - 1
    } else if value < 0 {
        len as i64 + value
    } else {
        -1
    };
    if index < 0 || index >= len as i64 {
        Err("OBJ index out of range".into())
    } else {
        Ok(index as usize)
    }
}

pub fn parse_obj(text: &str) -> Result<MeshData, String> {
    let mut positions = Vec::<[f32; 3]>::new();
    let mut normals = Vec::<[f32; 3]>::new();
    let mut uvs = Vec::<[f32; 2]>::new();
    let mut mesh = MeshData::default();
    let mut map = HashMap::new();
    let mut attributes = None;
    for (line_no, line) in text.lines().enumerate() {
        let fields: Vec<_> = line
            .split('#')
            .next()
            .unwrap_or("")
            .split_whitespace()
            .collect();
        if fields.is_empty() {
            continue;
        }
        let error = |s: &str| format!("OBJ line {}: {s}", line_no + 1);
        match fields[0] {
            "v" | "vn" | "vt" => {
                let count = if fields[0] == "vt" { 2 } else { 3 };
                if fields.len() != count + 1 {
                    return Err(error(
                        "Only xyz positions/normals and two-component UVs are supported; vertex colors and homogeneous coordinates are not supported.",
                    ));
                }
                let values: Result<Vec<f32>, _> =
                    fields[1..].iter().map(|v| v.parse::<f32>()).collect();
                let values = values.map_err(|_| error("Invalid coordinate"))?;
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(error("Non-finite coordinate"));
                }
                match fields[0] {
                    "v" => positions.push([values[0], values[1], values[2]]),
                    "vn" => normals.push([values[0], values[1], values[2]]),
                    _ => uvs.push([values[0], values[1]]),
                }
            }
            "f" => {
                if fields.len() != 4 {
                    return Err(error(
                        "Only triangular faces are supported. Triangulate polygons before import.",
                    ));
                }
                let mut triangle = [0; 3];
                for (corner, token) in fields[1..].iter().enumerate() {
                    let indices: Vec<_> = token.split('/').collect();
                    if indices.len() > 3 {
                        return Err(error("Invalid face index"));
                    }
                    let p = obj_index(indices[0], positions.len()).map_err(|e| error(&e))?;
                    let uv = indices
                        .get(1)
                        .filter(|s| !s.is_empty())
                        .map(|s| obj_index(s, uvs.len()))
                        .transpose()
                        .map_err(|e| error(&e))?;
                    let normal = indices
                        .get(2)
                        .filter(|s| !s.is_empty())
                        .map(|s| obj_index(s, normals.len()))
                        .transpose()
                        .map_err(|e| error(&e))?;
                    let present = (uv.is_some(), normal.is_some());
                    if attributes.is_some_and(|a| a != present) {
                        return Err(error(
                            "Normals and UVs must be consistently present on all face corners.",
                        ));
                    }
                    attributes = Some(present);
                    let key = (p, uv, normal);
                    triangle[corner] = if let Some(&index) = map.get(&key) {
                        index
                    } else {
                        let index = u32::try_from(mesh.positions.len())
                            .map_err(|_| error("Too many vertices"))?;
                        mesh.positions.push(positions[p]);
                        if let Some(uv) = uv {
                            if mesh.tex_coords.is_empty() {
                                mesh.tex_coords.push(Vec::new());
                            }
                            mesh.tex_coords[0].push(uvs[uv]);
                        }
                        if let Some(n) = normal {
                            mesh.normals.get_or_insert_with(Vec::new).push(normals[n]);
                        }
                        map.insert(key, index);
                        index
                    };
                }
                mesh.triangles.push(triangle);
            }
            "o" | "g" | "s" => {}
            other => {
                return Err(error(&format!(
                    "Unsupported OBJ directive '{other}' (materials, colors, curves and lines are not imported)."
                )));
            }
        }
    }
    if mesh.triangles.is_empty() {
        return Err("OBJ contains no triangle faces".into());
    }
    mesh.validate().map_err(|e| e.to_string())?;
    Ok(mesh)
}

fn obj_export_error(mesh: &MeshData) -> Option<String> {
    if mesh.tangents.is_some()
        || mesh.colors.is_some()
        || mesh.tex_coords.len() > 1
        || !mesh.materials.is_empty()
        || mesh.primitive_ids.iter().any(|&id| id != 0)
    {
        Some("OBJ export unavailable: tangents, colors, multiple UV sets, or material/primitive metadata would be lost. Save JKLM to preserve them.".into())
    } else {
        None
    }
}

pub fn export_obj(mesh: &MeshData) -> Result<String, String> {
    mesh.validate().map_err(|e| e.to_string())?;
    if let Some(error) = obj_export_error(mesh) {
        return Err(error);
    }
    let mut text = String::from("# JKL decoded mesh\n");
    for p in &mesh.positions {
        writeln!(text, "v {} {} {}", p[0], p[1], p[2]).unwrap();
    }
    if let Some(uvs) = mesh.tex_coords.first() {
        for p in uvs {
            writeln!(text, "vt {} {}", p[0], p[1]).unwrap();
        }
    }
    if let Some(normals) = &mesh.normals {
        for p in normals {
            writeln!(text, "vn {} {} {}", p[0], p[1], p[2]).unwrap();
        }
    }
    for t in &mesh.triangles {
        text.push('f');
        for &i in t {
            let i = i + 1;
            match (!mesh.tex_coords.is_empty(), mesh.normals.is_some()) {
                (false, false) => write!(text, " {i}"),
                (true, false) => write!(text, " {i}/{i}"),
                (false, true) => write!(text, " {i}//{i}"),
                (true, true) => write!(text, " {i}/{i}/{i}"),
            }
            .unwrap();
        }
        text.push('\n');
    }
    Ok(text)
}
