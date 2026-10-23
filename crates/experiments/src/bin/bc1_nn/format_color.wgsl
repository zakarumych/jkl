struct PackParams { count: u32, format: u32 }
@group(0) @binding(0) var<storage, read> color_input: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> full_output: array<vec4<u32>>;
var<immediate> pack_params: PackParams;
@compute @workgroup_size(64)
fn append_alpha(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= pack_params.count { return; }
    let color = color_input[id.x];
    var alpha = vec2<u32>(0xffffffffu);
    if pack_params.format == 3u { alpha = vec2<u32>(255u, 0u); }
    full_output[id.x] = vec4<u32>(alpha, color);
}
