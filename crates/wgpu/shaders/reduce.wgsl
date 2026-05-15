const REDUCE_MAX_SUBGROUPS: u32 = 32u;
const REDUCE_MAX_STAGES: u32 = 5u;
const REDUCE_POS_INF_F32: f32 = 3.4028235e38;
const REDUCE_NEG_INF_F32: f32 = -3.4028235e38;

var<workgroup> reduce: array<u32, REDUCE_MAX_SUBGROUPS>;

fn reduce_div_ceil_u32(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

fn workgroupAdd_f32(
    value: f32,
    workgroup_size: u32,
    subgroup_size: u32,
    local_index: u32,
) -> f32 {
    let subgroup_id = local_index / subgroup_size;
    let subgroup_invocation_id = local_index % subgroup_size;
    let subgroup_sum = subgroupAdd(value);
    if workgroup_size <= subgroup_size {
        return subgroup_sum;
    }

    if subgroup_invocation_id == 0u {
        reduce[subgroup_id] = bitcast<u32>(subgroup_sum);
    }

    workgroupBarrier();

    let subgroup_count = reduce_div_ceil_u32(workgroup_size, subgroup_size);

    var threads = subgroup_count;
    for (var stage = 0u; stage < REDUCE_MAX_STAGES; stage += 1u) {
        if threads > 1u {
            let half = threads / 2u;
            let rem = threads % 2u;
            let offset = half + rem;

            if subgroup_invocation_id == 0u && subgroup_id < half {
                let other = subgroup_id + offset;
                reduce[subgroup_id] = bitcast<u32>(bitcast<f32>(reduce[subgroup_id]) + bitcast<f32>(reduce[other]));
            }

            threads = half + rem;

            workgroupBarrier();
        }
    }
    return bitcast<f32>(reduce[0]);
}

fn workgroupMin_f32(
    value: f32,
    workgroup_size: u32,
    subgroup_size: u32,
    local_index: u32,
) -> f32 {
    let subgroup_id = local_index / subgroup_size;
    let subgroup_invocation_id = local_index % subgroup_size;
    let subgroup_min = subgroupMin(value);
    if workgroup_size <= subgroup_size {
        return subgroup_min;
    }

    if subgroup_invocation_id == 0u {
        reduce[subgroup_id] = bitcast<u32>(subgroup_min);
    }

    workgroupBarrier();

    let subgroup_count = reduce_div_ceil_u32(workgroup_size, subgroup_size);

    var threads = subgroup_count;
    for (var stage = 0u; stage < REDUCE_MAX_STAGES; stage += 1u) {
        if threads > 1u {
            let half = threads / 2u;
            let rem = threads % 2u;
            let offset = half + rem;

            if subgroup_invocation_id == 0u && subgroup_id < half {
                let other = subgroup_id + offset;
                reduce[subgroup_id] = bitcast<u32>(min(bitcast<f32>(reduce[subgroup_id]), bitcast<f32>(reduce[other])));
            }

            threads = half + rem;

            workgroupBarrier();
        }
    }
    return bitcast<f32>(reduce[0]);
}

fn workgroupMax_f32(
    value: f32,
    workgroup_size: u32,
    subgroup_size: u32,
    local_index: u32,
) -> f32 {
    let subgroup_id = local_index / subgroup_size;
    let subgroup_invocation_id = local_index % subgroup_size;
    let subgroup_max = subgroupMax(value);
    if workgroup_size <= subgroup_size {
        return subgroup_max;
    }

    if subgroup_invocation_id == 0u {
        reduce[subgroup_id] = bitcast<u32>(subgroup_max);
    }

    workgroupBarrier();

    let subgroup_count = reduce_div_ceil_u32(workgroup_size, subgroup_size);

    var threads = subgroup_count;
    for (var stage = 0u; stage < REDUCE_MAX_STAGES; stage += 1u) {
        if threads > 1u {
            let half = threads / 2u;
            let rem = threads % 2u;
            let offset = half + rem;

            if subgroup_invocation_id == 0u && subgroup_id < half {
                let other = subgroup_id + offset;
                reduce[subgroup_id] = bitcast<u32>(max(bitcast<f32>(reduce[subgroup_id]), bitcast<f32>(reduce[other])));
            }

            threads = half + rem;

            workgroupBarrier();
        }
    }
    return bitcast<f32>(reduce[0]);
}

fn workgroupAdd_u32(
    value: u32,
    workgroup_size: u32,
    subgroup_size: u32,
    local_index: u32,
) -> u32 {
    let subgroup_id = local_index / subgroup_size;
    let subgroup_invocation_id = local_index % subgroup_size;
    let subgroup_sum = subgroupAdd(value);
    if workgroup_size <= subgroup_size {
        return subgroup_sum;
    }

    if subgroup_invocation_id == 0u {
        reduce[subgroup_id] = subgroup_sum;
    }

    workgroupBarrier();

    let subgroup_count = reduce_div_ceil_u32(workgroup_size, subgroup_size);

    var threads = subgroup_count;
    for (var stage = 0u; stage < REDUCE_MAX_STAGES; stage += 1u) {
        if threads > 1u {
            let half = threads / 2u;
            let rem = threads % 2u;
            let offset = half + rem;

            if subgroup_invocation_id == 0u && subgroup_id < half {
                let other = subgroup_id + offset;
                reduce[subgroup_id] = reduce[subgroup_id] + reduce[other];
            }

            threads = half + rem;

            workgroupBarrier();
        }
    }
    return reduce[0];
}

fn workgroupMin_u32(
    value: u32,
    workgroup_size: u32,
    subgroup_size: u32,
    local_index: u32,
) -> u32 {
    let subgroup_id = local_index / subgroup_size;
    let subgroup_invocation_id = local_index % subgroup_size;
    let subgroup_min = subgroupMin(value);
    if workgroup_size <= subgroup_size {
        return subgroup_min;
    }

    if subgroup_invocation_id == 0u {
        reduce[subgroup_id] = subgroup_min;
    }

    workgroupBarrier();

    let subgroup_count = reduce_div_ceil_u32(workgroup_size, subgroup_size);

    var threads = subgroup_count;
    for (var stage = 0u; stage < REDUCE_MAX_STAGES; stage += 1u) {
        if threads > 1u {
            let half = threads / 2u;
            let rem = threads % 2u;
            let offset = half + rem;

            if subgroup_invocation_id == 0u && subgroup_id < half {
                let other = subgroup_id + offset;
                reduce[subgroup_id] = min(reduce[subgroup_id], reduce[other]);
            }

            threads = half + rem;

            workgroupBarrier();
        }
    }
    return reduce[0];
}

fn workgroupMax_u32(
    value: u32,
    workgroup_size: u32,
    subgroup_size: u32,
    local_index: u32,
) -> u32 {
    let subgroup_id = local_index / subgroup_size;
    let subgroup_invocation_id = local_index % subgroup_size;
    let subgroup_max = subgroupMax(value);
    if workgroup_size <= subgroup_size {
        return subgroup_max;
    }

    if subgroup_invocation_id == 0u {
        reduce[subgroup_id] = subgroup_max;
    }

    workgroupBarrier();

    let subgroup_count = reduce_div_ceil_u32(workgroup_size, subgroup_size);

    var threads = subgroup_count;
    for (var stage = 0u; stage < REDUCE_MAX_STAGES; stage += 1u) {
        if threads > 1u {
            let half = threads / 2u;
            let rem = threads % 2u;
            let offset = half + rem;

            if subgroup_invocation_id == 0u && subgroup_id < half {
                let other = subgroup_id + offset;
                reduce[subgroup_id] = max(reduce[subgroup_id], reduce[other]);
            }

            threads = half + rem;

            workgroupBarrier();
        }
    }
    return reduce[0];
}
