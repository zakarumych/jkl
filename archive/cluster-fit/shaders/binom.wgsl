const BINOM_MAX_N: u32 = 16u;
const BINOM_MAX_K: u32 = 8u;

const BINOM_COEFFICIENTS = array<array<u32, 8>, 16>(
    array<u32, 8>(1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u),
    array<u32, 8>(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u),
    array<u32, 8>(1u, 3u, 6u, 10u, 15u, 21u, 28u, 36u),
    array<u32, 8>(1u, 4u, 10u, 20u, 35u, 56u, 84u, 120u),
    array<u32, 8>(1u, 5u, 15u, 35u, 70u, 126u, 210u, 330u),
    array<u32, 8>(1u, 6u, 21u, 56u, 126u, 252u, 462u, 792u),
    array<u32, 8>(1u, 7u, 28u, 84u, 210u, 462u, 924u, 1716u),
    array<u32, 8>(1u, 8u, 36u, 120u, 330u, 792u, 1716u, 3432u),
    array<u32, 8>(1u, 9u, 45u, 165u, 495u, 1287u, 3003u, 6435u),
    array<u32, 8>(1u, 10u, 55u, 220u, 715u, 2002u, 5005u, 11440u),
    array<u32, 8>(1u, 11u, 66u, 286u, 1001u, 3003u, 8008u, 19448u),
    array<u32, 8>(1u, 12u, 78u, 364u, 1365u, 4368u, 12376u, 31824u),
    array<u32, 8>(1u, 13u, 91u, 455u, 1820u, 6188u, 18564u, 50388u),
    array<u32, 8>(1u, 14u, 105u, 560u, 2380u, 8568u, 27132u, 77520u),
    array<u32, 8>(1u, 15u, 120u, 680u, 3060u, 11628u, 38760u, 116280u),
    array<u32, 8>(1u, 16u, 136u, 816u, 3876u, 15504u, 54264u, 170544u)
);

struct BinomIterState {
    n: u32,
    k: u32,
    indices: array<u32, 8>,
}

fn binom_iter_init(n: u32, k: u32) -> BinomIterState {
    var state = BinomIterState();
    state.n = n;
    state.k = k;
    state.indices = array<u32, 8>();
    return state;
}

fn binom_iter_total_iterations(n: u32, k: u32) -> u32 {
    return BINOM_COEFFICIENTS[n - 1u][k - 1u];
}

fn binom_iter_suffix_count(value_count: u32, pick_count: u32) -> u32 {
    if pick_count == 0u {
        return 1u;
    }

    return BINOM_COEFFICIENTS[value_count - 1u][pick_count - 1u];
}

fn binom_iter_init_at(n: u32, k: u32, x: u32) -> BinomIterState {
    var state = binom_iter_init(n, k);
    let total = binom_iter_total_iterations(state.n, state.k);
    var rank = min(x, total - 1u);
    var prev = 0u;

    for (var pos = 0u; pos < state.k; pos += 1u) {
        let rem = state.k - pos - 1u;
        var value = prev;

        loop {
            let suffix_value_count = state.n - value;
            let count = binom_iter_suffix_count(suffix_value_count, rem);
            if rank < count {
                state.indices[pos] = value;
                prev = value;
                break;
            }

            rank -= count;
            value += 1u;
        }
    }

    return state;
}

fn binom_iter_next(state: ptr<function, BinomIterState>) -> u32 {
    let n = (*state).n;
    let k = (*state).k;
    let max_index = n - 1u;

    for (var i = k; i > 0u; i -= 1u) {
        let pos = i - 1u;
        if (*state).indices[pos] < max_index {
            let next_value = (*state).indices[pos] + 1u;
            (*state).indices[pos] = next_value;

            for (var j = pos + 1u; j < k; j += 1u) {
                (*state).indices[j] = next_value;
            }

            return 1u;
        }
    }

    return 0u;
}
