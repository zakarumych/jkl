//! Neural network for color prediction.

use std::io::{Read, Write};

const INPUT_SIZE: usize = 45;
const HIDDEN_SIZE: usize = 45;
const HIDDEN_LAYERS: usize = 0;
const OUTPUT_SIZE: usize = 3;

#[derive(Clone, Copy)]
struct Layer<const IN: usize, const OUT: usize> {
    weights: [[f32; IN]; OUT],
    biases: [f32; OUT],
}

impl<const IN: usize, const OUT: usize> Layer<IN, OUT> {
    const BYTES_SIZE: usize = OUT * (IN + 1) * size_of::<f32>();

    fn zero() -> Self {
        Layer {
            weights: [[0.0; IN]; OUT],
            biases: [0.0; OUT],
        }
    }

    fn from_rng(rng: &mut impl rand::Rng) -> Self {
        Layer {
            weights: [[0.0; IN]; OUT]
                .map(|row| row.map(|_| rng.gen_range(-1.0..=1.0) / (IN + 1) as f32)),
            biases: [0.0; OUT].map(|_| rng.gen_range(-1.0..=1.0) / (IN + 1) as f32),
        }
    }

    fn write_to(&self, mut write: impl Write) -> std::io::Result<()> {
        for row in self.weights.iter() {
            for w in row {
                write.write_all(&w.to_bits().to_le_bytes())?;
            }
        }
        for b in self.biases.iter() {
            write.write_all(&b.to_bits().to_le_bytes())?;
        }
        Ok(())
    }

    fn read_from(mut read: impl Read) -> std::io::Result<Self> {
        let mut weights = [[0.0; IN]; OUT];
        for row in weights.iter_mut() {
            for w in row.iter_mut() {
                let mut bytes = [0; 4];
                read.read_exact(&mut bytes)?;
                *w = f32::from_bits(u32::from_le_bytes(bytes));
            }
        }
        let mut biases = [0.0; OUT];
        for b in biases.iter_mut() {
            let mut bytes = [0; 4];
            read.read_exact(&mut bytes)?;
            *b = f32::from_bits(u32::from_le_bytes(bytes));
        }
        Ok(Layer { weights, biases })
    }
}

#[derive(Clone, Copy)]
pub struct Signals {
    input: [f32; INPUT_SIZE],
    hidden: [[f32; HIDDEN_SIZE]; HIDDEN_LAYERS],
    output: [f32; OUTPUT_SIZE],
}

impl Signals {
    pub fn output(&self) -> [f32; OUTPUT_SIZE] {
        self.output
    }
}

#[derive(Clone, Copy)]
pub struct Diffs {
    hidden_diff: [[[f32; HIDDEN_SIZE]; HIDDEN_SIZE]; HIDDEN_LAYERS],
    hidden_bias_diff: [[f32; HIDDEN_SIZE]; HIDDEN_LAYERS],
    output_diff: [[f32; HIDDEN_SIZE]; OUTPUT_SIZE],
    output_bias_diff: [f32; OUTPUT_SIZE],
}

impl Diffs {
    pub fn new() -> Self {
        Diffs {
            hidden_diff: [[[0.0; HIDDEN_SIZE]; HIDDEN_SIZE]; HIDDEN_LAYERS],
            hidden_bias_diff: [[0.0; HIDDEN_SIZE]; HIDDEN_LAYERS],
            output_diff: [[0.0; HIDDEN_SIZE]; OUTPUT_SIZE],
            output_bias_diff: [0.0; OUTPUT_SIZE],
        }
    }
}

#[derive(Clone, Copy)]
pub struct Model {
    hidden: [Layer<HIDDEN_SIZE, HIDDEN_SIZE>; HIDDEN_LAYERS],
    output: Layer<HIDDEN_SIZE, OUTPUT_SIZE>,
}

impl Model {
    pub const BYTES_SIZE: usize = Layer::<HIDDEN_SIZE, HIDDEN_SIZE>::BYTES_SIZE * HIDDEN_LAYERS
        + Layer::<HIDDEN_SIZE, OUTPUT_SIZE>::BYTES_SIZE;

    pub fn new() -> Self {
        let mut rng =
            <rand_chacha::ChaCha8Rng as rand::SeedableRng>::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);

        Model {
            hidden: [(); HIDDEN_LAYERS].map(|_| Layer::from_rng(&mut rng)),
            output: Layer::from_rng(&mut rng),
        }
    }

    pub fn forward(&self, input_signal: [f32; INPUT_SIZE]) -> Signals {
        // eprintln!("IN: {:?}", input_signal);

        let mut hidden_signal = [[0.0; HIDDEN_SIZE]; HIDDEN_LAYERS];
        let mut output_signal = [0.0; OUTPUT_SIZE];

        // Forward

        if HIDDEN_LAYERS > 0 {
            for n in 0..HIDDEN_SIZE {
                for w in 0..INPUT_SIZE {
                    hidden_signal[0][n] += input_signal[w] * self.hidden[0].weights[n][w];
                }
                hidden_signal[0][n] += self.hidden[0].biases[n];
                hidden_signal[0][n] = a(hidden_signal[0][n]);
            }

            for l in 1..HIDDEN_LAYERS {
                for n in 0..HIDDEN_SIZE {
                    for w in 0..HIDDEN_SIZE {
                        hidden_signal[l][n] +=
                            hidden_signal[l - 1][w] * self.hidden[l].weights[n][w];
                    }
                    hidden_signal[l][n] += self.hidden[l].biases[n];
                    hidden_signal[l][n] = a(hidden_signal[l][n]);
                }
            }
            for n in 0..OUTPUT_SIZE {
                for w in 0..HIDDEN_SIZE {
                    output_signal[n] +=
                        hidden_signal[HIDDEN_LAYERS - 1][w] * self.output.weights[n][w];
                }
                output_signal[n] += self.output.biases[n];
            }
        } else {
            for n in 0..OUTPUT_SIZE {
                for w in 0..INPUT_SIZE {
                    output_signal[n] += input_signal[w] * self.output.weights[n][w];
                }
                output_signal[n] += self.output.biases[n];
            }
        }

        // eprintln!("OUT: {}", output_signal);

        Signals {
            input: input_signal,
            hidden: hidden_signal,
            output: output_signal,
        }
    }

    pub fn backward(&mut self, signal: Signals, expected: [f32; OUTPUT_SIZE], diffs: &mut Diffs) {
        // eprintln!("EXP: {}", expected);

        let mut hidden_error = [[0.0; HIDDEN_SIZE]; HIDDEN_LAYERS];
        let mut output_error = [0.0; OUTPUT_SIZE];

        // Backward

        for n in 0..OUTPUT_SIZE {
            output_error[n] = signal.output[n] - expected[n];
        }

        if HIDDEN_LAYERS > 0 {
            for n in 0..OUTPUT_SIZE {
                diffs.output_bias_diff[n] += output_error[n];
                for w in 0..HIDDEN_SIZE {
                    diffs.output_diff[n][w] +=
                        output_error[n] * signal.hidden[HIDDEN_LAYERS - 1][w];

                    hidden_error[HIDDEN_LAYERS - 1][w] +=
                        self.output.weights[n][w] * output_error[n];
                    hidden_error[HIDDEN_LAYERS - 1][w] = b(
                        hidden_error[HIDDEN_LAYERS - 1][w],
                        signal.hidden[HIDDEN_LAYERS - 1][w],
                    );
                }
            }

            for l in (1..HIDDEN_LAYERS).rev() {
                for n in 0..HIDDEN_SIZE {
                    diffs.hidden_bias_diff[l][n] += hidden_error[l][n];

                    for w in 0..HIDDEN_SIZE {
                        diffs.hidden_diff[l][n][w] += hidden_error[l][n] * signal.hidden[l - 1][w];

                        hidden_error[l - 1][w] += self.hidden[l].weights[n][w] * hidden_error[l][n];
                        hidden_error[l - 1][w] = b(hidden_error[l - 1][w], signal.hidden[l - 1][w]);
                    }
                }
            }

            for n in 0..HIDDEN_SIZE {
                diffs.hidden_bias_diff[0][n] += hidden_error[0][n];
                for w in 0..INPUT_SIZE {
                    diffs.hidden_diff[0][n][w] += hidden_error[0][n] * signal.input[w];
                }
            }
        } else {
            for n in 0..OUTPUT_SIZE {
                diffs.output_bias_diff[n] += output_error[n];
                for w in 0..INPUT_SIZE {
                    diffs.output_diff[n][w] += output_error[n] * signal.input[w];
                }
            }
        }
    }

    pub fn optimize(&mut self, diffs: Diffs, size: usize) {
        // Optimize

        let learning_rate = 0.001 * (1.0 / size as f32);

        for l in 0..HIDDEN_LAYERS {
            for n in 0..HIDDEN_SIZE {
                for w in 0..INPUT_SIZE {
                    self.hidden[l].weights[n][w] -= diffs.hidden_diff[l][n][w] * learning_rate;
                }
                self.hidden[l].biases[n] -= diffs.hidden_bias_diff[l][n] * learning_rate;
            }
        }

        for n in 0..OUTPUT_SIZE {
            for w in 0..HIDDEN_SIZE {
                self.output.weights[n][w] -= diffs.output_diff[n][w] * learning_rate;
            }
            self.output.biases[n] -= diffs.output_bias_diff[n] * learning_rate;
        }
    }

    pub fn write_to(&self, mut write: impl Write) -> std::io::Result<()> {
        for layer in self.hidden.iter() {
            layer.write_to(&mut write)?;
        }
        self.output.write_to(&mut write)
    }

    pub fn read_from(mut read: impl Read) -> std::io::Result<Self> {
        let mut hidden = [Layer::zero(); HIDDEN_LAYERS];
        for layer in hidden.iter_mut() {
            *layer = Layer::read_from(&mut read)?;
        }
        let output = Layer::read_from(&mut read)?;
        Ok(Model { hidden, output })
    }
}

struct MT {
    hidden: [Layer<HIDDEN_SIZE, HIDDEN_SIZE>; HIDDEN_LAYERS],
    output: Layer<HIDDEN_SIZE, OUTPUT_SIZE>,
}

struct VT {
    hidden: [Layer<HIDDEN_SIZE, HIDDEN_SIZE>; HIDDEN_LAYERS],
    output: Layer<HIDDEN_SIZE, OUTPUT_SIZE>,
}

pub struct Adam {
    mt: MT,
    vt: VT,
    b1t: f32,
    b2t: f32,
}

impl Adam {
    pub fn new() -> Self {
        Adam {
            mt: MT {
                hidden: [0.0; HIDDEN_LAYERS].map(|_| Layer::zero()),
                output: Layer::zero(),
            },
            vt: VT {
                hidden: [0.0; HIDDEN_LAYERS].map(|_| Layer::zero()),
                output: Layer::zero(),
            },
            b1t: 0.9,
            b2t: 0.999,
        }
    }

    pub fn optimize(&mut self, model: &mut Model, diffs: &Diffs, size: usize) {
        // Optimize

        self.b1t *= 0.9;
        self.b2t *= 0.999;

        let learning_rate = 0.00001;

        for l in 0..HIDDEN_LAYERS {
            for n in 0..HIDDEN_SIZE {
                for w in 0..INPUT_SIZE {
                    let diff = diffs.hidden_diff[l][n][w] / size as f32;
                    let mt = self.mt.hidden[l].weights[n][w] * 0.9 + diff * 0.1;
                    let vt = self.vt.hidden[l].weights[n][w] * 0.999 + diff * diff * 0.001;

                    let corr = (mt / (1.0 - self.b1t)) / ((vt / (1.0 - self.b2t)).sqrt() + 1e-8);

                    self.mt.hidden[l].weights[n][w] = mt;
                    self.vt.hidden[l].weights[n][w] = vt;

                    model.hidden[l].weights[n][w] -= corr * learning_rate;
                }

                let diff = diffs.hidden_bias_diff[l][n] / size as f32;
                let mt = self.mt.hidden[l].biases[n] * 0.9 + diff * 0.1;
                let vt = self.vt.hidden[l].biases[n] * 0.999 + diff * diff * 0.001;

                let corr = (mt / (1.0 - self.b1t)) / (vt / (1.0 - self.b2t).sqrt() + 1e-8);

                self.mt.hidden[l].biases[n] = mt;
                self.vt.hidden[l].biases[n] = vt;

                model.hidden[l].biases[n] -= corr * learning_rate;
            }
        }

        for n in 0..OUTPUT_SIZE {
            for w in 0..HIDDEN_SIZE {
                let diff = diffs.output_diff[n][w] / size as f32;
                let mt = self.mt.output.weights[n][w] * 0.9 + diff * 0.1;
                let vt = self.vt.output.weights[n][w] * 0.999 + diff * diff * 0.001;

                let corr = (mt / (1.0 - self.b1t)) / (vt / (1.0 - self.b2t).sqrt() + 1e-8);

                self.mt.output.weights[n][w] = mt;
                self.vt.output.weights[n][w] = vt;

                model.output.weights[n][w] -= corr * learning_rate;
            }

            let diff = diffs.output_bias_diff[n] / size as f32;
            let mt = self.mt.output.biases[n] * 0.9 + diff * 0.1;
            let vt = self.vt.output.biases[n] * 0.999 + diff * diff * 0.001;

            let corr = (mt / (1.0 - self.b1t)) / (vt / (1.0 - self.b2t).sqrt() + 1e-8);

            self.mt.output.biases[n] = mt;
            self.vt.output.biases[n] = vt;

            model.output.biases[n] -= corr * learning_rate;
        }
    }
}

fn leaky_relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        x * 0.01
    }
}

fn leaky_relu_derivative(s: f32) -> f32 {
    if s > 0.0 {
        1.0
    } else {
        0.01
    }
}

fn sigmoid(x: f32) -> f32 {
    let y = x.exp();
    y / (1.0 + y)
}

fn sigmoid_derivative(x: f32) -> f32 {
    let y = sigmoid(x);
    y * (1.0 - y)
}

fn a(x: f32) -> f32 {
    sigmoid(x)
}

fn b(x: f32, s: f32) -> f32 {
    x * sigmoid_derivative(s)
}

// #[test]
// fn test() {
//     let mut model = Model::new();
//     let mut rng = rand::thread_rng();

//     let f = |x: f32| x * x + 2.0 * x;

//     let mut ew = 1.0;

//     for i in 0..100 {
//         let input = rng.gen_range(0.0..10.0);
//         let expected = f(input);
//         let predicted = model.run([input], expected);
//         let error = (expected - predicted) / expected;
//         ew = 0.9 * ew + 0.1 * error;

//         eprintln!(
//             "{:05} @ i: {:05.5} x: {:05.5} p: {:05.5}, e: {:05.5}: ew: {:05.5}",
//             i, input, expected, predicted, error, ew
//         );
//     }

//     model.run([0.0], 0.0);
// }
