//! Neural network for color prediction.

use rand::Rng;
use rand_chacha::rand_core::le;

const INPUT_SIZE: usize = 8;
const HIDDEN_SIZE: usize = 8;

struct Layer<const IN: usize, const OUT: usize> {
    weights: [[f32; IN]; OUT],
    biases: [f32; OUT],
}

impl<const IN: usize, const OUT: usize> Layer<IN, OUT> {
    fn from_rng(rng: &mut impl rand::Rng) -> Self {
        Layer {
            weights: [[0.0; IN]; OUT].map(|row| row.map(|_| 1.0 / (IN + 1) as f32)),
            biases: [0.0; OUT].map(|_| 1.0 / (IN + 1) as f32),
        }
    }
}

pub(crate) struct Model {
    hidden: Layer<INPUT_SIZE, HIDDEN_SIZE>,
    output: Layer<HIDDEN_SIZE, 1>,
}

pub(crate) struct Signals {
    input: [f32; INPUT_SIZE],
    hidden: [f32; HIDDEN_SIZE],
    output: f32,
}

impl Signals {
    pub fn output(&self) -> f32 {
        self.output
    }
}

impl Model {
    pub fn new() -> Self {
        let mut rng =
            <rand_chacha::ChaCha8Rng as rand::SeedableRng>::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);

        Model {
            hidden: Layer::from_rng(&mut rng),
            output: Layer::from_rng(&mut rng),
        }
    }

    pub fn forward(&mut self, input_signal: [f32; INPUT_SIZE]) -> Signals {
        // eprintln!("IN: {:?}", input_signal);

        let mut hidden_signal = [0.0; HIDDEN_SIZE];
        let mut output_signal = 0.0;

        // Forward

        for n in 0..HIDDEN_SIZE {
            for w in 0..INPUT_SIZE {
                hidden_signal[n] += input_signal[w] * self.hidden.weights[n][w];
            }
            hidden_signal[n] += self.hidden.biases[n];
            hidden_signal[n] = a(hidden_signal[n]);
        }

        for w in 0..HIDDEN_SIZE {
            output_signal += hidden_signal[w] * self.output.weights[0][w];
        }
        output_signal += self.output.biases[0];

        // eprintln!("OUT: {}", output_signal);

        Signals {
            input: input_signal,
            hidden: hidden_signal,
            output: output_signal,
        }
    }

    pub fn backward(&mut self, signals: Signals, expected: f32) {
        // eprintln!("EXP: {}", expected);

        let mut hidden_error = [0.0; HIDDEN_SIZE];
        let mut hidden_diff = [[0.0; INPUT_SIZE]; HIDDEN_SIZE];
        let mut hidden_bias_diff = [0.0; HIDDEN_SIZE];

        let output_error;
        let mut output_diff = [0.0; HIDDEN_SIZE];
        let output_bias_diff;

        // Backward

        output_error = signals.output - expected;

        output_bias_diff = output_error;
        for w in 0..HIDDEN_SIZE {
            output_diff[w] = output_error * signals.hidden[w];

            hidden_error[w] += self.output.weights[0][w] * output_error;
            hidden_error[w] = b(hidden_error[w], signals.hidden[w]);
        }

        for n in 0..HIDDEN_SIZE {
            hidden_bias_diff[n] = hidden_error[n];
            for w in 0..INPUT_SIZE {
                hidden_diff[n][w] = hidden_error[n] * signals.input[w];
            }
        }

        // Optimize

        let learning_rate = 0.1;

        for n in 0..HIDDEN_SIZE {
            for w in 0..INPUT_SIZE {
                self.hidden.weights[n][w] -= hidden_diff[n][w] * learning_rate;
            }
            self.hidden.biases[n] -= hidden_bias_diff[n] * learning_rate;
        }

        for w in 0..HIDDEN_SIZE {
            self.output.weights[0][w] -= output_diff[w] * learning_rate;
        }
        self.output.biases[0] -= output_bias_diff * learning_rate;
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
