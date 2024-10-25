//! This module contains Huffman prefix-codes compression for MOK format.

use std::{cmp::Reverse, collections::BinaryHeap, hash::Hash, ops::Range};

use bitvec::vec::BitVec;
use hashbrown::HashMap;

#[derive(Clone, Copy, Debug)]
enum TreeNode<T> {
    Leaf {
        frequency: usize,
        value: T,
    },
    InternalNode {
        frequency: usize,
        left_idx: usize,
        right_idx: usize,
    },
}

impl<T> TreeNode<T> {
    fn frequency(&self) -> usize {
        match *self {
            TreeNode::Leaf { frequency, .. } => frequency,
            TreeNode::InternalNode { frequency, .. } => frequency,
        }
    }
}

struct Tree<T> {
    nodes: Vec<TreeNode<T>>,
    root_idx: usize,
}

// # Panics if iterator is empty.
fn build_tree<T>(values: impl Iterator<Item = T>) -> Tree<T>
where
    T: Hash + Ord + Copy,
{
    let mut map = HashMap::new();
    let mut nodes = Vec::new();
    let mut heap = BinaryHeap::new();

    // Build frequency tree
    values.for_each(|value| {
        *map.entry(value).or_insert(0usize) += 1;
    });

    assert!(!map.is_empty());

    // Build nodes array.
    map.into_iter().for_each(|(value, frequency)| {
        // Reverse is needed because BinaryHeap is max-heap.
        nodes.push(TreeNode::Leaf { frequency, value });
    });

    // Insert all unique values into heap
    for (idx, node) in nodes.iter().enumerate() {
        // Reverse frequency because BinaryHeap is max-heap
        // and we need min-heap.
        heap.push((Reverse(node.frequency()), idx));
    }

    while heap.len() > 1 {
        let (Reverse(left_frequency), left_idx) = heap.pop().unwrap();
        let (Reverse(right_frequency), right_idx) = heap.pop().unwrap();

        let node = TreeNode::InternalNode {
            frequency: left_frequency + right_frequency,
            left_idx,
            right_idx,
        };

        let idx = nodes.len();
        nodes.push(node);

        heap.push((Reverse(node.frequency()), idx));
    }

    assert!(heap.len() == 1);

    let (Reverse(_), root_idx) = heap.pop().unwrap();

    Tree { nodes, root_idx }
}

struct Codewords<T> {
    bits: BitVec,
    ranges: HashMap<T, Range<usize>>,
}

fn visit_node<T>(
    node: usize,
    nodes: &[TreeNode<T>],
    codeword: &mut BitVec,
    bits: &mut BitVec,
    ranges: &mut HashMap<T, Range<usize>>,
) where
    T: Copy + Hash + Eq,
{
    match nodes[node] {
        TreeNode::Leaf { value, .. } => {
            ranges.insert(value, bits.len()..codeword.len());
            bits.extend_from_bitslice(&codeword);
        }
        TreeNode::InternalNode {
            left_idx,
            right_idx,
            ..
        } => {
            bits.push(false);
            visit_node(left_idx, nodes, codeword, bits, ranges);
            bits.pop();
            bits.push(true);
            visit_node(right_idx, nodes, codeword, bits, ranges);
            bits.pop();
        }
    }
}

fn build_codewords<T>(tree: &Tree<T>) -> Codewords<T>
where
    T: Copy + Hash + Eq,
{
    let mut codeword = Bits::new();
    let mut bits = Bits::new();
    let mut ranges = HashMap::new();

    visit_node(
        tree.root_idx,
        &tree.nodes,
        &mut codeword,
        &mut bits,
        &mut ranges,
    );

    Codewords { bits, ranges }
}
