use std::collections::{HashMap, HashSet};

use ordered_float::{OrderedFloat, Pow};

type Float64 = OrderedFloat<f64>;
type NodeIndex = usize;
type FeatureIndex = usize;
type Threshold = Float64;
type ClassIndex = usize;

#[allow(unused)]
pub struct Tree {
    root: NodeIndex,
    nodes: Vec<Node>,
    feature_names: Vec<String>,
    class_values: Vec<Float64>,
}

enum Node {
    Leaf(Leaf),
    Branch(Branch),
}

struct Leaf {
    prediction: ClassIndex,
}

impl Leaf {
    pub fn new(prediction: ClassIndex) -> Self {
        Self { prediction }
    }
}

#[allow(unused)]
struct Branch {
    feature: FeatureIndex,
    threshold: Float64,
    left: NodeIndex,
    right: NodeIndex,
    n_samples: usize,
    prediction: ClassIndex,
    class_distribution: Vec<usize>,
}

impl Branch {
    pub fn new(
        feature: FeatureIndex,
        threshold: Float64,
        left: NodeIndex,
        right: NodeIndex,
        n_samples: usize,
        prediction: ClassIndex,
        class_distribution: Vec<usize>,
    ) -> Self {
        Self {
            feature,
            threshold,
            left,
            right,
            n_samples,
            prediction,
            class_distribution,
        }
    }
}

/// Target feature
pub struct Y {
    datapoints: Vec<ClassIndex>,
    n_classes: usize,
    class_values: Vec<Float64>,
}

impl Y {
    pub fn new(class_values: Vec<Float64>) -> Self {
        let mut unique_values: Vec<Float64> = class_values
            .iter()
            .cloned()
            .collect::<HashSet<Float64>>()
            .into_iter()
            .collect();

        unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let value_to_index: HashMap<Float64, ClassIndex> = unique_values
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();

        let datapoints = class_values
            .iter()
            .map(|&val| *value_to_index.get(&val).unwrap())
            .collect();

        let n_classes = unique_values.len();

        Self {
            datapoints,
            n_classes,
            class_values: unique_values,
        }
    }

    pub fn datapoints(&self) -> &Vec<ClassIndex> {
        &self.datapoints
    }

    pub fn class_counts(&self, indices: &[usize]) -> Vec<usize> {
        let mut counts = vec![0; self.n_classes];
        for &idx in indices {
            counts[self.datapoints[idx]] += 1;
        }
        counts
    }

    pub fn majority_class(&self, indices: &[usize]) -> ClassIndex {
        self.class_counts(indices)
            .iter()
            .enumerate()
            .max_by_key(|&(_, count)| count)
            .map(|(class_idx, _)| class_idx)
            .unwrap_or(0)
    }
}

fn gini_impurity(counts: &[usize], total: usize) -> Float64 {
    if total == 0 {
        return Float64::from(0);
    }

    let impurity = 1.0
        - counts
            .iter()
            .map(|&count| (count as f64 / total as f64).pow(2))
            .sum::<f64>();

    Float64::from(impurity)
}

fn weighted_gini_impurity(left_counts: &[usize], right_counts: &[usize]) -> Float64 {
    let left_total: usize = left_counts.iter().sum();
    let right_total: usize = right_counts.iter().sum();
    let total = left_total + right_total;

    if total == 0 {
        return 0.into();
    }

    let left_weight = left_total as f64 / total as f64;
    let right_weight = right_total as f64 / total as f64;

    let left_impurity = gini_impurity(left_counts, left_total);
    let right_impurity = gini_impurity(right_counts, right_total);

    (left_weight * left_impurity.into_inner() + right_weight * right_impurity.into_inner()).into()
}

fn get_best_threshold(feature: &Vec<Float64>, target: &Y) -> (Threshold, Float64) {
    let mut combined = feature
        .iter()
        .enumerate()
        .collect::<Vec<(usize, &Float64)>>();

    combined.sort_by(|a, b| a.1.cmp(&b.1));

    let n_samples = combined.len();
    let all_indices: Vec<usize> = (0..n_samples).collect();
    let all_counts = target.class_counts(&all_indices);
    let all_gini = gini_impurity(&all_counts, n_samples);

    let mut best_threshold = Float64::from(0.0);
    let mut best_gain = Float64::from(0.0);

    for i in 0..n_samples - 1 {
        let threshold = (combined[i].1 + combined[i + 1].1) / 2.0;

        let left_indices: Vec<usize> = combined.iter().take(i + 1).map(|(i, _)| *i).collect();
        let right_indices: Vec<usize> = combined.iter().skip(i + 1).map(|(i, _)| *i).collect();

        let left_counts = target.class_counts(&left_indices);
        let right_counts = target.class_counts(&right_indices);

        let weighted_gini = weighted_gini_impurity(&left_counts, &right_counts);
        let gain = all_gini - weighted_gini;

        if gain > best_gain {
            best_gain = gain;
            best_threshold = threshold;
        }
    }

    (best_threshold, best_gain)
}

impl Tree {
    pub fn new(feature_names: Vec<String>) -> Self {
        Self {
            root: 0,
            nodes: Vec::with_capacity(feature_names.len()),
            feature_names,
            class_values: Vec::new(),
        }
    }

    fn push(&mut self, node: Node) -> NodeIndex {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    pub fn fit(&mut self, features: &[Vec<Float64>], target: &Y, max_depth: usize) {
        let sample_indices: Vec<usize> = (0..features[0].len()).collect();

        self.class_values = target.class_values.clone();

        self.root = self.grow(&features, target, &sample_indices, 0, max_depth)
    }

    fn grow(
        &mut self,
        features: &[Vec<Float64>],
        target: &Y,
        indices: &[usize],
        depth: usize,
        max_depth: usize,
    ) -> NodeIndex {
        let n_samples = indices.len();

        let class_counts = target.class_counts(indices);
        let majority_class = target.majority_class(indices);

        if depth >= max_depth
            || class_counts.iter().filter(|&&count| count > 0).count() <= 1
            || n_samples < 2
        {
            let leaf = Node::Leaf(Leaf::new(majority_class));
            return self.push(leaf);
        }

        let mut best_feature_idx = 0;
        let mut best_threshold = Float64::from(0);
        let mut best_gain = Float64::from(0);

        for (feature_idx, feature) in features.iter().enumerate() {
            let (threshold, gain) = get_best_threshold(feature, target);

            if gain > best_gain {
                best_gain = gain;
                best_threshold = threshold;
                best_feature_idx = feature_idx;
            }
        }

        if best_gain <= Float64::from(0) {
            let leaf = Node::Leaf(Leaf::new(majority_class));
            return self.push(leaf);
        }

        let best_feature = &features[best_feature_idx];
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for &idx in indices {
            if best_feature[idx] < best_threshold {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        if left_indices.is_empty() || right_indices.is_empty() {
            let leaf = Node::Leaf(Leaf::new(majority_class));
            return self.push(leaf);
        }

        let left = self.grow(features, target, &left_indices, depth + 1, max_depth);
        let right = self.grow(features, target, &right_indices, depth + 1, max_depth);

        let branch = Node::Branch(Branch::new(
            best_feature_idx,
            best_threshold,
            left,
            right,
            n_samples,
            majority_class,
            class_counts,
        ));

        self.push(branch)
    }

    pub fn predict_one(&self, sample: &[Float64]) -> ClassIndex {
        let mut node_idx = self.root;

        loop {
            match &self.nodes[node_idx] {
                Node::Leaf(leaf) => return leaf.prediction,
                Node::Branch(branch) => {
                    let feature_value = sample[branch.feature];
                    if feature_value < branch.threshold {
                        node_idx = branch.left;
                    } else {
                        node_idx = branch.right;
                    }
                }
            }
        }
    }

    pub fn predict(&self, samples: &[Vec<Float64>]) -> Vec<ClassIndex> {
        samples
            .iter()
            .map(|sample| self.predict_one(sample))
            .collect()
    }

    pub fn class_index_to_value(&self, index: ClassIndex) -> Float64 {
        self.class_values[index]
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(&[3, 3], Float64::from(0.5))]
    #[case(&[1, 0], Float64::from(0.0))]
    fn gini_impurity_output(#[case] counts: &[usize], #[case] expected: Float64) {
        let left_total: usize = counts.iter().sum();

        let impurity = gini_impurity(counts, left_total);

        assert_eq!(impurity, expected);
    }

    #[rstest]
    #[case(&[3, 3], &[1, 0], Float64::from(0.4285))]
    fn weighted_gini_impurity_output(
        #[case] left_counts: &[usize],
        #[case] right_counts: &[usize],
        #[case] expected: Float64,
    ) {
        let weighted_impurity = weighted_gini_impurity(left_counts, right_counts);

        let epsilon = 1e-4;
        assert!(
            (weighted_impurity.into_inner() - expected.into_inner()).abs() < epsilon,
            "result: {:?}, expected: {:?}",
            weighted_impurity,
            expected
        );
    }
}
