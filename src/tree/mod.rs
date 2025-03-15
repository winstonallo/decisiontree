use std::ops::Div;

use ordered_float::{FloatIsNan, OrderedFloat};

type Float64 = OrderedFloat<f64>;
type NodeIndex = usize;
type FeatureIndex = usize;
type Threshold = Float64;

struct Tree {
    root: NodeIndex,
    nodes: Vec<Node>,
    feature_names: Vec<String>,
}

enum Node {
    Leaf(Leaf),
    Branch(Branch),
}

struct Leaf {
    prediction: Float64,
    samples: usize,
}

impl Leaf {
    pub fn new(prediction: Float64, samples: usize) -> Self {
        Self { prediction, samples }
    }
}

struct Branch {
    feature: FeatureIndex,
    threshold: Float64,
    left: NodeIndex,
    right: NodeIndex,
    samples: usize,
    prediction: Float64,
}

impl Branch {
    pub fn new(
        feature: FeatureIndex,
        threshold: Float64,
        left: NodeIndex,
        right: NodeIndex,
        samples: usize,
        prediction: Float64,
    ) -> Self {
        Self {
            feature,
            threshold,
            left,
            right,
            samples,
            prediction,
        }
    }
}

struct Feature {
    name: String,
    features: Vec<Float64>,
}

impl Feature {
    pub fn new(name: &str, features: Vec<Float64>) -> Self {
        Self {
            name: name.to_string(),
            features,
        }
    }
}

fn gini_impurity(feature: &Feature, target: &Feature) -> Float64 {
    0.into()
}

fn get_best_feature(feature: &Feature, target: &Feature) -> Threshold {
    let mut combined = feature
        .features
        .clone()
        .into_iter()
        .zip(target.features.clone())
        .zip((0..feature.features.len()).collect::<Vec<FeatureIndex>>())
        .map(|((a, b), c)| (a, b, c))
        .collect::<Vec<(Float64, Float64, FeatureIndex)>>();
    combined.sort_by(|a, b| a.0.cmp(&b.0));

    let x: Vec<Float64> = combined.iter().map(|f| f.0).collect();
    let y: Vec<Float64> = combined.iter().map(|f| f.1).collect();
    let idx: Vec<FeatureIndex> = combined.iter().map(|f| f.2).collect();

    let mut lowest_impurity: Float64 = 0.into();
    for i in 0..combined.len() - 1 {
        let threshold: Threshold = (x[i] + x[i + 1]) / 2.0;
    }

    0.into()
}

impl Tree {
    pub fn new(feature_names: Vec<String>) -> Self {
        Self {
            root: 0,
            nodes: Vec::with_capacity(feature_names.len()),
            feature_names,
        }
    }

    fn push(&mut self, node: Node) -> NodeIndex {
        self.nodes.push(node);
        self.nodes.len() - 1
    }
    fn grow(&mut self, depth: usize) -> NodeIndex {
        if depth > 10 {
            let leaf = self.push(Node::Leaf(Leaf::new(1.into(), 2)));
            return leaf;
        }

        let left = self.grow(depth + 1);
        let right = self.grow(depth + 1);

        let name = &self.feature_names[depth % 3];
        let node = Node::Branch(Branch::new(1, 2.into(), left, right, 1, 2.into()));
        let node_index = self.push(node);

        node_index
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::sync::LazyLock;

    static X: LazyLock<Vec<Feature>> = LazyLock::new(|| {
        vec![
            Feature::new(
                "loves_popcorn",
                vec![1.into(), 1.into(), 0.into(), 0.into(), 1.into(), 1.into(), 0.into()],
            ),
            Feature::new(
                "loves_soda",
                vec![1.into(), 0.into(), 1.into(), 1.into(), 1.into(), 0.into(), 0.into()],
            ),
            Feature::new(
                "age",
                vec![
                    7.into(),
                    12.into(),
                    18.into(),
                    35.into(),
                    38.into(),
                    50.into(),
                    83.into(),
                ],
            ),
        ]
    });

    static Y: LazyLock<Feature> = LazyLock::new(|| {
        Feature::new(
            "loves_cool_as_ice",
            vec![0.into(), 0.into(), 1.into(), 1.into(), 1.into(), 0.into(), 0.into()],
        )
    });

    #[test]
    fn tree_new() {
        let tree = Tree::new(X.iter().map(|f| f.name.clone()).collect());
    }
}
