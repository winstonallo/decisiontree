use std::collections::HashMap;

use node::Node;
use ordered_float::OrderedFloat;

mod node;

pub struct DecisionTree {
    min_samples_split: usize,
    max_depth: usize,
    n_features: Option<usize>,
    root: Option<Node>,
}

impl DecisionTree {
    pub fn new(min_samples_split: usize, max_depth: usize, n_features: Option<usize>) -> Self {
        Self {
            min_samples_split,
            max_depth,
            n_features,
            root: None,
        }
    }

    pub fn fit(&mut self, x: Vec<Vec<OrderedFloat<f64>>>, y: Vec<OrderedFloat<f64>>) {
        self.n_features = if self.n_features.is_none() {
            Some(x[0].len())
        } else {
            Some(x[0].len().min(self.n_features.unwrap()))
        };

        self.root = self.grow(x, y, 0);
    }

    fn grow(
        &mut self,
        x: Vec<Vec<OrderedFloat<f64>>>,
        y: Vec<OrderedFloat<f64>>,
        depth: usize,
    ) -> Option<Node> {
        let (n_samples, n_features) = (x.len(), x[0].len());
        let n_labels = {
            let mut y0 = y.clone();
            y0.sort();
            y0.dedup();
            y0.len()
        };

        if depth >= self.max_depth || n_labels == 1 || n_samples < self.min_samples_split {
            return Some(Node::new(None, None, None, None, Some(self.leaf_value(y))));
        }

        None
    }

    fn leaf_value(&self, y: Vec<OrderedFloat<f64>>) -> OrderedFloat<f64> {
        y.into_iter()
            .fold(HashMap::<OrderedFloat<f64>, usize>::new(), |mut m, x| {
                *m.entry(x).or_default() += 1;
                m
            })
            .into_iter()
            .max_by_key(|(_, v)| *v)
            .map(|(k, _)| k)
            .unwrap()
    }
}
