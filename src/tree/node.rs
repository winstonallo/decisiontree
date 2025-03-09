use ordered_float::OrderedFloat;

pub struct Node {
    feature: Option<OrderedFloat<f64>>,
    threshold: Option<OrderedFloat<f64>>,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    value: Option<OrderedFloat<f64>>,
}

impl Node {
    pub fn new(
        feature: Option<OrderedFloat<f64>>,
        threshold: Option<OrderedFloat<f64>>,
        left: Option<Node>,
        right: Option<Node>,
        value: Option<OrderedFloat<f64>>,
    ) -> Self {
        Self {
            feature,
            threshold,
            left: left.map(Box::new),
            right: right.map(Box::new),
            value,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.value.is_some()
    }
}
