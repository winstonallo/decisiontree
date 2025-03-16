use std::vec;

use dct::tree::Y;
use ordered_float::OrderedFloat;

type Float64 = OrderedFloat<f64>;

fn main() {
    let x: Vec<Vec<Float64>> = vec![
        vec![
            1.into(),
            1.into(),
            0.into(),
            0.into(),
            1.into(),
            1.into(),
            0.into(),
        ],
        vec![
            1.into(),
            0.into(),
            1.into(),
            1.into(),
            1.into(),
            0.into(),
            0.into(),
        ],
        vec![
            7.into(),
            12.into(),
            18.into(),
            35.into(),
            38.into(),
            50.into(),
            83.into(),
        ],
    ];

    let y: Y = Y::new(vec![
        0.into(),
        0.into(),
        1.into(),
        1.into(),
        1.into(),
        0.into(),
        0.into(),
    ]);

    let mut tree = dct::tree::Tree::new(vec![
        "loves_popcorn".to_string(),
        "loves_soda".to_string(),
        "age".to_string(),
    ]);

    tree.fit(&x, &y, 2);

    let samples: Vec<Vec<Float64>> = vec![
        vec![1.into(), 1.into(), 7.into()],
        vec![1.into(), 0.into(), 12.into()],
        vec![0.into(), 1.into(), 18.into()],
        vec![0.into(), 1.into(), 35.into()],
        vec![1.into(), 1.into(), 38.into()],
        vec![1.into(), 0.into(), 50.into()],
        vec![0.into(), 0.into(), 83.into()],
    ];

    let y_pred = tree.predict(&samples);

    for (idx, pred) in y_pred.iter().enumerate() {
        println!(
            "predicted: {}, expected: {}",
            tree.class_index_to_value(*pred),
            tree.class_index_to_value(y.datapoints()[idx])
        )
    }
}
