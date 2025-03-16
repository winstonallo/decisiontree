use std::{error::Error, fs::File};

use dct::tree::{Tree, Y};
use ordered_float::OrderedFloat;
use polars::prelude::*;

fn dataframe_to_samples(df: &DataFrame) -> Result<Vec<Vec<OrderedFloat<f64>>>, Box<dyn Error>> {
    let height = df.height();
    let mut samples = Vec::with_capacity(height);

    // iterate over each row index
    for i in 0..height {
        let mut sample = Vec::with_capacity(df.width());
        // iterate over each column
        for col in df.get_columns() {
            // assuming the column type is f64
            let value = col.f64()?.get(i).ok_or("Failed to get value")?;
            sample.push(OrderedFloat::from(value));
        }
        samples.push(sample);
    }
    Ok(samples)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut df = CsvReader::new(File::open("breast-cancer.csv")?).finish()?;

    let y = Y::new(
        df.column("diagnosis")?
            .str()?
            .iter()
            .map(|s| if s.unwrap() == "M" { 1 } else { 0 })
            .into_iter()
            .map(|f| OrderedFloat::from(f))
            .collect::<Vec<OrderedFloat<f64>>>(),
    );

    df.drop_in_place("id")?;
    df.drop_in_place("diagnosis")?;

    let features = dataframe_to_samples(&df)?;

    let mut tree = Tree::new(vec![
        "radius_mean".to_string(),
        "texture_mean".to_string(),
        "perimeter_mean".to_string(),
        "area_mean".to_string(),
        "smoothness_mean".to_string(),
        "compactness_mean".to_string(),
        "concavity_mean".to_string(),
        "concave_points_mean".to_string(),
        "symmetry_mean".to_string(),
        "fractal_dimension_mean".to_string(),
        "radius_se".to_string(),
        "texture_se".to_string(),
        "perimeter_se".to_string(),
        "area_se".to_string(),
        "smoothness_se".to_string(),
        "compactness_se".to_string(),
        "concavity_se".to_string(),
        "concave_points_se".to_string(),
        "symmetry_se".to_string(),
        "fractal_dimension_se".to_string(),
        "radius_worst".to_string(),
        "texture_worst".to_string(),
        "perimeter_worst".to_string(),
        "area_worst".to_string(),
        "smoothness_worst".to_string(),
        "compactness_worst".to_string(),
        "concavity_worst".to_string(),
        "concave_points_worst".to_string(),
        "symmetry_worst".to_string(),
        "fractal_dimension_worst".to_string(),
    ]);

    tree.fit(&features, &y, 1000);

    let predictions = tree.predict(&features);

    let mut acc = 0.0;

    for (idx, pred) in predictions.iter().enumerate() {
        acc += (*pred == y.datapoints()[idx]) as u64 as f64;
    }

    acc /= features.len() as f64;

    println!("acc: {:.2}%", acc * 100 as f64);

    Ok(())
}
