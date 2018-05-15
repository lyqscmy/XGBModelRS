#![feature(test)]
extern crate test;
extern crate tree_model;

use std::fs::File;
use std::io::Read;
use test::Bencher;
use tree_model::{FVec, XGBModel};

#[bench]
fn bench_predict_leaf(b: &mut Bencher) {
    let model = setup_model();

    let (indices, values) = setup_input();
    let tree_limit = 0;
    let mut feats = FVec::new(model.num_features());
    feats.set(&indices, &values);
    let mut preds = vec![0; model.num_trees()];

    b.iter(|| {
        feats.set(&indices, &values);
        model.predict_leaf(&feats, tree_limit, &mut preds);
        feats.reset(&indices)
    });

    let output = setup_ouput();
    assert_eq!(output, preds);
}

pub fn add_two(a: i32) -> i32 {
    a + 2
}

#[bench]
fn bench_add_two(b: &mut Bencher) {
    b.iter(|| add_two(2));
}

fn setup_ouput() -> Vec<u32> {
    let mut f = File::open("./data/output.txt").expect("Failed to open output.txt");
    let mut line = String::new();
    f.read_to_string(&mut line)
        .expect("Failed to read output.txt");
    let line = line.trim();

    let xs: Vec<&str> = line.split(' ').collect();
    let length = xs.len() - 1;
    let mut preds = Vec::with_capacity(length);
    for x in &xs {
        let a: u32 = x.parse().unwrap();
        preds.push(a);
    }
    preds
}

fn setup_model() -> XGBModel {
    let mut f = File::open("./data/model.bin").expect("Failed to open model.bin");
    let metadata = f.metadata().expect("Failed to read metadata");
    let mut buffer = Vec::with_capacity((metadata.len() as usize) + 1);
    f.read_to_end(&mut buffer)
        .expect("Failed to read file to buffer");

    XGBModel::load(&buffer).expect("Failed to load model")
}

fn setup_input() -> (Vec<u32>, Vec<f32>) {
    let mut f = File::open("./data/input.txt").expect("Failed to open input.txt");
    let mut line = String::new();
    f.read_to_string(&mut line)
        .expect("Failed to read input.txt");
    let line = line.trim();

    let xs: Vec<&str> = line.split(' ').collect();
    let length = xs.len() - 1;
    let mut indices = Vec::with_capacity(length);
    let mut values = Vec::with_capacity(length);
    for t in &xs[1..] {
        let c: Vec<&str> = t.split(':').collect();
        let a: u32 = c[0].parse().unwrap();
        let b: f32 = c[1].parse().unwrap();
        indices.push(a);
        values.push(b);
    }

    (indices, values)
}
