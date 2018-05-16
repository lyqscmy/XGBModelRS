# xgboost预测过程Rust版
改过xgboost预测过程的C++接口，在线预测效果非常好，熟悉核心逻辑之后尝试过用Java重写，但是性能很不理想。本来想用C++重写，但是C++的坑不太熟，怕写出不安全的代码。正好在学Rust，于是拿来练手，重写过程非常顺利愉快。

这是我用Python解析二进制模型文件的[代码](https://github.com/lyqscmy/TreeModel)，和C++源码交叉对比过解析和预测过程，没毛病，只是解析过程没做异常处理。

# 单行预测Benchmark
i5-6500 CPU @ 3.20GHz

## xgboost python接口
```python
# pred_leaf:0.029471746901981535ms
number = 10000
predic_time = timeit.timeit(
    "bst.predict(dtest, pred_leaf=True)", globals=globals(), number=number)
print("pred_leaf:{}ms".format(predic_time / number * 1000))
```

## Rust
```Rust
// test bench_predict_leaf ... bench:       4,764 ns/iter (+/- 500) 
b.iter(|| {
    feats.set(&indices, &values);
    model.predict_leaf(&feats, tree_limit, &mut preds);
    feats.reset(&indices)
});
```