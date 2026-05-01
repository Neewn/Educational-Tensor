# Educational Tensor

A small tensor implementation written in pure Python for learning how tensor libraries work under the hood.

This project is not trying to replace NumPy, PyTorch, or TensorFlow. It is meant to make the core ideas visible:

- flat storage
- tensor shape, rank, and size
- contiguous strides
- offset-based views
- slicing without copying
- reshaping and permutation
- broadcasting with zero strides
- compacting non-contiguous views
- basic elementwise arithmetic

## Example

```python
from tensor import Tensor

x = Tensor([[1, 2, 3], [4, 5, 6]])

print(x.shape)
# (2, 3)

print(x[:, 1].tolist())
# [2, 5]

print(x[:, ::-1].tolist())
# [[3, 2, 1], [6, 5, 4]]

print((x + Tensor([10, 20, 30])).tolist())
# [[11, 22, 33], [14, 25, 36]]

print(x.transpose().tolist())
# [[1, 4], [2, 5], [3, 6]]
```

## Why This Exists

Most tensor libraries hide their internal mechanics behind highly optimized native kernels. That is great for real work, but less helpful when learning.

This project keeps the implementation small and readable so you can inspect how a tensor can be represented by:

```text
data + shape + stride + offset
```

For example, slicing a column from a matrix does not need to copy values. It can create a new tensor view that shares the same data but changes the shape, stride, and offset.

## Features

Current features:

- scalar, vector, matrix, and higher-rank tensors
- construction from nested Python lists
- explicit tensor descriptions for flat or zero-size data
- integer indexing and slicing
- negative indexing
- reversed slices
- view-based `permute`, `transpose`, `squeeze`, and `unsqueeze`
- `reshape`, with compaction for non-compact views
- broadcasting for elementwise arithmetic
- `+`, `-`, `*`, `/`, and `//`
- conversion back to Python lists with `tolist()`

## Limitations

This is an educational project, so several production features are intentionally missing or incomplete:

- no NumPy integration
- no automatic differentiation
- no matrix multiplication yet
- no GPU support
- only `int` and `float` tensors are supported
- no dtype promotion between `int` and `float`
- arithmetic is implemented with Python loops

## Running Tests

The project uses the Python standard library `unittest`, so there are no required third-party dependencies.

From the project root:

```bash
python testcases.py
```


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
