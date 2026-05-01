from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cached_property
from itertools import zip_longest


@dataclass(frozen=True)
class TensorDescription:

    supported_types = (int, float)

    dtype: type
    shape: tuple[int, ...]

    # Note: We accept 0 dimensions for shape
    def __post_init__(self):
        if not isinstance(self.shape, tuple):
            raise TypeError(f'Shape {self.shape} is not a valid tuple')

        if not all(isinstance(dim, int) for dim in self.shape):
            raise TypeError("Shape dimensions must all be integers")

        if not all(dim >= 0 for dim in self.shape):
            raise ValueError("Shape dimensions must be non-negative")

        if self.dtype is bool:
            raise TypeError("bool tensors are not supported")

        if self.dtype not in self.supported_types:
            raise TypeError(f"type {self.dtype!r} is not supported")

        for shape_dim in self.shape:
            if type(shape_dim) != int:
                raise TypeError(f'shape dimensions must be integers, dimension {shape_dim} has type {type(shape_dim).__name__}')

    @cached_property
    def size(self) -> int:
        if not self.shape:
            return 1

        total = 1
        for i in self.shape:
            total *= i

        return total

    @cached_property
    def rank(self) -> int:
        return len(self.shape)

    def get_contiguous_stride(self):
        # quick checkpoint
        if self.rank == 0: #scalar
            return ()
        #quick checkpoint
        if self.rank == 1:
            return (1,)

        temp_stride = [1] * self.rank
        cur_stride = 1

        for i, dim in enumerate(self.shape[:0:-1], 2):
            cur_stride *= dim
            temp_stride[-i] = cur_stride
        return tuple(temp_stride)

    def __repr__(self):
            return f'TensorDescription(shape={self.shape}, dtype={self.dtype.__name__}, rank={self.rank}, size={self.size})'


@dataclass
class Tensor:
    data : list | int | float = 0.0
    stride: tuple[int, ...] = None
    offset: int = 0
    description: TensorDescription = None
    is_flat : bool = False

    @property
    def dtype(self): return self.description.dtype
    @property
    def shape(self): return self.description.shape
    @property
    def rank(self): return self.description.rank
    @property
    def size(self): return self.description.size
    @property
    def is_scalar(self):
        return self.rank == 0
    @property
    def is_contiguous(self):
        return self.stride == self.description.get_contiguous_stride()
    @property
    def is_compact(self): #Either scalar or is compact
        return self.is_scalar or (self.is_contiguous and self.offset == 0 and len(self.data) == self.size)

    @classmethod
    # Skips the __init__ and __post_init__
    # Skips validation
    # Strictly for internal use
    def _view(cls, data, description, stride, offset) -> Tensor:
        obj = cls.__new__(cls)
        obj.data = data
        obj.description = description
        obj.stride = stride
        obj.offset = offset
        obj.is_flat = True
        return obj

    def _flatten_data(self): #Also validates shape
        if not isinstance(self.data, list):
            self.is_flat = True
            self.data = [self.data]
            return

        out = []
        q = deque()
        q.append((self.data, 0)) # data, layer
        while q:
            cur, layer = q.popleft()

            if layer >= self.rank:
                raise ValueError(f"Unexpected nested list at layer {layer + 1}")

            if len(cur) != self.shape[layer]:
               raise ValueError(f'Child length {len(cur)} is not consistent with shape {self.shape} at layer {layer}')

            for child in cur:
                if isinstance(child, list):
                    q.append((child, layer+1))

                else: #scalar
                    if layer != self.rank - 1:
                        raise ValueError(f'Unexpected scalar:{child} at layer {layer+1}')
                    out.append(child)

        self.data = out
        self.is_flat = True

    # If not supplied a stride, this method creates one from inference
    # Assume that tensor description has been supplied/made and validated
    # Assume that tensor data is stored contiguously. Meaning no unused gaps in data.
    def _infer_stride(self):
        self.stride = self.description.get_contiguous_stride()

    # If not supplied a description, this creates one from inference
    # May not be accurate to intention
    def _infer_description(self):
        if not isinstance(self.data, list):
            self.description = TensorDescription(dtype=type(self.data), shape=())
            return

        cur = self.data
        rank = 0
        shape = []
        while isinstance(cur, list):
            if len(cur) == 0: #Cannot infer 0 dimensions
                raise ValueError("Failed to create tensor description: Empty tensor lists are not supported")
            rank += 1
            shape.append(len(cur))
            cur = cur[0]

        dtype = type(cur)
        shape = tuple(shape)
        self.description = TensorDescription(dtype=dtype, shape=shape)

    #Assume data is flat
    #Assume tensor description is made
    def _validate_dtype(self):
        for element in self.data:
            if type(element) != self.dtype:
                raise TypeError(f'element={element}, dtype={type(element).__name__} is not consistent with description dtype={self.dtype.__name__}')

    def _validate_stride(self):
        if self.is_scalar:
            if self.stride != ():
                raise AttributeError(f'Scalar should not have a stride, current stride={self.stride}')

            if self.offset < 0:
                raise AttributeError(f'Invalid offset for scalar: {self.offset} < 0')

            if self.offset >= len(self.data):
                raise AttributeError(f'Invalid offset for scalar: {self.offset} >= data_size={len(self.data)}')
        else:
            if not isinstance(self.stride, tuple):
                raise TypeError(f'Stride must be a tuple')

            if self.rank != len(self.stride):
                raise AttributeError(f'Shape and strides ranks do not match, shape={self.shape}, stride={self.stride}')

            if self.size > 0:
                min_index = self.offset + sum(min(0, (dim - 1) * st) for dim, st in zip(self.shape, self.stride))
                if min_index < 0:
                    raise AttributeError(f"Invalid stride, min index={min_index} < 0")

                max_index = self.offset + sum(max(0, (dim - 1) * st) for dim, st in zip(self.shape, self.stride))
                if max_index >= len(self.data):
                    raise AttributeError(f"Invalid stride: max index={max_index} > data_size={len(self.data)}")

    def __post_init__(self):
        if self.description is None:
            self._infer_description()

        if not self.is_flat:
            self._flatten_data()

        self._validate_dtype()

        if self.stride is None:
            self._infer_stride()

        self._validate_stride()

    def __iter__(self):
        if self.rank == 0:
            raise TypeError('Scalar is not iterable')

        if self.rank == 1:
            for i in range(self.shape[0]):
                yield self.data[self.offset + i * self.stride[0]]

        if self.rank > 1:
            for i in range(self.shape[0]):
                next_tensor_description = TensorDescription(dtype=self.dtype, shape=self.shape[1:])
                next_offset = self.offset + i * self.stride[0]
                yield Tensor._view(data=self.data, stride=self.stride[1:], offset=next_offset, description=next_tensor_description)

    def __len__(self):
        if self.is_scalar:
            raise TypeError(f'object of type {self.dtype.__name__} has no len()')
        return self.shape[0]

    def __getitem__(self, key):
        if self.is_scalar:
            raise TypeError('Scalar is not subscriptable')

        if type(key) is not tuple:
            if type(key) is int or type(key) is slice:
                key = (key,)
            else:
                raise TypeError(f'list indices must be integers or slices, not {type(key).__name__}')

        if len(key) > self.rank:
            raise IndexError("Too many indices for tensor")

        no_slices = True
        cur_shape = []
        cur_stride = []
        cur_offset = self.offset
        for axis, indexer in enumerate(key):
            axis_shape_max = self.shape[axis]
            axis_stride = self.stride[axis]
            if type(indexer) is int:
                if indexer < 0:
                    if axis_shape_max + indexer < 0:  # out of range on the negative side
                        raise IndexError(f"Tensor index out of range {indexer} < -{axis_shape_max}")
                    indexer = axis_shape_max + indexer # normalise
                if indexer >= axis_shape_max:
                    raise IndexError(f"Tensor index out of range {indexer} > {axis_shape_max - 1}")
                cur_offset += indexer * axis_stride
                #uncomment the below 2 lines to enable keep_dim
                #cur_shape.append(1)
                #cur_stride.append(self.stride[axis])

            elif type(indexer) is slice:
                no_slices = False
                start, stop, step = indexer.indices(axis_shape_max)
                if step > 0:
                    new_shape = max((stop - start + step - 1) // step, 0)
                else:
                    step_abs = -step
                    new_shape = max((start - stop + step_abs - 1) // step_abs, 0)
                new_stride = self.stride[axis] * step
                cur_offset += start * self.stride[axis]
                cur_shape.append(new_shape)
                cur_stride.append(new_stride)
            else:
                raise TypeError(f'list indices must be integers or slices, not {type(indexer).__name__}')

        for axis in range(len(key), self.rank):
             cur_stride.append(self.stride[axis])
             cur_shape.append(self.shape[axis])

        if no_slices and len(key) == self.rank:
            return self.data[cur_offset]
        else:
            out_tensor_description = TensorDescription(dtype=self.dtype, shape=tuple(cur_shape))
            out_tensor = Tensor._view(data=self.data, description=out_tensor_description, stride=tuple(cur_stride), offset=cur_offset)
            return out_tensor

    #Makes data flat + removes values before offset
    def compact(self):
        if self.is_compact or self.is_scalar:
            return self

        new_data = [None] * self.size
        idx = [0] * self.rank
        cur_idx = self.offset

        for new_i in range(self.size):
            new_data[new_i] = self.data[cur_idx]
            for axis in range(self.rank-1, -1, -1):
                idx[axis] += 1
                cur_idx += self.stride[axis]
                if idx[axis] < self.shape[axis]:
                    break
                cur_idx -= self.shape[axis] * self.stride[axis]
                idx[axis] = 0

        return Tensor._view(data=new_data, stride=self.description.get_contiguous_stride(), offset=0, description=self.description)

    # current ordering is 0,1,2..n
    # we switch to the new axes
    def permute(self, *axes):
        if self.is_scalar:
            raise TypeError("Scalar cannot be permuted")

        if len(axes) == 1 and isinstance(axes[0], (list, tuple)):
            axes = tuple(axes[0])

        if not all(type(axis) is int for axis in axes):
            raise TypeError("Axes must be integers")

        if len(axes) != self.rank:
            raise ValueError(f"Invalid input axes. dim={len(axes)} is not equal to rank={self.rank}")

        if set(axes) != set(range(len(axes))):
            raise ValueError(f"Invalid input axes. Axis numbers should be distinct and within the range (0 to rank-1). current rank={self.rank}")

        new_shape = tuple(self.shape[axis] for axis in axes)
        new_stride = tuple(self.stride[axis] for axis in axes)
        new_description = TensorDescription(dtype=self.dtype, shape=new_shape)

        return Tensor._view(data=self.data, stride=new_stride, offset = self.offset, description=new_description)

    #Strictly only for Tensors of rank 2
    def transpose(self):
        if self.rank != 2:
            raise ValueError(f"Permute is only for tensors with rank 2. Current rank = {self.rank}")
        return self.permute(1,0)

    def reshape(self, *new_shape):
        if self.is_scalar:
            raise TypeError("Scalar cannot be reshaped")

        if len(new_shape) == 0:
            raise ValueError("Missing shape argument")

        if len(new_shape) == 1 and isinstance(new_shape[0], (list, tuple)):
            new_shape = tuple(new_shape[0])

        new_shape_size = 1
        for i in new_shape:
            new_shape_size *= i
        if new_shape_size != self.size:
            raise ValueError(f'Shapes do not match. input shape size={new_shape_size}, current size ={self.size}')

        if not self.is_compact:
            compact_tensor = self.compact()
        else:
            compact_tensor = self

        new_description = TensorDescription(dtype=compact_tensor.dtype, shape=new_shape)
        return Tensor._view(data=compact_tensor.data, stride=new_description.get_contiguous_stride(), offset=0, description=new_description)

    #Removes shape dimension at index if it is 1
    def squeeze(self, index=None):
        if self.is_scalar:
            raise TypeError("Scalar cannot be squeezed")

        if index is None:
            if 1 in self.shape:
                new_shape = tuple(shape_dim for shape_dim in self.shape if shape_dim != 1)
                new_stride = tuple(stride for stride, shape_dim in zip(self.stride, self.shape) if shape_dim != 1)
                if new_shape == ():
                    return Tensor._view(self.data, stride=(), offset=self.offset, description=TensorDescription(dtype=self.dtype, shape=()))
                new_description = TensorDescription(dtype=self.dtype, shape=new_shape)
                return Tensor._view(data=self.data, stride=new_stride, offset=self.offset, description=new_description)
            else:
                return self

        if type(index) is not int:
            raise TypeError("Index must be a int")

        if index < -self.rank or index >= self.rank:
            raise IndexError(f"Squeeze index ({index}) must be between -rank({-self.rank}) and rank-1({self.rank - 1}). Current rank={self.rank}")

        if index < 0:
            index = self.rank + index

        if self.shape[index] == 1:
            new_shape = self.shape[:index] + self.shape[index+1:]
            new_stride = self.stride[:index] + self.stride[index+1:]
            if new_shape == ():
                return Tensor._view(self.data, stride=(), offset=self.offset, description=TensorDescription(dtype=self.dtype, shape=()))
            new_description = TensorDescription(dtype=self.dtype, shape=new_shape)
            return Tensor._view(data=self.data, stride=new_stride, offset=self.offset, description=new_description)

        return self

    #Adds a 1 to shape at index
    def unsqueeze(self, index):
        if type(index) is not int:
            raise TypeError("Index must be a int")

        if index < -self.rank-1 or index > self.rank:
            raise IndexError(f"Unsqueeze index ({index}) must be between -rank-1({-self.rank-1}) and rank({self.rank}). Current rank={self.rank}")

        if index < 0:
            index = self.rank + index + 1

        if self.is_scalar:
            new_description = TensorDescription(dtype=self.dtype, shape=(1,))
            return Tensor._view(data=self.data, stride=(1,), offset=self.offset, description=new_description)

        if index == self.rank:
            new_shape = self.shape[:index] + (1,)
            new_stride = self.stride[:index] + (self.stride[index-1],)
            new_description = TensorDescription(dtype=self.dtype, shape=new_shape)
        else:
            new_shape = self.shape[:index] + (1,) + self.shape[index:]
            new_stride = self.stride[:index] + (self.stride[index] * self.shape[index] ,) + self.stride[index:]
            new_description = TensorDescription(dtype=self.dtype, shape=new_shape)

        return Tensor._view(data=self.data, stride=new_stride, offset=self.offset, description=new_description)

    @staticmethod
    def _broadcast_shape(left_shape, right_shape):
        out = []
        for left_shape_dim, right_shape_dim in zip_longest(left_shape[::-1], right_shape[::-1], fillvalue=1):
            if left_shape_dim == right_shape_dim:
                out_dim = left_shape_dim
            elif left_shape_dim == 1:
                out_dim = right_shape_dim
            elif right_shape_dim == 1:
                out_dim = left_shape_dim
            else:
                raise ValueError(f"Shapes are not compatible. {left_shape} vs {right_shape}")
            out.append(out_dim)
        return tuple(out[::-1])

    def _broadcast_to(self, new_shape):
        new_description = TensorDescription(dtype=self.dtype, shape=new_shape)
        padding = len(new_shape) - len(self.shape)
        new_stride = (0, ) * padding + tuple(0 if shape_dim == 1 else stride for shape_dim, stride in zip(self.shape, self.stride))
        return Tensor._view(data=self.data, description=new_description, stride = new_stride, offset=self.offset)

    def _binary_op(self, other, op, out_dtype = None):

        if type(other) in (int, float):
            other = Tensor(other)
        elif not isinstance(other, Tensor):
            raise TypeError(f"Invalid type {type(other).__name__}. Expected scalar or Tensor")

        if self.dtype != other.dtype:
            raise TypeError(f"Cannot do operations with tensors of different datatypes. {self.dtype.__name__} vs {other.dtype.__name__} ")

        if self.is_scalar and other.is_scalar:
            return Tensor(data=op(self.data[self.offset], other.data[other.offset]))

        X, Y = self, other
        broadcast_shape = Tensor._broadcast_shape(X.shape, Y.shape)
        if X.shape != broadcast_shape:
            X = X._broadcast_to(broadcast_shape)
        if Y.shape != broadcast_shape:
            Y = Y._broadcast_to(broadcast_shape)

        out_data = [0] * X.size
        idx = [0] * X.rank

        X_index = X.offset
        Y_index = Y.offset
        for out_i in range(X.size):
            out_data[out_i] = op(X.data[X_index], Y.data[Y_index])
            for axis in range(X.rank-1, -1, -1):
                idx[axis] += 1
                X_index += X.stride[axis]
                Y_index += Y.stride[axis]
                if idx[axis] < X.shape[axis]:
                    break
                X_index -= X.shape[axis] * X.stride[axis]
                Y_index -= Y.shape[axis] * Y.stride[axis]
                idx[axis] = 0

        out_dtype = out_dtype or X.dtype
        out_description = TensorDescription(dtype=out_dtype, shape=X.shape)
        return Tensor._view(data=out_data, stride=X.description.get_contiguous_stride(), offset=0, description=out_description)

    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b)

    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b)

    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b)

    def __truediv__(self, other):
        return self._binary_op(other, lambda a, b: a / b, out_dtype=float)

    def __floordiv__(self, other):
        return self._binary_op(other, lambda a, b: a // b)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        if type(other) in (int, float):
            other = Tensor(other)
        else:
            raise TypeError(f"Invalid type {type(other).__name__}. Expected scalar or Tensor")
        return other.__sub__(self)

    def __rtruediv__(self, other):
        if type(other) in (int, float):
            other = Tensor(other)
        else:
            raise TypeError(f"Invalid type {type(other).__name__}. Expected scalar or Tensor")
        return other.__truediv__(self)

    def __rfloordiv__(self, other):
        if type(other) in (int, float):
            other = Tensor(other)
        else:
            raise TypeError(f"Invalid type {type(other).__name__}. Expected scalar or Tensor")
        return other.__floordiv__(self)

    #Return the head few values. Will return a flat array of data. No nesting yet
    def head(self, head_length):
        if self.is_scalar:
            return self.data[self.offset]

        out = []
        idx = [0] * self.rank
        cur_idx = self.offset
        for i in range(min(head_length, self.size)):
            out.append(self.data[cur_idx])
            for axis in range(self.rank-1, -1, -1):
                idx[axis] += 1
                cur_idx += self.stride[axis]
                if idx[axis] < self.shape[axis]:
                    break
                cur_idx -= self.shape[axis] * self.stride[axis]
                idx[axis] = 0
        return out

    def get_head_str(self) -> str:
        if self.is_scalar:
            return str(self.data[self.offset])
        head_out = self.head(5)
        if len(head_out) < 5:
            return str(head_out)
        return "[" + ", ".join(str(i) for i in head_out) + "...]"

    def tolist(self):
        if self.is_scalar:
            return self.data[self.offset]
        return [x.tolist() if isinstance(x, Tensor) else x for x in self]

    def __repr__(self) -> str:
        return f'Tensor(shape={self.shape}, dtype={self.dtype.__name__}, stride={self.stride}, offset={self.offset}, ' + f'data={self.get_head_str()})'

    def __str__(self) -> str:
        return self.__repr__()
        
