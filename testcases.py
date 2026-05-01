import unittest

from tensor import Tensor, TensorDescription


class TensorDescriptionTests(unittest.TestCase):
    def test_scalar_description(self):
        desc = TensorDescription(dtype=int, shape=())

        self.assertEqual(desc.dtype, int)
        self.assertEqual(desc.shape, ())
        self.assertEqual(desc.rank, 0)
        self.assertEqual(desc.size, 1)
        self.assertEqual(desc.get_contiguous_stride(), ())

    def test_multidimensional_description(self):
        desc = TensorDescription(dtype=float, shape=(2, 3, 4))

        self.assertEqual(desc.rank, 3)
        self.assertEqual(desc.size, 24)
        self.assertEqual(desc.get_contiguous_stride(), (12, 4, 1))

    def test_rejects_invalid_description_inputs(self):
        invalid_cases = [
            lambda: TensorDescription(dtype=int, shape=[2, 3]),
            lambda: TensorDescription(dtype=int, shape=(2.0, 3)),
            lambda: TensorDescription(dtype=int, shape=(True, 3)),
            lambda: TensorDescription(dtype=int, shape=(-1,)),
            lambda: TensorDescription(dtype=bool, shape=(2,)),
            lambda: TensorDescription(dtype=str, shape=(2,)),
        ]

        for make_description in invalid_cases:
            with self.subTest(make_description=make_description):
                with self.assertRaises((TypeError, ValueError)):
                    make_description()


class TensorConstructionTests(unittest.TestCase):
    def test_constructs_scalar_tensors(self):
        int_tensor = Tensor(5)
        float_tensor = Tensor(1.5)

        self.assertEqual(int_tensor.shape, ())
        self.assertEqual(int_tensor.stride, ())
        self.assertEqual(int_tensor.tolist(), 5)
        self.assertTrue(int_tensor.is_scalar)

        self.assertEqual(float_tensor.dtype, float)
        self.assertEqual(float_tensor.tolist(), 1.5)

    def test_constructs_nested_tensor(self):
        tensor = Tensor([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(tensor.shape, (2, 3))
        self.assertEqual(tensor.stride, (3, 1))
        self.assertEqual(tensor.rank, 2)
        self.assertEqual(tensor.size, 6)
        self.assertEqual(tensor.tolist(), [[1, 2, 3], [4, 5, 6]])

    def test_constructs_from_flat_data_with_description(self):
        desc = TensorDescription(dtype=int, shape=(2, 2))
        tensor = Tensor(data=[1, 2, 3, 4], description=desc, is_flat=True)

        self.assertEqual(tensor.shape, (2, 2))
        self.assertEqual(tensor.stride, (2, 1))
        self.assertEqual(tensor.tolist(), [[1, 2], [3, 4]])

    def test_constructs_described_zero_size_tensor(self):
        desc = TensorDescription(dtype=int, shape=(2, 0))
        tensor = Tensor(data=[[], []], description=desc)

        self.assertEqual(tensor.shape, (2, 0))
        self.assertEqual(tensor.size, 0)
        self.assertEqual(tensor.tolist(), [[], []])

    def test_requires_description_to_infer_empty_tensor(self):
        with self.assertRaises(ValueError):
            Tensor([])

        with self.assertRaises(ValueError):
            Tensor([[], []])

    def test_rejects_shape_and_data_mismatches(self):
        with self.assertRaises(ValueError):
            Tensor(data=[1, 2, 3], description=TensorDescription(dtype=int, shape=(2, 2)))

        with self.assertRaises(ValueError):
            Tensor(data=[1, 2, 3, 4], description=TensorDescription(dtype=int, shape=()))

        with self.assertRaises(ValueError):
            Tensor([[1, 2], [3]])

    def test_rejects_dtype_mismatches(self):
        with self.assertRaises(TypeError):
            Tensor([[1, 2], [3, 4.0]])

        with self.assertRaises(TypeError):
            Tensor(data=[1, 2.0], description=TensorDescription(dtype=int, shape=(2,)), is_flat=True)

        with self.assertRaises(TypeError):
            Tensor(True)


class TensorIndexingTests(unittest.TestCase):
    def setUp(self):
        self.matrix = Tensor([[1, 2, 3], [4, 5, 6]])
        self.cube_data = [
            [[0, 1, 2], [10, 11, 12], [20, 21, 22], [30, 31, 32]],
            [[100, 101, 102], [110, 111, 112], [120, 121, 122], [130, 131, 132]],
        ]
        self.cube = Tensor(self.cube_data)

    def test_integer_indexing_returns_values_or_lower_rank_views(self):
        self.assertEqual(Tensor([10, 20, 30])[0], 10)
        self.assertEqual(Tensor([10, 20, 30])[-1], 30)
        self.assertEqual(self.matrix[1, 2], 6)
        self.assertEqual(self.matrix[1].shape, (3,))
        self.assertEqual(self.matrix[1].tolist(), [4, 5, 6])
        self.assertEqual(self.cube[1].shape, (4, 3))
        self.assertEqual(self.cube[1, 2].shape, (3,))
        self.assertEqual(self.cube[1, 2, 1], 121)

    def test_mixed_integer_and_slice_indexing_drops_integer_axes(self):
        self.assertEqual(self.matrix[0, :].shape, (3,))
        self.assertEqual(self.matrix[0, :].tolist(), [1, 2, 3])
        self.assertEqual(self.matrix[:, 1].shape, (2,))
        self.assertEqual(self.matrix[:, 1].tolist(), [2, 5])
        self.assertEqual(self.cube[1, :, 2].shape, (4,))
        self.assertEqual(self.cube[1, :, 2].tolist(), [102, 112, 122, 132])
        self.assertEqual(self.cube[:, 2, 1].shape, (2,))
        self.assertEqual(self.cube[:, 2, 1].tolist(), [21, 121])

    def test_slice_indexing_preserves_sliced_axes(self):
        vector = Tensor([10, 20, 30])

        self.assertEqual(vector[0:1].shape, (1,))
        self.assertEqual(vector[0:1].tolist(), [10])
        self.assertEqual(self.matrix[0:1, :].shape, (1, 3))
        self.assertEqual(self.matrix[0:1, :].tolist(), [[1, 2, 3]])
        self.assertEqual(self.matrix[:, 1:2].shape, (2, 1))
        self.assertEqual(self.matrix[:, 1:2].tolist(), [[2], [5]])

    def test_empty_and_reversed_slices(self):
        vector = Tensor([10, 20, 30])

        self.assertEqual(vector[3:3].shape, (0,))
        self.assertEqual(vector[3:3].tolist(), [])
        self.assertEqual(vector[::-1].shape, (3,))
        self.assertEqual(vector[::-1].tolist(), [30, 20, 10])
        self.assertEqual(self.matrix[:, ::-1].tolist(), [[3, 2, 1], [6, 5, 4]])

    def test_empty_tuple_index_returns_equivalent_view(self):
        view = self.matrix[()]

        self.assertEqual(view.shape, self.matrix.shape)
        self.assertEqual(view.stride, self.matrix.stride)
        self.assertEqual(view.offset, self.matrix.offset)
        self.assertEqual(view.tolist(), self.matrix.tolist())

    def test_rejects_invalid_indexing(self):
        vector = Tensor([10, 20, 30])

        with self.assertRaises(TypeError):
            Tensor(5)[0]

        for key in [1.2, [0], "x", True]:
            with self.subTest(key=key):
                with self.assertRaises(TypeError):
                    vector[key]

        for key in [(0, "x"), ("x", 0), (0, 1.2), (0, True)]:
            with self.subTest(key=key):
                with self.assertRaises(TypeError):
                    self.matrix[key]

        with self.assertRaises(IndexError):
            vector[3]

        with self.assertRaises(IndexError):
            vector[-4]

        with self.assertRaises(IndexError):
            self.matrix[0, 1, 2]

        with self.assertRaises(ValueError):
            vector[::0]


class TensorViewAndCompactionTests(unittest.TestCase):
    def test_iteration_and_tolist_use_tensor_values(self):
        self.assertEqual(list(Tensor([10, 20, 30])), [10, 20, 30])
        self.assertEqual(Tensor([[1, 2], [3, 4]]).tolist(), [[1, 2], [3, 4]])

    def test_views_share_backing_storage(self):
        tensor = Tensor([[1, 2, 3], [4, 5, 6]])
        column = tensor[:, 1]

        self.assertIs(column.data, tensor.data)
        self.assertEqual(column.offset, 1)
        self.assertEqual(column.stride, (3,))
        self.assertEqual(column.tolist(), [2, 5])

    def test_view_creation_does_not_revalidate_storage(self):
        tensor = Tensor([[1, 2], [3, 4]])
        original_validate_dtype = Tensor._validate_dtype

        def fail_if_called(_self):
            raise AssertionError("view construction should not revalidate storage")

        try:
            Tensor._validate_dtype = fail_if_called
            self.assertEqual(tensor[0].tolist(), [1, 2])
            self.assertEqual(tensor[:, 1].tolist(), [2, 4])
        finally:
            Tensor._validate_dtype = original_validate_dtype

    def test_compact_copies_non_compact_views(self):
        tensor = Tensor([[1, 2, 3], [4, 5, 6]])
        view = tensor[:, ::-1]
        compact = view.compact()

        self.assertIsNot(compact.data, tensor.data)
        self.assertTrue(compact.is_compact)
        self.assertEqual(compact.stride, (3, 1))
        self.assertEqual(compact.offset, 0)
        self.assertEqual(compact.tolist(), [[3, 2, 1], [6, 5, 4]])

    def test_compact_returns_self_for_already_compact_tensor(self):
        tensor = Tensor([[1, 2], [3, 4]])

        self.assertIs(tensor.compact(), tensor)

    def test_head_respects_view_strides(self):
        tensor = Tensor([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(tensor[:, 1].head(5), [2, 5])
        self.assertEqual(tensor[:, ::-1].head(4), [3, 2, 1, 6])


class TensorShapeOperationTests(unittest.TestCase):
    def setUp(self):
        self.matrix = Tensor([[1, 2, 3], [4, 5, 6]])
        self.cube_data = [
            [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]],
            [[100, 101, 102, 103], [110, 111, 112, 113], [120, 121, 122, 123]],
        ]
        self.cube = Tensor(self.cube_data)

    def test_transpose_returns_rank_two_view_with_swapped_axes(self):
        transposed = self.matrix.transpose()

        self.assertIs(transposed.data, self.matrix.data)
        self.assertEqual(transposed.shape, (3, 2))
        self.assertEqual(transposed.stride, (1, 3))
        self.assertEqual(transposed.offset, 0)
        self.assertEqual(transposed.tolist(), [[1, 4], [2, 5], [3, 6]])

    def test_transpose_rejects_non_matrix_tensors(self):
        with self.assertRaises(ValueError):
            Tensor([1, 2, 3]).transpose()

        with self.assertRaises(ValueError):
            self.cube.transpose()

    def test_permute_reorders_rank_three_axes_as_view(self):
        permuted = self.cube.permute(2, 0, 1)

        self.assertIs(permuted.data, self.cube.data)
        self.assertEqual(permuted.shape, (4, 2, 3))
        self.assertEqual(permuted.stride, (1, 12, 4))
        self.assertEqual(
            permuted.tolist(),
            [
                [[0, 10, 20], [100, 110, 120]],
                [[1, 11, 21], [101, 111, 121]],
                [[2, 12, 22], [102, 112, 122]],
                [[3, 13, 23], [103, 113, 123]],
            ],
        )

    def test_permute_accepts_single_list_or_tuple_of_axes(self):
        self.assertEqual(self.matrix.permute([1, 0]).tolist(), [[1, 4], [2, 5], [3, 6]])
        self.assertEqual(self.matrix.permute((1, 0)).tolist(), [[1, 4], [2, 5], [3, 6]])

    def test_permute_rejects_invalid_axes(self):
        invalid_axis_cases = [
            lambda: self.matrix.permute(0),
            lambda: self.matrix.permute(0, 0),
            lambda: self.matrix.permute(0, 2),
            lambda: self.cube.permute(0, 1),
        ]
        invalid_type_cases = [
            lambda: self.matrix.permute(True, False),
            lambda: self.matrix.permute(0, "x"),
        ]

        with self.assertRaises(TypeError):
            Tensor(5).permute()

        for permute_tensor in invalid_type_cases:
            with self.subTest(permute_tensor=permute_tensor):
                with self.assertRaises(TypeError):
                    permute_tensor()

        for permute_tensor in invalid_axis_cases:
            with self.subTest(permute_tensor=permute_tensor):
                with self.assertRaises(ValueError):
                    permute_tensor()

    def test_reshape_changes_shape_and_contiguous_stride_without_reordering_values(self):
        reshaped = self.matrix.reshape(3, 2)

        self.assertIs(reshaped.data, self.matrix.data)
        self.assertEqual(reshaped.shape, (3, 2))
        self.assertEqual(reshaped.stride, (2, 1))
        self.assertEqual(reshaped.offset, 0)
        self.assertEqual(reshaped.tolist(), [[1, 2], [3, 4], [5, 6]])

    def test_reshape_accepts_single_list_or_tuple_shape(self):
        self.assertEqual(self.matrix.reshape([6]).tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(self.matrix.reshape((1, 6)).tolist(), [[1, 2, 3, 4, 5, 6]])

    def test_reshape_compacts_non_compact_views_before_reshaping(self):
        view = self.matrix[:, ::-1]
        reshaped = view.reshape(3, 2)

        self.assertIsNot(reshaped.data, self.matrix.data)
        self.assertTrue(reshaped.is_compact)
        self.assertEqual(reshaped.shape, (3, 2))
        self.assertEqual(reshaped.stride, (2, 1))
        self.assertEqual(reshaped.tolist(), [[3, 2], [1, 6], [5, 4]])

    def test_reshape_rejects_invalid_shapes(self):
        with self.assertRaises(TypeError):
            Tensor(5).reshape(1)

        with self.assertRaises(ValueError):
            self.matrix.reshape(4, 2)

        with self.assertRaises(ValueError):
            self.matrix.reshape(2, -3)

    def test_squeeze_removes_all_singleton_axes_as_view(self):
        tensor = Tensor([[[1, 2, 3]]])
        squeezed = tensor.squeeze()

        self.assertIs(squeezed.data, tensor.data)
        self.assertEqual(squeezed.shape, (3,))
        self.assertEqual(squeezed.stride, (1,))
        self.assertEqual(squeezed.offset, 0)
        self.assertEqual(squeezed.tolist(), [1, 2, 3])

    def test_squeeze_specific_positive_and_negative_axes(self):
        tensor = Tensor([[[1, 2, 3]], [[4, 5, 6]]])
        middle_squeezed = tensor.squeeze(1)
        trailing_singleton = Tensor([[[1], [2]], [[3], [4]]])
        trailing_squeezed = trailing_singleton.squeeze(-1)

        self.assertIs(middle_squeezed.data, tensor.data)
        self.assertEqual(middle_squeezed.shape, (2, 3))
        self.assertEqual(middle_squeezed.stride, (3, 1))
        self.assertEqual(middle_squeezed.tolist(), [[1, 2, 3], [4, 5, 6]])
        self.assertIs(trailing_squeezed.data, trailing_singleton.data)
        self.assertEqual(trailing_squeezed.shape, (2, 2))
        self.assertEqual(trailing_squeezed.tolist(), [[1, 2], [3, 4]])

    def test_squeeze_keeps_zero_sized_axes_and_noops_without_singletons(self):
        desc = TensorDescription(dtype=int, shape=(1, 0, 2, 1))
        zero_size_tensor = Tensor(data=[], description=desc, is_flat=True)
        squeezed = zero_size_tensor.squeeze()

        self.assertIs(squeezed.data, zero_size_tensor.data)
        self.assertEqual(squeezed.shape, (0, 2))
        self.assertEqual(squeezed.stride, (2, 1))
        self.assertEqual(squeezed.tolist(), [])
        self.assertIs(self.matrix.squeeze(), self.matrix)
        self.assertIs(self.matrix.squeeze(0), self.matrix)

    def test_squeeze_can_return_scalar_tensor(self):
        squeezed = Tensor([[7]]).squeeze()

        self.assertTrue(squeezed.is_scalar)
        self.assertEqual(squeezed.shape, ())
        self.assertEqual(squeezed.stride, ())
        self.assertEqual(squeezed.tolist(), 7)

    def test_squeeze_rejects_invalid_inputs(self):
        invalid_type_cases = ["x", 1.2, True]
        invalid_index_cases = [self.matrix.rank, -self.matrix.rank - 1]

        with self.assertRaises(TypeError):
            Tensor(5).squeeze()

        for index in invalid_type_cases:
            with self.subTest(index=index):
                with self.assertRaises(TypeError):
                    self.matrix.squeeze(index)

        for index in invalid_index_cases:
            with self.subTest(index=index):
                with self.assertRaises(IndexError):
                    self.matrix.squeeze(index)

    def test_unsqueeze_inserts_axes_as_view(self):
        front = self.matrix.unsqueeze(0)
        middle = self.matrix.unsqueeze(1)
        end = self.matrix.unsqueeze(2)

        self.assertIs(front.data, self.matrix.data)
        self.assertEqual(front.shape, (1, 2, 3))
        self.assertEqual(front.stride, (6, 3, 1))
        self.assertEqual(front.tolist(), [[[1, 2, 3], [4, 5, 6]]])
        self.assertEqual(middle.shape, (2, 1, 3))
        self.assertEqual(middle.stride, (3, 3, 1))
        self.assertEqual(middle.tolist(), [[[1, 2, 3]], [[4, 5, 6]]])
        self.assertEqual(end.shape, (2, 3, 1))
        self.assertEqual(end.stride, (3, 1, 1))
        self.assertEqual(end.tolist(), [[[1], [2], [3]], [[4], [5], [6]]])

    def test_unsqueeze_accepts_negative_axes_and_non_compact_views(self):
        front = self.matrix.unsqueeze(-3)
        end = self.matrix.unsqueeze(-1)
        column = self.matrix[:, 1]
        column_unsqueezed = column.unsqueeze(1)

        self.assertEqual(front.shape, (1, 2, 3))
        self.assertEqual(front.tolist(), [[[1, 2, 3], [4, 5, 6]]])
        self.assertEqual(end.shape, (2, 3, 1))
        self.assertEqual(end.tolist(), [[[1], [2], [3]], [[4], [5], [6]]])
        self.assertIs(column_unsqueezed.data, self.matrix.data)
        self.assertEqual(column_unsqueezed.offset, 1)
        self.assertEqual(column_unsqueezed.shape, (2, 1))
        self.assertEqual(column_unsqueezed.stride, (3, 3))
        self.assertEqual(column_unsqueezed.tolist(), [[2], [5]])

    def test_unsqueeze_scalar_returns_size_one_tensor(self):
        unsqueezed = Tensor(7).unsqueeze(0)
        negative_unsqueezed = Tensor(7).unsqueeze(-1)

        self.assertEqual(unsqueezed.shape, (1,))
        self.assertEqual(unsqueezed.stride, (1,))
        self.assertEqual(unsqueezed.tolist(), [7])
        self.assertEqual(negative_unsqueezed.tolist(), [7])

    def test_unsqueeze_rejects_invalid_inputs(self):
        invalid_type_cases = ["x", 1.2, True]
        invalid_index_cases = [self.matrix.rank + 1, -self.matrix.rank - 2]

        for index in invalid_type_cases:
            with self.subTest(index=index):
                with self.assertRaises(TypeError):
                    self.matrix.unsqueeze(index)

        for index in invalid_index_cases:
            with self.subTest(index=index):
                with self.assertRaises(IndexError):
                    self.matrix.unsqueeze(index)


class TensorArithmeticOperationTests(unittest.TestCase):
    def setUp(self):
        self.matrix = Tensor([[1, 2, 3], [4, 5, 6]])
        self.other_matrix = Tensor([[10, 20, 30], [40, 50, 60]])

    def test_add_subtract_and_multiply_same_shape_tensors(self):
        added = self.matrix + self.other_matrix
        subtracted = self.other_matrix - self.matrix
        multiplied = self.matrix * self.other_matrix

        self.assertEqual(added.shape, (2, 3))
        self.assertEqual(added.stride, (3, 1))
        self.assertEqual(added.offset, 0)
        self.assertEqual(added.tolist(), [[11, 22, 33], [44, 55, 66]])
        self.assertEqual(subtracted.tolist(), [[9, 18, 27], [36, 45, 54]])
        self.assertEqual(multiplied.tolist(), [[10, 40, 90], [160, 250, 360]])

    def test_arithmetic_creates_independent_output_storage(self):
        result = self.matrix + self.other_matrix

        self.assertIsNot(result.data, self.matrix.data)
        self.assertIsNot(result.data, self.other_matrix.data)
        self.assertTrue(result.is_compact)

    def test_arithmetic_with_right_hand_python_scalars(self):
        self.assertEqual((self.matrix + 10).tolist(), [[11, 12, 13], [14, 15, 16]])
        self.assertEqual((self.matrix - 1).tolist(), [[0, 1, 2], [3, 4, 5]])
        self.assertEqual((self.matrix * 2).tolist(), [[2, 4, 6], [8, 10, 12]])

    def test_arithmetic_with_scalar_tensors(self):
        scalar = Tensor(10)

        self.assertEqual((scalar + self.matrix).tolist(), [[11, 12, 13], [14, 15, 16]])
        self.assertEqual((scalar - self.matrix).tolist(), [[9, 8, 7], [6, 5, 4]])
        self.assertEqual((scalar * self.matrix).tolist(), [[10, 20, 30], [40, 50, 60]])

    def test_scalar_tensor_operations_use_the_requested_operator(self):
        left = Tensor(10)
        right = Tensor(3)

        self.assertEqual((left + right).tolist(), 13)
        self.assertEqual((left - right).tolist(), 7)
        self.assertEqual((left * right).tolist(), 30)
        self.assertEqual((left / right).tolist(), 10 / 3)
        self.assertEqual((left // right).tolist(), 3)

    def test_reverse_python_scalar_arithmetic(self):
        self.assertEqual((10 + self.matrix).tolist(), [[11, 12, 13], [14, 15, 16]])
        self.assertEqual((10 - self.matrix).tolist(), [[9, 8, 7], [6, 5, 4]])
        self.assertEqual((2 * self.matrix).tolist(), [[2, 4, 6], [8, 10, 12]])
        self.assertEqual((12 / self.matrix).tolist(), [[12.0, 6.0, 4.0], [3.0, 2.4, 2.0]])
        self.assertEqual((12 // self.matrix).tolist(), [[12, 6, 4], [3, 2, 2]])

    def test_broadcasts_trailing_vector_over_matrix_rows(self):
        vector = Tensor([10, 20, 30])

        self.assertEqual((self.matrix + vector).shape, (2, 3))
        self.assertEqual((self.matrix + vector).tolist(), [[11, 22, 33], [14, 25, 36]])
        self.assertEqual((vector * self.matrix).tolist(), [[10, 40, 90], [40, 100, 180]])

    def test_broadcasts_singleton_row_and_column_dimensions(self):
        row = Tensor([[10, 20, 30]])
        column = Tensor([[10], [20]])

        self.assertEqual((self.matrix + row).tolist(), [[11, 22, 33], [14, 25, 36]])
        self.assertEqual((self.matrix + column).tolist(), [[11, 12, 13], [24, 25, 26]])

    def test_broadcasts_two_tensors_with_complementary_singleton_dimensions(self):
        column = Tensor([[1], [2]])
        row = Tensor([[10, 20, 30]])

        self.assertEqual((column + row).shape, (2, 3))
        self.assertEqual((column + row).tolist(), [[11, 21, 31], [12, 22, 32]])

    def test_arithmetic_respects_non_compact_view_strides(self):
        reversed_columns = self.matrix[:, ::-1]
        result = reversed_columns + self.matrix

        self.assertEqual(reversed_columns.tolist(), [[3, 2, 1], [6, 5, 4]])
        self.assertEqual(result.tolist(), [[4, 4, 4], [10, 10, 10]])

    def test_true_division_returns_float_tensor(self):
        divisor = Tensor([[2, 2, 2], [2, 2, 2]])
        divided = self.matrix / divisor

        self.assertEqual(divided.dtype, float)
        self.assertEqual(divided.shape, (2, 3))
        self.assertEqual(divided.stride, (3, 1))
        self.assertEqual(divided.tolist(), [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])

    def test_floor_division_keeps_integer_tensor_for_integer_operands(self):
        divisor = Tensor([[2, 2, 2], [2, 2, 2]])
        divided = self.matrix // divisor

        self.assertEqual(divided.dtype, int)
        self.assertEqual(divided.tolist(), [[0, 1, 1], [2, 2, 3]])

    def test_division_broadcasts_scalars_and_vectors(self):
        vector = Tensor([1, 2, 3])

        self.assertEqual((self.matrix / vector).tolist(), [[1.0, 1.0, 1.0], [4.0, 2.5, 2.0]])
        self.assertEqual((self.matrix // 2).tolist(), [[0, 1, 1], [2, 2, 3]])
        self.assertEqual((Tensor(12) / self.matrix).tolist(), [[12.0, 6.0, 4.0], [3.0, 2.4, 2.0]])

    def test_division_respects_non_compact_view_strides(self):
        reversed_columns = self.matrix[:, ::-1]
        result = reversed_columns / self.matrix

        self.assertEqual(reversed_columns.tolist(), [[3, 2, 1], [6, 5, 4]])
        self.assertEqual(result.tolist(), [[3.0, 1.0, 1 / 3], [1.5, 1.0, 4 / 6]])

    def test_division_rejects_zero_divisors(self):
        with self.assertRaises(ZeroDivisionError):
            self.matrix / Tensor([[1, 0, 1], [1, 1, 1]])

        with self.assertRaises(ZeroDivisionError):
            self.matrix // 0

    def test_rejects_incompatible_shapes_and_operand_types(self):
        with self.assertRaises(ValueError):
            self.matrix + Tensor([1, 2])

        with self.assertRaises(TypeError):
            self.matrix + "x"

        with self.assertRaises(TypeError):
            self.matrix * True


if __name__ == "__main__":
    unittest.main(verbosity=2)
