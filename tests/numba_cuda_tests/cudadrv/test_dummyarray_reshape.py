# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import numpy as np

from numba_cuda_mlir.numba_cuda.cudadrv import dummyarray


def make_dummy(arr):
    return dummyarray.Array.from_desc(0, arr.shape, arr.strides, arr.itemsize)


class TestDummyArrayReshape(unittest.TestCase):
    """Regression tests for Array.reshape() honoring the requested order.

    ``attempt_nocopy_reshape`` (and the fast paths above it) must only
    produce a zero-copy view when the requested `order` actually matches
    how the array is laid out in memory. Taking the fast path whenever the
    array is contiguous in *either* order - regardless of the requested
    order - yields a view with the wrong strides: the read succeeds, but
    silently returns the wrong data instead of raising or copying.
    """

    def _check_reshape(self, base_shape, newshape, order):
        base = np.arange(int(np.prod(base_shape)), dtype=np.int64).reshape(base_shape)
        for real in (np.ascontiguousarray(base), np.asfortranarray(base)):
            dummy = make_dummy(real)
            try:
                newarr, extents = dummy.reshape(*newshape, order=order)
            except NotImplementedError:
                # No zero-copy view is possible for this layout/order
                # combination; a copy would be required, which is correct.
                continue

            # A no-copy reshape was reported. The strides it produced must
            # describe a view into `real`'s actual memory that reproduces
            # NumPy's own order-aware reshape - never merely reinterpreted
            # memory in the wrong order.
            view = np.lib.stride_tricks.as_strided(real, shape=newshape, strides=newarr.strides)
            expected = real.reshape(newshape, order=order)
            np.testing.assert_array_equal(
                view,
                expected,
                err_msg=(
                    f"reshape({newshape}, order={order!r}) on a "
                    f"{'C' if real.flags['C_CONTIGUOUS'] else 'F'}-contiguous "
                    f"{base_shape} array produced a no-copy view with the wrong data"
                ),
            )

    def test_reshape_flatten_order_c(self):
        self._check_reshape((2, 3), (6,), "C")

    def test_reshape_flatten_order_f(self):
        self._check_reshape((2, 3), (6,), "F")

    def test_reshape_2d_to_2d_order_c(self):
        self._check_reshape((2, 3), (3, 2), "C")

    def test_reshape_2d_to_2d_order_f(self):
        self._check_reshape((2, 3), (3, 2), "F")

    def test_reshape_3d(self):
        self._check_reshape((2, 3, 4), (4, 3, 2), "C")
        self._check_reshape((2, 3, 4), (4, 3, 2), "F")

    def test_matching_order_reshape_is_still_zero_copy(self):
        # Guard against an overly conservative fix: reshaping with the
        # order that already matches the array's own layout must still
        # take the no-copy fast path.
        c_arr = np.ascontiguousarray(np.arange(6, dtype=np.int64).reshape(2, 3))
        f_arr = np.asfortranarray(np.arange(6, dtype=np.int64).reshape(2, 3))

        dummy_c = make_dummy(c_arr)
        newarr, _ = dummy_c.reshape(6, order="C")
        np.testing.assert_array_equal(
            np.lib.stride_tricks.as_strided(c_arr, shape=(6,), strides=newarr.strides),
            c_arr.reshape(6, order="C"),
        )

        dummy_f = make_dummy(f_arr)
        newarr, _ = dummy_f.reshape(6, order="F")
        np.testing.assert_array_equal(
            np.lib.stride_tricks.as_strided(f_arr, shape=(6,), strides=newarr.strides),
            f_arr.reshape(6, order="F"),
        )


if __name__ == "__main__":
    unittest.main()
