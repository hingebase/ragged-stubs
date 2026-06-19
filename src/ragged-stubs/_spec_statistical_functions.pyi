# Copyright 2025-2026 hingebase

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import sys
from typing import Any, Literal, TypeAlias, TypeVar, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from typing_extensions import Unpack

from ._spec_array_object import array
from ._typing import Dtype

@overload
def max(  # noqa: A001
    x: array[Any, _DTypeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: Literal[False] = ...,
) -> array[Any, _DTypeT]: ...
@overload
def max(  # noqa: A001
    x: array[_ShapeT, _DTypeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: Literal[True],
) -> array[_ShapeT, _DTypeT]: ...
@overload
def max(  # noqa: A001
    x: array[Any, _DTypeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: bool,
) -> array[Any, _DTypeT]: ...

@overload
def mean(
    x: array[_AtLeast1D, _InexactDTypeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: Literal[False] = ...,
) -> array[Any, _InexactDTypeT]: ...
@overload
def mean(
    x: array[_AtLeast1DT, _InexactDTypeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: Literal[True],
) -> array[_AtLeast1DT, _InexactDTypeT]: ...
@overload
def mean(
    x: array[_AtLeast1D, _InexactDTypeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: bool,
) -> array[Any, _InexactDTypeT]: ...
@overload
def mean(
    x: array[_AtLeast1D, Dtype[_Integer | np.bool_]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.float64]]: ...
@overload
def mean(
    x: array[_AtLeast1DT, Dtype[_Integer | np.bool_]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: Literal[True],
) -> array[_AtLeast1DT, Dtype[np.float64]]: ...
@overload
def mean(
    x: array[_AtLeast1D, Dtype[_Integer | np.bool_]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: bool,
) -> array[Any, Dtype[np.float64]]: ...

@overload
def min(  # noqa: A001
    x: array[Any, _DTypeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: Literal[False] = ...,
) -> array[Any, _DTypeT]: ...
@overload
def min(  # noqa: A001
    x: array[_ShapeT, _DTypeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: Literal[True],
) -> array[_ShapeT, _DTypeT]: ...
@overload
def min(  # noqa: A001
    x: array[Any, _DTypeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: bool,
) -> array[Any, _DTypeT]: ...

@overload
def prod(
    x: array[Any, Dtype[_RegularTypeT]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[_RegularTypeT]]: ...
@overload
def prod(
    x: array[_ShapeT, Dtype[_RegularTypeT]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[_RegularTypeT]]: ...
@overload
def prod(
    x: array[Any, Dtype[_RegularTypeT]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: bool,
) -> array[Any, Dtype[_RegularTypeT]]: ...
@overload
def prod(
    x: array[Any, Dtype[_SignedInteger | np.bool_]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.int64]]: ...
@overload
def prod(
    x: array[_ShapeT, Dtype[_SignedInteger | np.bool_]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.int64]]: ...
@overload
def prod(
    x: array[Any, Dtype[_SignedInteger | np.bool_]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: bool,
) -> array[Any, Dtype[np.int64]]: ...
@overload
def prod(
    x: array[Any, Dtype[_UnsignedInteger]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.uint64]]: ...
@overload
def prod(
    x: array[_ShapeT, Dtype[_UnsignedInteger]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.uint64]]: ...
@overload
def prod(
    x: array[Any, Dtype[_UnsignedInteger]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: bool,
) -> array[Any, Dtype[np.uint64]]: ...
@overload
def prod(
    x: array[Any, Dtype[_Floating]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.float64]]: ...
@overload
def prod(
    x: array[_ShapeT, Dtype[_Floating]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.float64]]: ...
@overload
def prod(
    x: array[Any, Dtype[_Floating]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: bool,
) -> array[Any, Dtype[np.float64]]: ...
@overload
def prod(
    x: array[Any, Dtype[np.complex64]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.complex128]]: ...
@overload
def prod(
    x: array[_ShapeT, Dtype[np.complex64]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.complex128]]: ...
@overload
def prod(
    x: array[Any, Dtype[np.complex64]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: bool,
) -> array[Any, Dtype[np.complex128]]: ...
@overload
def prod(
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnySignedIntegerDType | onp.AnyBoolDType,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.int64]]: ...
@overload
def prod(
    x: array[_ShapeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnySignedIntegerDType | onp.AnyBoolDType,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.int64]]: ...
@overload
def prod(
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnySignedIntegerDType | onp.AnyBoolDType,
    keepdims: bool,
) -> array[Any, Dtype[np.int64]]: ...
@overload
def prod(
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyUnsignedIntegerDType,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.uint64]]: ...
@overload
def prod(
    x: array[_ShapeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyUnsignedIntegerDType,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.uint64]]: ...
@overload
def prod(
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyUnsignedIntegerDType,
    keepdims: bool,
) -> array[Any, Dtype[np.uint64]]: ...
@overload
def prod(
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyFloat32DType,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.float32]]: ...
@overload
def prod(
    x: array[_ShapeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyFloat32DType,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.float32]]: ...
@overload
def prod(
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyFloat32DType,
    keepdims: bool,
) -> array[Any, Dtype[np.float32]]: ...
@overload
def prod(
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyFloat64DType,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.float64]]: ...
@overload
def prod(
    x: array[_ShapeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyFloat64DType,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.float64]]: ...
@overload
def prod(
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyFloat64DType,
    keepdims: bool,
) -> array[Any, Dtype[np.float64]]: ...
@overload
def prod(
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyComplex64DType,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.complex64]]: ...
@overload
def prod(
    x: array[_ShapeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyComplex64DType,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.complex64]]: ...
@overload
def prod(
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyComplex64DType,
    keepdims: bool,
) -> array[Any, Dtype[np.complex64]]: ...
@overload
def prod(
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyComplex128DType,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.complex128]]: ...
@overload
def prod(
    x: array[_ShapeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyComplex128DType,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.complex128]]: ...
@overload
def prod(
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyComplex128DType,
    keepdims: bool,
) -> array[Any, Dtype[np.complex128]]: ...
@overload
def prod(
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: npt.DTypeLike,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.int64 | np.uint64 | _Inexact]]: ...
@overload
def prod(
    x: array[_ShapeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: npt.DTypeLike,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.int64 | np.uint64 | _Inexact]]: ...
@overload
def prod(
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: npt.DTypeLike,
    keepdims: bool,
) -> array[Any, Dtype[np.int64 | np.uint64 | _Inexact]]: ...

@overload
def std(
    x: array[_AtLeast1D, _InexactDTypeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: Literal[False] = ...,
) -> array[Any, _InexactDTypeT]: ...
@overload
def std(
    x: array[_AtLeast1D, Dtype[np.int8 | np.uint8 | np.bool_]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.float16]]: ...
@overload
def std(
    x: array[_AtLeast1D, Dtype[np.int16 | np.uint16]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.float32]]: ...
@overload
def std(
    x: array[_AtLeast1D, Dtype[np.int32 | np.uint32 | np.int64 | np.uint64]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.float64]]: ...
@overload
def std(
    x: array[_AtLeast1DT, _InexactDTypeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: Literal[True],
) -> array[_AtLeast1DT, _InexactDTypeT]: ...
@overload
def std(
    x: array[_AtLeast1DT, Dtype[np.int8 | np.uint8 | np.bool_]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: Literal[True],
) -> array[_AtLeast1DT, Dtype[np.float16]]: ...
@overload
def std(
    x: array[_AtLeast1DT, Dtype[np.int16 | np.uint16]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: Literal[True],
) -> array[_AtLeast1DT, Dtype[np.float32]]: ...
@overload
def std(
    x: array[_AtLeast1DT, Dtype[np.int32 | np.uint32 | np.int64 | np.uint64]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: Literal[True],
) -> array[_AtLeast1DT, Dtype[np.float64]]: ...
@overload
def std(
    x: array[_AtLeast1D, _InexactDTypeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: bool,
) -> array[Any, _InexactDTypeT]: ...
@overload
def std(
    x: array[_AtLeast1D, Dtype[np.int8 | np.uint8 | np.bool_]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: bool,
) -> array[Any, Dtype[np.float16]]: ...
@overload
def std(
    x: array[_AtLeast1D, Dtype[np.int16 | np.uint16]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: bool,
) -> array[Any, Dtype[np.float32]]: ...
@overload
def std(
    x: array[_AtLeast1D, Dtype[np.int32 | np.uint32 | np.int64 | np.uint64]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: bool,
) -> array[Any, Dtype[np.float64]]: ...

@overload
def sum(  # noqa: A001
    x: array[Any, Dtype[_RegularTypeT]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[_RegularTypeT]]: ...
@overload
def sum(  # noqa: A001
    x: array[_ShapeT, Dtype[_RegularTypeT]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[_RegularTypeT]]: ...
@overload
def sum(  # noqa: A001
    x: array[Any, Dtype[_RegularTypeT]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: bool,
) -> array[Any, Dtype[_RegularTypeT]]: ...
@overload
def sum(  # noqa: A001
    x: array[Any, Dtype[_SignedInteger | np.bool_]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.int64]]: ...
@overload
def sum(  # noqa: A001
    x: array[_ShapeT, Dtype[_SignedInteger | np.bool_]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.int64]]: ...
@overload
def sum(  # noqa: A001
    x: array[Any, Dtype[_SignedInteger | np.bool_]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: bool,
) -> array[Any, Dtype[np.int64]]: ...
@overload
def sum(  # noqa: A001
    x: array[Any, Dtype[_UnsignedInteger]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.uint64]]: ...
@overload
def sum(  # noqa: A001
    x: array[_ShapeT, Dtype[_UnsignedInteger]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.uint64]]: ...
@overload
def sum(  # noqa: A001
    x: array[Any, Dtype[_UnsignedInteger]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: bool,
) -> array[Any, Dtype[np.uint64]]: ...
@overload
def sum(  # noqa: A001
    x: array[Any, Dtype[_Floating]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.float64]]: ...
@overload
def sum(  # noqa: A001
    x: array[_ShapeT, Dtype[_Floating]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.float64]]: ...
@overload
def sum(  # noqa: A001
    x: array[Any, Dtype[_Floating]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: bool,
) -> array[Any, Dtype[np.float64]]: ...
@overload
def sum(  # noqa: A001
    x: array[Any, Dtype[np.complex64]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.complex128]]: ...
@overload
def sum(  # noqa: A001
    x: array[_ShapeT, Dtype[np.complex64]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.complex128]]: ...
@overload
def sum(  # noqa: A001
    x: array[Any, Dtype[np.complex64]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: None = ...,
    keepdims: bool,
) -> array[Any, Dtype[np.complex128]]: ...
@overload
def sum(  # noqa: A001
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnySignedIntegerDType | onp.AnyBoolDType,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.int64]]: ...
@overload
def sum(  # noqa: A001
    x: array[_ShapeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnySignedIntegerDType | onp.AnyBoolDType,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.int64]]: ...
@overload
def sum(  # noqa: A001
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnySignedIntegerDType | onp.AnyBoolDType,
    keepdims: bool,
) -> array[Any, Dtype[np.int64]]: ...
@overload
def sum(  # noqa: A001
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyUnsignedIntegerDType,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.uint64]]: ...
@overload
def sum(  # noqa: A001
    x: array[_ShapeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyUnsignedIntegerDType,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.uint64]]: ...
@overload
def sum(  # noqa: A001
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyUnsignedIntegerDType,
    keepdims: bool,
) -> array[Any, Dtype[np.uint64]]: ...
@overload
def sum(  # noqa: A001
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyFloat32DType,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.float32]]: ...
@overload
def sum(  # noqa: A001
    x: array[_ShapeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyFloat32DType,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.float32]]: ...
@overload
def sum(  # noqa: A001
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyFloat32DType,
    keepdims: bool,
) -> array[Any, Dtype[np.float32]]: ...
@overload
def sum(  # noqa: A001
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyFloat64DType,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.float64]]: ...
@overload
def sum(  # noqa: A001
    x: array[_ShapeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyFloat64DType,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.float64]]: ...
@overload
def sum(  # noqa: A001
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyFloat64DType,
    keepdims: bool,
) -> array[Any, Dtype[np.float64]]: ...
@overload
def sum(  # noqa: A001
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyComplex64DType,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.complex64]]: ...
@overload
def sum(  # noqa: A001
    x: array[_ShapeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyComplex64DType,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.complex64]]: ...
@overload
def sum(  # noqa: A001
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyComplex64DType,
    keepdims: bool,
) -> array[Any, Dtype[np.complex64]]: ...
@overload
def sum(  # noqa: A001
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyComplex128DType,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.complex128]]: ...
@overload
def sum(  # noqa: A001
    x: array[_ShapeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyComplex128DType,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.complex128]]: ...
@overload
def sum(  # noqa: A001
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: onp.AnyComplex128DType,
    keepdims: bool,
) -> array[Any, Dtype[np.complex128]]: ...
@overload
def sum(  # noqa: A001
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: npt.DTypeLike,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.int64 | np.uint64 | _Inexact]]: ...
@overload
def sum(  # noqa: A001
    x: array[_ShapeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: npt.DTypeLike,
    keepdims: Literal[True],
) -> array[_ShapeT, Dtype[np.int64 | np.uint64 | _Inexact]]: ...
@overload
def sum(  # noqa: A001
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    dtype: npt.DTypeLike,
    keepdims: bool,
) -> array[Any, Dtype[np.int64 | np.uint64 | _Inexact]]: ...

@overload
def var(
    x: array[_AtLeast1D, _InexactDTypeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: Literal[False] = ...,
) -> array[Any, _InexactDTypeT]: ...
@overload
def var(
    x: array[_AtLeast1DT, _InexactDTypeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: Literal[True],
) -> array[_AtLeast1DT, _InexactDTypeT]: ...
@overload
def var(
    x: array[_AtLeast1D, _InexactDTypeT],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: bool,
) -> array[Any, _InexactDTypeT]: ...
@overload
def var(
    x: array[_AtLeast1D, Dtype[_Integer | np.bool_]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: Literal[False] = ...,
) -> array[Any, Dtype[np.float64]]: ...
@overload
def var(
    x: array[_AtLeast1DT, Dtype[_Integer | np.bool_]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: Literal[True],
) -> array[_AtLeast1DT, Dtype[np.float64]]: ...
@overload
def var(
    x: array[_AtLeast1D, Dtype[_Integer | np.bool_]],
    /,
    *,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: bool,
) -> array[Any, Dtype[np.float64]]: ...

if sys.version_info >= (3, 14):
    _Floating: TypeAlias = np.floating
    _Inexact: TypeAlias = np.inexact
    _Integer: TypeAlias = np.integer
    _SignedInteger: TypeAlias = np.signedinteger
    _UnsignedInteger: TypeAlias = np.unsignedinteger
else:
    _Floating: TypeAlias = np.floating[Any]
    _Inexact: TypeAlias = np.inexact[Any, Any]
    _Integer: TypeAlias = np.integer[Any]
    _SignedInteger: TypeAlias = np.signedinteger[Any]
    _UnsignedInteger: TypeAlias = np.unsignedinteger[Any]

_AtLeast1D: TypeAlias = tuple[int, Unpack[tuple[int | None, ...]]]
_Axis: TypeAlias = int | None
_AtLeast1DT = TypeVar(
    "_AtLeast1DT",
    tuple[int],
    tuple[int, int],
    tuple[int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int, Unpack[tuple[int, ...]]],
    tuple[int, int, int, Unpack[tuple[int, ...]]],
    tuple[int, int, Unpack[tuple[int, ...]]],
    tuple[int, Unpack[tuple[int, ...]]],
    tuple[int, _Axis],
    tuple[int, _Axis, _Axis],
    tuple[int, _Axis, _Axis, _Axis],
    tuple[int, _Axis, _Axis, _Axis, Unpack[tuple[_Axis, ...]]],
    tuple[int, _Axis, _Axis, Unpack[tuple[_Axis, ...]]],
    tuple[int, _Axis, Unpack[tuple[_Axis, ...]]],
    tuple[int, Unpack[tuple[_Axis, ...]]],
)
_DTypeT = TypeVar("_DTypeT", bound=Dtype)
_InexactDTypeT = TypeVar("_InexactDTypeT", bound=Dtype[np.inexact[Any]])
_RegularTypeT = TypeVar("_RegularTypeT", np.int64, np.uint64, np.float64, np.complex128)
_ShapeT = TypeVar(
    "_ShapeT",
    tuple[()],
    tuple[int],
    tuple[int, int],
    tuple[int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int, Unpack[tuple[int, ...]]],
    tuple[int, int, int, Unpack[tuple[int, ...]]],
    tuple[int, int, Unpack[tuple[int, ...]]],
    tuple[int, Unpack[tuple[int, ...]]],
    tuple[int, ...],
    tuple[int, _Axis],
    tuple[int, _Axis, _Axis],
    tuple[int, _Axis, _Axis, _Axis],
    tuple[int, _Axis, _Axis, _Axis, Unpack[tuple[_Axis, ...]]],
    tuple[int, _Axis, _Axis, Unpack[tuple[_Axis, ...]]],
    tuple[int, _Axis, Unpack[tuple[_Axis, ...]]],
    tuple[int, Unpack[tuple[_Axis, ...]]],
)
