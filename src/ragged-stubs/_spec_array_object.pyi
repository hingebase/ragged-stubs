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

import enum
import sys
import types
from collections.abc import Callable, Iterable, Iterator, Mapping, Sized
from typing import Generic, Literal, SupportsAbs, SupportsComplex, SupportsFloat, SupportsIndex, SupportsInt, TypeAlias

import numpy as np
import optype as op
import optype.numpy as onp
from optype.dlpack import CanDLPackCompat, CanDLPackDevice
from typing_extensions import Any, CapsuleType, Self, TypeVar, Unpack, overload, override

from ._typing import Device, Dtype, Shape

class array(  # noqa: N801
    CanDLPackCompat,
    CanDLPackDevice[enum.Enum, int],
    SupportsAbs[array[_ShapeT_co, _DTypeT_co]],
    SupportsComplex,
    SupportsFloat,
    SupportsIndex,
    SupportsInt,
    Sized,
    Generic[_ShapeT_co, _DTypeT_co],
):
    @overload
    def __init__(
        self: array[_ShapeT, Dtype[_SCT]],
        obj: array[_ShapeT],
        dtype: onp.ToDType[_SCT],
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        obj: array[_ShapeT_co, _DTypeT_co],
        dtype: None = ...,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_ShapeT, Dtype[np.bool_]],
        obj: array[_ShapeT],
        dtype: onp.AnyBoolDType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_ShapeT, Dtype[np.int8]],
        obj: array[_ShapeT],
        dtype: onp.AnyInt8DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_ShapeT, Dtype[np.int16]],
        obj: array[_ShapeT],
        dtype: onp.AnyInt16DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_ShapeT, Dtype[np.int32]],
        obj: array[_ShapeT],
        dtype: onp.AnyInt32DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_ShapeT, Dtype[np.int64]],
        obj: array[_ShapeT],
        dtype: onp.AnyInt64DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_ShapeT, Dtype[np.uint8]],
        obj: array[_ShapeT],
        dtype: onp.AnyUInt8DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_ShapeT, Dtype[np.uint16]],
        obj: array[_ShapeT],
        dtype: onp.AnyUInt16DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_ShapeT, Dtype[np.uint32]],
        obj: array[_ShapeT],
        dtype: onp.AnyUInt32DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_ShapeT, Dtype[np.uint64]],
        obj: array[_ShapeT],
        dtype: onp.AnyUInt64DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_ShapeT, Dtype[np.float32]],
        obj: array[_ShapeT],
        dtype: onp.AnyFloat32DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_ShapeT, Dtype[np.float64]],
        obj: array[_ShapeT],
        dtype: onp.AnyFloat64DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_ShapeT, Dtype[np.complex64]],
        obj: array[_ShapeT],
        dtype: onp.AnyComplex64DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_ShapeT, Dtype[np.complex128]],
        obj: array[_ShapeT],
        dtype: onp.AnyComplex128DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_ShapeT],
        obj: array[_ShapeT],
        dtype: type[Any] | str,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_RegularShapeT, Dtype[_SCT]],
        obj: np.ndarray[_RegularShapeT, _NumPyDType],
        dtype: onp.ToDType[_SCT],
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_RegularShapeT, _DTypeT],
        obj: np.ndarray[_RegularShapeT, _DTypeT],
        dtype: None = ...,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_RegularShapeT, Dtype[np.bool_]],
        obj: np.ndarray[_RegularShapeT, _NumPyDType],
        dtype: onp.AnyBoolDType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_RegularShapeT, Dtype[np.int8]],
        obj: np.ndarray[_RegularShapeT, _NumPyDType],
        dtype: onp.AnyInt8DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_RegularShapeT, Dtype[np.int16]],
        obj: np.ndarray[_RegularShapeT, _NumPyDType],
        dtype: onp.AnyInt16DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_RegularShapeT, Dtype[np.int32]],
        obj: np.ndarray[_RegularShapeT, _NumPyDType],
        dtype: onp.AnyInt32DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_RegularShapeT, Dtype[np.int64]],
        obj: np.ndarray[_RegularShapeT, _NumPyDType],
        dtype: onp.AnyInt64DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_RegularShapeT, Dtype[np.uint8]],
        obj: np.ndarray[_RegularShapeT, _NumPyDType],
        dtype: onp.AnyUInt8DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_RegularShapeT, Dtype[np.uint16]],
        obj: np.ndarray[_RegularShapeT, _NumPyDType],
        dtype: onp.AnyUInt16DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_RegularShapeT, Dtype[np.uint32]],
        obj: np.ndarray[_RegularShapeT, _NumPyDType],
        dtype: onp.AnyUInt32DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_RegularShapeT, Dtype[np.uint64]],
        obj: np.ndarray[_RegularShapeT, _NumPyDType],
        dtype: onp.AnyUInt64DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_RegularShapeT, Dtype[np.float32]],
        obj: np.ndarray[_RegularShapeT, _NumPyDType],
        dtype: onp.AnyFloat32DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_RegularShapeT, Dtype[np.float64]],
        obj: np.ndarray[_RegularShapeT, _NumPyDType],
        dtype: onp.AnyFloat64DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_RegularShapeT, Dtype[np.complex64]],
        obj: np.ndarray[_RegularShapeT, _NumPyDType],
        dtype: onp.AnyComplex64DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_RegularShapeT, Dtype[np.complex128]],
        obj: np.ndarray[_RegularShapeT, _NumPyDType],
        dtype: onp.AnyComplex128DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[_RegularShapeT],
        obj: np.ndarray[_RegularShapeT, _NumPyDType],
        dtype: type[Any] | str,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[tuple[()], Dtype[_SCT]],
        obj: complex | np.number[Any] | np.timedelta64,
        dtype: onp.ToDType[_SCT],
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[tuple[()], Dtype[_NumberT]],
        obj: _NumberT,
        dtype: None = ...,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[tuple[()], Dtype[np.bool_]],
        obj: bool,
        dtype: None = ...,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[tuple[()], Dtype[np.int64]],
        obj: op.JustInt,
        dtype: None = ...,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[tuple[()], Dtype[np.float64]],
        obj: op.JustFloat,
        dtype: None = ...,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[tuple[()], Dtype[np.complex128]],
        obj: op.JustComplex,
        dtype: None = ...,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(self: array, obj: object, dtype: None = ..., device: Device | None = ..., copy: bool | None = ...) -> None: ...
    @overload
    def __init__(
        self: array[tuple[()], Dtype[np.bool_]],
        obj: complex | np.number[Any] | np.timedelta64,
        dtype: onp.AnyBoolDType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[tuple[()], Dtype[np.int8]],
        obj: complex | np.number[Any] | np.timedelta64,
        dtype: onp.AnyInt8DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[tuple[()], Dtype[np.int16]],
        obj: complex | np.number[Any] | np.timedelta64,
        dtype: onp.AnyInt16DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[tuple[()], Dtype[np.int32]],
        obj: complex | np.number[Any] | np.timedelta64,
        dtype: onp.AnyInt32DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[tuple[()], Dtype[np.int64]],
        obj: complex | np.number[Any] | np.timedelta64,
        dtype: onp.AnyInt64DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[tuple[()], Dtype[np.uint8]],
        obj: complex | np.number[Any] | np.timedelta64,
        dtype: onp.AnyUInt8DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[tuple[()], Dtype[np.uint16]],
        obj: complex | np.number[Any] | np.timedelta64,
        dtype: onp.AnyUInt16DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[tuple[()], Dtype[np.uint32]],
        obj: complex | np.number[Any] | np.timedelta64,
        dtype: onp.AnyUInt32DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[tuple[()], Dtype[np.uint64]],
        obj: complex | np.number[Any] | np.timedelta64,
        dtype: onp.AnyUInt64DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[tuple[()], Dtype[np.float32]],
        obj: complex | np.number[Any] | np.timedelta64,
        dtype: onp.AnyFloat32DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[tuple[()], Dtype[np.float64]],
        obj: complex | np.number[Any] | np.timedelta64,
        dtype: onp.AnyFloat64DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[tuple[()], Dtype[np.complex64]],
        obj: complex | np.number[Any] | np.timedelta64,
        dtype: onp.AnyComplex64DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[tuple[()], Dtype[np.complex128]],
        obj: complex | np.number[Any] | np.timedelta64,
        dtype: onp.AnyComplex128DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[tuple[()]],
        obj: complex | np.number[Any] | np.timedelta64,
        dtype: type[Any] | str,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[Any, Dtype[_SCT]],
        obj: object,
        dtype: onp.ToDType[_SCT],
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[Any, Dtype[np.bool_]],
        obj: object,
        dtype: onp.AnyBoolDType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[Any, Dtype[np.int8]],
        obj: object,
        dtype: onp.AnyInt8DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[Any, Dtype[np.int16]],
        obj: object,
        dtype: onp.AnyInt16DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[Any, Dtype[np.int32]],
        obj: object,
        dtype: onp.AnyInt32DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[Any, Dtype[np.int64]],
        obj: object,
        dtype: onp.AnyInt64DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[Any, Dtype[np.uint8]],
        obj: object,
        dtype: onp.AnyUInt8DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[Any, Dtype[np.uint16]],
        obj: object,
        dtype: onp.AnyUInt16DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[Any, Dtype[np.uint32]],
        obj: object,
        dtype: onp.AnyUInt32DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[Any, Dtype[np.uint64]],
        obj: object,
        dtype: onp.AnyUInt64DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[Any, Dtype[np.float32]],
        obj: object,
        dtype: onp.AnyFloat32DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[Any, Dtype[np.float64]],
        obj: object,
        dtype: onp.AnyFloat64DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[Any, Dtype[np.complex64]],
        obj: object,
        dtype: onp.AnyComplex64DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array[Any, Dtype[np.complex128]],
        obj: object,
        dtype: onp.AnyComplex128DType,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: array,
        obj: object,
        dtype: type[Any] | str,
        device: Device | None = ...,
        copy: bool | None = ...,
    ) -> None: ...

    @override
    def __str__(self) -> str: ...  # noqa: PYI029
    @override
    def __repr__(self) -> str: ...  # noqa: PYI029
    def __contains__(self, other: complex) -> bool: ...
    @override
    def __len__(self) -> int: ...
    def __iter__(self: array[tuple[int, Unpack[tuple[_Axis, ...]]], _DTypeT]) -> Iterator[array[Any, _DTypeT]]: ...

    @overload
    def item(self: array[tuple[()], Dtype[_ComplexFloating]]) -> complex: ...
    @overload
    def item(self: array[tuple[()], Dtype[_Floating]]) -> float: ...
    @overload
    def item(self: array[tuple[()], Dtype[_Integer]]) -> int: ...
    @overload
    def item(self: array[tuple[()], Dtype[np.bool_]]) -> bool: ...
    @overload
    def item(self: array[tuple[()]]) -> complex: ...

    @overload
    def tolist(self: array[Any, Dtype[_ComplexFloating]]) -> complex | onp.SequenceND[complex]: ...
    @overload
    def tolist(self: array[Any, Dtype[_Floating]]) -> float | onp.SequenceND[float]: ...
    @overload
    def tolist(self: array[Any, Dtype[_Integer]]) -> int | onp.SequenceND[int]: ...
    @overload
    def tolist(self: array[Any, Dtype[np.bool_]]) -> bool | onp.SequenceND[bool]: ...
    @overload
    def tolist(self: array[tuple[()]]) -> complex: ...
    @overload
    def tolist(self: array[tuple[int, Unpack[tuple[_Axis, ...]]]]) -> onp.SequenceND[complex]: ...

    @property
    def dtype(self) -> _DTypeT_co: ...
    @property
    def device(self) -> Device: ...
    @property
    def mT(self: array[_AtLeast2DT, _DTypeT]) -> array[_AtLeast2DT, _DTypeT]: ...  # noqa: N802
    @property
    def ndim(self) -> int: ...
    @property
    def shape(self) -> _ShapeT_co: ...
    @property
    def size(self) -> int: ...
    @property
    def T(self: array[_2DT, _DTypeT]) -> array[_2DT, _DTypeT]: ...  # noqa: N802
    @override
    def __abs__(self) -> Self: ...

    @overload
    def __add__(self, other: array | np.ndarray[Any, Dtype]) -> array: ...
    @overload
    def __add__(self: array[_ShapeT], other: complex | np.number[Any]) -> array[_ShapeT]: ...
    @overload
    def __add__(self, other: object) -> array: ...

    @overload
    def __and__(self: _BoolOrIntArray, other: _BoolOrIntArray | np.ndarray[Any, _BoolOrIntDType]) -> _BoolOrIntArray: ...
    @overload
    def __and__(self: array[_ShapeT, _BoolOrIntDType], other: int | _Integer) -> array[_ShapeT, _BoolOrIntDType]: ...
    @overload
    def __and__(self: _BoolOrIntArray, other: object) -> _BoolOrIntArray: ...

    # https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__array_namespace__.html
    def __array_namespace__(self, *, api_version: str | None = ...) -> Any: ...  # noqa: ANN401

    # These special methods are not meant to be called explicitly,
    # and we haven't seen any necessity to return `Any`
    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "at"],
        *inputs: object,
        **kwargs: object,
    ) -> object: ...
    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Iterable[type[Any]],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> object: ...

    def __bool__(self) -> bool: ...
    @override
    def __complex__(self) -> complex: ...

    # https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html
    @override
    def __dlpack__(self, *, stream: int | Any | None = ...) -> CapsuleType: ...

    @override
    def __dlpack_device__(self) -> tuple[enum.Enum, int]: ...

    @overload
    @override
    def __eq__(  # pyrefly: ignore[bad-override]
        self,
        other: array | np.ndarray[Any, Dtype],
        /,
    ) -> array[Any, Dtype[np.bool_]]: ...
    @overload
    def __eq__(self: array[_ShapeT], other: complex | np.number[Any], /) -> array[_ShapeT, Dtype[np.bool_]]: ...
    @overload
    def __eq__(self, other: object, /) -> array[Any, Dtype[np.bool_]]: ...  # pyright: ignore[reportIncompatibleMethodOverride]  # ty: ignore[invalid-method-override]

    @override
    def __float__(self) -> float: ...

    @overload
    def __floordiv__(
        self: array[Any, _RealDType],
        other: array[Any, _RealDType] | np.ndarray[Any, _RealDType],
        /,
    ) -> array[Any, Dtype[_RealNumber]]: ...
    @overload
    def __floordiv__(self: array[_ShapeT, _RealDType], other: float | _RealNumber, /) -> array[_ShapeT, Dtype[_RealNumber]]: ...
    @overload
    def __floordiv__(self: array[Any, _RealDType], other: object, /) -> array[Any, Dtype[_RealNumber]]: ...

    @overload
    def __ge__(self, other: array | np.ndarray[Any, Dtype], /) -> array[Any, Dtype[np.bool_]]: ...
    @overload
    def __ge__(self: array[_ShapeT], other: complex | np.number[Any], /) -> array[_ShapeT, Dtype[np.bool_]]: ...
    @overload
    def __ge__(self, other: object, /) -> array[Any, Dtype[np.bool_]]: ...

    def __getitem__(self, key: _GetSliceKey | tuple[_GetSliceKey, ...] | _BoolOrIntArray) -> array[Any, _DTypeT_co]: ...

    @overload
    def __gt__(self, other: array | np.ndarray[Any, Dtype], /) -> array[Any, Dtype[np.bool_]]: ...
    @overload
    def __gt__(self: array[_ShapeT], other: complex | np.number[Any], /) -> array[_ShapeT, Dtype[np.bool_]]: ...
    @overload
    def __gt__(self, other: object, /) -> array[Any, Dtype[np.bool_]]: ...

    @override
    def __index__(self) -> int: ...
    @override
    def __int__(self) -> int: ...
    def __invert__(self: array[_ShapeT, _BoolOrIntDTypeT]) -> array[_ShapeT, _BoolOrIntDTypeT]: ...

    @overload
    def __le__(self, other: array | np.ndarray[Any, Dtype], /) -> array[Any, Dtype[np.bool_]]: ...
    @overload
    def __le__(self: array[_ShapeT], other: complex | np.number[Any], /) -> array[_ShapeT, Dtype[np.bool_]]: ...
    @overload
    def __le__(self, other: object, /) -> array[Any, Dtype[np.bool_]]: ...

    @overload
    def __lshift__(
        self: _BoolOrIntArray,
        other: _BoolOrIntArray | np.ndarray[Any, _BoolOrIntDType],
    ) -> array[Any, Dtype[_Integer]]: ...
    @overload
    def __lshift__(self: array[_ShapeT, _BoolOrIntDType], other: int | _Integer) -> array[_ShapeT, Dtype[_Integer]]: ...
    @overload
    def __lshift__(self: _BoolOrIntArray, other: object) -> array[Any, Dtype[_Integer]]: ...

    @overload
    def __lt__(self, other: array | np.ndarray[Any, Dtype], /) -> array[Any, Dtype[np.bool_]]: ...
    @overload
    def __lt__(self: array[_ShapeT], other: complex | np.number[Any], /) -> array[_ShapeT, Dtype[np.bool_]]: ...
    @overload
    def __lt__(self, other: object, /) -> array[Any, Dtype[np.bool_]]: ...

    def __matmul__(self: array[_AtLeast2D], other: array[_AtLeast2D]) -> array[_AtLeast2D]: ...

    @overload
    def __mod__(
        self: array[Any, _RealDType],
        other: array[Any, _RealDType] | np.ndarray[Any, _RealDType],
        /,
    ) -> array[Any, Dtype[_RealNumber]]: ...
    @overload
    def __mod__(self: array[_ShapeT, _RealDType], other: float | _RealNumber, /) -> array[_ShapeT, Dtype[_RealNumber]]: ...
    @overload
    def __mod__(self: array[Any, _RealDType], other: object, /) -> array[Any, Dtype[_RealNumber]]: ...

    @overload
    def __mul__(self, other: array | np.ndarray[Any, Dtype]) -> array: ...
    @overload
    def __mul__(self: array[_ShapeT], other: complex | np.number[Any]) -> array[_ShapeT]: ...
    @overload
    def __mul__(self, other: object) -> array: ...

    @overload
    @override
    def __ne__(  # pyrefly: ignore[bad-override]
        self,
        other: array | np.ndarray[Any, Dtype],
        /,
    ) -> array[Any, Dtype[np.bool_]]: ...
    @overload
    def __ne__(self: array[_ShapeT], other: complex | np.number[Any], /) -> array[_ShapeT, Dtype[np.bool_]]: ...
    @overload
    def __ne__(self, other: object, /) -> array[Any, Dtype[np.bool_]]: ...  # pyright: ignore[reportIncompatibleMethodOverride]  # ty: ignore[invalid-method-override]

    def __neg__(self: array[_ShapeT, _NumericDTypeT]) -> array[_ShapeT, _NumericDTypeT]: ...

    @overload
    def __or__(self: _BoolOrIntArray, other: _BoolOrIntArray | np.ndarray[Any, _BoolOrIntDType]) -> _BoolOrIntArray: ...
    @overload
    def __or__(self: array[_ShapeT, _BoolOrIntDType], other: int | _Integer) -> array[_ShapeT, _BoolOrIntDType]: ...
    @overload
    def __or__(self: _BoolOrIntArray, other: object) -> _BoolOrIntArray: ...

    def __pos__(self: array[_ShapeT, _NumericDTypeT]) -> array[_ShapeT, _NumericDTypeT]: ...

    @overload
    def __pow__(self, other: array | np.ndarray[Any, Dtype]) -> array[Any, _NumericDType]: ...
    @overload
    def __pow__(self: array[_ShapeT], other: complex | np.number[Any]) -> array[_ShapeT, _NumericDType]: ...
    @overload
    def __pow__(self, other: object) -> array[Any, _NumericDType]: ...

    @overload
    def __rshift__(
        self: _BoolOrIntArray,
        other: _BoolOrIntArray | np.ndarray[Any, _BoolOrIntDType],
    ) -> array[Any, Dtype[_Integer]]: ...
    @overload
    def __rshift__(self: array[_ShapeT, _BoolOrIntDType], other: int | _Integer) -> array[_ShapeT, Dtype[_Integer]]: ...
    @overload
    def __rshift__(self: _BoolOrIntArray, other: object) -> array[Any, Dtype[_Integer]]: ...

    def __setitem__(self, key: _ToIndex | tuple[_ToIndex, ...] | _BoolOrIntArray, value: object) -> None: ...

    @overload
    def __sub__(self: array[Any, _NumericDType], other: array | np.ndarray[Any, Dtype]) -> array[Any, _NumericDType]: ...
    @overload
    def __sub__(self, other: array[Any, _NumericDType] | np.ndarray[Any, _NumericDType]) -> array[Any, _NumericDType]: ...
    @overload
    def __sub__(self: array[_ShapeT, _NumericDType], other: complex | np.number[Any]) -> array[_ShapeT, _NumericDType]: ...
    @overload
    def __sub__(
        self: array[_ShapeT],
        other: op.JustInt | op.JustFloat | op.JustComplex | np.number[Any],
    ) -> array[_ShapeT, _NumericDType]: ...
    @overload
    def __sub__(self, other: object) -> array[Any, _NumericDType]: ...

    @overload
    def __truediv__(self, other: array | np.ndarray[Any, Dtype], /) -> array[Any, _InexactDType]: ...
    @overload
    def __truediv__(self: array[_ShapeT], other: complex | np.number[Any], /) -> array[_ShapeT, _InexactDType]: ...
    @overload
    def __truediv__(self, other: object, /) -> array[Any, _InexactDType]: ...

    @overload
    def __xor__(self: _BoolOrIntArray, other: _BoolOrIntArray | np.ndarray[Any, _BoolOrIntDType]) -> _BoolOrIntArray: ...
    @overload
    def __xor__(self: array[_ShapeT, _BoolOrIntDType], other: int | _Integer) -> array[_ShapeT, _BoolOrIntDType]: ...
    @overload
    def __xor__(self: _BoolOrIntArray, other: object) -> _BoolOrIntArray: ...

    # https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.to_device.html
    def to_device(self, device: Device, /, *, stream: int | Any | None = ...) -> Self: ...  # noqa: ANN401

    @property
    def at(self) -> _AtIndexHelper[_ShapeT_co, _DTypeT_co]: ...

    @overload
    def __iadd__(self, other: array | np.ndarray[Any, Dtype]) -> array: ...
    @overload
    def __iadd__(self: array[_ShapeT], other: complex | np.number[Any]) -> array[_ShapeT]: ...
    @overload
    def __iadd__(self, other: object) -> array: ...

    @overload
    def __isub__(self: array[Any, _NumericDType], other: array | np.ndarray[Any, Dtype]) -> array[Any, _NumericDType]: ...
    @overload
    def __isub__(self: array, other: array[Any, _NumericDType] | np.ndarray[Any, _NumericDType]) -> array[Any, _NumericDType]: ...
    @overload
    def __isub__(self: array[_ShapeT, _NumericDType], other: complex | np.number[Any]) -> array[_ShapeT, _NumericDType]: ...
    @overload
    def __isub__(
        self: array[_ShapeT],
        other: op.JustInt | op.JustFloat | op.JustComplex | np.number[Any],
    ) -> array[_ShapeT, _NumericDType]: ...
    @overload
    def __isub__(self, other: object) -> array[Any, _NumericDType]: ...

    @overload
    def __imul__(self, other: array | np.ndarray[Any, Dtype]) -> array: ...
    @overload
    def __imul__(self: array[_ShapeT], other: complex | np.number[Any]) -> array[_ShapeT]: ...
    @overload
    def __imul__(self, other: object) -> array: ...

    @overload
    def __itruediv__(self, other: array | np.ndarray[Any, Dtype], /) -> array[Any, _InexactDType]: ...
    @overload
    def __itruediv__(self: array[_ShapeT], other: complex | np.number[Any], /) -> array[_ShapeT, _InexactDType]: ...
    @overload
    def __itruediv__(self, other: object, /) -> array[Any, _InexactDType]: ...

    @overload
    def __ifloordiv__(
        self: array[Any, _RealDType],
        other: array[Any, _RealDType] | np.ndarray[Any, _RealDType],
        /,
    ) -> array[Any, Dtype[_RealNumber]]: ...
    @overload
    def __ifloordiv__(self: array[_ShapeT, _RealDType], other: float | _RealNumber, /) -> array[_ShapeT, Dtype[_RealNumber]]: ...
    @overload
    def __ifloordiv__(self: array[Any, _RealDType], other: object, /) -> array[Any, Dtype[_RealNumber]]: ...

    @overload
    def __ipow__(self, other: array | np.ndarray[Any, Dtype]) -> array[Any, _NumericDType]: ...
    @overload
    def __ipow__(self: array[_ShapeT], other: complex | np.number[Any]) -> array[_ShapeT, _NumericDType]: ...
    @overload
    def __ipow__(self, other: object) -> array[Any, _NumericDType]: ...

    @overload
    def __imod__(
        self: array[Any, _RealDType],
        other: array[Any, _RealDType] | np.ndarray[Any, _RealDType],
        /,
    ) -> array[Any, Dtype[_RealNumber]]: ...
    @overload
    def __imod__(self: array[_ShapeT, _RealDType], other: float | _RealNumber, /) -> array[_ShapeT, Dtype[_RealNumber]]: ...
    @overload
    def __imod__(self: array[Any, _RealDType], other: object, /) -> array[Any, Dtype[_RealNumber]]: ...

    def __imatmul__(self: array[_AtLeast2D], other: array[_AtLeast2D]) -> array[_AtLeast2D]: ...

    @overload
    def __iand__(self: _BoolOrIntArray, other: _BoolOrIntArray | np.ndarray[Any, _BoolOrIntDType]) -> _BoolOrIntArray: ...
    @overload
    def __iand__(self: array[_ShapeT, _BoolOrIntDType], other: int | _Integer) -> array[_ShapeT, _BoolOrIntDType]: ...
    @overload
    def __iand__(self: _BoolOrIntArray, other: object) -> _BoolOrIntArray: ...

    @overload
    def __ior__(self: _BoolOrIntArray, other: _BoolOrIntArray | np.ndarray[Any, _BoolOrIntDType]) -> _BoolOrIntArray: ...
    @overload
    def __ior__(self: array[_ShapeT, _BoolOrIntDType], other: int | _Integer) -> array[_ShapeT, _BoolOrIntDType]: ...
    @overload
    def __ior__(self: _BoolOrIntArray, other: object) -> _BoolOrIntArray: ...

    @overload
    def __ixor__(self: _BoolOrIntArray, other: _BoolOrIntArray | np.ndarray[Any, _BoolOrIntDType]) -> _BoolOrIntArray: ...
    @overload
    def __ixor__(self: array[_ShapeT, _BoolOrIntDType], other: int | _Integer) -> array[_ShapeT, _BoolOrIntDType]: ...
    @overload
    def __ixor__(self: _BoolOrIntArray, other: object) -> _BoolOrIntArray: ...

    @overload
    def __ilshift__(
        self: _BoolOrIntArray,
        other: _BoolOrIntArray | np.ndarray[Any, _BoolOrIntDType],
    ) -> array[Any, Dtype[_Integer]]: ...
    @overload
    def __ilshift__(self: array[_ShapeT, _BoolOrIntDType], other: int | _Integer) -> array[_ShapeT, Dtype[_Integer]]: ...
    @overload
    def __ilshift__(self: _BoolOrIntArray, other: object) -> array[Any, Dtype[_Integer]]: ...

    @overload
    def __irshift__(
        self: _BoolOrIntArray,
        other: _BoolOrIntArray | np.ndarray[Any, _BoolOrIntDType],
    ) -> array[Any, Dtype[_Integer]]: ...
    @overload
    def __irshift__(self: array[_ShapeT, _BoolOrIntDType], other: int | _Integer) -> array[_ShapeT, Dtype[_Integer]]: ...
    @overload
    def __irshift__(self: _BoolOrIntArray, other: object) -> array[Any, Dtype[_Integer]]: ...

    __radd__ = __add__
    __rmul__ = __mul__
    __rand__ = __and__
    __ror__ = __or__
    __rxor__ = __xor__

    @overload
    def __rsub__(self: array[Any, _NumericDType], other: array | np.ndarray[Any, Dtype]) -> array[Any, _NumericDType]: ...
    @overload
    def __rsub__(self, other: array[Any, _NumericDType] | np.ndarray[Any, _NumericDType]) -> array[Any, _NumericDType]: ...
    @overload
    def __rsub__(self: array[_ShapeT, _NumericDType], other: complex | np.number[Any]) -> array[_ShapeT, _NumericDType]: ...
    @overload
    def __rsub__(
        self: array[_ShapeT],
        other: op.JustInt | op.JustFloat | op.JustComplex | np.number[Any],
    ) -> array[_ShapeT, _NumericDType]: ...
    @overload
    def __rsub__(self, other: object) -> array[Any, _NumericDType]: ...

    @overload
    def __rtruediv__(self, other: array | np.ndarray[Any, Dtype], /) -> array[Any, _InexactDType]: ...
    @overload
    def __rtruediv__(self: array[_ShapeT], other: complex | np.number[Any], /) -> array[_ShapeT, _InexactDType]: ...
    @overload
    def __rtruediv__(self, other: object, /) -> array[Any, _InexactDType]: ...

    @overload
    def __rfloordiv__(
        self: array[Any, _RealDType],
        other: array[Any, _RealDType] | np.ndarray[Any, _RealDType],
        /,
    ) -> array[Any, Dtype[_RealNumber]]: ...
    @overload
    def __rfloordiv__(self: array[_ShapeT, _RealDType], other: float | _RealNumber, /) -> array[_ShapeT, Dtype[_RealNumber]]: ...
    @overload
    def __rfloordiv__(self: array[Any, _RealDType], other: object, /) -> array[Any, Dtype[_RealNumber]]: ...

    @overload
    def __rpow__(self, other: array | np.ndarray[Any, Dtype]) -> array[Any, _NumericDType]: ...
    @overload
    def __rpow__(self: array[_ShapeT], other: complex | np.number[Any]) -> array[_ShapeT, _NumericDType]: ...
    @overload
    def __rpow__(self, other: object) -> array[Any, _NumericDType]: ...

    @overload
    def __rmod__(
        self: array[Any, _RealDType],
        other: array[Any, _RealDType] | np.ndarray[Any, _RealDType],
        /,
    ) -> array[Any, Dtype[_RealNumber]]: ...
    @overload
    def __rmod__(self: array[_ShapeT, _RealDType], other: float | _RealNumber, /) -> array[_ShapeT, Dtype[_RealNumber]]: ...
    @overload
    def __rmod__(self: array[Any, _RealDType], other: object, /) -> array[Any, Dtype[_RealNumber]]: ...

    @overload
    def __rlshift__(
        self: _BoolOrIntArray,
        other: _BoolOrIntArray | np.ndarray[Any, _BoolOrIntDType],
    ) -> array[Any, Dtype[_Integer]]: ...
    @overload
    def __rlshift__(self: array[_ShapeT, _BoolOrIntDType], other: int | _Integer) -> array[_ShapeT, Dtype[_Integer]]: ...
    @overload
    def __rlshift__(self: _BoolOrIntArray, other: object) -> array[Any, Dtype[_Integer]]: ...

    @overload
    def __rrshift__(
        self: _BoolOrIntArray,
        other: _BoolOrIntArray | np.ndarray[Any, _BoolOrIntDType],
    ) -> array[Any, Dtype[_Integer]]: ...
    @overload
    def __rrshift__(self: array[_ShapeT, _BoolOrIntDType], other: int | _Integer) -> array[_ShapeT, Dtype[_Integer]]: ...
    @overload
    def __rrshift__(self: _BoolOrIntArray, other: object) -> array[Any, Dtype[_Integer]]: ...

    def __rmatmul__(self: array[_AtLeast2D], other: array[_AtLeast2D]) -> array[_AtLeast2D]: ...

class _AtIndexHelper(Generic[_ShapeT_co, _DTypeT_co]):
    def __init__(self, x: array[_ShapeT_co, _DTypeT_co]) -> None: ...
    def __getitem__(
        self,
        idx: _GetSliceKey | tuple[_GetSliceKey, ...] | _BoolOrIntArray,
    ) -> _AtIndexer[_ShapeT_co, _DTypeT_co]: ...

class _AtIndexer(Generic[_ShapeT_co, _DTypeT_co]):
    def __init__(
        self,
        x: array[_ShapeT_co, _DTypeT_co],
        idx: _GetSliceKey | tuple[_GetSliceKey, ...] | _BoolOrIntArray,
    ) -> None: ...

    @overload
    def set(self: _AtIndexer[_AtLeast1DT, _DTypeT], val: object) -> array[_AtLeast1DT, _DTypeT]: ...
    @overload
    def set(self: _AtIndexer[tuple[()], _DTypeT], val: object) -> array[tuple[Literal[1]], _DTypeT]: ...

    @overload
    def add(self: _AtIndexer[_AtLeast1DT, _DTypeT], val: object) -> array[_AtLeast1DT, _DTypeT]: ...
    @overload
    def add(self: _AtIndexer[tuple[()], _DTypeT], val: object) -> array[tuple[Literal[1]], _DTypeT]: ...

    @overload
    def subtract(self: _AtIndexer[_AtLeast1DT, _DTypeT], val: object) -> array[_AtLeast1DT, _DTypeT]: ...
    @overload
    def subtract(self: _AtIndexer[tuple[()], _DTypeT], val: object) -> array[tuple[Literal[1]], _DTypeT]: ...

    @overload
    def multiply(self: _AtIndexer[_AtLeast1DT, _DTypeT], val: object) -> array[_AtLeast1DT, _DTypeT]: ...
    @overload
    def multiply(self: _AtIndexer[tuple[()], _DTypeT], val: object) -> array[tuple[Literal[1]], _DTypeT]: ...

    @overload
    def divide(self: _AtIndexer[_AtLeast1DT, _DTypeT], val: object) -> array[_AtLeast1DT, _DTypeT]: ...
    @overload
    def divide(self: _AtIndexer[tuple[()], _DTypeT], val: object) -> array[tuple[Literal[1]], _DTypeT]: ...

    @overload
    def power(self: _AtIndexer[_AtLeast1DT, _DTypeT], val: object) -> array[_AtLeast1DT, _DTypeT]: ...
    @overload
    def power(self: _AtIndexer[tuple[()], _DTypeT], val: object) -> array[tuple[Literal[1]], _DTypeT]: ...

    @overload
    def min(self: _AtIndexer[_AtLeast1DT, _DTypeT], val: object) -> array[_AtLeast1DT, _DTypeT]: ...
    @overload
    def min(self: _AtIndexer[tuple[()], _DTypeT], val: object) -> array[tuple[Literal[1]], _DTypeT]: ...

    @overload
    def max(self: _AtIndexer[_AtLeast1DT, _DTypeT], val: object) -> array[_AtLeast1DT, _DTypeT]: ...
    @overload
    def max(self: _AtIndexer[tuple[()], _DTypeT], val: object) -> array[tuple[Literal[1]], _DTypeT]: ...

if sys.version_info >= (3, 14):
    _ComplexFloating: TypeAlias = np.complexfloating
    _Floating: TypeAlias = np.floating
    _Integer: TypeAlias = np.integer
else:
    _ComplexFloating: TypeAlias = np.complexfloating[Any, Any]
    _Floating: TypeAlias = np.floating[Any]
    _Integer: TypeAlias = np.integer[Any]

_Axis: TypeAlias = int | None
_2DT = TypeVar("_2DT", tuple[int, int], tuple[int, _Axis])
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
_AtLeast2D: TypeAlias = tuple[int, int | None, Unpack[tuple[int | None, ...]]]
_AtLeast2DT = TypeVar(
    "_AtLeast2DT",
    tuple[int, int],
    tuple[int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int, Unpack[tuple[int, ...]]],
    tuple[int, int, int, Unpack[tuple[int, ...]]],
    tuple[int, int, Unpack[tuple[int, ...]]],
    tuple[int, _Axis],
    tuple[int, _Axis, _Axis],
    tuple[int, _Axis, _Axis, _Axis],
    tuple[int, _Axis, _Axis, _Axis, Unpack[tuple[_Axis, ...]]],
    tuple[int, _Axis, _Axis, Unpack[tuple[_Axis, ...]]],
    tuple[int, _Axis, Unpack[tuple[_Axis, ...]]],
)
_BoolOrIntDType: TypeAlias = Dtype[np.bool_ | _Integer]
_BoolOrIntArray: TypeAlias = array[Any, _BoolOrIntDType]
_BoolOrIntDTypeT = TypeVar("_BoolOrIntDTypeT", bound=_BoolOrIntDType)
_DTypeT = TypeVar("_DTypeT", bound=Dtype)
_DTypeT_co = TypeVar("_DTypeT_co", bound=Dtype, covariant=True, default=np.dtype[Any])
_GetSliceKey: TypeAlias = int | slice | types.EllipsisType | None
_InexactDType: TypeAlias = Dtype[np.inexact[Any]]
_NumberT = TypeVar("_NumberT", bound=np.number[Any])
_NumericDType: TypeAlias = Dtype[np.number[Any]]
_NumericDTypeT = TypeVar("_NumericDTypeT", bound=_NumericDType)
_NumPyDType: TypeAlias = np.dtype[np.bool_ | np.number[Any] | np.datetime64 | np.timedelta64]
_RealNumber: TypeAlias = _Integer | _Floating
_RealDType: TypeAlias = Dtype[np.bool_ | _RealNumber]
_RegularShapeT = TypeVar("_RegularShapeT", bound=tuple[int, ...])
_SCT = TypeVar("_SCT", bound=np.bool_ | np.number[Any])
_ShapeT = TypeVar("_ShapeT", bound=Shape)
_ShapeT_co = TypeVar("_ShapeT_co", bound=Shape, covariant=True, default=tuple[Any, ...])
_ToIndex: TypeAlias = SupportsIndex | slice | types.EllipsisType | onp.ToIntND | None
