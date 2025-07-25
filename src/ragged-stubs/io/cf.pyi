# Copyright 2025 hingebase

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

__all__ = [
    "from_cf_contiguous",
    "from_cf_indexed",
    "to_cf_contiguous",
    "to_cf_indexed",
]

from typing import Any, TypeVar

import numpy as np
from ragged._spec_array_object import array
from ragged._typing import Dtype  # noqa: PLC2701

def to_cf_contiguous(
    x: array[tuple[int, int | None], _DTypeT],
) -> tuple[array[tuple[int], _DTypeT], array[tuple[int], Dtype[np.int64]]]: ...
def from_cf_contiguous(
    content: array[tuple[int], _DTypeT],
    counts: array[tuple[int], Dtype[np.integer[Any]]],
) -> array[tuple[int, None], _DTypeT]: ...
def to_cf_indexed(
    x: array[tuple[int, int | None], _DTypeT],
) -> tuple[array[tuple[int], _DTypeT], array[tuple[int], Dtype[np.int64]]]: ...
def from_cf_indexed(
    content: array[tuple[int], _DTypeT],
    counts: array[tuple[int], Dtype[np.integer[Any]]],
) -> array[tuple[int, None], _DTypeT]: ...

_DTypeT = TypeVar("_DTypeT", bound=Dtype)
