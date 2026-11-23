`geometry.hpp` contains `geometry_hypot`, a two-coordinate C++ adaptation of
the compensated `vector_norm` algorithm in CPython 3.10.12's
[`Modules/mathmodule.c`](https://github.com/python/cpython/blob/v3.10.12/Modules/mathmodule.c).
The adaptation preserves the arithmetic order and specializes the input length
to two coordinates; it replaces CPython macros and argument handling with C++
standard-library operations.

Copyright (c) 2001-2023 Python Software Foundation. All Rights Reserved.

The original CPython license and historical notices are reproduced in
[`CPython_LICENSE.txt`](CPython_LICENSE.txt).
