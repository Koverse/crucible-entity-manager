"""Type aliases shared across the package."""

import numpy as np
import numpy.typing as npt

type JSONScalar = bool | int | float | str | None
type JSONValue = JSONScalar | list[JSONValue] | dict[str, JSONValue]
type JSONObject = dict[str, JSONValue]

type FloatArray = npt.NDArray[np.float64]
