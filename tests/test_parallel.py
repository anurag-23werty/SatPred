import math

import pytest

from sathealth.utils.parallel import parallel_map


def test_parallel_map_keeps_order_in_both_modes() -> None:
    items = [1.0, 4.0, 9.0, 16.0]
    assert list(parallel_map(math.sqrt, items, jobs=1)) == [1.0, 2.0, 3.0, 4.0]
    assert list(parallel_map(math.sqrt, items, jobs=2)) == [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(ValueError, match="jobs"):
        list(parallel_map(math.sqrt, items, jobs=0))
