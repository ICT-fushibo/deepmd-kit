"""CPU contract checks for the DPA4 Opt4 route."""

import pytest

from deepmd.md_stages.dpa4 import opt4


def test_opt4_rejects_other_route() -> None:
    with pytest.raises(ValueError, match="DPA4 Opt4 route"):
        opt4.run_md(type("Request", (), {"model": "dpa3", "stage": "opt4"})())
