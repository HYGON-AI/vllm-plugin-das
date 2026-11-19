# SPDX-License-Identifier: Apache-2.0
"""Check the DSpark compatibility contract at the Adapter dependency floor."""
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace as NS

import pytest

from hcu_vllm_das import dspark_capacity


def _config(anchor, budget):
    return NS(
        speculative_config=NS(
            num_speculative_tokens=5,
            draft_model_config=NS(hf_config=NS(sample_from_anchor=anchor)),
        ),
        scheduler_config=NS(max_num_seqs=16, max_num_batched_tokens=budget),
    )


@pytest.mark.parametrize('anchor, query', [(True, 5), (False, 6)])
def test_existing_dspark_capacity_without_adapter_helper(monkeypatch, anchor, query):
    monkeypatch.setitem(sys.modules, 'hcu_adapter.dspark_capacity', None)
    legacy = runpy.run_path(dspark_capacity.__file__)['token_capacity']
    config = _config(anchor, 128)
    expected = dict(query_per_request=query, query_rows=16 * query, verify_rows=96, budget=128)
    assert vars(legacy(config)) == vars(dspark_capacity.token_capacity(config)) == expected
    with pytest.raises(ValueError, match='exceeds original budget'):
        legacy(_config(anchor, 95))


def test_pcp_dspark_requires_the_companion_adapter_kernel(monkeypatch):
    monkeypatch.setitem(sys.modules, 'hcu_adapter.kernels.pcp_dspark', None)
    facade = Path(dspark_capacity.__file__).with_name('pcp_dspark.py')
    with pytest.raises(RuntimeError, match='companion hcu-adapter PCP DSpark'):
        runpy.run_path(str(facade))
