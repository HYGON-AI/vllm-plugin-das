# SPDX-License-Identifier: Apache-2.0
"""Native CPU differential coverage without importing vLLM or initializing HCU."""
import importlib.util
import os
from pathlib import Path
import random
import unittest

from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers


SOURCE = Path(os.environ.get(
    "HY4_DETOKENIZER_TAIL_SOURCE",
    str(Path(__file__).resolve().parents[2] /
        "vllm_hcu/runtime_compat/bytelevel_detokenizer.py"),
))


def tokenizer():
    vocab = {c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens([AddedToken("[SPECIAL]", special=True)])
    return backend


def replay(backend, ids, output, skip, burst=1):
    stream = decoders.DecodeStream(ids=ids, skip_special_tokens=skip)
    return [stream.step(backend, output[i:i+burst] if burst > 1 else output[i])
            for i in range(0, len(output), burst)]


class RecordingTokenizer:
    """Count real native decode work without changing ByteLevel semantics."""

    def __init__(self, backend):
        self.backend = backend
        self.decoder = backend.decoder
        self.decode_sizes = []

    def decode(self, ids, **kwargs):
        self.decode_sizes.append(len(ids))
        return self.backend.decode(ids, **kwargs)


class ByteLevelPromptTailTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SOURCE.is_file(), "prompt-tail helper has not been implemented")
        spec = importlib.util.spec_from_file_location("tail_under_test", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.choose = module.choose_prompt_tail
        self.backend = tokenizer()

    def test_long_clean_prompt_uses_bounded_context_without_mutating_ids(self):
        ids = self.backend.encode("ordinary prefix " * 100).ids
        before = ids[:]
        tail = self.choose(self.backend, ids, True)
        self.assertEqual(len(tail), 32)
        self.assertEqual(ids, before)
        output = self.backend.encode("你好 😀 abc").ids
        for skip in [True, False]:
            self.assertEqual(replay(self.backend, ids, output, skip),
                             replay(self.backend, tail, output, skip))

    def test_partial_character_behind_long_skipped_special_run_falls_back(self):
        prefix = self.backend.encode("ordinary prefix " * 100).ids
        char = self.backend.encode("😀").ids
        special = self.backend.token_to_id("[SPECIAL]")
        ids = prefix + char[:1] + [special] * 65
        output = char[1:] + self.backend.encode("abc").ids
        for skip in [True, False]:
            tail = self.choose(self.backend, ids, skip)
            self.assertEqual(replay(self.backend, ids, output, skip),
                             replay(self.backend, tail, output, skip))
        self.assertIs(self.choose(self.backend, ids, True), ids)

    def test_empty_skipped_tail_widens_until_it_contains_real_context(self):
        ids = self.backend.encode("abc " * 100).ids
        ids += [self.backend.token_to_id("[SPECIAL]")] * 40
        tail = self.choose(self.backend, ids, True)
        self.assertGreater(len(tail), 40)
        output = self.backend.encode("你好 abc").ids
        self.assertEqual(replay(self.backend, ids, output, True),
                         replay(self.backend, tail, output, True))

    def test_short_empty_and_missing_prompt_keep_original_object(self):
        for ids in [None, [], [1, 2, 3]]:
            self.assertIs(self.choose(self.backend, ids, True), ids)

    def test_literal_replacement_character_keeps_conservative_fallback(self):
        ids = self.backend.encode("ordinary prefix " * 100 + "\ufffd end").ids
        self.assertIs(self.choose(self.backend, ids, True), ids)

    def test_three_token_character_finds_adjacent_clean_boundary(self):
        char = self.backend.encode("汉").ids
        self.assertEqual(len(char), 3)
        ids = char * 28333
        self.assertEqual(self.backend.decode(ids), "汉" * 28333)
        self.assertIn("\ufffd", self.backend.decode(ids[-32:]))
        self.assertNotIn("\ufffd", self.backend.decode(ids[-33:]))
        output = self.backend.encode("你好 😀 abc").ids
        for skip in [True, False]:
            with self.subTest(skip=skip):
                recorded = RecordingTokenizer(self.backend)
                tail = self.choose(recorded, ids, skip)
                self.assertEqual(len(tail), 33)
                self.assertEqual(recorded.decode_sizes, [32, 33])
                for burst in [1, 3, 7]:
                    self.assertEqual(replay(self.backend, ids, output, skip, burst),
                                     replay(self.backend, tail, output, skip, burst))

    def test_fallback_decode_work_has_a_prompt_length_independent_budget(self):
        special = self.backend.token_to_id("[SPECIAL]")
        partial_char = self.backend.encode("😀").ids[:1]
        for repeats in [28333, 56666]:
            prefix = self.backend.encode("abc").ids * repeats
            cases = {
                "literal_replacement": prefix + self.backend.encode("\ufffd end").ids,
                "incomplete_character": prefix + partial_char,
                "long_skipped_run": prefix + [special] * 4096,
            }
            for name, ids in cases.items():
                with self.subTest(case=name, prompt_tokens=len(ids)):
                    recorded = RecordingTokenizer(self.backend)
                    self.assertTrue(self.choose(recorded, ids, True) is ids,
                                    "budget-exhausted selection must keep the original IDs")
                    self.assertLessEqual(len(recorded.decode_sizes), 8)
                    self.assertLessEqual(sum(recorded.decode_sizes), 512)
                    self.assertLessEqual(max(recorded.decode_sizes), 67)

    def test_other_decoder_keeps_full_prompt(self):
        ids = self.backend.encode("ordinary prefix " * 100).ids
        self.backend.decoder = decoders.BPEDecoder()
        self.assertIs(self.choose(self.backend, ids, True), ids)

    def test_arbitrary_byte_ids_before_clean_suffix_and_output_bursts(self):
        rng = random.Random(0)
        special = self.backend.token_to_id("[SPECIAL]")
        output = self.backend.encode("你好 👩🏽‍💻 á abc\n").ids
        for _ in range(20):
            # Valid vocabulary IDs can contain malformed UTF-8 byte runs,
            # unlike prompts obtained by encoding a complete Unicode string.
            ids = [rng.randrange(256) for _ in range(128)]
            ids += self.backend.encode(" stable clean suffix " * 10).ids
            ids += [special] * rng.randrange(66)
            for skip in [True, False]:
                tail = self.choose(self.backend, ids, skip)
                for burst in [1, 3, 7]:
                    self.assertEqual(replay(self.backend, ids, output, skip, burst),
                                     replay(self.backend, tail, output, skip, burst))


if __name__ == "__main__":
    unittest.main()
