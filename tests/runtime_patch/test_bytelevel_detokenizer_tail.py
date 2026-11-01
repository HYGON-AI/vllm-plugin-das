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
