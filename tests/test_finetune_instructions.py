import unittest

from finetune_instructions import encode_examples


class RecordingTokenizer:
    def __init__(self):
        self.seen = []

    def encode(self, text):
        self.seen.append(text)
        return list(range(len(text)))


class InstructionPromptFormatTests(unittest.TestCase):
    def test_owner_prefix_is_not_duplicated(self):
        tokenizer = RecordingTokenizer()
        encode_examples([("Owner: Say hello", "Orbit: Hello there.")], tokenizer, 128)
        self.assertEqual(tokenizer.seen[0], "Owner: Say hello\nOrbit:")

    def test_specialist_prefix_is_preserved(self):
        tokenizer = RecordingTokenizer()
        encode_examples([("Tool planner: calculate 2+2", '{"steps":[]}')], tokenizer, 128)
        self.assertEqual(tokenizer.seen[0], "Tool planner: calculate 2+2\nJSON:")

    def test_plain_instruction_gets_owner_prefix(self):
        tokenizer = RecordingTokenizer()
        encode_examples([("Explain gravity", "It attracts masses.")], tokenizer, 128)
        self.assertEqual(tokenizer.seen[0], "Owner: Explain gravity\nOrbit:")


if __name__ == "__main__":
    unittest.main()
