"""CPU-only checks of the boundary channel mechanics; no models or qualification data."""

import unittest

import torch

from selffold.message_boundary import (
    BoundaryMemory,
    bits_to_int,
    boundary_bits,
    convention_bits,
    harden,
)


class MessageBoundaryTests(unittest.TestCase):
    def test_harden_is_onehot_and_idempotent(self):
        table = torch.tensor([[[0.1, 0.9], [0.6, 0.4]]])
        hard = harden(table)
        self.assertTrue(torch.equal(hard, torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])))
        self.assertTrue(torch.equal(harden(hard), hard))

    def test_convention_carries_only_the_offoff_bit(self):
        masks = torch.tensor([0b0001, 0b1110, 0b0101])
        bits = convention_bits(masks, 3)
        self.assertEqual(bits.shape, (3, 3))
        self.assertEqual(bits[:, 0].tolist(), [1, 0, 1])
        self.assertFalse(bits[:, 1:].any())

    def test_dropped_and_supplied_channels(self):
        out = torch.ones(4, 2, dtype=torch.long)
        mask = torch.tensor([0b0001, 0b0010, 0b0100, 0b1000])
        self.assertFalse(boundary_bits(out, mask, "none", 2).any())
        supplied = boundary_bits(out, mask, "supplied", 2)
        self.assertTrue(torch.equal(supplied, convention_bits(mask, 2)))

    def test_message_widths_index_the_pointer(self):
        model = BoundaryMemory(17, 2)
        table = torch.rand(3, 4, 16)
        for message in (0, 1, 2, 3):
            result, out_bits, *_ = model.step(
                table, torch.full((3,), message), torch.tensor([5, 6, 9])
            )
            self.assertEqual(result.shape, (3, 4, 16))
            self.assertEqual(out_bits.shape, (3, 2))
            self.assertEqual(bits_to_int(out_bits).shape, (3,))

    def test_pointer_rows_come_from_the_input_table(self):
        model = BoundaryMemory(41, 1)
        rows = torch.rand(5, 2, 16)
        table = rows[:, [0, 0, 0, 1]]
        result, *_ = model.step(
            table, torch.zeros(5, dtype=torch.long), torch.tensor([6] * 5)
        )
        for row in result.reshape(-1, 16):
            self.assertTrue(any(torch.equal(row, r) for pair in rows for r in pair))


if __name__ == "__main__":
    unittest.main()
