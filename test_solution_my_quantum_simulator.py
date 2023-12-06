import unittest

import numpy as np

import my_quantum_simulator as mqs


def assert_gates_equivalent(gate1, gate2):
    prod = gate1 @ gate2.T.conj()

    np.testing.assert_allclose(prod @ prod.conj(), np.eye(prod.shape[0]), atol=1e-12)


class TestMQS(unittest.TestCase):
    def test_build_ry_gate(self):
        y_gate = np.array([[0, -1j], [1j, 0]])

        new_y_gate = mqs.build_ry_gate(np.pi)

        assert_gates_equivalent(new_y_gate, y_gate)

    def test_control(self):
        x_gate = np.array([[0, 1], [1, 0]])
        ref_cx_gate = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

        cx_gate = mqs.control(x_gate)

        assert_gates_equivalent(cx_gate, ref_cx_gate)

    def test_reorder_qubits_gate(self):
        ref_c0x1_gate = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

        c1x0_gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

        c0x1_gate = mqs.reorder_qubits_gate(c1x0_gate, [1, 0])

        assert_gates_equivalent(c0x1_gate, ref_c0x1_gate)

        ref_i0c1x2_gate = np.kron(
            np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]), np.array([[1, 0], [0, 1]])
        )

        c0x1i2_gate = np.kron(
            np.array([[1, 0], [0, 1]]), np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
        )

        i0c1x2 = mqs.reorder_qubits_gate(c0x1i2_gate, [1, 2, 0])

        assert_gates_equivalent(i0c1x2, ref_i0c1x2_gate)

    def test_reorder_qubits_cx_gate(self):
        ref_c0x1_gate = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

        ref_c0x1i2_gate = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
            ]
        )

        c0x1i2_gate = mqs.convert_to_more_qubits(ref_c0x1_gate, 3, [0, 1])
        assert_gates_equivalent(c0x1i2_gate, ref_c0x1i2_gate)

        ref_c0i1x2_gate = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )

        c0i1x2_gate = mqs.reorder_qubits_gate(ref_c0x1i2_gate, [0, 2, 1])
        assert_gates_equivalent(c0i1x2_gate, ref_c0i1x2_gate)

        ref_x0i1c2_gate = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ]
        )

        x0i1c2_gate = mqs.reorder_qubits_gate(ref_c0x1i2_gate, [2, 0, 1])
        assert_gates_equivalent(x0i1c2_gate, ref_x0i1c2_gate)

        ref_i0x1c2_gate = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
            ]
        )

        i0x1c2_gate = mqs.reorder_qubits_gate(ref_c0x1i2_gate, [2, 1, 0])
        assert_gates_equivalent(i0x1c2_gate, ref_i0x1c2_gate)

        ref_x0c1x2_gate = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ]
        )

        x0c1x2_gate = mqs.reorder_qubits_gate(ref_c0x1i2_gate, [1, 0, 2])
        assert_gates_equivalent(x0c1x2_gate, ref_x0c1x2_gate)

        ref_i0c1x2_gate = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )

        i0c1x2_gate = mqs.reorder_qubits_gate(ref_c0x1i2_gate, [1, 2, 0])
        assert_gates_equivalent(i0c1x2_gate, ref_i0c1x2_gate)

    def test_convert_to_more_qubits(self):
        ref_c0x1_gate = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

        ref_c2x0 = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ]
        )

        ref_c2x1 = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
            ]
        )

        c2x0 = mqs.convert_to_more_qubits(ref_c0x1_gate, 3, [2, 0])

        np.testing.assert_allclose(c2x0, ref_c2x0, atol=1e-12)

        c2x1 = mqs.convert_to_more_qubits(ref_c0x1_gate, 3, [2, 1])

        np.testing.assert_allclose(c2x1, ref_c2x1, atol=1e-12)

    def test_prepare_init_state(self):
        ref_init_state = np.array([1, 0, 0, 0, 0, 0, 0, 0])

        init_state = mqs.prepare_init_state(3)

        np.testing.assert_allclose(init_state, ref_init_state, atol=1e-12)

    def test_sample_state(self):
        h_gate = np.sqrt(0.5) * np.array([[1, 1], [1, -1]])
        c0x1_gate = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
        ih_gate = np.kron(np.eye(2), h_gate)

        bell_circuit = c0x1_gate @ ih_gate

        init_state = mqs.prepare_init_state(2)

        final_state = bell_circuit @ init_state

        counts = mqs.sample_state(final_state, 1000)

        self.assertTrue("01" not in counts)
        self.assertTrue("10" not in counts)

        values = list(counts.values())

        self.assertEqual(np.round(values[0] / values[1]), 1)

    def test_c0x1c2(self):
        ref_c0x1c2_gate = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
            ]
        )

        x_gate = np.array([[0, 1], [1, 0]])
        c0x1_gate = mqs.control(x_gate)
        c0c1x2_gate = mqs.control(c0x1_gate)

        c0x1c2_gate = mqs.reorder_qubits_gate(c0c1x2_gate, [0, 2, 1])

        assert_gates_equivalent(c0x1c2_gate, ref_c0x1c2_gate)

    def test_c0x2(self):
        ref_c0x2_gate = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )

        x_gate = np.array([[0, 1], [1, 0]])
        c0x1_gate = mqs.control(x_gate)

        c0x2_gate = mqs.convert_to_more_qubits(c0x1_gate, 3, [0, 2])

        assert_gates_equivalent(c0x2_gate, ref_c0x2_gate)


if __name__ == "__main__":
    unittest.main()
