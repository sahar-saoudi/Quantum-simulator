import math
import numpy as np
from numpy.typing import NDArray

X_GATE = np.array([[0, 1], [1, 0]])
Y_GATE = np.array([[0, -1j], [1j, 0]])
Z_GATE = np.array([[1, 0], [0, -1]])
H_GATE = np.sqrt(0.5) * np.array([[1, 1], [1, -1]])


def prepare_init_state(number_of_qubits: int) -> NDArray:
    """
    Prepares the initial statevector (all qubits in state zero)

    Args:
        number_of_qubits (int): The number of qubits.

    Returns:
        NDArray: The statevector
    """

    init_state = np.zeros(2**number_of_qubits)
    init_state[0] = 1

    return init_state


def build_ry_gate(angle: float) -> NDArray:
    """
    Builds the matrix of a Ry gate for a given angle.

    Args:
        angle (float): The angle of the rotation around the y axis.

    Returns:
        NDArray: Matrix representation of a Ry rotation.
    """

    ry_gate = np.array([[np.cos(angle / 2), (-np.sin(angle / 2))], [(np.sin(angle / 2)), (np.cos(angle / 2))]])

    return ry_gate


def control(gate: NDArray) -> NDArray:
    """
    Convert a gate into a controlled gate.

    Args:
        gate (NDArray): Array representation of a gate. Basis using little-endian.

    Returns:
        NDArray: Array representation of the controlled gate controlled by the 0th qubit. Basis using little-endian.
    """

    gate_size = gate.shape[0]

    control_gate = np.zeros((2 * gate_size,) * 2)

    control_gate[::2, ::2] = np.eye(gate_size)
    control_gate[1::2, 1::2] = gate

    return control_gate


def reorder_qubits_gate(gate: NDArray, order: "list[int]") -> NDArray:
    """
    Transforms the Array of a gate by changing the qubit order.
    Ex : [2,0,1] means qubit 0 goes to position 2, qubit 1 goes to position 0 and qubit 2 goes to 1.

    Args:
        gate (NDArray): Array representation of a gate. Basis using little-endian.
        order (list[int]): Order in which to place the qubits.

    Returns:
        NDArray: Array representation of the gate with qubits moved around.
    """
    order = np.array(order)
    number_of_qubits = int(np.log2(gate.shape[0]))

    assert(number_of_qubits == len(order))

    from_index = np.flip(np.arange(2 * number_of_qubits))
    to_index = (np.concatenate((2*number_of_qubits - 1 - order, number_of_qubits - 1 - order)))

    gate_shape = gate.shape
    reshaped_gate = gate.reshape((2,) * (2 * number_of_qubits))
    reordered_gate = np.moveaxis(reshaped_gate, from_index, to_index).reshape(gate_shape)

    return reordered_gate


def convert_to_more_qubits(gate: NDArray, number_of_qubits: int, apply_on_qubits: "list[int]") -> NDArray:
    """
    Convert a gate to a higher number of qubits adding identity operator to additional qubits. Apply the gate given a set of qubits.

    Args:
        gate (NDArray): Array representation of a gate. Basis using little-endian.
        number_of_qubits (int): The total number to convert the gate to.
        apply_on_qubits (list[int]): On which qubits the original gate should be applied

    Returns:
        NDArray: Array representation of the gate with additional qubits.
    """

    initial_number_of_qubits = int(np.log2(gate.shape[0]))

    assert len(apply_on_qubits) == initial_number_of_qubits
    assert number_of_qubits > initial_number_of_qubits

    matrix_size_to_add = 2 ** (number_of_qubits - initial_number_of_qubits)
    tmp_gate = np.kron(np.eye(matrix_size_to_add), gate)

    qubits_index_list = list(range(number_of_qubits))
    new_list = np.setdiff1d(qubits_index_list, apply_on_qubits, assume_unique=True).tolist()

    new_gate = reorder_qubits_gate(tmp_gate, apply_on_qubits + new_list)

    return new_gate


def sample_state(state_vector: NDArray, shots: int) -> dict:
    """
    Simulate the sampling for a given statevector

    Args:
        state_vector (NDArray): A statevector
        shots (int): The total number of samples to take.

    Returns:
        dict: Results of the sampling given as a dictionary. Keys are bit strings (ex : "0110") and values integers.
    """

    position_vector = np.arange(len(state_vector))
    probability_vector = np.square(np.absolute(state_vector))

    arr = np.random.choice(position_vector, shots, p=probability_vector)
    unique, counts = np.unique(arr, return_counts=True)

    # Convert unique integer positions to binary strings with leading zeros, matching the maximum bit length
    max_length = int(np.log2(len(state_vector)))
    unique_binary = [format(u, f"0{max_length}b") for u in unique] 

    counts = dict(zip(unique_binary, counts))

    return counts
