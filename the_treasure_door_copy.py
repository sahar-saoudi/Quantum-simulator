# %% imports and definitions
import numpy as np

import my_quantum_simulator as mqs
from my_quantum_simulator import X_GATE, H_GATE
from my_quantum_simulator import convert_to_more_qubits

I = np.array([[1, 0], [0, 1]])
CX_GATE = mqs.control(mqs.X_GATE)
SWAP_GATE = CX_GATE @ mqs.reorder_qubits_gate(CX_GATE, [1, 0]) @ CX_GATE

# %% Circuit construction
# setup_circuit = (convert_to_more_qubits(CX_GATE, 3, [0,1])) @ (convert_to_more_qubits((np.kron(H_GATE ,H_GATE)), 3, [0, 2]))
# lying_circuit =  (convert_to_more_qubits(X_GATE, 3, [2])) @ (convert_to_more_qubits(CX_GATE, 3, [2,0])) @ (convert_to_more_qubits(X_GATE, 3, [2])) @ (convert_to_more_qubits(CX_GATE, 3, [2,1]))
# do_not_circuit =  (convert_to_more_qubits((np.kron(X_GATE, X_GATE)), 3, [0, 1])) @ (convert_to_more_qubits(SWAP_GATE, 3, [0, 1]))

# final_circuit = lying_circuit @ do_not_circuit @ lying_circuit @ setup_circuit
#Voici la version simplifiée du circuit
final_circuit = (convert_to_more_qubits(SWAP_GATE, 3, [0, 1])) @ (convert_to_more_qubits(CX_GATE, 3, [0,1])) @ (convert_to_more_qubits(H_GATE, 3, [0])) @ (convert_to_more_qubits(H_GATE, 3, [2]))
# %% Circuit simulation

final_state = final_circuit @ mqs.prepare_init_state(3)

# %% Measurement simulation

results = mqs.sample_state(final_state, shots=1000)

print(results)

# %%
