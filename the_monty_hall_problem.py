# %% imports and definitions
import numpy as np
import math

import my_quantum_simulator as mqs
from my_quantum_simulator import convert_to_more_qubits as ctmq

CX_GATE = mqs.control(mqs.X_GATE)
CH_GATE = mqs.control(mqs.H_GATE)

W_ANGLE = 2 * math.acos(1/math.sqrt(3))
RY_W = mqs.build_ry_gate(W_ANGLE)

# %% Circuit construction   


first_part = ctmq(mqs.X_GATE, 3, [0]) @ ctmq(CX_GATE, 3, [0, 1]) @ ctmq(CX_GATE, 3, [1, 2]) @ ctmq(CH_GATE, 3, [0, 1]) @ ctmq(RY_W, 3, [0])

# %% Circuit simulation

second_part = ctmq(CH_GATE, 3, [1, 2]) @ ctmq(CX_GATE, 3, [0, 2])
final_circuit = ctmq(second_part, 4, [0, 2, 3]) @ ctmq(first_part, 4, [0, 1, 2])

final_state = final_circuit @ mqs.prepare_init_state(4)

# %% Measurement simulation

results = mqs.sample_state(final_state, 1000)

# %% print

print(results)



# %% défis

# Circuit construction

x_gate_4 = ctmq(mqs.X_GATE, 4, [2])

# Circuit simulation
second_part = ctmq(CH_GATE, 4, [2, 3]) @ ctmq(CX_GATE, 4, [0, 3])
final_circuit = second_part @ x_gate_4

final_state = final_circuit @ mqs.prepare_init_state(4)

# Measurement simulation

results = mqs.sample_state(final_state, 1000)

# print

print(results)
