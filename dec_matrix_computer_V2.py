import numpy as np
import itertools
from typing import Iterable, Optional, Any

from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.providers.backend import Backend
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library import IGate, XGate, YGate, ZGate, CXGate
from qiskit.quantum_info import Operator

# -------------------------------------------------------------------
#  SECTION 1: THE PAULI TWIRL TRANSPILER PASS
# -------------------------------------------------------------------

from qiskit.circuit.library import IGate, XGate, YGate, ZGate
import random
from qiskit.transpiler import PassManager, TransformationPass
from qiskit.dagcircuit import DAGCircuit

# --- Define Pauli gates and conjugation lookup tables ---

PAULIS = {'I': IGate(), 'X': XGate(), 'Y': YGate(), 'Z': ZGate()}
PAULI_NAMES = ['I', 'X', 'Y', 'Z']

# ---
# These are the correct maps, verified by P_out = G * P_in * G_dagger
# (ignoring phase)
# ---

# P_out = X * P_in * X
# X*Y*X = -Y, X*Z*X = -Z
X_CONJUGATION_MAP = {'I': 'I', 'X': 'X', 'Y': 'Y', 'Z': 'Z'}

# P_out = SX * P_in * SX_dagger
# SX*X*SX_dag = X
# SX*Y*SX_dag = Z
# SX*Z*SX_dag = -Y
SX_CONJUGATION_MAP = {'I': 'I', 'X': 'X', 'Y': 'Z', 'Z': 'Y'} 

# P_out = CNOT * P_in * CNOT
CNOT_CONJUGATION_MAP = {
    'II': ('I', 'I'), 'IX': ('I', 'X'), 'IY': ('Z', 'Y'), 'IZ': ('Z', 'Z'),
    'XI': ('X', 'X'), 'XX': ('X', 'I'), 'XY': ('Y', 'Z'), 'XZ': ('Y', 'Y'),
    'YI': ('Y', 'X'), 'YX': ('Y', 'I'), 'YY': ('X', 'Z'), 'YZ': ('X', 'Y'),
    'ZI': ('Z', 'I'), 'ZX': ('Z', 'X'), 'ZY': ('I', 'Y'), 'ZZ': ('I', 'Z'),
}

class ManualPauliTwirl(TransformationPass):
    """
    Randomly inserts Pauli gates around 'cx', 'sx', and 'x' gates
    to twirl the noise channels.
    
    This pass should be run ON A TRANSPILED CIRCUIT.
    """
    def __init__(self, seed=None):
        super().__init__()
        self.rng = random.Random(seed)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        new_dag = dag.copy_empty_like()
        
        for node in dag.topological_op_nodes():
            if node.op.name == 'cx':
                # --- Twirl CNOT ---
                p_c_in_str = self.rng.choice(PAULI_NAMES)
                p_t_in_str = self.rng.choice(PAULI_NAMES)
                p_in_key = p_c_in_str + p_t_in_str
                p_c_out_str, p_t_out_str = CNOT_CONJUGATION_MAP[p_in_key]
                
                if p_c_in_str != 'I': new_dag.apply_operation_back(PAULIS[p_c_in_str], [node.qargs[0]])
                if p_t_in_str != 'I': new_dag.apply_operation_back(PAULIS[p_t_in_str], [node.qargs[1]])
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
                if p_c_out_str != 'I': new_dag.apply_operation_back(PAULIS[p_c_out_str], [node.qargs[0]])
                if p_t_out_str != 'I': new_dag.apply_operation_back(PAULIS[p_t_out_str], [node.qargs[1]])
            
            elif node.op.name == 'sx':
                # --- Twirl SX ---
                p_in_str = self.rng.choice(PAULI_NAMES)
                p_out_str = SX_CONJUGATION_MAP[p_in_str]
                
                if p_in_str != 'I': new_dag.apply_operation_back(PAULIS[p_in_str], node.qargs)
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
                if p_out_str != 'I': new_dag.apply_operation_back(PAULIS[p_out_str], node.qargs)
                    
            elif node.op.name == 'x':
                # --- Twirl X ---
                p_in_str = self.rng.choice(PAULI_NAMES)
                p_out_str = X_CONJUGATION_MAP[p_in_str]
                
                if p_in_str != 'I': new_dag.apply_operation_back(PAULIS[p_in_str], node.qargs)
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
                if p_out_str != 'I': new_dag.apply_operation_back(PAULIS[p_out_str], node.qargs)
                    
            elif node.op.name not in ['barrier', 'measure']:
                # --- Keep all other gates (like rz, id) ---
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        return new_dag


# -------------------------------------------------------------------
#  SECTION 2: YOUR ADJUSTED CLASS
#  (This now uses the Pauli twirling functions from above)
# -------------------------------------------------------------------

class DECMatrixComputer:
    """
    Computes the A matrix for Distribution Error Correction (DEC).
    """
    
    def __init__(self, backend: Backend = None, basis=None, shots=10000, twirl=False):
        """
        Initializes the computer.

        Args:
            backend (Backend): The Qiskit backend to run simulations on.
                               (e.g., AerSimulator(noise_model=nm) or FakeAthensV2())
            shots (int): The total number of shots to use for sampling.
            twirl (bool): Whether to use Pauli twirling.
        """
        if backend is None:
            self.backend = AerSimulator()
        else:
            self.backend = backend

        if basis is None:
            self.basis = ['cx', 'id', 'rz', 'sx', 'x']  # IBM basis
        else:
            self.basis = basis
            
        self.shots = shots
        self.twirl = twirl


    def sample_circuit(self, circuit, shots=None):
        """
        Sample a quantum circuit and return the probability distribution.
        
        If self.twirl is True, this returns the *averaged* distribution
        from many randomly twirled circuits. Assumes the input circuit
        is NOT transpiled. Transpilation occurs within get_full_twirled_distribution.
        
        If self.twirl is False, this returns the distribution from a
        single run of the original circuit. Assumes the input circuit 
        is already transpiled.
        """
        if shots is None:
            shots = self.shots
            
        n_qubits = circuit.num_qubits
        final_distribution = np.zeros(2**n_qubits)

        # --- BRANCH 1: EXECUTE WITH PAULI TWIRLING ---
        if self.twirl:
            final_distribution = self.get_full_twirled_distribution(circuit, 
                                                              num_gate_randomizations=20, 
                                                              num_meas_randomizations=20)
        
        # --- BRANCH 2: EXECUTE NORMALLY (NO TWIRLING) ---
        else:
            circuit_with_meas = circuit.copy()
            if not circuit_with_meas.cregs:
                circuit_with_meas.add_register(ClassicalRegister(n_qubits, 'meas'))
            circuit_with_meas.measure_all()
            
            job = self.backend.run(circuit_with_meas, shots=shots)
            result = job.result()
            counts = result.get_counts()

            for state_str, count in counts.items():
                clean_state_str = state_str.replace(' ', '')
                if len(clean_state_str) > n_qubits:
                     clean_state_str = clean_state_str[:n_qubits]
                     
                state_int = int(clean_state_str, 2)
                if state_int < len(final_distribution):
                    if shots > 0:
                        final_distribution[state_int] = count / shots
                    else:
                        final_distribution[state_int] = 0.0

        return final_distribution
    
    def create_noise_estimation_circuit(self, payload_circuit):
        """
        Create a Noise Estimation Circuit (NEC) by replacing 
        superposition-creating gates with classical equivalents.
        
        Following the official implementation's approach.
        """
        nec = QuantumCircuit(payload_circuit.num_qubits)
        
        for instruction in payload_circuit.data:
            gate_name = instruction.operation.name
            qubits = instruction.qubits
            
            if gate_name == 'h':
                # Hadamard creates superposition - remove it (identity)
                pass
            elif gate_name == 'sx':
                # SX creates superposition - replace with X
                nec.x(qubits[0])
            elif gate_name == 's':
                # S gate: |0⟩ -> |0⟩, |1⟩ -> i|1⟩
                # From computational basis, doesn't create superposition
                # But in the official code, they seem to handle this differently
                nec.s(qubits[0])
            elif gate_name == 'sdg':
                # In official implementation: replace S† with Z
                nec.z(qubits[0])
            elif gate_name == 't':
                # T gate creates phases - remove it in NEC
                pass
            elif gate_name == 'tdg':
                # T† gate creates phases - remove it in NEC
                pass
            elif gate_name in ['x', 'y', 'z']:
                # Pauli gates don't create superposition
                getattr(nec, gate_name)(qubits[0])
            elif gate_name in ['cx', 'cy', 'cz']:
                # Controlled Paulis don't create superposition
                getattr(nec, gate_name)(qubits[0], qubits[1])
            elif gate_name == 'swap':
                nec.swap(qubits[0], qubits[1])
            elif gate_name == 'rz':
                # RZ doesn't create superposition from computational basis
                nec.rz(instruction.operation.params[0], qubits[0])
            elif gate_name in ['rx', 'ry']:
                # RX and RY create superposition - skip them
                pass
            elif gate_name in ['barrier', 'measure']:
                # Skip these
                pass
            else:
                print(f"Warning: Unhandled gate in NEC creation: {gate_name}")
        
        return nec

    def get_ideal_nec_output(self, nec, input_state_index=0):
        """
        Compute the ideal output of the NEC classically.
        Since NEC doesn't create superposition from |0⟩, 
        we can determine its output deterministically.
        """
        n_qubits = nec.num_qubits
        
        # Start with the input state (usually |0...0⟩)
        state = input_state_index
        
        # Simulate the NEC classically
        for instruction in nec.data:
            if instruction.operation.name == 'x':
                qubit_idx = nec.qubits.index(instruction.qubits[0])
                state ^= (1 << qubit_idx)  # Flip the bit
            elif instruction.operation.name == 'cx':
                control_idx = nec.qubits.index(instruction.qubits[0])
                target_idx = nec.qubits.index(instruction.qubits[1])
                if (state >> control_idx) & 1:
                    state ^= (1 << target_idx)
            elif instruction.operation.name == 'cz':
                # CZ doesn't change computational basis states
                pass
            # Add other gates as needed
            elif instruction.operation.name == 'measure':
                pass  # Skip measurements in classical simulation
        
        return state

    def compute_first_column_fast(self, payload_circuit):
        """
        Compute just the first column of A matrix.
        Need to account for the ideal NEC output state.

        Sample from twirled NEC circuit is self.twirl=True
        """
        # Create the noise estimation circuit
        nec = self.create_noise_estimation_circuit(payload_circuit)
        
        # CRITICAL: Determine the ideal output of the NEC
        ideal_nec_output = self.get_ideal_nec_output(nec, input_state_index=0)
        
        # Sample the NEC with noise
        if self.twirl: # Twirl NEC circuit so that it has the same error channel as the payload circuit
            noisy_distribution = self.get_full_twirled_distribution(nec, 
                                                              num_gate_randomizations=20, 
                                                              num_meas_randomizations=20)
        else:
            noisy_distribution = self.sample_circuit(nec)
        
        # The first column represents how noise spreads from |0⟩ to all states
        # But our NEC might not output |0⟩ ideally - it outputs |ideal_nec_output⟩
        # So we need to "shift" the distribution back
        first_column = np.zeros_like(noisy_distribution)
        
        for i in range(len(noisy_distribution)):
            # Map from actual NEC output space back to error pattern space
            error_pattern_index = i ^ ideal_nec_output
            first_column[error_pattern_index] = noisy_distribution[i]
        
        return first_column

    def correct_distribution_properly(self, noisy_payload_dist, nec_first_column):
        """
        Correct distribution using the proper DEC method from the paper.
        
        The key insight: 
        - noisy_payload_dist: noisy distribution from actual payload circuit
        - nec_first_column: first column of A matrix from NEC sampling
        
        The paper uses: x = IFWHT(FWHT(z) ./ FWHT(a))
        where z is noisy payload, a is first column of A, x is corrected
        """
        z = noisy_payload_dist.copy()
        a = nec_first_column.copy()
        
        # Ensure same length and power of 2
        max_len = max(len(z), len(a))
        if max_len & (max_len - 1) != 0:
            next_power = 2**int(np.ceil(np.log2(max_len)))
            max_len = next_power
            
        # Pad both to same power-of-2 length
        z = np.pad(z, (0, max_len - len(z)))
        a = np.pad(a, (0, max_len - len(a)))
        
        # Apply Fast Walsh-Hadamard Transform
        fwht_z = self.fwht(z)
        fwht_a = self.fwht(a)
        
        # Element-wise division with zero handling
        # This is the key step from Eq. (10) in the paper
        corrected_fwht = np.zeros_like(fwht_z)
        mask = np.abs(fwht_a) > 1e-10  # Avoid division by very small numbers
        corrected_fwht[mask] = fwht_z[mask] / fwht_a[mask]
        
        # Inverse FWHT
        corrected = self.ifwht(corrected_fwht)
        
        # Truncate back to original size
        corrected = corrected[:len(noisy_payload_dist)]
        
        # Post-process to make it a valid probability distribution
        corrected = self.make_valid_distribution(corrected)

        return corrected

    def make_valid_distribution(self, dist):
        """
        Post-process the corrected distribution to be a valid probability distribution.
        This follows the approach mentioned in the paper for handling quasi-distributions.
        """
        # Method from Ref. [26] in the paper for correcting quasi-distributions
        # Simple approach: project negative values to zero and renormalize
        
        corrected = np.maximum(dist, 0)  # Remove negative values
        
        # Renormalize to sum to 1
        total = np.sum(corrected)
        if total > 1e-10:
            corrected = corrected / total
        else:
            # If all values are zero/negative, return uniform distribution
            corrected = np.ones_like(corrected) / len(corrected)
            
        return corrected
    
    def fwht(self, x):
        """Fast Walsh-Hadamard Transform"""
        x = x.copy()
        n = len(x)
        step = 1
        while step < n:
            for i in range(0, n, step * 2):
                for j in range(step):
                    u, v = x[i + j], x[i + j + step]
                    x[i + j], x[i + j + step] = u + v, u - v
            step *= 2
        return x / n
    
    def ifwht(self, x):
        """Inverse Fast Walsh-Hadamard Transform"""
        return self.fwht(x) * len(x)

    def classically_flip_distribution(self, distribution, n_qubits, flip_mask_array):
        """
        Classically flips the bits of a probability distribution array.
        
        Args:
            distribution (np.ndarray): The 2^n array of probabilities.
            n_qubits (int): The total number of qubits (e.g., 5 for Athens).
            flip_mask_array (list[int]): A list of 0s and 1s, e.g., [0, 1, 0, 0, 1].
                                         (Must be length n_qubits)
        
        Returns:
            np.ndarray: A new distribution with probabilities re-mapped.
        """
        
        # Convert the mask array [0, 1, 0, 1] into a single integer (e.g., 5)
        # We reverse the list for Qiskit's little-endian bit ordering (q4,q3,q2,q1,q0)
        mask_str = "".join(map(str, flip_mask_array[::-1]))
        mask_int = int(mask_str, 2)
        
        new_distribution = np.zeros_like(distribution)
        
        for state_int, prob in enumerate(distribution):
            if prob > 0:
                # This is the classical bit-flip
                flipped_state_int = state_int ^ mask_int 
                if flipped_state_int < len(new_distribution):
                    new_distribution[flipped_state_int] = prob
                
        return new_distribution
    
    def get_twirled_distribution(self, circuit, num_randomizations=100):
        """
        Generates the average probability distribution from N twirled circuits.
        
        Args:
            circuit (QuantumCircuit): The *transpiled* circuit to twirl.
            num_randomizations (int): Number of twirled circuits to average (N).
            
        Returns:
            np.ndarray: The final, averaged probability distribution.
        """
        total_shots = self.shots
        
        if total_shots < num_randomizations:
            print(f"Warning: total_shots ({total_shots}) < num_randomizations ({num_randomizations}).")
            print("Running each randomization with 1 shot.")
            shots_per_randomization = 1
        else:
            shots_per_randomization = total_shots // num_randomizations
        
        n_qubits = circuit.num_qubits
        summed_distribution = np.zeros(2**n_qubits)
       
        #print(f"Running {num_randomizations} twirled randomizations with {shots_per_randomization} shots each...")
        
        for i in range(num_randomizations):
            # 1. Create a PassManager with a NEW seed for this loop
            # This generates one random twirled circuit
            twirl_pass = ManualPauliTwirl(seed=i)
            pm = PassManager([twirl_pass])
            qc_twirled = pm.run(circuit)
            
            # 2. Run this single twirled circuit
            # We use your existing, verified sample_circuit function
            dist = self.sample_circuit(qc_twirled, shots=shots_per_randomization)
            
            # 3. Add its result to the sum
            summed_distribution += dist
            
        # 4. Average the final distribution
        averaged_distribution = summed_distribution / num_randomizations
        
        #print("Twirling complete.")
        return averaged_distribution
    
    def get_full_twirled_distribution(
        self,
        logical_circuit, 
        num_gate_randomizations=100, 
        num_meas_randomizations=100
    ):
        """
        Performs both gate twirling and measurement twirling, then averages.
        
        This version explicitly maps logical qubits to physical qubits
        to avoid layout-checking errors.
        """
        
        n_qubits = logical_circuit.num_qubits 
        
        summed_distribution = np.zeros(2**n_qubits)
    
        #print(f"Running {num_meas_randomizations} measurement randomizations...")
        
        for i in range(num_meas_randomizations):
            
            qc_with_mask = logical_circuit.copy()
            
            # --- 1. ADD MEASUREMENT MASK (QUANTUM) ---
            
            # Create a random mask just for the logical qubits
            meas_flip_mask_logical = [random.choice([0, 1]) for _ in range(n_qubits)]
            
            for q_idx, flip in enumerate(meas_flip_mask_logical):
                if flip == 1:
                    qc_with_mask.x(q_idx) # Flips logical qubits
            
            # --- 2. TRANSPILE (with explicit layout) ---
            
            # We create a simple layout map, e.g., [0, 1, 2...]
            # For your 1-qubit test, this will be [0]
            logical_to_physical_map = list(range(n_qubits))
            
            qc_transpiled_with_mask = transpile(
                qc_with_mask, 
                basis_gates=self.basis,
                optimization_level=0,
                initial_layout=logical_to_physical_map # Force logical 0 -> physical 0
            )
    
            # --- 3. CALL GATE-TWIRLING FUNCTION ---
            
            gate_twirled_dist = self.get_twirled_distribution(
                qc_transpiled_with_mask, 
                num_randomizations=num_gate_randomizations
            )
            
            # --- 4. UN-FLIP (CLASSICAL) ---
            
            # Create the full 5-bit mask for the classical flip
            full_meas_mask = [0] * n_qubits
            for logical_q_idx, flip in enumerate(meas_flip_mask_logical):
                if flip == 1:
                    # We know logical_q_idx maps to physical_q_idx
                    physical_q_idx = logical_to_physical_map[logical_q_idx]
                    full_meas_mask[physical_q_idx] = 1
            
            # Only flip if the mask is not all zeros
            if sum(full_meas_mask) > 0:
                dist_unflipped = self.classically_flip_distribution(
                    gate_twirled_dist, 
                    n_qubits, 
                    full_meas_mask
                )
            else:
                dist_unflipped = gate_twirled_dist # No flip needed
            
            # --- 5. ADD TO SUM ---
            summed_distribution += dist_unflipped
    
        # --- 6. RETURN FINAL AVERAGE ---
        print("Full twirling complete.")
        return summed_distribution / num_meas_randomizations
        
