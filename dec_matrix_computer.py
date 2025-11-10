import numpy as np
import itertools
from typing import Iterable, Optional, Any

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.providers.backend import Backend
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library import IGate, XGate, YGate, ZGate, CXGate
from qiskit.quantum_info import Operator

# -------------------------------------------------------------------
#  SECTION 1: THE PAULI TWIRL TRANSPILER PASS
#  (This is the class you were missing)
# -------------------------------------------------------------------

class PauliTwirl(TransformationPass):
    """A transpiler pass to add Pauli twirls to a circuit.
    
    This pass will replace any gates specified in `gates_to_twirl`
    with a randomly twirled equivalent: G -> P_j * G * P_i
    
    The Paulis P_i, P_j are chosen from a pre-computed set such that
    the ideal operation is preserved.
    """

    def __init__(
        self,
        gates_to_twirl: Optional[Iterable[str]] = None,
        seed: Any = None,
    ):
        """
        Args:
            gates_to_twirl: Names of gates to twirl. Defaults to ["cx"].
            seed: Seed for the pseudorandom number generator.
        """
        super().__init__()
        
        if gates_to_twirl is None:
            gates_to_twirl = ["cx"]
        self.gates_to_twirl = set(gates_to_twirl)
        
        # Build the twirling sets
        self._twirl_sets = {}
        for gate_name in self.gates_to_twirl:
            if gate_name == "cx":
                self._twirl_sets["cx"] = self._build_cx_twirl_set()
            # You could add other gates here, e.g., self._build_cz_twirl_set()
        
        # Initialize RNG
        self.rng = np.random.default_rng(seed)

    def _build_cx_twirl_set(self):
        """Builds the 16 (pre, post) Pauli pairs that twirl a CX gate."""
        paulis = [IGate(), XGate(), YGate(), ZGate()]
        pauli_pairs = list(itertools.product(paulis, paulis))
        twirl_set = []

        # Ideal CX operator
        cx_op = Operator(CXGate()) 

        for pre in pauli_pairs:
            # Operator for (P_pre_q0 tensor P_pre_q1)
            #pre_op = Operator(pre[0]) & Operator(pre[1])
            pre_op = Operator(pre[0]).tensor(Operator(pre[1]))
            
            for post in pauli_pairs:
                # Operator for (P_post_q0 tensor P_post_q1)
                #post_op = Operator(post[0]) & Operator(post[1])
                post_op = Operator(post[0]).tensor(Operator(post[1]))
                
                # Check if P_post * CX * P_pre == CX (up to global phase)
                twirled_op = post_op.compose(cx_op).compose(pre_op)
                
                if twirled_op.equiv(cx_op):
                    # The twirl is valid, store it
                    # ( (P_pre_q0, P_pre_q1), (P_post_q0, P_post_q1) )
                    twirl_set.append((pre, post))
        return twirl_set

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the PauliTwirl pass on `dag`.
        
        Args:
            dag: The DAG to map.
        
        Returns:
            A new DAG with twirled gates.
        """
        if not self._twirl_sets:
            return dag

        new_dag = dag.copy_empty_like()

        for node in dag.topological_op_nodes():
            if node.op.name in self.gates_to_twirl:
                # This gate needs to be twirled
                
                # Get the twirl set for this gate
                twirl_set = self._twirl_sets[node.op.name]
                
                # Pick a random twirl from the set
                idx = self.rng.integers(len(twirl_set))
                pre_gates, post_gates = twirl_set[idx]
                
                # Apply pre-Paulis
                for i, qubit in enumerate(node.qargs):
                    if pre_gates[i].name != "id": # Don't add Identity gates
                        new_dag.apply_operation_back(pre_gates[i], (qubit,))
                
                # Apply the original gate
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

                # Apply post-Paulis
                for i, qubit in enumerate(node.qargs):
                    if post_gates[i].name != "id": # Don't add Identity gates
                        new_dag.apply_operation_back(post_gates[i], (qubit,))
            else:
                # This gate is not twirled, just add it
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        return new_dag


# -------------------------------------------------------------------
#  SECTION 2: YOUR ADJUSTED CLASS
#  (This now uses the PauliTwirl class from above)
# -------------------------------------------------------------------

class DECMatrixComputer:
    """
    Computes the A matrix for Distribution Error Correction (DEC).
    """
    
    def __init__(self, backend: Backend = None, shots=10000, twirl=False):
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
            
        self.shots = shots
        self.twirl = twirl


    def sample_circuit(self, circuit, shots=None):
        """
        Sample a quantum circuit and return the probability distribution.
        
        Assumes the input circuit is already transpiled.
        
        If self.twirl is True, this returns the *averaged* distribution
        from many randomly twirled circuits.
        
        If self.twirl is False, this returns the distribution from a
        single run of the original circuit.
        """
        if shots is None:
            shots = self.shots
            
        n_qubits = circuit.num_qubits
        final_distribution = np.zeros(2**n_qubits)

        # --- BRANCH 1: EXECUTE WITH PAULI TWIRLING ---
        if self.twirl:
            
            num_twirl_circs = 30  # Number of random circuits to average
            shots_per_twirl = int(np.ceil(shots / num_twirl_circs))
            total_shots_run = 0

            for i in range(num_twirl_circs):
                
                # a. Create a new PassManager with a different seed
                #    to get a *different* random twirled circuit each time.
                twirl_pass = PauliTwirl(gates_to_twirl=['cx'], seed=i)
                pm = PassManager(twirl_pass)
                
                # b. Generate one randomly twirled version of the circuit
                twirled_circ = pm.run(circuit)
                
                # c. Add measurements
                twirled_circ_with_meas = twirled_circ.copy()
                if not twirled_circ_with_meas.cregs:
                    twirled_circ_with_meas.add_register(ClassicalRegister(n_qubits, 'meas'))
                twirled_circ_with_meas.measure_all()
                
                # d. Execute this single twirled circuit
                job = self.backend.run(twirled_circ_with_meas, shots=shots_per_twirl)
                result = job.result()
                counts = result.get_counts()
                
                # Sum the counts
                for state_str, count in counts.items():
                    clean_state_str = state_str.replace(' ', '')
                    if len(clean_state_str) > n_qubits:
                         clean_state_str = clean_state_str[:n_qubits]
                         
                    state_int = int(clean_state_str, 2)
                    if state_int < len(final_distribution):
                        final_distribution[state_int] += count
                
                total_shots_run += shots_per_twirl

            # e. Normalize the final averaged distribution
            if total_shots_run > 0:
                final_distribution /= total_shots_run
        
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



    #def sample_circuit(self, circuit, shots=None):
        """
        Sample a quantum circuit and return the probability distribution.
        """
    #    if shots is None:
    #        shots = self.shots
            
        # Add measurements if not present
    #    circuit_with_meas = circuit.copy()
    #    if not circuit_with_meas.cregs:  # No classical registers exist
    #        circuit_with_meas.add_register(ClassicalRegister(circuit.num_qubits, 'meas'))
    #        circuit_with_meas.measure_all()
    #    elif not any(instr.operation.name == 'measure' for instr in circuit_with_meas.data):
            # Classical register exists but no measurements
    #        circuit_with_meas.measure_all()
        
        # Execute with noise model
    #    if self.noise_model:
    #        simulator = AerSimulator(noise_model=self.noise_model)
    #    else:
    #        simulator = AerSimulator()
            
    #    job = simulator.run(circuit_with_meas, shots=shots)
    #    result = job.result()
    #    counts = result.get_counts()  # THIS LINE WAS MISSING
        
        # Convert to probability distribution
    #    n_qubits = circuit.num_qubits
    #    distribution = np.zeros(2**n_qubits)
        
    #    for state_str, count in counts.items():
            # Remove spaces from the bit string and take only the first n_qubits bits
    #        clean_state_str = state_str.replace(' ', '')
            # Ensure we only use the number of bits corresponding to qubits
    #        if len(clean_state_str) > n_qubits:
    #            clean_state_str = clean_state_str[:n_qubits]
                    
    #        state_int = int(clean_state_str, 2)
    #        if state_int < len(distribution):  # Safety check
    #            distribution[state_int] = count / shots
    
    #    return distribution

    def compute_first_column_fast(self, payload_circuit):
        """
        Compute just the first column of A matrix.
        Need to account for the ideal NEC output state.
        """
        # Create the noise estimation circuit
        nec = self.create_noise_estimation_circuit(payload_circuit)
        
        # CRITICAL: Determine the ideal output of the NEC
        ideal_nec_output = self.get_ideal_nec_output(nec, input_state_index=0)
        
        # Sample the NEC with noise
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

