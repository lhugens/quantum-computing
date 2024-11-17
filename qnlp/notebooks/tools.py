import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

import qiskit as qk
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.circuit.library import MCMT

######################################## 
# jobs
######################################## 

def list_jobs(backend):
    jobs = backend.jobs()
    for i, job in enumerate(jobs):
        time = job.creation_date().strftime('%Y-%m-%d %H:%M:%S')
        if hasattr(job.queue_info(), 'position'):
            status = f'pos {job.queue_info().position}'
        else:
            status = f'{job.status().name}'
        print(f'{i:>3} {time} {job.job_id()} {status}')

class execute_big: 
    def __init__(self, qc, backend=None, shots=40000):
        if not isinstance(qc, list):
            qc = [qc]

        max_shots = 20000
        self.no_copies = shots // max_shots
        new_qc = [qc[i] for i in range(len(qc)) for j in range(no_copies)]
        new_qc_transpiled = qk.transpile(new_qc, backend=backend)
        self.job_set = IBMQJobManager.run(new_qc_transpiled, name=name, backend=backend, shots=max_shots)
        return None

    def get_counts(self):
        dics = self.job_set.results().combine_results().get_counts()
        return [merge_dicts(a) for a in [dics[x:x + self.no_copies] for x in range(0, len(dics), self.no_copies)]]

def job_id(job):
    print(f'job_id=\'{job.job_id()}\'')

def counts(job):
    return flip_keys(job.result().get_counts())

def job_counts_heavy(no_copies, job_set):
    dics = job_set.results().combine_results().get_counts()
    return [merge_dicts(a) for a in [dics[x:x + no_copies] for x in range(0, len(dics), no_copies)]]

#def counts(qc, shots=20000):
#    backend = qk.Aer.get_backend('qasm_simulator')
#    job = qk.execute(qc, backend=backend, shots=shots)
#    return flip_keys(job.result().get_counts())


######################################## 
# circuits
######################################## 

def circ(n_qubits):
    '''
    Returns empty QuantumCircuit and QuantumRegister.
    example:
        >>> circ(3)
        (QuantumRegister(3, 'q'), 
        <qiskit.circuit.quantumcircuit.QuantumCircuit at 0x7f8fd3813df0>)
    '''
    qr = qk.QuantumRegister(n_qubits, 'q')
    qc = qk.QuantumCircuit(qr)
    return qr, qc

def circ_measured(n_qubits, n_bits):
    '''
    Returns empty QuantumCircuit, QuantumRegister and ClassicalRegister.
    example:
        >>> circ_measured(3)
        (QuantumRegister(3, 'q'),
        ClassicalRegister(2, 'c'),
        <qiskit.circuit.quantumcircuit.QuantumCircuit at 0x7f8fd38136a0>)
    '''
    qr = qk.QuantumRegister(n_qubits, 'q')
    cr = qk.ClassicalRegister(n_bits, 'c')
    qc = qk.QuantumCircuit(qr, cr)
    return qr, cr, qc

def isdagger(gate, dag_gate):
    '''
    Returns True if dag_gate is the dagger version of gate.
    '''
    qr, qc = circ(gate.num_qubits)
    qc.compose(gate, qr, inplace=True)
    qc.compose(dag_gate, qr, inplace=True)
    qc.measure_all()
    c = get_counts(qc)
    return list(c.keys()) == ['0'*gate.num_qubits]

def powers(gate, n):
    '''
    Returns a generator that gives successive powers 
    up to n of a given gate.
    example:
        >>> g = powers(gate, 2)
        >>> next(g).draw()
             ┌────────────┐
        q_0: ┤0           ├
             │  circuit-3 │
        q_1: ┤1           ├
             └────────────┘
        >>> next(g).draw()
             ┌────────────┐┌────────────┐
        q_0: ┤0           ├┤0           ├
             │  circuit-3 ││  circuit-3 │
        q_1: ┤1           ├┤1           ├
             └────────────┘└────────────┘
    '''
    qr, qc = circ(gate.num_qubits)
    qc.compose(gate, qr, inplace=True)
    power = 1

    while power <= n:
        yield qc
        power += 1
        qc.compose(gate, qr, inplace=True)

def S0_circuit(n_qubits):
    '''
    Returns a circuit that flips the sign of the 0 state.
    example:
        >>> print(S0_circuit(3).draw())
             ┌───┐               ┌───┐
        q_0: ┤ X ├───────■───────┤ X ├
             ├───┤       │       ├───┤
        q_1: ┤ X ├───────■───────┤ X ├
             ├───┤┌───┐┌─┴─┐┌───┐├───┤
        q_2: ┤ X ├┤ H ├┤ X ├┤ H ├┤ X ├
             └───┘└───┘└───┘└───┘└───┘
    '''
    qr, qc = circ(n_qubits)
    qc.x(qr)
    qc.h(-1)
    qc.compose(MCMT('cx', n_qubits-1, 1), qr, inplace=True)
    qc.h(-1)
    qc.x(qr)
    return qc

def oracle_circuit(bit_str):
    '''
    Returns the oracle (without ancillary qubits) for a given bit string of the form '0.10'.
    example:
        >>> print(oracle('0.10').draw())
             ┌───┐               ┌───┐
        q_0: ┤ X ├───────■───────┤ X ├
             └───┘       │       └───┘
        q_1: ────────────┼────────────
                         │            
        q_2: ────────────■────────────
             ┌───┐┌───┐┌─┴─┐┌───┐┌───┐
        q_3: ┤ X ├┤ H ├┤ X ├┤ H ├┤ X ├
             └───┘└───┘└───┘└───┘└───┘
    '''
    s_arr = np.array(list(bit_str), dtype=object)
    desired0 = list(np.where(s_arr == '0')[0])
    affected = list(np.where(s_arr != '.')[0])
    n_qubits = len(bit_str)
    qr, qc = circ(n_qubits)
    qc.x(qr[desired0])
    qc.h(-1)
    qc.compose(MCMT('cx', len(affected)-1, 1), qr[affected], inplace=True)
    qc.h(-1)
    qc.x(qr[desired0])
    return qc

def oracle_ancilla_circuit(bit_str):
    '''
    Returns the oracle (with 1 ancillary qubit) for a given bit string of the form '0.10'.
    example:
        >>> print(oracle_ancilla('0.10').draw())
             ┌───┐     ┌───┐
        q_0: ┤ X ├──■──┤ X ├
             └───┘  │  └───┘
        q_1: ───────┼───────
                    │       
        q_2: ───────■───────
             ┌───┐  │  ┌───┐
        q_3: ┤ X ├──■──┤ X ├
             └───┘┌─┴─┐└───┘
        q_4: ─────┤ X ├─────
                  └───┘     
    '''
    s_arr = np.array(list(bit_str), dtype=object)
    desired0 = list(np.where(s_arr == '0')[0])
    affected = list(np.where(s_arr != '.')[0])
    n_qubits = len(bit_str)
    qr, qc = circ(n_qubits+1)
    qc.x(qr[desired0])
    qc.compose(MCMT('cx', len(affected), 1), qr[affected] + [qr[-1]], inplace=True)
    qc.x(qr[desired0])
    return qc

def Q_circuit(U_circ, desired_str):
    n_qubits = U_circ.num_qubits

    # build Udag_circ
    Udag_circ = U_circ.reverse_ops()
    #assert(isdagger(U_circ, Udag_circ))

    # build O_circ (oracle)
    assert(len(desired_str)==n_qubits)

    O_circ = oracle_circuit(desired_str)
    S0_circ = S0_circuit(n_qubits)

    # build Q_gate (grover operator)
    qr, Q_circ = circ(n_qubits)
    Q_circ.compose(O_circ, qr, inplace=True)
    Q_circ.compose(Udag_circ, qr, inplace=True)
    Q_circ.compose(S0_circ, qr, inplace=True)
    Q_circ.compose(U_circ, qr, inplace=True)

    return Q_circ

def Q_ancilla_circuit(U_circ, desired_str):
    n_qubits = U_circ.num_qubits

    # build Udag_circ
    Udag_circ = U_circ.reverse_ops()
    assert(isdagger(U_circ, Udag_circ))

    # build O_circ (oracle)
    assert(len(desired_str)==n_qubits)

    O_circ = oracle_ancilla_circuit(desired_str)
    S0_circ = S0_circuit(n_qubits)

    # build Q_gate (grover operator)
    qr, Q_circ = circ(n_qubits+1)
    Q_circ.compose(O_circ, qr, inplace=True)
    Q_circ.compose(Udag_circ, qr[:-1], inplace=True)
    Q_circ.compose(S0_circ, qr[:-1], inplace=True)
    Q_circ.compose(U_circ, qr[:-1], inplace=True)

    return Q_circ

def ampl_circs(U_circ, desired_str, m=10):
    n_qubits = U_circ.num_qubits
    
    Q_circ = Q_circuit(U_circ, desired_str)
    
    # build list of circuits
    circ_lst = []
    shots = 20000
    qr, cr, qc = circ_measured(n_qubits, n_qubits)
    qc.compose(U_circ, qr, inplace=True)

    ## iteration 0 (does not have Q operator)
    qc_measured = deepcopy(qc)
    qc_measured.measure(qr, cr)
    circ_lst.append(qc_measured)
    
    for m in range(m):
        qc.compose(Q_circ, qr, inplace=True)
        qc_measured = deepcopy(qc)
        qc_measured.measure(qr, cr)
        circ_lst.append(qc_measured)

    return circ_lst

def ampl_ancilla_circs(U_circ, desired_str, M=10, ibm=False):
    n_qubits = U_circ.num_qubits
    
    Q_circ = Q_ancilla_circuit(U_circ, desired_str)
    
    # build list of circuits
    circ_lst = []
    qr, cr, qc = circ_measured(n_qubits+1, n_qubits)
    qc.compose(U_circ, qr[:-1], inplace=True)
    qc.x(-1)
    qc.h(-1)

    ## iteration 0 (does not have Q operator)
    qc_measured = deepcopy(qc)
    qc_measured.measure(qr[:-1], cr)
    circ_lst.append(qc_measured)
    
    for m in range(M):
        qc.compose(Q_circ, qr, inplace=True)
        qc_measured = deepcopy(qc)
        qc_measured.measure(qr[:-1], cr)
        circ_lst.append(qc_measured)

    return circ_lst

def a_arr(counts_lst, desired_str):
    assert(len(desired_str)==len(list(counts_lst[0].keys())[0]))
    shots = sum(list(counts_lst[0].values()))
    desired_basis_states = bit_combinations(desired_str)
    arr = np.zeros(len(counts_lst))
    for i, dic in enumerate(counts_lst):
        for state in desired_basis_states:
            #print(state, dic.get(state, 0))
            arr[i] += dic.get(state, 0)/shots
    return arr

def plot_amplification(a_arr, filename=None):
    a = a_arr[0]
    theta = np.arcsin(np.sqrt(a))
    m = np.arange(len(a_arr))
    ms = np.linspace(m[0], m[-1], 1000)
    
    fig = plt.figure(figsize = (7, 4))
    plt.xlabel(r'$m$')
    plt.ylabel('Prob of measuring good basis state')
    plt.plot(m, a_arr, 'o', label='experimental')
    plt.plot(ms, np.sin((2*ms+1)*theta)**2, label='theoretical')
    plt.axvline(x=np.floor(np.pi/(4*theta)), c='k', ls='--', label=r'$m_{floor}$')
    plt.axhline(y=max(a, 1-a), c='b', ls='--', label=r'max$(1-a,a)$')
    plt.legend()
    if filename != None:
        plt.savefig(time_name(filename), format='png', dpi=600)
    plt.plot()
    return

######################################## 
# utils
######################################## 

def generalize(fn):
    '''
    This decorator allows a function f(a, b) to be called 
    with a list of type(a) like so: f([a0, a1,..., an], b) 
    and it returns the list [f(a0, b), f(a1, b),...,f(an, b)].
    '''
    def wrapped(a, *args):
        if isinstance(a, list):
            lst = []
            for i in a:
                lst.append(fn(i, *args))
            return lst
        else:
            return(fn(a, *args))
    return wrapped

def n2(complex_scalar: complex) -> np.float64:
    ''' 
    Returns the |z|**2 of a complex number z.
    example:
        >>> n2(1+1j)
        2.0
    '''
    return np.absolute(complex_scalar)**2

def canonical(state: np.ndarray) -> np.ndarray:
    '''
    Rotates a certain 2D complex vector such that the
    first entry is real. 
    example:
        >>> canonical(np.array([1+1j, 2]))
        array([1.41421356+0.j, 1.41421356-1.41421356j])
    '''
    norm_a, norm_b = np.absolute(state)
    theta_a, theta_b = np.angle(state)
    phi = theta_b - theta_a
    return np.array([norm_a, norm_b*np.exp(1j*phi)])

@generalize
def substr(stri: str, indices: list) -> str:
    '''
    Mimics numpy index broadcasting.
    example:
        >>> substr('abcd', [1, 3])
        'bd'
    '''
    return ''.join(list(np.array(list(stri), dtype=object)[indices]))

@generalize
def substr_kill(stri: str, indices: list) -> str:
    '''
    The opposite of substr: it eliminates string elements.
    example:
        >>> substr_kill('abcd', [1, 3])   
        'ac'
    '''
    a = np.arange(0, len(stri))
    indices = np.delete(a, indices)
    return ''.join(list(np.array(list(stri), dtype=object)[indices]))

@generalize
def flip_keys(dic: dict) -> dict:
    ''' 
    Flips the strings in the keys of dictionary 'dic'. 
    example:
        >>> flip_keys({'011': 3, '001': 4})
        {'110': 3, '100': 4}
    '''
    new_dic = {}
    for key, value in dic.items():
        new_dic[key[::-1]] = value
    return new_dic

'''
def flip_keys(dic):
    for bit in list(dic.keys()):
        dic[bit[::-1]] = dic.pop(bit)
    return dic
'''

@generalize
def bit_index(s: str) -> str:
    '''
    Returns the index of the bit string.
    example:
        >>> bit_index('01')
        1
    '''
    n_qubits = len(s)
    canonical = 2**np.arange(0,n_qubits)[::-1]
    return np.sum(canonical*np.array(list(s), dtype=int))

# change the name of this list to bitdict_toarray
@generalize
def bit_array(dic: dict) -> np.ndarray:
    ''' 
    Takes a dictionary whose keys are bit strings (e.g. '101') and
    returns an array with the values of the dic in the indices of the bit strings.
    example:
        >>> bit_array({'00': 2, '01': 3})
        array([2., 3., 0., 0.])
    '''
    n_measured_qbits = len(list(dic.keys())[0])
    lst = np.zeros(2**n_measured_qbits)
    canonical = 2**np.arange(0,n_measured_qbits)[::-1]

    for key, value in dic.items():
        entry = np.sum(canonical*np.array(list(key), dtype=int))
        lst[entry] = value                   
    return lst

@generalize
def probs(dic: dict) -> np.ndarray:
    ''' 
    Takes a dictionary whose keys are bit strings (e.g. '101') and
    returns an array with the normalized values of the dic in the indices of the bit strings.
    example:
        >>> probs({'0': 19835, '1': 165})
        array([0.99175, 0.00825])
    '''
    n_measured_qbits = len(list(dic.keys())[0])
    shots = sum(list(dic.values()))
    lst = np.zeros(2**n_measured_qbits)
    canonical = 2**np.arange(0,n_measured_qbits)[::-1]

    for key, value in dic.items():
        entry = np.sum(canonical*np.array(list(key), dtype=int))
        lst[entry] = value/shots               
    return lst

def bit_lst_toarray(lst: list) -> np.ndarray:
    ''' 
    Takes a list whose entries are bit strings (e.g. '101') and
    returns an array with 1's in the indices of the bit strings.
    example:
        >>> bit_lst_toarray(['00', '01'])
        array([1., 1., 0., 0.])
    '''
    n_qubits = len(lst[0])
    arr = np.zeros(2**n_qubits)
    canonical = 2**np.arange(0,n_qubits)[::-1]

    for bit_str in lst:
        entry = np.sum(canonical*np.array(list(bit_str), dtype=int))
        arr[entry] = 1.
    return arr

@generalize
def permutation_filter(dic: dict, i: int, j: int) -> dict:
    '''
    Returns a copy of dic where entries that have 
    dic[i] != dic[j] are removed.
    example:
        >>> permutation_filter({'100': 243, '001': 35}, 0, 1)
        {'001': 35}
    '''
    new_dic = {}
    for key, value in dic.items():
        if key[i] == key[j]:
            new_dic[key] = value
    return new_dic

@generalize
def kill(dic: dict, bit_indices: list) -> dict:
    '''
    Returns a dictionary of the total counts in each different state
    the qubits with expect the qubits with "bit_indices" are in.
    example:
        >>> kill({'000': 2735, '011': 2749, '100': 13}, [0])
        {'00': 2748, '11': 2749}
    '''
    new_dic = {}
    for key, value in dic.items():
        entry = substr_kill(key, bit_indices)
        new_dic[entry] = new_dic.get(entry, 0) + value
    return new_dic

def norm(arr: np.ndarray) -> np.float64:
    '''
    Returns the the norm of a complex array:
        >>> norm(np.array([1+1j, 2]))
        6.0
    '''
    return np.sum(np.absolute(arr)**2)

def is_normalized(arr: np.ndarray) -> bool:
    eps = 1e-12
    return np.abs(1 - norm(arr)) < eps

def normalize(arr: np.ndarray) -> np.ndarray:
    ''' 
    Returns the normalized version of a vector, where its norm
    is given by the sum of the squared norm of its entries.
    example:
        >>> normalize([1+3j, 2, 3])
        array([0.20851441+0.62554324j, 0.41702883+0.j, 0.62554324+0.j])
    '''
    return arr / np.sqrt(norm(arr))

def proj_matrix(state):
    '''
    Returns the projection matrix |psi><psi|.
    example:
        >>> proj_matrix(np.array([1, 1j])/np.sqrt(2))
        array([[0.5+0.j , 0. -0.5j],
               [0. +0.5j, 0.5+0.j ]])
    '''
    return np.outer(state, state.conjugate())

def isunitary(M):
    '''
    Checks if M is a (complex) unitary matrix.
    example:
        >>> isunitary(np.array([[-0.2776-0.5307j, -0.1057+0.7938j],
                                [-0.1057-0.7938j,  0.2776-0.5307j]]))
        True
    '''
    return np.allclose(np.eye(len(M)), M.dot(M.T.conj()))

def bit_combinations(s: str) -> list:
    from itertools import product
    '''
    Input must be in the form '00.1000000', returns
    all possible bit strings of that form, having 0's
    and 1's in the place of '.'.
    example:
        >>> bit_combinations('00.1000000')
        ['0001000000', '0011000000']
    '''
    arr = np.array(list(s), dtype=object)
    dots_pos = np.where(arr == '.')[0]
    bit_lst = [list(''.join(str(bit) for bit in bits)) for bits in product([0, 1], repeat=len(dots_pos))]
    result = []
    for bit in bit_lst:
        arr[dots_pos] = bit
        result.append(''.join(str(b) for b in arr))
    return result

def merge_dicts(dic_lst):
    from functools import reduce
    from collections import Counter
    return dict(reduce(lambda x,y: x.update(y) or x, (Counter(dict(x)) for x in dic_lst)))

def time_name(s: str) -> str:
    from datetime import datetime
    '''
    Adds a timestamp to a given path.
    example:
        >>> filename('figures/nova.png')
        'figures/2021-12-07--19:38:55--nova.png'
    '''
    l = s.rsplit('/', 1)
    now = datetime.now()
    return l[0]+now.strftime('/%Y-%m-%d--%H:%M:%S--')+l[1]

def save_obj(obj, filename):
    '''
    Saves the object to a text file, for later loading.
    '''
    import pickle

    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    return 

def load_obj(filename):
    '''
    Loads a saved object.
    '''
    import pickle

    with open(filename, 'rb') as inp:
        obj = pickle.load(inp)
    return obj
