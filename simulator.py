import numpy as np
from qiskit import QuantumCircuit, transpile
import warnings
warnings.filterwarnings('ignore')
from qiskit import *
from qiskit_aer.noise import NoiseModel
import qiskit_aer.noise as noise
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.fake_provider import *
import pickle
import pandas as pd
from qiskit import QuantumCircuit, transpile
import time
from qiskit_aer.primitives import Estimator
import pickle
from utils import *  
from zne.noise_amplification import LocalFoldingAmplifier   
from mitiq import zne
import argparse

class Color:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m' 

def evaluate(noise_id, seeds):

    noise_model_lst = []
    with open('fakekolkata.pkl', 'rb') as file:
        noise_model_lst.append(pickle.load(file))
    with open('fakemontreal.pkl', 'rb') as file:
        noise_model_lst.append(pickle.load(file))
    with open('fakecairo.pkl', 'rb') as file:
        noise_model_lst.append(pickle.load(file))

    noise_model1 = noise.NoiseModel()
    noise_model = noise_model1.from_dict(noise_model_lst[noise_id - 1])  #[kata, mont, cairo]

    if noise_id == 1:
        shot = 6000
        noise_factors = range(1, 6, 2)  #[1,3,5]
    elif noise_id ==2:
        shot = 6000
        noise_factors = range(1, 10, 2)  #[1,3,5,7,9]
    else:
        shot = 1000
        noise_factors = range(3, 8, 2)  #[3,5,7]

    circuit = QuantumCircuit.from_qasm_file('circuit_{}'.format(noise_id))
    system_model = FakeMontreal()
    transpiled_circuit = transpile(circuit, backend=system_model, seed_transpiler=seeds)

     # Obtain the Duration of Quantum Circuit
    from qiskit import pulse

    with pulse.build(system_model) as my_program1:
        with pulse.transpiler_settings(optimization_level=0):
            pulse.call(transpiled_circuit)

    res=transpiled_circuit.layout.initial_layout.get_virtual_bits()
    maptable={}
    for s in res:    
        if 'ancilla' not in s.register.name:
            maptable[s.index] = res[s]    
    # generate the observable
    with open('Hamiltonian/observable', 'rb') as file:
        [op_list, coeffs] = pickle.load(file)

    labels= list(map(lambda s:rerange(s,maptable),op_list))
    observable = SparsePauliOp(labels, coeffs)

    # group observables using Qiskit built-it function group_commuting()
    grouped_observable = observable.group_commuting()   # 39 groups
    
        
    estimator = Estimator(
        backend_options = {
            'method': 'statevector',
            'device': 'CPU',
            'noise_model': noise_model
        },
        run_options = {
            'shots': shot,
            'seed': seeds,
        },
        skip_transpilation=True
    ) 

    # Specify ZNE strategy  
    local_amplifier = LocalFoldingAmplifier() #(gates_to_fold=2, random_seed=seeds)
    circuits = [] 
    for i in noise_factors:
        noisy_circuit = local_amplifier.amplify_circuit_noise(circuit=transpiled_circuit, noise_factor=i)      
        circuits.append(noisy_circuit)

    s = time.time()    
   
    results = []    
    for i in range(len(noise_factors)):
        result = 0
        for num_group in range(len(grouped_observable)):
            job = estimator.run(circuits[i], grouped_observable[num_group])            
            result_values = job.result().values[0]            
            result += result_values
        results.append(result)
    if noise_id != 2:
        result = avs(results)
    else:
        result = zne.RichardsonFactory.extrapolate(noise_factors, results)
    e = time.time()

    print('\nseed:', seeds)
    print('Energy:', result)
    print('Duration:', my_program1.duration)
    print('Time:', e-s) 

    return [result, my_program1.duration], results    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument(
        '--noise', type=int, default=1, help='noise id')
    parser.add_argument(
        '--seeds', nargs='+', type=int, default=[20, 21, 30, 33, 36, 42, 43, 55, 67, 170], help='Seeds')
    args = parser.parse_args()
    
    noise_id = args.noise    
    seeds = args.seeds

    df = pd.DataFrame(columns=['Seed', 'Energy'])
    df['Seed'] = seeds + ['Mean', 'Duration']

    result_file = 'results_{}.csv'.format(noise_id)    
    results = []
        
    for j in range(len(seeds)):
        result, lst = evaluate(noise_id, seeds[j])   # [energy, duration], result_list            
        results.append(result)                           
        df.loc[j, 'Energy'] = result[0].item()
        df.to_csv(result_file, index=False)

    mean = np.mean(results, axis = 0)
    df.loc[df['Seed'] == 'Mean', 'Energy'] = mean[0].item()
    df.loc[df['Seed'] == 'Duration', 'Energy'] = mean[1] 
    df.to_csv(result_file, index=False)    

    print(Color.YELLOW + 'Noise {} Energy: {}'.format(noise_id, mean[0]) + Color.RESET)  
    print(Color.YELLOW + 'Noise {} Duration: {}'.format(noise_id, mean[1]) + Color.RESET)

    



