# QISKit ACQUA Chemistry

QISKit ACQUA Chemistry is a set of tools, algorithms and software for use with quantum computers
to carry out research and investigate how to take advantage of quantum computing power to solve chemistry
problems. QISKit ACQUA Chemistry translates chemistry-specific problem inputs into inputs for a quantum algorithm
supplied by QISKit ACQUA, and then uses [QISKit](https://www.qiskit.org/) for the actual quantum computation. 

QISKit ACQUA Chemistry allows users with different levels of experience to execute chemistry experiments and 
contribute to the software stack.  Users with pure chemistry background can continue to configure chemistry 
problems according to their favorite software packages, called *drivers*.  These users do not need to learn the 
details of quantum computing; QISKit ACQUA Chemistry translates any chemistry program configuration entered by 
any end user in their favorite driver into quantum-specific input.

QISKit ACQUA allows also users more knowledgeable in the area of quantum computing to plug their contributions in.
For example, new quantum algorithms, optimizers and variational forms can easily be plugged in,
thereby allowing algorithm providers to contribute new quantum algorithms or more efficient implementations of
existing ones. Such algorithms may then be utilized by QISKit ACQUA Chemistry as applicable.

You can follow the [installation](#installation) instructions to install this software and its dependencies.

Once you have it installed you can experiment with the library using either the supplied [GUI](#gui) or
[command line](#command-line) tools.

More advanced users and [developers](#developers) may wish to develop and add their own algorithms or other code.
As mentioned above, the library has been designed with several extension points where, for example, new algorithms,
optimizers and variational forms can be created, and simply dropping the code into the appropriate folder will
seamlessly integrate them and make them available for use and configuration via the input file.

## GUI and command line tools

QISKit ACQUA Chemistry has both GUI and command line tools which may be used when solving chemistry
problems. Both can load and run an [input file](#input-file) specifying the molecule, an algorithm to be used and its
configuration, and various other options to tailor the experiment. If you are new to the library we highly recommend
getting started with the GUI.  

### GUI 

The GUI allows provides an easy means to load and run an input file specifying your chemistry problem. An input file
can also be created, edited and saved with validation of values to provide ease of configuring the chemistry problem
using the input file.

The following will run the GUI

`python qiskit_acqua_chemistry/ui`:

### Command line

Summary of qiskit_acqua_chemistry command line options:

`python qiskit_acqua_chemistry`:
```
usage: qiskit_acqua_chemistry [-h] [-o output | -jo json output] input

Quantum Chemistry Program.

positional arguments:
  input            Chemistry Driver input or Algorithm JSON input file

optional arguments:
  -h, --help       show this help message and exit
  -o output        Algorithm Results Output file name
  -jo json output  Algorithm JSON Output file name
```

## Installation

QISKit ACQUA Chemistry requires Python 3.5 or newer to be installed.

The library can currently be obtained either by cloning this repository, or by downloading
the zip of the source and unpacking it locally on your machine. The library requires several additional packages
to be installed in your Python environment. They can simply be installed using the following:

`pip install -r requirements.txt` 

Additionally if you want to develop/run the unit tests then further packages are needed and can be installed as follows: 

`pip install -r requirements-dev.txt` 

Finally you will need to install a chemistry driver. See the chemistry drivers [readme](qiskit_acqua_chemistry/drivers/readme.md)
for more detail as the installation varies by driver due to their use of a separate chemistry program or chemistry 
library specific to that driver.

## Input file

An input file is used to define your chemistry problem. It contains at a minimum a definition of the molecule and 
associated configuration, such as a basis set, in order to compute the electronic structure using an external ab-initio
chemistry program or chemistry library via a chemistry driver. Further configuration can also be supplied to explicitly
control the processing and the quantum algorithm, used for the computation, instead of using defaulted values when
none are supplied.

Several sample input files can be found in the [examples](examples) folder 

An input file comprises the following main sections, although not all are mandatory:

#### NAME

NAME is an optional free format text section. Here you can name and describe the problem solved by the input file. For
example:

```
&NAME
H2 molecule experiment
Ground state energy computed via Variational Quantum Eigensolver
&END
``` 

#### DRIVER

DRIVER is a mandatory section. This section defines the molecule and associated configuration for the electronic
structure computation by the chosen driver via its external chemistry program or library. The exact form on the
configuration depends on the specific driver being used. See the chemistry drivers [readme](qiskit_acqua_chemistry/drivers/readme.md)
for more information about the drivers and their configuration. You will need to look at the readme of the driver you
are using to find out about the configuration. Here are a couple of examples. Note that the DRIVER section names
which specific chemistry driver will be used and that a subsequent section, in the name of the driver, then supplies
the driver specific configuration.

Here is an example using the [PYSCF driver](qiskit_acqua_chemistry/drivers/pyscfd/readme.md). The DRIVER section names PYSCF as the
driver and then a PYSCF section, corresponding to the name, provides the molecule and basis set that will be used by
the PYSCF driver and hence the PySCF library to compute the electronic structure.

```
&DRIVER
  name=PYSCF
&END

&PYSCF
  atom=H .0 .0 .0; H .0 .0 0.74
  basis=sto3g
&END
```

Here is another example using the [PSI4 driver](qiskit_acqua_chemistry/drivers/psi4d/readme.md). Here PSI4 is named as the driver to be
used and the PSI4 section contains the molecule and basis set directly in a form that PSI4 understands. This is the
Psithon input file language for PSI4, and thus should be familiar to existing users of PSI4.

```
&DRIVER
  name=PSI4
&END

&PSI4
molecule h2 {
   0 1
   H       .0000000000       0.0          0.0
   H       .0000000000       0.0           .2
}

set {
  basis sto-3g
  scf_type pk
}
&END
```

#### OPERATOR

OPERATOR is an optional section. This section can be configured to control the specific way the electronic
structure information, from the driver, is converted to QuBit operator form in order to be processed by the ALGORITHM.
The following parameters may be set:

* name=hamiltonian

  Currently 'hamiltonian' should be used as the name since there only one operator entity at present 

* transformation=**full** | particle_hole

  Do *full* transformation or use *particle_hole* 
  
  The 'standard' second quantized Hamiltonian can be transformed using the particle-hole (p/h) option, which makes the
  expansion of the trial wavefunction from the HartreeFock reference state more natural. For trial wavefunctions
  in QISKit ACQUA, such as UCCSD, the p/h Hamiltonian can improve the speed of convergence of the
  VQE algorithm for the calculation of the electronic ground state properties.
  For more information on the p/h formalism see: [P. Barkoutsos, arXiv:1805.04340](https://arxiv.org/abs/1805.04340).

* qubit_mapping=jordan_wigner | **parity** | bravyi_kitaev

  Desired mapping from fermion to qubit. Note: bravyi_kitaev is also known as the binary-tree-based qubit mapping. 

* two_qubit_reduction=**true** | false

  With parity mapping the operator can be reduced by two qubits 

* max_workers=*integer, default 4*

  Processing of the hamiltonian from fermionic to qubit can take advantage of multiple cpu cores to run parallel
  processes to carry out the transformation. The number of such worker processes used will not exceed the
  actual number of cup cores or this max_workers number, whichever is the smaller.

* freeze_core=true | **false**

  Whether to freeze core orbitals in the computation or not. Frozen core orbitals are removed from the
  subsequent computation by the ALGORITHM and a corresponding offset from this removal is added back into the
  final computed result. This may be combined with orbital_reduction below.

* orbital_reduction=[]

  The orbitals from the electronic structure can be simplified for the subsequent computation.
   
  With this parameter you can specify a list of orbitals, the default being an empty list, to be removed from the 
  subsequent computation. The list should be indices of the orbitals from 0 to n-1, where the electronic structure
  has n orbitals. Note: for ease of referring to the higher orbitals the list also supports negative values with -1
  being the highest unoccupied orbital, -2 the next one down and so on. Also note, that while orbitals may be listed to
  reduce the overall size of the problem, the result can be less accurate as a result of using this simplification.
  
  Any orbitals in the list that are occupied are frozen and an offset is computed from their removal. This is the same
  procedure as happens when freeze_core is specified except here you can specify exactly the orbitals you want.
  
  Any orbitals in the list that are unoccupied virtual orbitals are simply eliminated entirely from the
  subsequent computation.
  
  When a list is specified along with freeze_core of true the effective orbitals being acted up is the set
  from freeze_core combined with those specified here.     

Here is an example below where in addition to freezing the core orbitals a couple of other orbitals are listed. In this
example it assumes there were 10 orbitals so the highest two, unoccupied virtual orbitals, will be eliminated from the
subsequent computation in addition to the frozen core treatment:

```
&OPERATOR
 name=hamiltonian
 qubit_mapping=jordan_wigner
 freeze_core=true
 orbital_reduction=[8,9]
&END
```

Note the above could be specified the following way, which simplifies expressing the higher orbitals since the numbering 
is relative to the highest orbital and will always refer to the highest two orbitals. 
```
&OPERATOR
 name=hamiltonian
 qubit_mapping=jordan_wigner
 freeze_core=true
 orbital_reduction=[-2,-1]
&END
```

#### ALGORITHM

ALGORITHM is an optional section that allows you to define which quantum algorithm will be used by the computation.
The algorithm defaults to VQE (Variational Quantum Eigensolver), with a set of default parameters. 

According to each ALGORITHM you may add further sections to optionally configure the algorithm further. These sections
correspond to the pluggable entities that [developers](#developers) may choose to create and add more to the set
currently provided.

Here is an example showing the VQE algorithm along with OPTIMIZER and VARIATIONAL_FORM sections for the optimizer and
variational forms that are used by VQE. 

```
&ALGORITHM
  name=VQE
  shots=1
  operator_mode=matrix
&END

&OPTIMIZER
  name=L_BFGS_B
  factr=10
&END

&VARIATIONAL_FORM
  name=RYRZ
  entangler_map={0: [1]}
&END
``` 

For more information on algorithms, and any pluggable entities it may use, see QISKit ACQUA for more specifics about 
them and their configuration options.


#### BACKEND   

BACKEND is an optional section that includes naming the [QISKit](https://www.qiskit.org/) quantum computational
backend to be used for the quantum computation. This defaults to a local quantum simulator backend. 

* name=*'qiskit backend name'* 

  Defaults to 'local_statevector_simulator' but any suitable quantum backend can be selected. The QConfig.py file
  may need to be setup for QISKit to access remote devices.
  See [QISKit installation](https://qiskit.org/documentation/install.html#installation) for information on how to
  configure the QConfig.py
  
* shots=*integer defaults to 1024*

  With a backend such as local_qasm_simulator, or a real device, this is number of repetitions of each circuit
  for sampling to be used.

* skip_translation=**false** | true

  Skip circuit translation phase. If the algorithm uses only basis gates directly supported then no translation of
  the circuit into basis gates is required. Skipping the translation may improve overall performance a little
  especially when many circuits are used repeatedly such as is teh case with the VQE algorithm.
   
  *Note: use with caution as if the algorithm does not restrict itself to the set of basis gates supported by the
   backend then the circuit (algorithm) will fail to run.*     

#### PROBLEM

PROBLEM is an optional section that includes the overall problem being solved and overall problem level configuration

* name=**energy** | excited_states

  Specifies the problem being solved. Ensures that algorithms that can handle this class of problem are used.
  
* enable_substitutions=**true** | false

  During configuration some items may require matching their settings e.g. UCCSD variation form and HartreeFock
  initial state configuration need qubit_mapping and two_qubit_reduction to match what is set in [OPERATOR](#operator)
  section hamiltonian. Also some objects, like the aforementioned, may require the user to know number of particles,
  number of orbitals etc. for their configuration. To assist the user in this regard configuration substitutions
  are enabled by default.
  
  Substitutions use a predefined set of intra-section and computed values that are used to substitute (overwrite)
  any values in the targeted fields appropriately. If enable_substitutions is set false then the end user has the
  full responsibility for the entire configuration.
  
* random_seed=*An integer, default None*  

  Aspects of the computation may include use of random numbers. For instance VQE will often use a random initial
  point if the variation form does not supply any preference based on the initial state (and not overridden by a user
  supplied initial point). In this case each run of VQE, for an otherwise a constant problem, can result in a different
  result. And even if the final value might be the same the number of evaluations may differ. To enable repeatable
  experiments, with the exact same outcome, an integer random seed can be set so as the (pseudo-)random numbers will
  be generated the same each time the experiment is run.
    

## Developers

### Programming interface

The UI and Command line tools use qischem.py when solving the chemistry problem given by the supplied
input file. A programmatic interface is also available that can be called using a dictionary in the same formula as the
input file. Like the input file its parameters take on the same values and same defaults.

The dictionary can be manipulated programmatically, if desired, to vary the problem e.g. changing the interatomic
distance of the molecule, changing basis set, algorithm etc. You can find notebooks in the [examples](examples) folder
demonstrating this usage.

The code fragment below also shows such a dictionary and a simple usage.     

```
qischem_dict = {
    'driver': {'name': 'PYSCF'},
    'PYSCF': {'atom': '', 'basis': 'sto3g'},
    'algorithm': {'name': 'VQE'}
}
molecule = 'H .0 .0 -{0}; H .0 .0 {0}'
d = 0.74

qischem_dict['PYSCF']['atom'] = molecule.format(d/2) 
solver = QISChem()
result = solver.run(qischem_dict)
print('Ground state energy {}'.format(result['energy']))
```

Note: the [GUI](#gui) tool can export a dictionary from an [input file](#input-file). You can load an existing input
file or create a new one and then simply export it as a dictionary for use in a program.

### Result dictionary

As can be seen in the programming interface example above the QISChem run() method returns a result dictionary. The
dictionary contains the following fields of note:

* *energy*

  The ground state energy

* *energies*

  An array of energies comprising the ground state energy and any excited states if they were computed
  
* *nuclear_repulsion_energy*

  The nuclear repulsion energy 
  
* *hf_energy*

  The Hartree-Fock ground state energy as computed by the driver
  
* *nuclear_dipole_moment*, *electronic_dipole_moment*, *dipole_moment*

  Nuclear, electronic and combined dipole moments for x, y and z
  
* *total_dipole_moment*

  Total dipole moment 
 
* *algorithm_retvals* 

  The result dictionary of the algorithm that ran for the above values. See the algorithm for any further information.  

### For writers of algorithms and other utilities such as optimizers and variational forms:

QISKit ACQUA is the library of cross-domain algorithms and pluggable utilities. Please refer to the documentation 
there for more information on how to write and contribute such objects to QISKit ACQUA. Such objects are then available
to be used by QISKit ACQUA Chemistry.  

### For unit test writers:

Unit tests should go under "test" folder and be classes derived from QISKitAcquaChemistryTestCase class.

They should not have print statements, instead use self.log.debug. If they use assert, they should be from the unittest
package like self.AssertTrue, self.assertRaises etc.

For guidance look at the tests cases implemented at https://github.com/QISKit/qiskit-sdk-py/tree/master/test/python


### For unit test running:

To run all unit tests: `python -m unittest discover`

To run a particular unit test module: `python -m unittest test/test_end2end.py`

For help: `python -m unittest -h`

There are other running options at: https://docs.python.org/3/library/unittest.html#command-line-options

In order to see unit test log messages you need to set environment variable:
```
LOG_LEVEL=DEBUG
export LOG_LEVEL
```

The example above will save all results from "self.log.debug()" to a ".log" file with same name as the module used to
run. For instance "test_end2end.log" in the test folder.

## Additional Reading

Here are some references to other useful materials that may be helpful

* [Quantum optimization using variational algorithms on near-term quantum devices](https://arxiv.org/abs/1710.01022)
* [Fermionic Quantum Computation](https://arxiv.org/abs/quant-ph/0003137v2)