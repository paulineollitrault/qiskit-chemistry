import itertools
import logging
import sys

import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.tools import parallel_map
from qiskit.chemistry import FermionicOperator
from qiskit import QuantumRegister, QuantumCircuit
from .qiskit_chemistry_error import QiskitChemistryError
from .bksf import bksf_mapping
from qiskit.aqua import Operator, aqua_globals
from qiskit.tools.events import TextProgressBar

logger = logging.getLogger(__name__)


class RotatedFermionicOperator(FermionicOperator):
    r"""
    A set of functions to rotate the fermionic Hamiltonians to
    a better basis for qubit Hamiltonians.

    References:
     - M. Motta, E. Ye, J. R. McClean, Z. Li, A. J. Minnich, \
        R. Babbush and G. Kin-Lic Chan \
        arXiv:1808.02625
    """

    def __init__(self, h1, h2, cholesky_threshold = 1e-10, svd_threshold = 1e-10):
        super().__init__(h1, h2=h2)

        h1_new, h2_new, rotations, Us = self.rotate_basis(h1, h2, cholesky_threshold, svd_threshold)
        self.h1 = h1_new
        self.h2 = h2_new
        self.rotations = rotations
        self.Us = Us


    def rotate_basis(self, h1, h2, cholesky_thresold = 1e-10, svd_threshold = 1e-10):

        num_qubits = len(h1)
        l = num_qubits ** 2

        # supermatrix (of eq. 5)
        V = np.zeros((l, l))

        # state initialization single
        qr = QuantumRegister(num_qubits, 'q')

        # build ps, qr matrix
        # indexs: ps=s+N*p , qr=r+N*q

        # implementation of eq.5

        # for chem notation
        for p in range(num_qubits):
            for s in range(num_qubits):
                for r in range(num_qubits):
                    for q in range(num_qubits):

                        if (q == s):
                            h1[p, r] += 0.5 * 2 * h2[p, r, q, s]

                        if h2[p, r, q, s] != 0:
                            V[s + num_qubits * p, r + num_qubits * q] -= 2 * h2[p, r, q, s]

        # calculate cholesky vectors
        # corresponds to the number of measurement (grouped Pauli needed)

        L_tot = RotatedFermionicOperator._cholesky_decomposition(V, threshold=cholesky_thresold)

        self.num_basis = len(L_tot)

        rotations = [[]]
        Us = [np.diag(np.ones(num_qubits))]
        h2_new = []
        count = 0
        num_givens = 0

        for L in L_tot:

            rotations.append([])
            h2_new.append(np.zeros((num_qubits, num_qubits)))
            # state preparation double

            eig_val, U = RotatedFermionicOperator._SVD(L, threshold=svd_threshold) #singular value decomposition: eq. 7

            Us.append(U)

            # calculate all needed givens rotation and R
            Givens_array, Q, R = self._qr_decomposition(U)

            num_givens += len(Givens_array)

            for i in range(len(R)):
                for j in range(len(R)):
                    if i != j:
                        R[i][j] = 0
                    else:
                        R[i][j] = np.sign(R[i][j])


            # rotate state
            for rot in reversed(Givens_array):
                rotations[count+1].append(RotatedFermionicOperator._givens_to_hilbertspace(rot, qr))

            # add R to circuit has no effect
            # circuit_qis_L += R_to_hilbert(R, N, qr)
            #this can be commented out but check in case but should just add a global phase

            # build double Hamiltonian see eq.3
            for i in range(len(eig_val)):
                for j in range(len(eig_val)):

                    h2_new[count][i][j] = eig_val[i] * eig_val[j] / 2.

            count+=1

        return h1, h2_new, rotations, Us



    def mapping(self, map_type, threshold=1e-10):
        """Map fermionic operator to qubit operator.

        Using multiprocess to speedup the mapping, the improvement can be
        observed when h2 is a non-sparse matrix.

        Args:
            map_type (str): case-insensitive mapping type.
                            "jordan_wigner", "parity", "bravyi_kitaev", "bksf"
            threshold (float): threshold for Pauli simplification

        Returns:
            Operator: create an Operator object in Paulis form.

        Raises:
            QiskitChemistryError: if the `map_type` can not be recognized.
        """
        """
        ####################################################################
        ############   DEFINING MAPPED FERMIONIC OPERATORS    ##############
        ####################################################################
        """

        self._map_type = map_type
        n = self._modes  # number of fermionic modes / qubits
        map_type = map_type.lower()
        if map_type == 'jordan_wigner':
            a = self._jordan_wigner_mode(n)
        elif map_type == 'parity':
            a = self._parity_mode(n)
        elif map_type == 'bravyi_kitaev':
            a = self._bravyi_kitaev_mode(n)
        elif map_type == 'bksf':
            return bksf_mapping(self)
        else:
            raise QiskitChemistryError('Please specify the supported modes: '
                                       'jordan_wigner, parity, bravyi_kitaev, bksf')
        """
        ####################################################################
        ############    BUILDING THE MAPPED HAMILTONIAN     ################
        ####################################################################
        """
        operator_list = []
        pauli_list = Operator(paulis=[])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Mapping one-body terms to Qubit Hamiltonian:")
            TextProgressBar(output_handler=sys.stderr)
        results = parallel_map(FermionicOperator._one_body_mapping,
                               [(self.h1[i, j], a[i], a[j])
                                for i, j in itertools.product(range(n), repeat=2) if self.h1[i, j] != 0],
                               task_args=(threshold,), num_processes=aqua_globals.num_processes)
        for result in results:
            pauli_list += result
        pauli_list.chop(threshold=threshold)

        operator_list.append(pauli_list)

        for _h2 in self.h2:
            pauli_list = Operator(paulis=[])
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Mapping two-body terms to Qubit Hamiltonian:")
                TextProgressBar(output_handler=sys.stderr)
            results = parallel_map(RotatedFermionicOperator._two_body_mapping,
                                   [(_h2[i, j], a[i], a[j])
                                    for i, j in itertools.product(range(n), repeat=2) if _h2[i, j] != 0],
                                   task_args=(threshold,), num_processes=aqua_globals.num_processes)
            for result in results:
                pauli_list += result
            pauli_list.chop(threshold=threshold)

            operator_list.append(pauli_list)

        return operator_list

    @staticmethod
    def _two_body_mapping(h2_ij_a_ij, threshold):
        """
        Subroutine for two body mapping. We use the chemists notation
        for the two-body term, h2(i,j,k,m) adag_i a_i adag_j a_j.

        Args:
            h2_ijkm_aijkm (tuple): value of h2 at index (i,j,k,m),
                                   pauli at index i, pauli at index j,
                                   pauli at index k, pauli at index m
            threshold: (float): threshold to remove a pauli

        Returns:
            Operator: Operator for those paulis
        """
        h2_ij, a_i, a_j = h2_ij_a_ij
        pauli_list = []

        for alpha in range(2):
            for beta in range(2):
                for gamma in range(2):
                    for delta in range(2):
                        pauli_prod_1 = Pauli.sgn_prod(a_i[alpha], a_i[beta])
                        pauli_prod_2 = Pauli.sgn_prod(pauli_prod_1[0], a_j[gamma])
                        pauli_prod_3 = Pauli.sgn_prod(pauli_prod_2[0], a_j[delta])

                        phase1 = pauli_prod_1[1] * pauli_prod_2[1] * pauli_prod_3[1]
                        phase2 = np.power(-1j, alpha + gamma) * np.power(1j, beta + delta)
                        pauli_term = [h2_ij / 16 * phase1 * phase2, pauli_prod_3[0]]
                        if np.absolute(pauli_term[0]) > threshold:
                            pauli_list.append(pauli_term)
        return Operator(paulis=pauli_list)

    @staticmethod
    def _cholesky_decomposition(V, threshold=1.e-20):
        ''' following :Chapter 13
                        Cholesky decomposition techniques in electronic
                        structure theory
                        Francesco Aquilante, Linus Boman, Jonas Bostr¨om, Henrik Koch, Roland Lindh,
                        Alfredo S´anchez de Mer´as and Thomas Bondo Pedersen
            retuns [Cholesky vector of the matrix], ...
            Bem: biggest diagonal elements is choosen first
            numerically more staple
                        '''
        # inizialisation
        exact_V = V
        L = []
        L_tot = []
        approx_V = np.zeros_like(V)

        while np.any((exact_V - approx_V) > threshold):
            diag = np.diag(V)
            i = np.unravel_index(diag.argmax(), diag.shape) # i is the index of the biggest number in the diagonal of V

            if V[i][i] > 0: #2-electron supermatrix should always be positive definite

                for j in range(len(V)):
                    L.append(V[j][i] / np.sqrt(V[i][i]))

                dif = np.outer(np.transpose(np.conjugate(L)), L) # matrix out of the choleski vector
                V = V - dif #substract the contribution of this vector to the matrix V
                approx_V += dif
                L_tot.append(L)
                L = []

            else:
                raise Exception("Algorithm was not able to converge")
        L_tot = np.array(L_tot)

        return L_tot

    @staticmethod
    def _SVD(L, threshold=1.e-30):
        N = int(np.sqrt(len(L)))
        L = L.reshape((N, N))

        # make sure L is hermitian but it should be already
        L = 0.5 * (L + np.transpose(np.conjugate(L)))

        eig_val, U = np.linalg.eigh(L)

        # sort U
        ind = abs(eig_val).argsort()
        ind = ind[::-1]
        U = U[:, ind]
        eig_val = eig_val[ind] #sorting to get biggest eigenvalue first

        sum = 0
        for j in range(len(eig_val) - 1, -1, -1):
            sum += abs(eig_val[j])
            if sum < threshold:
                eig_val[j] = 0 # sum the eigenvalues till we reach a threshold (page 3 second column first line)

        return eig_val, U

    @staticmethod
    def _givens_rot(a, b):
        if abs(b) == 0:
            c = 1.
            s = 0.


        elif abs(b) > abs(a):

            r = a / b
            s = 1. / np.sqrt(1. + r ** 2)
            c = s * r
        else:

            r = b / a
            c = 1. / np.sqrt(1 + r ** 2)
            s = c * r

        return c, s

    @staticmethod
    def _qr_decomposition(A):
        # see link https://stackoverflow.com/questions/13438073/qr-decomposition-algorithm-using-givens-rotations
        Givens_array = []
        n = len(A)
        Q = np.eye(n)
        R = A

        for j in range(n):
            for i in range(n - 1, j, -1):
                G = np.eye(n)
                c, s = RotatedFermionicOperator._givens_rot(R[i - 1][j], R[i][j])
                G[i - 1][i - 1] = c
                G[i][i] = c
                G[i][i - 1] = s
                G[i - 1][i] = -s

                if c != 1:
                    Givens_array.append(G)

                R = np.dot(np.transpose(np.conjugate(G)), R)
                Q = np.dot(Q, G)  # not needed

        return Givens_array, Q, R

    @staticmethod
    def _givens_to_hilbertspace(U, qr):  # U : rotations, qr: quantum register
        diag = np.diag(U)
        # look for the indices of interest (the cosines are always on the diagonal so then we also know the sinus)
        i = next((i for i, x in enumerate(diag) if x != 1), None)
        j = next((i for i, x in enumerate(diag[i:]) if x != 1), None) + 1 + i

        a = U[i][i]
        b = U[i][j]
        c = U[j][i]
        d = U[j][j]

        return RotatedFermionicOperator._to_hilbert(i, j, a, b, c, d, qr)

    @staticmethod
    def _find_angle(a, b):
        # we have a = cos(theta) and we want theta
        angle = 0
        if a >= 0 <= b:  # check in which part of the trigonometric circle we are
            angle = np.arcsin(a)
        if a >= 0 > b:
            angle = np.arccos(b)
        if a < 0 > b:
            angle = np.arccos(-b) + np.pi
        if a < 0 <= b:
            angle = np.arcsin(a) + 2 * np.pi

        return angle

    @staticmethod
    def _to_hilbert(i, j, a, b, c, d, qr):
        omega = RotatedFermionicOperator._find_angle(c, a)

        if omega != np.pi / 2 and omega != 3 * np.pi / 2:
            d /= np.cos(omega)
            phi = RotatedFermionicOperator._find_angle(np.imag(d), np.real(d))
        else:
            b /= -np.sin(omega)
            phi = RotatedFermionicOperator._find_angle(np.imag(b), np.real(b))

        if phi != 0:
            print('/!\ COMPLEX')

        #(AB).T = B.T A.T

        circuit = QuantumCircuit(qr)
        circuit.rz(phi, qr[j])
        circuit.cx(qr[j], qr[i])
        circuit.cu3(-2 * omega, 0., 0., qr[i], qr[j])
        circuit.cx(qr[j], qr[i])

        return circuit
