from mpi4py import MPI
import numpy as np
from scipy.sparse import csr_matrix
import petsc_funcs as pet

def create(nsize):
    A = np.zeros((nsize, nsize))

    # Fill the matrix with some values
    for i in range(nsize):
        for j in range(nsize):
            if i == j or i == j - 1 or i == j + 1:  # Example pattern
                A[i, j] = i + j + 1

    # Convert the dense matrix to CSR format
    A_csr = csr_matrix(A)

    return A_csr


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    if rank == 0:
        nsize = 100
        CSR = create(nsize)

        shape = CSR.shape
        nrows = shape[0]

        N_pertask, extra = divmod(nrows, nprocs)

        count = np.array([N_pertask + 1 if i < extra else N_pertask for i in range(nprocs)], dtype=int)

        displ = np.array([sum(count[:i]) for i in range(nprocs)])

        for i in range(1,nprocs):
            
            rstart = displ[i]
            rend   = displ[i] + count[i]

            CSR_indptr  = CSR.indptr[rstart:rend+1] - CSR.indptr[rstart]
            
            pstart      = CSR.indptr[rstart]
            pend        = CSR.indptr[rend]

            CSR_indices = CSR.indices[pstart:pend]
            CSR_data    = CSR.data[pstart:pend]

            lengths = {
                'CSR_indptr' : len(CSR_indptr),
                'CSR_indices': len(CSR_indices),
                'CSR_data'   : len(CSR_data)
            }

            comm.send(lengths,    dest=i)
            comm.Send([CSR_indptr, MPI.INT], dest=i)
            comm.Send([CSR_indices, MPI.INT], dest=i)
            comm.Send([CSR_data, MPI.DOUBLE], dest=i)

        rstart = displ[0]
        rend   = displ[0] + count[0]

        CSR_indptr  = CSR.indptr[rstart:rend+1] - CSR.indptr[rstart]
        pstart      = CSR.indptr[rstart]
        pend        = CSR.indptr[rend]
        CSR_indices = CSR.indices[pstart:pend]
        CSR_data    = CSR.data[pstart:pend]

    else:
        lengths     = comm.recv(source=0)
        CSR_indptr  = np.empty(lengths['CSR_indptr'],  dtype=np.int32)
        CSR_indices = np.empty(lengths['CSR_indices'], dtype=np.int32)
        CSR_data    = np.empty(lengths['CSR_data'],    dtype=np.double)

        comm.Recv(CSR_indptr,  source=0)
        comm.Recv(CSR_indices, source=0)
        comm.Recv(CSR_data,    source=0)

        shape = None


    shape = comm.bcast(shape, root=0)

    A = pet.createPETScMat(comm, shape, (CSR_indptr, CSR_indices, CSR_data))
    E = pet.solveSLEPcEigenvalues(comm, A)

    nconv = E.getConverged()
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()

    if rank == 0:
        for i in range(nconv):
            k = E.getEigenpair(i, vr, vi)
            print("Eigenvalue: ", k)


if __name__=="__main__":
    main()
    