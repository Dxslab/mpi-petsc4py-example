import numpy as np
import petsc4py
import sys

petsc4py.init(sys.argv)

from mpi4py import MPI
from petsc4py import PETSc
from scipy.sparse import random

# Create a set of matrices and vectors for testing that satisfy: AX = B
def create_system(size, seed=42, density=0.1):
  rng = np.random.default_rng(seed=seed)
  A = random(size, size, density=density, format='csr', dtype=np.double, random_state=rng)
  X = rng.random(size)
  B = A.dot(X)
  return A, X, B

def solver_petsc_i(csr, shape, rhs, solver, comm=None):
  # Uppercase for numpy sparse arrays
  # Lowercase for PETSc arrays/matrices

  #---- Set up arrays -----
  a = PETSc.Mat().createAIJ(comm=comm, size=shape, csr=csr)

  a.setUp()
  a.assemblyBegin()
  a.assemblyEnd()
  x, b = a.getVecs()
  b.setArray(rhs)

  #---- Set up solver -----
  ksp = PETSc.KSP().create(comm=comm)

  # This implements a method that applies ONLY the preconditioner exactly once.
  # This may be used in inner iterations, where it is desired to allow multiple iterations as well as the "0-iteration" case.
  # It is commonly used with the direct solver preconditioners like PCLU and PCCHOLESKY
  ksp.setType('preonly')

  pc = ksp.getPC()
  pc.setType('lu')

  pc.setFactorSolverType(solver)

  ksp.setOperators(a)
  ksp.setFromOptions() # Apply any command line options
  ksp.setUp()

  #---- Solve -----
  ksp.solve(b, x)

  return x

#---- Get MPI info ----
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank   = comm.Get_rank()

if rank == 0:
  # Set up matrix & vector on master MPI task, as well as solution vector
  size = 100
  A, X_actual, B_all = create_system(size=size, seed=42, density=0.1)

  shape = A.shape
  nrows = shape[0]

  # split the number of rows evenly (as possible) among the MPI tasks
  N_pertask, extra = divmod(nrows, nprocs)

  # count: the size of each sub-task
  count = np.array([N_pertask + 1 if i < extra else N_pertask for i in range(nprocs)], dtype=int)

  # displacement: the starting index of each sub-task
  displ = np.array([sum(count[:i]) for i in range(nprocs)])

  #---- Send the relevant subsets of A and B to each slave MPI task ----
  for i in range(1,nprocs):

    # Get the start and end row index for this MPI task
    rstart = displ[i]
    rend   = displ[i] + count[i]

    #---- Get the subsets of A and B using these rows ----
    A_indptr  = A.indptr[rstart:rend+1] - A.indptr[rstart]  # modified row-pointer array that will be consistent on the MPI task

    pstart    = A.indptr[rstart]    # starting row-pointer index
    pend      = A.indptr[rend]      # end      row-pointer index

    A_indices = A.indices[pstart:pend]
    A_data    = A.data[pstart:pend]
    B         = B_all[rstart:rend]

    # Save the lengths of each array
    lengths = {
      'A_indptr' : len(A_indptr),
      'A_indices': len(A_indices),
      'A_data'   : len(A_data),
      'B'        : len(B)
    }

    # Send the arrays and their lenghts to the relevant MPI task
    comm.send(lengths,  dest=i)
    comm.Send(A_indptr, dest=i)
    comm.Send(A_indices,dest=i)
    comm.Send(A_data,   dest=i)
    comm.Send(B,        dest=i)

  #---- Set the relevant subsets of A and B for the master MPI task (we don't need to do an MPI Send)
  rstart = displ[0]
  rend   = displ[0] + count[0]

  A_indptr  = A.indptr[rstart:rend+1] - A.indptr[rstart]
  pstart    = A.indptr[rstart]
  pend      = A.indptr[rend]
  A_indices = A.indices[pstart:pend]
  A_data    = A.data[pstart:pend]
  B         = B_all[rstart:rend]

else:
  # Receive the array lengths
  lengths   = comm.recv(source=0)
  # Initialise the buffers
  A_indptr  = np.empty(lengths['A_indptr'],  dtype=np.int32)
  A_indices = np.empty(lengths['A_indices'], dtype=np.int32)
  A_data    = np.empty(lengths['A_data'],    dtype=np.double)
  B         = np.empty(lengths['B'],         dtype=np.double)
  # Receive the arrays
  comm.Recv(A_indptr,  source=0)
  comm.Recv(A_indices, source=0)
  comm.Recv(A_data,    source=0)
  comm.Recv(B   ,      source=0)

  shape = None

# Broadcast matrix shape to each MPI task
shape = comm.bcast(shape, root=0)

x = solver_petsc_i(csr=(A_indptr,A_indices,A_data), shape=shape, rhs=B, solver='mumps', comm=comm)

#---- Gather the solution onto a single array on the master MPI task
if rank == 0:
  X = np.empty(nrows,dtype=np.double)
else:
  X = None
comm.Gatherv(x.array,X)

#---- Compare solution to expected answer
if rank == 0:
  print(np.allclose(X,X_actual))
