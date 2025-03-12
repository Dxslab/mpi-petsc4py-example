from petsc4py import PETSc
from slepc4py import SLEPc


def createPETScMat(comm, shape, csr):
    A = PETSc.Mat().createAIJ(comm=comm, size=shape, csr=csr)
    A.assemble()
    # A.view()

    return A


def solveSLEPcEigenvalues(comm, A):
    E = SLEPc.EPS().create(comm=comm)
    E.setOperators(A)
    E.setProblemType(SLEPc.EPS.ProblemType.HEP)
    E.setFromOptions()
    E.solve()

    return E
