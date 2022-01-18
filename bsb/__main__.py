from . import cli

print("Fragile MPI test")
from mpi4py import MPI

print("MPI ramk:", MPI.COMM_WORLD.Get_rank())
print("MPI size:", MPI.COMM_WORLD.Get_size())
cli.scaffold_cli()
