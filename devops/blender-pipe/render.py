import mpi4py.MPI, os, sys, subprocess

rank = mpi4py.MPI.COMM_WORLD.rank + 1
size = mpi4py.MPI.COMM_WORLD.size
f = sys.argv[1]
o = sys.argv[2]
e = "CYCLES" if len(sys.argv) < 4 or sys.argv[3] != "EEVEE" else "BLENDER_EEVEE"
if o[-1] != "/":
    o += "/"

if rank == 1:
    from pathlib import Path

    op = Path(o)
    print("pathinfo:", op.parts, len(op.parts), op.parts[:-1])
    if len(op.parts) > 1:
        Path(os.path.join(*op.parts[:-1])).mkdir(parents=True, exist_ok=True)

print(f"Starting blender {e} job")

subprocess.check_call(
    [
        "blender",
        "-b",
        f,
        "-E",
        e,
        "-o",
        o,
        "-s",
        str(rank),
        "-j",
        str(size),
        "-a",
    ]
)
mpi4py.MPI.COMM_WORLD.Barrier()
