import sys, argparse, h5py
from .morphologies import MorphologyRepository
from .plotting import plot_morphology, plot_voxel_morpho_map
import matplotlib.pyplot as plt

class ReplState:
    def __init__(self):
        self.exit = False
        self.state = 'base'
        self.question = None
        self.prefix = None
        self.reply = None
        self.next = None
        self.command = None

    def repl(self):
        self.command = input('{}{}{}> '.format(
            (self.reply or '') + ('\n' if self.reply else ''),
            (self.question or '') + ('\n' if self.question else ''),
            self.prefix or ''
        ))
        self.exit = self.command == "exit"
        self.state = self.next or self.state
        self.next = None
        self.question = None
        self.prefix = None
        self.reply = None

def debug_voxel_cloud(scaffold):
    # REPL
    # Bullshit hacky code, just needed to debug. We should make a real REPL at some point.
    repo_file = input("Specify the morphology repository: ")
    repo = MorphologyRepository(repo_file if len(repo_file) > 0 else None)
    state = ReplState()
    while not state.exit:
        state.repl()
        if state.command == 'list all':
            state.reply = repo.list_all_morphologies()
            if len(state.reply) == 0:
                state.reply = "<None>"
            else:
                state.reply = "All available morphologies: " + ", ".join(state.reply)
        elif state.command == 'list voxelized':
            state.reply = repo.list_all_voxelized()
            if len(state.reply) == 0:
                state.reply = "<None>"
            else:
                state.reply = "All voxelized morphologies: " + ", ".join(state.reply)
        elif state.command[:10] == 'import swc':
            eof_name = state.command[11:].find(' ') + 11
            name = state.command[11:eof_name]
            file = state.command[(eof_name+1):]
            repo.import_swc(file, name, overwrite=True)
            state.reply = "Added '{}' as '{}' to the repository.".format(file, name)
        elif state.command[:6] == "remove":
            repo.remove_morphology(state.command[7:])
        elif state.command[:4] == "plot":
            name = state.command[5:]
            if repo.morphology_exists(name):
                morphology = repo.get_morphology(name)
                if repo.voxel_cloud_exists(name):
                    plot_voxel_morpho_map(morphology)
                else:
                    plot_morphology(morphology)
                plt.show()
            else:
                state.reply = "Unknown morphology '{}'".format(name)

def debug_hdf5(scaffold):
    df = chr(172)
    def format_level(lvl, sub=None):
        return ' ' * (lvl * 3) + (sub or df)

    def format_self(obj, name, lvl):
        print(format_level(lvl) + name)
        if hasattr(obj, "attrs"):
            for attr in obj.attrs.keys():
                print(' ' + format_level(lvl, '.') + attr + ' = ' + str(obj.attrs[attr])[:min(100, len(str(obj.attrs[attr])))] + ('...' if len(str(obj.attrs[attr])) > 100 else ''))
        if hasattr(obj, "keys"):
            for key in obj.keys():
                format_self(obj[key], obj[key].name, lvl + 1)

    hdf5_file = input("Specify the hdf5 file: ")
    with h5py.File(hdf5_file, 'r+') as f:
        format_self(f, str(f.file), 0)
