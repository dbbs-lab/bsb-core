import sys, argparse
from .output import MorphologyRepository

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
    repo_file = input("Specify the morphology repository: ")
    repo = MorphologyRepository(repo_file if len(repo_file) > 0 else None)
    repo.initialise(scaffold)
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
