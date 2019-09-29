from scaffold.helpers import dimensions, origin
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from .plotting import plot_voxelize_results
from scipy import ndimage
from random import choice as random_element
from time import sleep
from sklearn.neighbors import KDTree

class VoxelCloud:
    def __init__(self, bounds, voxels, grid_size, map, occupancies=None):
        self.bounds = bounds
        self.grid_size = grid_size
        self.voxels = voxels
        self.map = map
        self.occupancies = occupancies

    def get_boxes(self):
        return m_grid(self.bounds, self.grid_size)

    def get_occupancies(self):
        if self.occupancies is None:
            voxel_occupancy = np.array(list(map(lambda x: len(x), self.map)))
            max_voxel_occupancy = max(voxel_occupancy)
            normalized_voxel_occupancy = voxel_occupancy / (max_voxel_occupancy)
            self.occupancies = normalized_voxel_occupancy
        return self.occupancies

    def center_of_mass(self):
        boxes = self.get_boxes()
        points = boxes + self.grid_size / 2
        voxels = self.voxels
        occupancies = self.get_occupancies()
        point_positions = np.column_stack(points[:, voxels]).T
        return center_of_mass(point_positions, occupancies)

    @staticmethod
    def create(morphology, N):
        hit_detector, box_data = morphology_detector_factory(morphology)
        bounds, voxels, length, error = voxelize(N, box_data, hit_detector)
        # plot_voxelize_results(bounds, voxels, length)
        voxel_map = morphology.get_compartment_map(m_grid(bounds, length), voxels, length)
        if error == 0:
            return VoxelCloud(bounds, voxels, length, voxel_map)
        else:
            raise NotImplementedError("Pick random voxels and distribute their compartments to random neighbours")

_class_dimensions = dimensions
_class_origin = origin
class Box(dimensions, origin):
    def __init__(self, dimensions=None, origin=None):
        _class_dimensions.__init__(self, dimensions)
        _class_origin.__init__(self, origin)

    @staticmethod
    def from_bounds(bounds):
        dimensions = np.amax(bounds, axis=1) - np.amin(bounds,axis=1)
        origin = np.amin(bounds,axis=1) + dimensions / 2
        return Box(dimensions=dimensions, origin=origin)

def m_grid(bounds, size):
    return np.mgrid[
        bounds[0, 0]:bounds[0, 1]:size,
        bounds[1, 0]:bounds[1, 1]:size,
        bounds[2, 0]:bounds[2, 1]:size
    ]

def voxelize(N, box_data, hit_detector, max_iterations=200, precision_iterations=30):
    # Initialise
    bounds = np.column_stack((box_data.origin - box_data.dimensions / 2, box_data.origin + box_data.dimensions / 2))
    box_length = np.max(box_data.dimensions) # Size of the edge of a cube in the box counting grid
    best_length, best_error = box_length, N # Keep track of our best results so far
    last_box_count, last_box_length = 0., 0. # Keep track of the previous iteration for binary search jumps
    precision_i, i = 0., 0. # Keep track of the iterations
    crossed_treshold = False # Should we consider each next iteration as merely increasing precision?

    # Refine the grid size each iteration to find the right amount of boxes that trigger the hit_detector
    while i < max_iterations and precision_i < precision_iterations:
        i += 1
        if crossed_treshold: # Are we doing these iterations just to increase precision, or still trying to find a solution?
            precision_i += 1
        box_count = 0 # Reset box count
        boxes_x, boxes_y, boxes_z = m_grid(bounds, box_length) # Create box counting grid
        # Create a voxel grid where voxels are switched on if they trigger the hit_detector
        voxels = np.zeros((boxes_x.shape[0], boxes_x.shape[1], boxes_x.shape[2]), dtype=bool)
        # Iterate over all the boxes in the total grid.
        for x_i in range(boxes_x.shape[0]):
            for y_i in range(boxes_x.shape[1]):
                for z_i in range(boxes_x.shape[2]):
                    # Get the lower corner of the query box
                    x = boxes_x[x_i,y_i,z_i]
                    y = boxes_y[x_i,y_i,z_i]
                    z = boxes_z[x_i,y_i,z_i]
                    hit = hit_detector(np.array([x, y, z]), box_length) # Is this box a hit? (Does it cover some part of the object?)
                    voxels[x_i, y_i, z_i] = hit # If its a hit, turn on the voxel
                    box_count += int(hit) # If its a hit, increase the box count
        if last_box_count < N and box_count >= N:
            # We've crossed the treshold from overestimating to underestimating
            # the box_length. A solution is found, but more precise values lie somewhere in between,
            # so start counting the precision iterations
            crossed_treshold = True
        if box_count < N: # If not enough boxes cover the object we should decrease the box length (and increase box count)
            new_box_length = box_length - np.abs(box_length - last_box_length) / 2
        else: # If too many boxes cover the object we should increase the box length (and decrease box count)
            new_box_length = box_length + np.abs(box_length - last_box_length) / 2
        # Store the results of this iteration and prepare variables for the next iteration.
        last_box_length, last_box_count = box_length, box_count
        box_length = new_box_length
        if abs(N - box_count) <= best_error: # Only store the following values if they improve the previous best results.
            best_error, best_length = abs(N - box_count), last_box_length
            best_bounds, best_voxels = bounds, voxels

    # Return best results and error
    return best_bounds, best_voxels, best_length, best_error

def detect_box_compartments(tree, box_origin, box_size):
    '''
        Given a tree of compartment locations and a box, it will return the ids of all compartments in the box

        :param box_origin: The lowermost corner of the box.
    '''
    # Get the outer sphere radius of the cube by taking the length of a diagonal through the cube divided by 2
    search_radius = np.sqrt(np.sum([box_size ** 2 for i in range(len(box_origin))])) / 2
    # Translate the query point to the middle of the box and search within the outer sphere radius.
    return tree.query_radius([box_origin + box_size / 2], search_radius)[0]

def morphology_detector_factory(morphology):
    '''
        Will return a hit detector and outer box required to perform voxelization on the morphology.
    '''
    # Transform the compartment object list into a compartment position 3D numpy array
    tree = morphology.compartment_tree
    compartments = tree.get_arrays()[0]
    n_dimensions = range(compartments.shape[1])
    # Create an outer detection box
    outer_box = Box()
    # The outer box dimensions are equal to the maximum distance between compartments in each of n dimensions
    outer_box.dimensions = np.array([np.max(compartments[:, i]) - np.min(compartments[:, i]) for i in n_dimensions])
    # The outer box origin is in the middle of the outer bounds. (So lowermost point + half of dimensions)
    outer_box.origin = np.array([np.min(compartments[:, i]) + outer_box.dimensions[i] / 2 for i in n_dimensions])
    # Create the detector function
    def morphology_detector(box_origin, box_size):
        # Report a hit if more than 0 compartments are within the box.
        return len(detect_box_compartments(tree, box_origin, box_size)) > 0
    # Return the morphology detector function and box data as the factory products
    return morphology_detector, outer_box

def center_of_mass(points, weights = None):
    if weights is None:
        cog = [np.sum(points[dim, :]) / points.shape[1] for dim in range(points.shape[0])]
    else:
        cog = [np.sum(points[dim, :] * weights) for dim in range(points.shape[0])] / np.sum(weights)
    return cog

def set_attraction(attractor, voxels):
    attraction_voxels = np.indices(voxels.shape)[:, voxels].T
    attraction_map = np.zeros(voxels.shape)
    dist = np.sqrt(np.sum((attraction_voxels - attractor + np.ones(len(attractor)) * 0.5)**2, axis=1))
    distance_sorting = dist.argsort()[::-1]
    attraction = 1
    first_voxel = distance_sorting[0]
    attraction_map[attraction_voxels[first_voxel,0],attraction_voxels[first_voxel,1],attraction_voxels[first_voxel,2]] = 1
    last_distance = dist[first_voxel]
    for v in distance_sorting[1:]:
        distance = dist[v]
        attraction += int(distance < last_distance)
        attraction_map[attraction_voxels[v,0],attraction_voxels[v,1],attraction_voxels[v,2]] = attraction
        last_distance = distance
    return attraction_map

class AttractionGame:
    def __init__(self, attractor, field):
        self.players = []
        self.eliminated_players = []
        self.attractor = attractor
        self.field = field
        self.active_players = 0
        self.occupied = {}
        self.turn = 0
        self.artists = []
        self.paused = False

    def clear(self):
        self.players = []
        self.eliminated_players = []
        self.turn = 0
        self.occupied = {}
        for artist in self.artists:
            artist.remove()
        self.artists = []

    def occupy(self, player, new_position):
        if new_position in self.occupied:
            raise Exception("Position already occupied")
        self.unoccupy(player.position)
        self.occupied[new_position] = True
        player.position = new_position

    def unoccupy(self, position):
        if position in self.occupied:
            del self.occupied[position]

    def add_player(self, payload, position):
        if position in self.occupied:
            raise Exception("Position already occupied")
        player = AttractionPlayer(self, payload, position)
        self.players.append(player)
        # self.occupied[position] = True
        self.active_players += 1
        return player

    def eliminate_player(self, player):
        self.players.remove(player)
        self.eliminated_players.append(player)
        self.active_players -= 1

    def is_unoccupied(self, position):
        return not tuple(position) in self.occupied

    def is_out_of_bounds(self, position):
        p = np.array(position)
        return np.sum((p < 0) | (p >= self.field.shape)) > 0

    def get_attraction(self, position):
        p = tuple(position)
        if self.is_out_of_bounds(p):
            return 0
        return self.field[p]

    def get_attractions(self, candidates):
        return [self.get_attraction(p) for p in candidates]

    def play(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set(xlabel='x', ylabel='z', zlabel='y')
        ax.voxels(np.swapaxes(self.field > 0, 1, 2), facecolors=(0.,0.,0.,0.0), edgecolor='k', linewidth=.25)
        com = self.attractor
        ax.scatter([com[0]],[com[2]],[com[1]], s=30, c=[(1., 0., 0.)])
        plt.interactive(True)
        plt.show()
        input()
        self.turn = 0
        self.set_plot_limits(ax)
        self.plot_turn(fig, ax)
        ind = np.indices(self.field.shape)[:,self.field > 0].T
        while self.active_players > 0:
            self.turn += 1
            # print('####### TURN ', self.turn)
            furthest_player_first = np.flip(self.get_closest_players()).tolist()
            for p in furthest_player_first:
                dists = np.array(get_distances(ind, p.position))
                dist_sort = dists.argsort()
                new_pos = None
                for try_pos in range(len(dist_sort)):
                    best_pos = ind[dist_sort[try_pos]]
                    if self.is_unoccupied(best_pos):
                        new_pos = best_pos
                        print('player {} should move to {} after {} tries'.format(p.id, best_pos, try_pos))
                        break
                if not new_pos is None:
                    p.move(tuple(new_pos))
                    p.eliminate()
                self.plot_turn(fig, ax)
        plt.show(block=True)


    def get_closest_players(self):
        positions = list(map(lambda p: p.position, self.players))
        distances = self.get_attractor_distances(positions)
        return np.array(self.players)[np.argsort(distances)].tolist()

    def get_attractor_distances(self, candidates):
        dists = get_distances(candidates, self.attractor - 0.5)
        return dists

    def get_plot_voxels(self):
        pass

    def plot_turn(self, fig, ax):
        for artist in self.artists:
            artist.remove()
        self.artists = []
        for p in self.players:
            # Draw player
            self.artists.append(ax.scatter([p.position[0] + 0.5],[p.position[2] + 0.5],[p.position[1] + 0.5], c=[p.color]))
            # self.artists.append(ax.plot(xs=[p.last_move[0] + 0.5, p.position[0] + 0.5],ys=[p.last_move[2] + 0.5, p.position[2] + 0.5], zs=[p.last_move[1] + 0.5, p.position[1] + 0.5], c=(0., 1., 0.), linewidth=.25)[0])
        for p in self.eliminated_players:
            self.artists.append(ax.scatter([p.position[0] + 0.5],[p.position[2] + 0.5],[p.position[1] + 0.5], c=[(0.5, 0.5, 0.5, 0.5)]))
        fig.canvas.draw()
        fig.canvas.flush_events()

    def set_plot_limits(self, ax):
        positions = list(map(lambda p: p.position, self.players))
        if len(positions) == 0:
            min_ax = [0., 0., 0.]
            max_ax = np.array(self.field.shape) + 1
        else:
            np_pos = np.array(positions)
            min_ax = np.array([min(np.min(np_pos[:,0]),0), min(np.min(np_pos[:,1]),0), min(np.min(np_pos[:,2]),0)]) + 1
            max_ax = np.array([max(np.max(np_pos[:,0]),self.field.shape[0]), max(np.max(np_pos[:,1]),self.field.shape[1]), max(np.max(np_pos[:,2]),self.field.shape[2])]) + 1
        ax.set(xlim=(min_ax[0], max_ax[0]), ylim=(min_ax[2], max_ax[2]), zlim=(min_ax[1], max_ax[1]))




class AttractionPlayer:
    def __init__(self, game, payload, position):
        pos = tuple(position)
        self.game = game
        self.id = len(game.players)
        self.payload = payload
        self.position = pos
        self.attraction = game.get_attraction(pos)
        self.eliminated = False
        self.moves = set([pos])
        self.last_move = pos
        self.color = np.random.rand(3)

    def eliminate(self):
        self.eliminated = True
        self.game.eliminate_player(self)

    def neighbors(self):
        x = self.position[0] - 1
        y = self.position[1] - 1
        z = self.position[2] - 1
        neighbors = np.indices((3,3,3))
        neighbors[0,:,:,:] += x
        neighbors[1,:,:,:] += y
        neighbors[2,:,:,:] += z
        return neighbors

    def filter_unoccupied(self, candidates):
        return list(filter(lambda c: self.game.is_unoccupied(c), candidates))

    def filter_attractive(self, candidates):
        if self.attraction == 0: # You're losing the game buddy, you can't be picky
            return candidates
        return list(filter(lambda c: self.game.get_attraction(c) > self.attraction, candidates))

    def filter_unvisited(self, candidates):
        return list(set(candidates) - self.moves)

    def candidates_from_indices(self, indices):
        all_mask = np.ones(indices.shape[1:], dtype=bool)
        coords = indices[:, all_mask]
        candidates = [(coords[0, i], coords[1, i], coords[2, i]) for i in range(coords.shape[1])]
        # print('cand:', candidates)
        return candidates

    def get_available_moves(self):
        candidates = self.candidates_from_indices(self.neighbors())
        unvisited_candidates = self.filter_unvisited(candidates)
        open_candidates = self.filter_unoccupied(unvisited_candidates)
        attractive_candidates = self.filter_attractive(open_candidates)
        return attractive_candidates

    def get_best_moves(self):
        available_moves = self.get_available_moves()
        if len(available_moves) == 0:
            return None
        attractions = self.game.get_attractions(available_moves)
        return find_best_candidates(available_moves, attractions)

    def get_move(self):
        best_moves = self.get_best_moves()
        if best_moves is None:
            return None
        distances = self.game.get_attractor_distances([(p[0] + 0.5, p[1] + 0.5, p[2] + 0.5) for p in best_moves])
        closest = find_best_candidates(best_moves, distances, metric=np.min)
        closest_to_me = find_best_candidates(closest, get_distances(closest, self.position), metric=np.min)
        return random_element(closest_to_me)

    def move(self, pos):
        print('player {} moves to {}'.format(self.id, pos))
        # m = self.get_move()
        # if m is None:
        #     # print('player {}: eliminated at {} with score {}'.format(self.id, self.position, self.game.turn, self.game.get_attraction(self.position)))
        #     self.eliminate()
        #     return
        # print('player {}: moved from {} to {}'.format(self.id, self.position, m))
        # self.last_move = self.position
        # self.moves.add(m)
        self.game.occupy(self, pos)
        self.attraction = self.game.get_attraction(pos)
        # print('last move is now:',self.last_move, self.position)

def find_best_candidates(candidates, results, metric=np.max):
    best_result = metric(results)
    highest_scoring = [candidates[i] if results[i] == best_result else False for i in range(len(results))]
    return list(filter(bool, highest_scoring))

def get_distances(candidates, point):
    return [np.sqrt(np.sum((np.array(c) - point) ** 2)) for c in candidates]
