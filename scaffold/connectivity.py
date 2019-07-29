import abc
from .helpers import ConfigurableClass
import numpy as np
from pprint import pprint

class ConnectionStrategy(ConfigurableClass):
	@abc.abstractmethod
	def connect(self):
		pass

class ReciprocalGolgiGlomerulus(ConnectionStrategy):
	def validate(self):
		pass

	def connect(self):
		pass

class TouchingConvergenceDivergence(ConnectionStrategy):
	casts = {
		'divergence': int,
		'convergence': int
	}

	required = ['divergence', 'convergence']
	def validate(self):
		pass

	def connect(self):
		pass

class TouchConnect(ConnectionStrategy):
	def validate(self):
		pass

	def connect(self):
		pass

class ConnectomeGlomGranule(TouchingConvergenceDivergence):

	def validate(self):
		pass

	def connect(self):
		from_celltype = self.from_celltype
		to_celltype = self.to_celltype
		glomeruli = self.scaffold.cells_by_type[from_celltype.name]
		granules = self.scaffold.cells_by_type[to_celltype.name]
		dend_len = to_celltype.geometry.dendrite_length
		n_conn_glom = 4
		glom_grc = np.zeros((0,3))
		first_glomerulus = glomeruli[0,0]

		def connectome_glom_grc(first_glomerulus, glomeruli, granules, dend_len, n_conn_glom, glom_grc):
			glom_x = glomeruli[:,2]
			glom_y = glomeruli[:,3]
			glom_z = glomeruli[:,4]
			for i in granules:	# for all granules: calculate which glomeruli can be connected, then choose 4 of them

				# find all glomeruli at a maximum distance of 40micron
				volume_matrix = (((glom_x-i[2])**2)+((glom_y-i[3])**2)+((glom_z-i[4])**2)-(dend_len**2)).__le__(0)
				good_gloms = np.where(volume_matrix==True)[0]	# indexes of glomeruli that can potentially be connected

				if (len(good_gloms))>n_conn_glom:
					gloms_distance = np.sqrt((glomeruli[good_gloms,2]-i[2])**2+(glomeruli[good_gloms,3]-i[3])**2+(glomeruli[good_gloms,4]-i[4])**2)
					dist_matrix = np.zeros((len(good_gloms), 2))
					dist_matrix[:,0] = good_gloms + first_glomerulus
					dist_matrix[:,1] = gloms_distance
					sc_dist = dist_matrix[dist_matrix[:,1].argsort()]	# sorting of the resulting vector on the distances
					connected_f = sc_dist[0:n_conn_glom,0]
					connected_dist = sc_dist[0:n_conn_glom,1]
					connected_provv = connected_f.astype(int)
					connected_gloms = connected_provv

					# construction of the output matrix: the first column has the  glomerulus index, while the second column has the connected granule index
					matrix = np.zeros((n_conn_glom, 3))
					matrix[:,1] = i[0]
					matrix[:,0] = connected_gloms
					matrix[:,2] = gloms_distance[0:n_conn_glom]
					glom_grc = np.vstack((glom_grc, matrix))

				else:
					gloms_distance = np.sqrt((glomeruli[good_gloms,2]-i[2])**2+(glomeruli[good_gloms,3]-i[3])**2+(glomeruli[good_gloms,4]-i[4])**2)
					connected_gloms = good_gloms + first_glomerulus

					matrix = np.zeros((len(connected_gloms), 3))
					matrix[:,1] = i[0]
					matrix[:,0] = connected_gloms
					matrix[:,2] = gloms_distance
					glom_grc = np.vstack((glom_grc, matrix))

			glom_grc = glom_grc[1:-1,:]

			return glom_grc

		glom_grc = connectome_glom_grc(first_glomerulus, glomeruli, granules, dend_len, n_conn_glom, glom_grc)
		return glom_grc
