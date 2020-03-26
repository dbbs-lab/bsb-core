import numpy as np
from ..strategy import ConnectionStrategy
from ...reporting import report, warn


class ConnectomeMossyGlomerulus(ConnectionStrategy):
    """
        Implementation for the connections between mossy fibers and glomeruli.
        The connectivity is somatotopic and
    """

    def validate(self):
        pass

    def connect(self):
        def probability_mapping(input, center, std):
            # input: input array that has to be transformed
            # center: center of the sigmoid
            # std: value at which the sigmoid reaches the 54% of its value
            output = np.empty(input.size, dtype=float)
            input_rect = np.fabs(input - center)
            output[np.where(input <= center)] = (
                0.5 + 0.5 * (input[np.where(input <= center)]) / center
            )
            output[np.where(input > center)] = 2.0 * (
                1.0
                - 1.0
                / (1.0 + np.exp(-input_rect[np.where(input > center)] * (1.0 / std)))
            )
            return output

        def compute_likelihood(x, z, gloms):
            # Based on the distance between the x and z position of each
            # MF and the x z positions of the glomeruli
            # the likelihood of a glomerulus to belong to the MF
            # is computed
            dist_x = np.fabs(gloms[:, 0] - x)
            dist_z = np.fabs(gloms[:, 1] - z)

            prob_x = probability_mapping(
                dist_x, center=30.0, std=3.0
            )  # As in Sultan, 2001 for the parasagittal axis
            prob_z = probability_mapping(
                dist_z, center=10.0, std=1.0
            )  # As in Sultan, 2001 for the mediolateral axis

            probabilities = prob_x * prob_z
            return probabilities

        # Source and target neurons are extracted
        mossy_cell_type = self.from_cell_types[0]
        glomerulus_cell_type = self.to_cell_types[0]
        mossy = self.scaffold.entities_by_type[mossy_cell_type.name].astype(int)
        glomeruli = self.scaffold.cells_by_type[glomerulus_cell_type.name]
        # Number of MFs placed and ID of the first MF
        MF_num = np.shape(mossy)[0]
        First_MF = np.min(mossy)

        # Glom x, y and ID
        Glom_xzID = glomeruli[:, [2, 4, 0]]
        total_glom = np.shape(Glom_xzID)[0]

        # Boundaries of X and Z space for glomeruli
        BoundsX = np.array([np.min(Glom_xzID[:, 0]), np.max(Glom_xzID[:, 0])])
        BoundsZ = np.array([np.min(Glom_xzID[:, 1]), np.max(Glom_xzID[:, 1])])

        # Computation of how many MFs do we need to "place" for the two axes
        XZ_Area = (BoundsX[1] - BoundsX[0]) * (BoundsZ[1] - BoundsZ[0])
        MF_per_Area = MF_num / XZ_Area
        MF_per_X = np.ceil((BoundsX[1] - BoundsX[0]) * np.sqrt(MF_per_Area)).astype(int)
        MF_per_Z = np.ceil((BoundsZ[1] - BoundsZ[0]) * np.sqrt(MF_per_Area)).astype(int)

        # Create uniform grid in the X-Z plane
        MF_X = np.linspace(BoundsX[0], BoundsX[1], num=MF_per_X)
        MF_Z = np.linspace(BoundsZ[0], BoundsZ[1], num=MF_per_Z)
        xv, zv = np.meshgrid(MF_X, MF_Z, sparse=False, indexing="ij")
        xv = xv.flatten()
        zv = zv.flatten()

        # Limit the number of MFs (xv and zv) to MF_num
        if np.size(xv) > MF_num:
            delete_points = np.random.randint(0, np.size(xv), size=np.size(xv) - MF_num)
            xv = np.delete(xv, delete_points)
            zv = np.delete(zv, delete_points)

        # labels store the assigned MF to each glomerulus
        labels = -1 * np.ones(np.shape(Glom_xzID)[0], dtype=int)
        best_glom = -1 * np.ones(MF_num * MF_num, dtype=int)
        best_prob = np.zeros(MF_num, dtype=float)
        min_glom = np.min(Glom_xzID[:, 2]).astype(int)

        # This loop iterates associating at each time one glomeurlus to the MF
        # that has the maximum likelihood to be connected to it
        while np.shape(Glom_xzID)[0] > 0:
            # Every time the array is shuffled to avoid bias toward the first glumeruli in the list
            np.random.shuffle(Glom_xzID)
            # For each MF, the highest probability (best_prob) and the corresponding glumerulus (best_blom)
            # are computed
            for i in range(MF_num):
                probabilities = compute_likelihood(xv[i], zv[i], Glom_xzID)
                best_glom[i] = np.argmax(probabilities)
                best_prob[i] = np.max(probabilities)
            # We select the best glomerulus among the best ones for each MF
            highest_glom_MF = np.argmax(best_prob)
            # The label of that glomerulus is assigned
            labels[
                int(Glom_xzID[best_glom[highest_glom_MF], 2]) - min_glom
            ] = highest_glom_MF
            # That glomerulus is deleted from the list
            Glom_xzID = np.delete(Glom_xzID, best_glom[highest_glom_MF], axis=0)
            report(
                "Associated "
                + str(int(100 * (1 - np.shape(Glom_xzID)[0] / total_glom)))
                + "% glomeruli",
                ongoing=True,
                level=3,
            )
            if np.shape(Glom_xzID)[0] == 0:
                break
        # Labels range from 0 to MF_num, while they should range from First_MF to First_MF+MF_num
        labels += First_MF
        connections = np.column_stack((labels, glomeruli[:, 0]))
        self.scaffold.connect_cells(self, connections)
