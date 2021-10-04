import numpy as np
from bsb.connectivity import ConnectionStrategy
from bsb.reporting import report, warn
from time import time  # usalo per time sttart stop

# import time    #usalo per il time.sleep


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
            # print(' dist_x shape', input.shape)
            input_rect = np.fabs(input - center)
            output[np.where(input <= center)] = (
                0.5 + 0.5 * (input[np.where(input <= center)]) / center
            )
            output[np.where(input > center)] = 2.0 * (
                1.0
                - 1.0
                / (1.0 + np.exp(-input_rect[np.where(input > center)] * (1.0 / std)))
            )  # sigmoide rovesciata
            return output

        def compute_likelihood(x, z, gloms):
            # Based on the distance between the x and z position of each
            # MF and the x z positions of the glomeruli
            # the likelihood of a glomerulus to belong to the MF
            # is computed
            dist_x = np.fabs(
                gloms[:, 0] - x
            )  # np.fabs returns the absolute value di (coordinata x del glom - valore x)
            dist_z = np.fabs(gloms[:, 1] - z)

            # Glomeruli receiving signals from the same mf were
            # grouped together in anisotropic clusters that extended 60 µm along the parasagittal direction (x-axis)
            # and 20 µm along the transverse direction (z-axis)

            prob_x = probability_mapping(
                dist_x, center=30.0, std=3.0
            )  # As in Sultan, 2001 for the parasagittal axis
            prob_z = probability_mapping(
                dist_z, center=10.0, std=1.0
            )  # As in Sultan, 2001 for the mediolateral axis

            probabilities = prob_x * prob_z
            return probabilities

        t_start = time()

        # Source and target neurons are extracted
        mossy_cell_type = self.from_cell_types[0]
        glomerulus_cell_type = self.to_cell_types[0]

        # glomeruli contiene per ogni glom, l'id del glo, l'id del cell type, le coordinate x y z
        glomeruli = self.scaffold.cells_by_type[glomerulus_cell_type.name]

        # I'm shamelessly aiming for the path of least resistance here and will
        # produce a most horrible workaround to quickly converge on the code:
        #
        # mossy fiber positions will be sampled from a uniform distribution,
        # and merged with the mossy fiber (entity) IDs. These will be padded
        # together in the awful 5-column old-style placement matrices so that
        # they can be plugged into the `mossy` variable and the rest of the code
        # will run as before.

        _mossy_ids = mossy_cell_type.get_placement_set().identifiers
        _layer = mossy_cell_type.placement.layer_instance
        _og, _dims = _layer.origin, _layer.dimensions
        _rng = np.random.default_rng()
        _uni_pos = []

        mossy = np.column_stack(
            (
                _mossy_ids,
                np.broadcast_to(0, (len(_mossy_ids),)),
                *(_rng.random(len(_mossy_ids)) * d + o for o, d in zip(_og, _dims)),
            )
        )

        first_MF = int(mossy[0, 0])
        report("id first mf:", first_MF, level=3)
        num_mf = len(mossy[:, 0])
        report("num mf  :", num_mf, level=3)

        # Glom x, y and ID
        Glom_xzID = glomeruli[:, [2, 4, 0]]
        Mf_xzID = mossy[:, [2, 4, 0]]
        mossy_id = (np.array(mossy[:, 0])).tolist()
        glom_id = (np.array(glomeruli[:, 0])).tolist()
        # time.sleep(4)
        total_glom = np.shape(Glom_xzID)[0]
        report("num glom", total_glom, level=3)
        first_glom = int(glomeruli[0, 0])
        ind_associated_glom = []
        deleted_gloms = []
        index = 0
        num_glom = 0
        connections = []
        mo = []
        gl = []

        n = 50  # prendo in modo casuale i 'num_glom' glomeruli tra i primi 100 a prob maggiore
        fine = 0
        w = 0
        ind_non_associated_glom = []
        non_associated_mf = []
        for i in np.random.permutation(range(num_mf)):
            mf_id = i + first_MF
            w = w + 1
            report(
                "--> connecting mf num : ", w, "/", num_mf, "-id mf=", mf_id, i, level=3
            )
            # time.sleep(1)
            mf_x = mossy[i, 2]
            mf_z = mossy[i, 4]
            probabilities = compute_likelihood(mf_x, mf_z, Glom_xzID)
            # print('prob prima', len(probabilities), probabilities )   #probabilità per ogni glomerulo, lunghezza: 20085

            (ind_prob,) = np.where(
                probabilities > 0.0001
            )  # indici con prob > 0.001, da 0 a 20084
            # print('numero indici a prob > 0.001 : ', len(ind_prob))
            # print('indici non ordinati:', ind_prob)

            ind_prob_ord_dec = (ind_prob[np.argsort(probabilities[ind_prob])])[
                ::-1
            ]  # ordino in ordine decrescente: il primo indice corrisponde al glom con probabilità maggiore
            # da questa lista devo prendere gli indici dei gloms dopo i primi n
            # print('numero indici a prob> 0.001', len(ind_prob_ord_dec))
            # print('indici ordinati per probabilità decrescente: ', ind_prob_ord_dec)

            ind_prob_ord_dec_n = ind_prob_ord_dec[:n]
            # print('primi n glomeruli (', len(ind_prob_ord_dec_n), ') :', ind_prob_ord_dec_n)
            ind_prob_ord_dec_n_random = np.random.permutation(ind_prob_ord_dec_n)
            # print('primi n glomeruli mescolati (', len(ind_prob_ord_dec_n_random), ') :', ind_prob_ord_dec_n_random)

            num_glom = min(
                int(20 + 3 * np.random.randn()), len(ind_prob_ord_dec)
            )  # distribuzione standard con media 0 e varianza 1
            # print('num glomeruli da associare: ', num_glom)
            ind_associated_glom = ind_prob_ord_dec_n_random[:num_glom]
            ind = 1
            fine_sub_for_current_mf = 0
            ind_associated_glom_def = []
            for k in range(num_glom):  # per il num di glom da associare
                glom_associati = k
                # print('k=',k,'- analizzo il glomerulo numero ', ind_associated_glom[k], '(' , ind_associated_glom[k]+first_glom, ')' )
                if (
                    ind_associated_glom[k] not in deleted_gloms
                ):  # se il glom non è gia associato
                    # print('il glomerulo va bene ')
                    ind_associated_glom_def.append(ind_associated_glom[k])
                    glom_id.remove(glomeruli[ind_associated_glom[k]][0])
                else:  # il glomerulo è già associato
                    # print('glomerulo gia associato, devo sostituirlo')
                    if fine_sub_for_current_mf == 1:
                        # print('non posso sostituire il glom con niente, quindi lo elimino')
                        continue  # salta le conse scritte sotto e va alla prossima iterazione del for
                    else:  # trovo un sostituto per il glomerulo
                        substitute_glom = True
                        while substitute_glom:
                            # print(' - indice ', ind+num_glom, 'len probabilities', len(probabilities))
                            # print('indice numero sostituzioni,', ind)
                            if (
                                k + index
                            ) == total_glom:  # ho finito i glomeruli a disposizione
                                # print('non ho più glomeruli a disposizione')
                                fine = 1
                                substitute_glom = False
                                break  # esce dal while
                            elif (ind + num_glom - 1) == len(ind_prob_ord_dec):
                                # print (' finiti i gloms a prob > 0.001')
                                fine_sub_for_current_mf = 1
                                substitute_glom = False
                            else:
                                if (ind + num_glom) <= n:  # arrivo fino a n = 50
                                    # print('cerco nei gloms tra i primi 50 a prob maggiore ')
                                    # print('ind + num glo =', ind+num_glom, 'lunghezza associated glom random = ', len(associated_glom_random))
                                    # new_glom = associated_glom_random[num_glom+ind-1]   #nuova prova
                                    ind_new_glom = ind_prob_ord_dec_n_random[
                                        num_glom + ind - 1
                                    ]
                                else:  # entro per n>50
                                    # print('cerco nei glom dopo i primi 50')
                                    # new_glom = associated_glom_all[num_glom+ind-1]
                                    ind_new_glom = ind_prob_ord_dec[num_glom + ind - 1]

                                # print('analizzo', ind_new_glom, '(', ind_new_glom+first_glom, ')' )
                                if ind_new_glom not in deleted_gloms:
                                    # print('sostituisco' , ind_associated_glom[k], 'con il nuovo glom  ', ind_new_glom, '(' , ind_new_glom+first_glom, ')')
                                    substitute_glom = False
                                    ind_associated_glom_def.append(ind_new_glom)
                                    glom_id.remove(glomeruli[ind_new_glom][0])
                                ind = (
                                    ind + 1
                                )  # indice per prendere nuovi glomeruli, nel caso di selezione glom già usati.

                    if fine == 1:
                        # print('condizione fine')
                        break  # esco dal ciclo for

            deleted_gloms.extend(
                list(ind_associated_glom_def)
            )  # se esco prima dal for, devo agiungere solo i gloms
            # print('num glomeruli associati: ', len(deleted_gloms))

            # print('associo', len(ind_associated_glom_def), 'nuovi glomeruli')

            if len(ind_associated_glom_def) == 0:
                non_associated_mf.append(mf_id)
            for j in range(len(ind_associated_glom_def)):
                mo.append(mf_id)
                gl.append(glomeruli[ind_associated_glom_def[j]][0])
                # print(' - connection ' , len(mo), '/',total_glom,'-associate mf ', mf_id, 'to glom', glomeruli[ind_associated_glom_def[j]][0] )
                # print (' len mo e gl' , len(mo), len(gl))
                if (len(mo)) == total_glom:
                    # print ('non associo indice ', index+j, '(' , j+index+1, '==', total_glom, ')' )
                    break
            # index = index + j + 1

            if fine == 1:
                # print('esco dal secondo for')
                break  # esco dal ciclo for
        # print (' index = ', index)
        # print (' len mo e gl' , len(mo), len(gl))
        # print (' num non associated mf ', len(non_associated_mf))
        # print('id mf non associate: ', non_associated_mf)
        report(first_MF, level=3)
        ind_non_associated_mf = []
        for i in range(len(non_associated_mf)):
            ind_non_associated_mf.append(non_associated_mf[i] - first_MF)
        # print('indici mf non associate', ind_non_associated_mf)
        report("connetto i glomeruli non associati", level=3)
        num_non_ass_glom = len(glom_id)
        report("num glomeruli non associati", len(glom_id), level=3)
        ind_non_associated_glom = []
        # glom_x = []
        # glom_y =[]
        # glom_z = []
        for i in range(len(glom_id)):
            ind_non_associated_glom.append(glom_id[i] - first_glom)

        # print(' index non associated gloms', ind_non_associated_glom )
        c = 0

        # for i in np.random.permutation(range(len(ind_non_associated_glom))):
        for i in range(len(ind_non_associated_glom)):
            # print('incice', i )
            # print (ind_non_associated_glom[i])
            gl_id = ind_non_associated_glom[i] + first_glom
            c = c + 1
            report(
                "--> connecting glom num : ",
                c,
                "/",
                num_non_ass_glom,
                "-id glom=",
                gl_id,
                ind_non_associated_glom[i],
                level=3,
            )
            # time.sleep(1)
            glom_x = glomeruli[int(ind_non_associated_glom[i])][2]
            glom_z = glomeruli[int(ind_non_associated_glom[i])][4]
            # print ( ' glomeruli x e z ', glom_x, glom_z)
            probabilities = compute_likelihood(glom_x, glom_z, Mf_xzID)
            # print( 'len probabilities', len(probabilities))  #lunhezza par al numero di mf
            ind_max_prob = int(np.argmax(probabilities))
            # print('probabilities', probabilities)
            # print('indice massima probabilità', ind_max_prob, probabilities[ind_max_prob], 'corrisponde alla mf', mossy_id[ind_max_prob] )
            mo.append(mossy_id[ind_max_prob])
            gl.append(gl_id)
            # print(' - connection ' , index+i, '/',total_glom,'-associate mf ',  mossy_id[ind_max_prob], 'to glom', gl_id )

        connections = np.column_stack((mo, gl))
        report("tot connessioni", len(connections), level=3)
        t_end = time()
        tot_time = t_end - t_start
        report("Total  time: ", tot_time, level=3)
        report("random 50, probabilità = 0.0001", level=3)
        self.scaffold.connect_cells(self, connections)
