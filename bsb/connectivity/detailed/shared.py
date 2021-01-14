from random import choice as random_element


class MorphologyStrategy:
    def list_all_morphologies(self, cell_type):
        return cell_type.list_all_morphologies()

    def get_random_morphology(self, cell_type):
        """
        Return a morphology suited to represent a cell of the given `cell_type`.
        """
        available_morphologies = self.list_all_morphologies(cell_type)
        if len(available_morphologies) == 0:
            raise MissingMorphologyError(
                "Can't perform touch detection without detailed morphologies for {}".format(
                    cell_type.name
                )
            )
        m_name = random_element(available_morphologies)
        if not m_name in self.morphology_cache:
            mr = self.scaffold.morphology_repository
            self.morphology_cache[m_name] = mr.get_morphology(
                m_name, scaffold=self.scaffold
            )
        return self.morphology_cache[m_name]

    def get_all_morphologies(self, cell_type):
        all_morphologies = []
        for m_name in self.list_all_morphologies(cell_type):
            if not m_name in self.morphology_cache:
                mr = self.scaffold.morphology_repository
                self.morphology_cache[m_name] = mr.get_morphology(
                    m_name, scaffold=self.scaffold
                )
            all_morphologies.append(self.morphology_cache[m_name])
        return all_morphologies
