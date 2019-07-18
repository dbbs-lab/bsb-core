import abc

def copyIniKey(obj, section, key_config):
    ini_key = key_config['key']
    if not ini_key in section: # Only copy values that exist in the config
        return

    def micrometer(value):
        return float(value) * 10 ** -6

    # Process the config values based on the type in their key_config.
    morph_map = {'micrometer': micrometer, 'float': float, 'string': str}
    obj.__dict__[ini_key] = morph_map[key_config['type']](section[ini_key])


class ConfigurableClass(abc.ABC):
    '''
        A class that can be configured.
    '''

    def initialise(self, scaffold):
        self.scaffold = scaffold
        self.castConfig()
        self.validate()

    @abc.abstractmethod
    def validate(self):
        '''
            Must be implemented by child classes. Raise exceptions when invalid configuration parameters
            are received.
        '''
        pass

    def castConfig(self):
        '''
            Casts/validates values imported onto this object from configuration files to their final form.
            The `casts` dictionary should contain the key of the attribute and a function that takes
            a value as only argument. This dictionary will be used to cast the attributes when castConfig
            is called.
        '''
        castingDict = getattr(self.__class__, 'casts', {})
        for attr, cast in castingDict.items():
            if hasattr(self, attr):
                try:
                    self.__dict__[attr] = cast(self.__dict__[attr])
                except Exception as e:
                    if not hasattr(self, 'name'):
                        raise Exception("Could not cast configured attribute '{}' for '{}'".format(attr, self))
                    else:
                        raise Exception("Could not cast configured attribute '{}' for '{}'".format(attr, self.name))
