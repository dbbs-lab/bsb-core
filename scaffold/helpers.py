import abc
from .quantities import parseToMicrometer

def copyIniKey(obj, section, key_config):
    ini_key = key_config['key']
    if not ini_key in section: # Only copy values that exist in the config
        return

    # Process the config values based on the type in their key_config.
    morph_map = {'micrometer': parseToMicrometer, 'float': float, 'string': str}
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
        if hasattr(self, 'name'):
            name = self.name
        else:
            name = str(self)
        castingDict = getattr(self.__class__, 'casts', {})
        defaultDict = getattr(self.__class__, 'defaults', {})
        required =    getattr(self.__class__, 'required', [])
        # Get unique keys
        attrKeys = set([*castingDict.keys(), *defaultDict.keys(), *required])
        for attr in attrKeys:
            isRequired = attr in required
            hasDefault = attr in defaultDict
            shouldCast = attr in castingDict
            if not hasattr(self, attr):
                if hasDefault:
                    self.__dict__[attr] = defaultDict[attr]
                elif isRequired:
                    raise Exception("Required attribute '{}' missing from '{}' section.".format(attr, self.name))
            elif shouldCast:
                cast = castingDict[attr]
                try:
                    self.__dict__[attr] = cast(self.__dict__[attr])
                except Exception as e:
                    if not hasattr(self, 'name'):
                        raise Exception("Could not cast configured attribute '{}' with value '{}' for '{}'".format(
                            attr,
                            self.__dict__[attr],
                            self.name)
                        )
