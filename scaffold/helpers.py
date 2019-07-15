

def copyIniKey(obj, section, key_config):
    ini_key = key_config['key']
    if not ini_key in section: # Only copy values that exist in the config
        return

    def micrometer(value):
        return float(value) * 10 ** -6

    # Process the config values based on the type in their key_config.
    morph_map = {'micrometer': micrometer, 'float': float, 'string': str}
    obj.__dict__[ini_key] = morph_map[key_config['type']](section[ini_key])
