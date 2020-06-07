def in_(container):
    def validator(instance, value):
        return value in container

    return validator
