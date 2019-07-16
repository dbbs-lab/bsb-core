# Waiting for https://github.com/nielstron/quantulum3/issues/131 to be resolved
from quantulum3 import parser as UnitParser
from pint import UnitRegistry
from pprint import pprint

units = UnitRegistry()

def parseToMicrometer(text):
    try:
        quantities = UnitParser.parse(text.replace('Âµm', 'µm'))
        if len(quantities) == 0:
            raise Exception("No quantity found where micrometers were expected")
        result = quantities[0]
        if result.unit.name == 'dimensionless':
            result.unit.name = 'micrometer'
        pq = units.Quantity(result.value, result.unit.name)
        return pq.to(units.micrometer).magnitude
    except Exception as e:
        raise Exception("Unable to parse '{}' to micrometer.".format(text))

def parseToDensity(text):
    try:
        quantities = UnitParser.parse(text.replace('Âµm', 'µm'))
        if len(quantities) == 0:
            raise Exception("No quantity found where micrometers were expected")
        result = quantities[0]
        if result.unit.name.find('cubic') == -1:
            raise Exception("A cubic unit is expected as density in '{}'".format(text))
        unitName = list(filter(lambda x: x != 'cubic' and x != 'per', result.unit.name.split(' ')))[-1]
        unit = units.Unit(unitName)
        pq = result.value * unit ** -3
        return pq.to(units.millimeter ** -3).magnitude
    except Exception as e:
        raise Exception("Unable to parse '{}' to density.".format(text))
