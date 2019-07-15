# Waiting for https://github.com/nielstron/quantulum3/issues/131 to be resolved
# from quantulum3 import parser as UnitParser
# from pprint import pprint

def parseToMicrometer(text):
    # quantity = UnitParser.parse(text)
    #
    # pprint(quantity)

    return float(text) * 10 ** -6

def parseToDensity(text):
    parts = text.split('e', 1)
    try:
        if len(parts) == 1:
            return float(parts[0])
        return float(parts[0]) * 10 ** float(parts[1])
    except Exception as e:
        raise Exception("Unable to parse string '{}' to a density.".format(text))
