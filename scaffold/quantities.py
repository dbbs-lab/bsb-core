# Waiting for https://github.com/nielstron/quantulum3/issues/131 to be resolved
# from quantulum3 import parser as UnitParser
# from pprint import pprint

def parseToMicrometer(text):
    # quantity = UnitParser.parse(text)
    #
    # pprint(quantity)

    return float(text) * 10 ** -6
