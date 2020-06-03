import json


class parsedict(dict):
    pass


class JsonParser:
    def __init__(self, content):
        self.content = content

    def parse(self):
        return json.loads(self.content)
