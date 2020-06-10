import json


class JsonMeta:
    pass


class JsonParser:
    def parse(self, content, path=None):
        meta = JsonMeta()
        meta.path = path
        return json.loads(content), meta


__plugin__ = JsonParser
