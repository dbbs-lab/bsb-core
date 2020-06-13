import json


class JsonMeta:
    pass


class JsonParser:
    data_description = "JSON"

    def parse(self, content, path=None):
        meta = JsonMeta()
        meta.path = path
        if isinstance(content, str):
            content = json.loads(content)
        return content, meta


__plugin__ = JsonParser
