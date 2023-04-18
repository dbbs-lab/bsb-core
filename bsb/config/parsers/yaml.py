import yaml

from ._parser import Parser


class YAMLParser(Parser):
    """
    Parser plugin class to parse YAML configuration files.
    """

    data_description = "YAML"
    data_extensions = ("yaml", "yml")

    def parse(self, content, path=None):
        """
        Parse the YAML

        :param content: File contents
        :type content: str
        :param path: Path the content came from
        :type path: str
        """

        content = yaml.safe_load(content)
        meta = {"path": path}
        return content, meta

    def generate(self, tree, pretty=False):
        return yaml.dump(tree)


__plugin__ = YAMLParser
