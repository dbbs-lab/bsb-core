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

        :param content: file stream
        :type content: stream
        :param path: path to file to store in metadata
        :type path: str
        """

        content = yaml.safe_load(content)
        meta = {"path": path}
        return content, meta
