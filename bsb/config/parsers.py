import abc
import functools
import os

from ..exceptions import FileReferenceError, PluginError
from ._parse_types import file_imp, file_ref, parsed_dict, recurse_handlers


class ConfigurationParser(abc.ABC):
    @abc.abstractmethod
    def parse(self, content, path=None):
        """
        Parse configuration file content.

        :param content: str or dict content of the file to parse
        :param path: path to the file containing the configuration.
        :return: configuration tree and metadata attached as dictionaries
        """
        pass

    @abc.abstractmethod
    def generate(self, tree, pretty=False):
        """
        Generate a string representation of the configuration tree (dictionary).

        :param dict tree: configuration tree
        :param bool pretty: if True, will add indentation to the output string
        :return: str representation of the configuration tree
        :rtype: str
        """
        pass


class ReferenceParser(ConfigurationParser):
    """
    Parser plugin class to parse configuration files with references and imports.
    """

    def parse_content(self, content):
        if isinstance(content, str):
            content = parsed_dict(self.from_str(content))
        elif isinstance(content, dict):
            content = parsed_dict(content)
        return content

    @abc.abstractmethod
    def from_str(self, content):
        """
        Parse dictionary from string content.

        :param str content: content to parse
        :return: parsed dictionary
        :rtype: dict
        """
        pass

    @abc.abstractmethod
    def load_content(self, stream):
        """
        Read content from file object and return parsed dictionary.

        :param stream: python file object
        :return: parsed dictionary
        :rtype: dict
        """
        pass

    def parse(self, content, path=None):
        # Parses the content. If path is set it's used as the root for the multi-document
        # features. During parsing the references (refs & imps) are stored. After parsing
        # the other documents are parsed by the standard file loader (so no recursion yet)
        # After loading all required documents the references are resolved and all values
        # copied over to their final destination.
        content = self.parse_content(content)
        self.root = content
        self.path = path or os.getcwd()
        self.is_doc = path and not os.path.isdir(path)
        self.references = []
        self.documents = {}
        self._traverse(content, content.items())
        self.resolved_documents = {}
        self._resolve_documents()
        self._resolve_references()
        meta = {"path": path}
        return content, meta

    def _traverse(self, node, iter):
        # Iterates over all values in `iter` and checks for import keys, recursion or refs
        # Also wraps all nodes in their `parsed_*` counterparts.
        for key, value in iter:
            if self._is_import(key):
                self._store_import(node)
            elif type(value) in recurse_handlers:
                # The recurse handlers wrap the dicts and lists and return appropriate
                # iterators for them.
                value, iter = recurse_handlers[type(value)](value, node)
                # Set some metadata on the wrapped recursable objects.
                value._key = key
                value._parent = node
                # Overwrite the reference to the original object with a reference to the
                # wrapped object.
                node[key] = value
                # Recurse a level deeper
                self._traverse(value, iter)
            elif self._is_reference(key):
                self._store_reference(node, value)

    def _is_reference(self, key):
        return key == "$ref"

    def _is_import(self, key):
        return key == "$import"

    @staticmethod
    def _get_ref_document(ref, base=None):
        if "#" not in ref or ref.split("#")[0] == "":
            return None
        doc = ref.split("#")[0]
        if not os.path.isabs(doc):
            if not base:
                base = os.getcwd()
            elif not os.path.isdir(base):
                base = os.path.dirname(base)
                if not os.path.exists(base):
                    raise IOError("Can't find reference directory '{}'".format(base))
            doc = os.path.abspath(os.path.join(base, doc))
        return doc

    @staticmethod
    def _get_absolute_ref(node, ref):
        ref = ref.split("#")[-1]
        if ref.startswith("/"):
            path = ref
        else:
            path = os.path.join(node.location(), ref)
        return os.path.normpath(path).replace(os.path.sep, "/")

    def _store_reference(self, node, ref):
        # Analyzes the reference and creates a ref object from the given data
        doc = self._get_ref_document(ref, self.path)
        ref = self._get_absolute_ref(node, ref)
        if doc not in self.documents:
            self.documents[doc] = set()
        self.documents[doc].add(ref)
        self.references.append(file_ref(node, doc, ref))

    def _store_import(self, node):
        # Analyzes the import node and creates a ref object from the given data
        imp = node["$import"]
        ref = imp["ref"]
        doc = self._get_ref_document(ref)
        ref = self._get_absolute_ref(node, ref)
        if doc not in self.documents:
            self.documents[doc] = set()
        self.documents[doc].add(ref)
        if "values" not in imp:
            e = RuntimeError(f"Import node {node} is missing a 'values' list.")
            e._bsbparser_show_user = True
            raise e
        self.references.append(file_imp(node, doc, ref, imp["values"]))

    def _resolve_documents(self):
        # Iterates over the list of stored documents parses them and fetches the content
        # of each reference node.
        for file, refs in self.documents.items():
            if file is None:
                content = self.root
            else:
                # We could open another ReferenceParser to easily recurse.
                with open(file, "r") as f:
                    content = self.load_content(f)
            try:
                self.resolved_documents[file] = self._resolve_document(content, refs)
            except FileReferenceError as jre:
                if not file:
                    raise
                raise FileReferenceError(
                    str(jre) + " in document '{}'".format(file)
                ) from None

    def _resolve_document(self, content, refs):
        resolved = {}
        for ref in refs:
            resolved[ref] = self._fetch_reference(content, ref)
        return resolved

    def _fetch_reference(self, content, ref):
        parts = [p for p in ref.split("/")[1:] if p]
        n = content
        loc = ""
        for part in parts:
            loc += "/" + part
            try:
                n = n[part]
            except KeyError:
                raise FileReferenceError(
                    "'{}' in File reference '{}' does not exist".format(loc, ref)
                ) from None
            if not isinstance(n, dict):
                raise FileReferenceError(
                    "File references can only point to dictionaries. '{}' is a {}".format(
                        "{}' in '{}".format(loc, ref) if loc != ref else ref,
                        type(n).__name__,
                    )
                )
        return n

    def _resolve_references(self):
        for ref in self.references:
            target = self.resolved_documents[ref.doc][ref.ref]
            ref.resolve(self, target)


@functools.cache
def get_configuration_parser_classes():
    from ..plugins import discover

    return discover("config.parsers")


def get_configuration_parser(parser, **kwargs):
    """
    Create an instance of a configuration parser that can parse configuration
    strings into configuration trees, or serialize trees into strings.

    Configuration trees can be cast into Configuration objects.
    """
    parsers = get_configuration_parser_classes()
    if parser not in parsers:
        raise PluginError(f"Configuration parser '{parser}' not found")
    return parsers[parser](**kwargs)


__all__ = [
    "ConfigurationParser",
    "ReferenceParser",
    "get_configuration_parser",
    "get_configuration_parser_classes",
]
