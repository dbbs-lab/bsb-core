"""
JSON parsing module. Built on top of the Python ``json`` module. Adds JSON imports and
references.
"""

import json
import os
from ...exceptions import JsonImportError, ConfigurationWarning, JsonReferenceError
from ...reporting import warn
from ._parser import Parser


def _json_iter(obj):  # pragma: nocover
    if isinstance(obj, dict):
        return obj.items()
    elif isinstance(obj, list):
        return iter(obj)
    else:
        return iter(())


class parsed_node:
    def location(self):
        return "/" + "/".join(str(part) for part in self._location_parts([]))

    def _location_parts(self, carry):
        if hasattr(self, "_parent"):
            self._parent._location_parts(carry)
            carry.append(self._key)
        return carry

    def __str__(self):
        return f"<parsed json config '{super().__str__()}' at '{self.location()}'>"

    def __repr__(self):
        return super().__str__()


def _traverse_wrap(node, iter):
    for key, value in iter:
        if type(value) in recurse_handlers:
            value, iter = recurse_handlers[type(value)](value, node)
            value._key = key
            value._parent = node
            node[key] = value
            _traverse_wrap(value, iter)


class parsed_dict(dict, parsed_node):
    def merge(self, other):
        """
        Recursively merge the values of another dictionary into us
        """
        for key, value in other.items():
            if key in self and isinstance(self[key], dict) and isinstance(value, dict):
                if not isinstance(self[key], parsed_dict):  # pragma: nocover
                    self[key] = d = parsed_dict(self[key])
                    d._key = key
                    d._parent = self
                self[key].merge(value)
            elif isinstance(value, dict):
                self[key] = d = parsed_dict(value)
                d._key = key
                d._parent = self
                _traverse_wrap(d, d.items())
            else:
                if isinstance(value, list):
                    value = parsed_list(value)
                    value._key = key
                    value._parent = self
                self[key] = value

    def rev_merge(self, other):
        """
        Recursively merge ourself onto another dictionary
        """
        m = parsed_dict(other)
        _traverse_wrap(m, m.items())
        m.merge(self)
        self.clear()
        self.update(m)
        for v in self.values():
            if hasattr(v, "_parent"):
                v._parent = self


class parsed_list(list, parsed_node):
    pass


class json_ref:
    def __init__(self, node, doc, ref):
        self.node = node
        self.doc = doc
        self.ref = ref

    def resolve(self, parser, target):
        del self.node["$ref"]
        self.node.rev_merge(target)

    def __str__(self):
        return "<json ref '{}'>".format(((self.doc + "#") if self.doc else "") + self.ref)


class json_imp(json_ref):
    def __init__(self, node, doc, ref, values):
        super().__init__(node, doc, ref)
        self.values = values

    def resolve(self, parser, target):
        del self.node["$import"]
        for key in self.values:
            if key not in target:
                raise JsonImportError(
                    "'{}' does not exist in import node '{}'".format(key, self.ref)
                )
            if isinstance(target[key], dict):
                imported = parsed_dict()
                imported.merge(target[key])
                imported._key = key
                imported._parent = self.node
                if key in self.node:
                    if isinstance(self.node[key], dict):
                        imported.merge(self.node[key])
                    else:
                        warn(
                            f"Importkey '{key}' of {self} is ignored because the parent"
                            f" already contains a key '{key}'"
                            f" with value '{self.node[key]}'.",
                            ConfigurationWarning,
                            stacklevel=3,
                        )
                        continue
                self.node[key] = imported
            elif isinstance(target[key], list):
                imported, iter = _prep_list(target[key], self.node)
                imported._key = key
                imported._parent = self.node
                self.node[key] = imported
                _traverse_wrap(imported, iter)
            else:
                self.node[key] = target[key]


class JsonParser(Parser):
    """
    Parser plugin class to parse JSON configuration files.
    """

    data_description = "JSON"
    data_extensions = ("json",)

    def parse(self, content, path=None):
        # Parses the content. If path is set it's used as the root for the multi-document
        # features. During parsing the references (refs & imps) are stored. After parsing
        # the other documents are parsed by the standard json module (so no recursion yet)
        # After loading all required documents the references are resolved and all values
        # copied over to their final destination.
        if isinstance(content, str):
            content = parsed_dict(json.loads(content))
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

    def _store_reference(self, node, ref):
        # Analyzes the reference and creates a ref object from the given data
        doc = _get_ref_document(ref, self.path)
        ref = _get_absolute_ref(node, ref)
        if doc not in self.documents:
            self.documents[doc] = set()
        self.documents[doc].add(ref)
        self.references.append(json_ref(node, doc, ref))

    def _store_import(self, node):
        # Analyzes the import node and creates a ref object from the given data
        imp = node["$import"]
        ref = imp["ref"]
        doc = _get_ref_document(ref)
        ref = _get_absolute_ref(node, ref)
        if doc not in self.documents:
            self.documents[doc] = set()
        self.documents[doc].add(ref)
        self.references.append(json_imp(node, doc, ref, imp["values"]))

    def _resolve_documents(self):
        # Iterates over the list of stored documents parses them and fetches the content
        # of each reference node.
        for file, refs in self.documents.items():
            if file is None:
                content = self.root
            else:
                # We could open another JsonParser to easily recurse.
                with open(file, "r") as f:
                    content = json.load(f)
            try:
                self.resolved_documents[file] = self._resolve_document(content, refs)
            except JsonReferenceError as jre:
                if not file:
                    raise
                raise JsonReferenceError(
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
                raise JsonReferenceError(
                    "'{}' in JSON reference '{}' does not exist".format(loc, ref)
                ) from None
            if not isinstance(n, dict):
                raise JsonReferenceError(
                    "JSON references can only point to dictionaries. '{}' is a {}".format(
                        "{}' in '{}".format(loc, ref) if loc != ref else ref,
                        type(n).__name__,
                    )
                )
        return n

    def _resolve_references(self):
        for ref in self.references:
            target = self.resolved_documents[ref.doc][ref.ref]
            ref.resolve(self, target)


def _prep_dict(node, parent):
    return parsed_dict(node), node.items()


def _prep_list(node, parent):
    return parsed_list(node), enumerate(node)


recurse_handlers = {
    dict: _prep_dict,
    parsed_dict: _prep_dict,
    list: _prep_list,
    parsed_list: _prep_list,
}


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


def _get_absolute_ref(node, ref):
    ref = ref.split("#")[-1]
    if ref.startswith("/"):
        path = ref
    else:
        path = os.path.join(node.location(), ref)
    return os.path.normpath(path).replace(os.path.sep, "/")


__plugin__ = JsonParser
