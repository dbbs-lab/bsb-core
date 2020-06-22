import json, os
from ...exceptions import *


class parsed_node:
    def location(self):
        return "/" + "/".join(self._location_parts([]))

    def _location_parts(self, carry):
        if hasattr(self, "_parent"):
            self._parent._location_parts(carry)
            carry.append(self._key)
        return carry

    def merge(self, merge_parser, other):
        for key, value in other.items():
            if key not in self:
                mutdict = {key: value}
                # Use _traverse to recursively parse the merge value, the `mutdict` will
                # be mutated and the new parsed value can be extracted.
                merge_parser._traverse(mutdict, mutdict.items())
                self[key] = mutdict[key]

    def __str__(self):
        return self.location()


class parsed_dict(dict, parsed_node):
    pass


class parsed_list(list, parsed_node):
    pass


class json_ref:
    def __init__(self, node, doc, ref):
        self.node = node
        self.doc = doc
        self.ref = ref

    def resolve(self, parser, target):
        del self.node["$ref"]
        self.node.merge(parser, target)


class json_imp:
    def __init__(self, node, doc, ref):
        self.node = node
        self.doc = doc
        self.ref = ref

    def resolve(self, parser, target):
        del self.node["$ref"]
        self.node.merge(parser, target)


class JsonMeta:
    pass


class JsonParser:
    data_description = "JSON"

    def parse(self, content, path=None):
        meta = JsonMeta()
        meta.path = path
        if isinstance(content, str):
            content = parsed_dict(json.loads(content))
        self.root = content
        self.path = path or os.getcwd()
        self.meta = meta
        self.references = []
        self.documents = {}
        self._traverse(content, content.items())
        self.resolved_documents = {}
        self._resolve_documents()
        self._resolve_references()
        return content, meta

    def _traverse(self, node, iter):
        for key, value in iter:
            if type(value) in self.recurse_handlers:
                value, iter = self.recurse_handlers[type(value)](self, value, node)
                value._key = key
                value._parent = node
                node[key] = value
                self._traverse(value, iter)
            elif self._is_reference(key):
                self._store_reference(node, value)
            elif self._is_import(key):
                self._store_import(node, key, value)

    def _prep_dict(self, node, parent):
        return parsed_dict(node), node.items()

    def _prep_list(self, node, parent):
        return parsed_list(node), map(lambda t: (str(t[0]), t[1]), enumerate(node))

    def _is_reference(self, key):
        return key == "$ref"

    def _is_import(self, key):
        return key == "$import"

    def _store_reference(self, node, ref):
        doc = _get_ref_document(ref)
        ref = _get_absolute_ref(node, ref)
        if doc not in self.documents:
            self.documents[doc] = set()
        self.documents[doc].add(ref)
        self.references.append(json_ref(node, doc, ref))

    def _store_import(self, node):
        ref = node["ref"]
        doc = _get_ref_document(ref)
        ref = _get_absolute_ref(node, ref)
        if doc not in self.documents:
            self.documents[doc] = set()
        self.documents[doc].add(ref)
        self.references.append(json_imp(node, doc, ref))

    def _resolve_documents(self):
        for file, refs in self.documents.items():
            if file is None:
                content = self.root
            else:
                # We could open another JsonParser to easily recurse.
                with open(file, "r") as f:
                    content = json.load(f)
            self.resolved_documents[file] = self._resolve_document(content, refs)

    def _resolve_document(self, content, refs):
        resolved = {}
        for ref in refs:
            resolved[ref] = self._fetch_reference(content, ref)
        return resolved

    def _fetch_reference(self, content, ref):
        parts = ref.split("/")[1:]
        n = content
        loc = ""
        for part in parts:
            loc += "/" + part
            try:
                n = n[part]
            except KeyError:
                raise JsonReferenceError(
                    "'{}' in JSON reference '{}' does not exist.".format(loc, ref)
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

    recurse_handlers = {dict: _prep_dict, list: _prep_list}


def _get_ref_document(ref):
    if "#" not in ref:
        return None
    doc = ref.split("#")[0]
    if not os.path.isabs(doc):
        doc = os.path.abspath(os.path.join(self.path, doc))
    return doc


def _get_absolute_ref(node, ref):
    if ref.startswith("/"):
        path = ref
    else:
        path = os.path.join(node.location(), ref)
    return os.path.normpath(path).replace(os.path.sep, "/")


__plugin__ = JsonParser
