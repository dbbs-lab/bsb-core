from __future__ import annotations

from .. import warn
from ..exceptions import ConfigurationWarning, FileImportError


class parsed_node:
    _key = None
    """Key to reference the node"""
    _parent: parsed_node = None
    """Parent node"""

    def location(self):
        return "/" + "/".join(str(part) for part in self._location_parts([]))

    def _location_parts(self, carry):
        parent = self
        while (parent := parent._parent) is not None:
            if parent._parent is not None:
                carry.insert(0, parent._key)
        carry.append(self._key or "")
        return carry

    def __str__(self):
        return f"<parsed file config '{super().__str__()}' at '{self.location()}'>"

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
        Recursively merge ourselves onto another dictionary
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


class file_ref:
    def __init__(self, node, doc, ref):
        self.node = node
        self.doc = doc
        self.ref = ref
        self.key_path = node.location()

    def resolve(self, parser, target):
        del self.node["$ref"]
        self.node.rev_merge(target)

    def __str__(self):
        return "<file ref '{}'>".format(((self.doc + "#") if self.doc else "") + self.ref)


class file_imp(file_ref):
    def __init__(self, node, doc, ref, values):
        super().__init__(node, doc, ref)
        self.values = values

    def resolve(self, parser, target):
        del self.node["$import"]
        for key in self.values:
            if key not in target:
                raise FileImportError(
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
                self._fix_references(self.node[key], parser)
            elif isinstance(target[key], list):
                imported, iter = _prep_list(target[key], self.node)
                imported._key = key
                imported._parent = self.node
                self.node[key] = imported
                self._fix_references(self.node[key], parser)
                _traverse_wrap(imported, iter)
            else:
                self.node[key] = target[key]

    def _fix_references(self, node, parser):
        # fix parser's references after the import.
        if hasattr(parser, "references"):
            for ref in parser.references:
                node_loc = node.location()
                if node_loc in ref.key_path:
                    # need to update the reference
                    # we descend the tree from the node until we reach the ref
                    # It should be here because of the merge.
                    loc_node = node
                    while node_loc != ref.key_path:
                        key = ref.key_path.split(node_loc, 1)[-1].split("/", 1)[-1]
                        if key not in loc_node:  # pragma: nocover
                            raise ParserError(
                                f"Reference {ref.key_path} not found in {node_loc}. "
                                f"Should have been merged."
                            )
                        loc_node = node[key]
                        node_loc += "/" + key
                    ref.node = loc_node
