import copy
import os
import pprint
import random
from collections import defaultdict
from typing import List

import lxml
import lxml.html
from graphviz import Digraph


HIERARCHICAL = "hierarchical"
SEQUENTIAL = "sequential"


def generate_random_colors(n: int) -> List[str]:
    """
    # todo(unittest)
    Returns:
        list of size `n` with colors in format RGB in HEX: `1A2B3C`
    """
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += (0.85 + 0.30 * random.random()) * step + int(
            random.random() * 256 * 0.2
        )
        g += (0.85 + 0.30 * random.random()) * step + int(
            random.random() * 256 * 0.2
        )
        b += (0.85 + 0.30 * random.random()) * step + int(
            random.random() * 256 * 0.2
        )
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append("{:0>2X}{:0>2X}{:0>2X}".format(r, g, b))
    return ret


# todo: solve problem of typing mdr bc of circular reference
def paint_data_records(mdr, doc: lxml.html.HtmlElement):
    doc_copy = copy.deepcopy(doc)
    data_records = mdr.get_data_records_as_node_lists(doc_copy)
    colors = generate_random_colors(len(data_records))
    for record, color in zip(data_records, colors):
        for gnode in record:
            for e in gnode:
                e.set(
                    "style",
                    e.attrib.get("style", "")
                    + " background-color: #{}!important;".format(color),
                )
    return doc_copy


def open_html_document(directory, file):
    directory = os.path.abspath(directory)
    filepath = os.path.join(directory, file)
    with open(filepath, "r") as file:
        html_document = lxml.html.fromstring(
            lxml.etree.tostring(lxml.html.parse(file), method="html")
        )
    return html_document


def html_to_dot_sequential_name(html, with_text=False):
    graph = Digraph(name="html")
    tag_counts = defaultdict(int)

    def add_node(html_node):
        tag = html_node.tag
        tag_sequential = tag_counts[tag]
        tag_counts[tag] += 1
        node_name = "{}-{}".format(tag, tag_sequential)
        graph.node(node_name, node_name)

        if len(html_node) > 0:
            for child in html_node.iterchildren():
                child_name = add_node(child)
                graph.edge(node_name, child_name)
        elif with_text:
            child_name = "-".join([node_name, "txt"])
            graph.node(child_name, html_node.text)
            graph.edge(node_name, child_name)
        return node_name

    add_node(html)
    return graph


def html_to_dot_hierarchical_name(html, with_text=False):
    graph = Digraph(name="html")

    def add_node(html_node, parent_suffix, brotherhood_index):
        tag = html_node.tag
        if parent_suffix is None and brotherhood_index is None:
            node_suffix = ""
            node_name = tag
        else:
            node_suffix = (
                "-".join([parent_suffix, str(brotherhood_index)])
                if parent_suffix
                else str(brotherhood_index)
            )
            node_name = "{}-{}".format(tag, node_suffix)
        graph.node(node_name, node_name, path=node_suffix)

        if len(html_node) > 0:
            for child_index, child in enumerate(html_node.iterchildren()):
                child_name = add_node(child, node_suffix, child_index)
                graph.edge(node_name, child_name)
        elif with_text:
            child_name = "-".join([node_name, "txt"])
            child_path = "-".join([node_suffix, "txt"])
            graph.node(child_name, html_node.text, path=child_path)
            graph.edge(node_name, child_name)
        return node_name

    add_node(html, None, None)
    return graph


def html_to_dot(html, name_option="hierarchical", with_text=False):
    if name_option == SEQUENTIAL:
        return html_to_dot_sequential_name(html, with_text=with_text)
    elif name_option == HIERARCHICAL:
        return html_to_dot_hierarchical_name(html, with_text=with_text)
    else:
        raise Exception("No name option `{}`".format(name_option))


class FormatPrinter(pprint.PrettyPrinter):
    def __init__(self, formats):
        super(FormatPrinter, self).__init__()
        self.formats = formats

    def format(self, obj, ctx, max_lvl, lvl):
        obj_type = type(obj)
        if obj_type in self.formats:
            type_format = self.formats[obj_type]
            return "{{0:{}}}".format(type_format).format(obj), 1, 0
        return pprint.PrettyPrinter.format(self, obj, ctx, max_lvl, lvl)
