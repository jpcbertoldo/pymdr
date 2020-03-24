import lxml
import lxml.html
import lxml.etree
import os
from collections import defaultdict
from graphviz import Digraph
import yaml


HIERARCHICAL = "hierarchical"
SEQUENTIAL = "sequential"


def open_doc(folder, filename):
    folder = os.path.abspath(folder)
    filepath = os.path.join(folder, filename)

    with open(filepath, "r") as file:
        doc = lxml.html.fromstring(
            lxml.etree.tostring(lxml.html.parse(file), method="html")
        )
    return doc


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
        else:
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
        else:
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


def depth(node):
    d = 0
    while node is not None:
        d += 1
        node = node.getparent()
    return d


def serialize_distances(dictionary):
    return yaml.dump(dictionary)


def unserialize_distances(string):
    return yaml.full_load(string)
