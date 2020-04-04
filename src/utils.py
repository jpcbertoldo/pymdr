"""
Module dependencies:
    all -> utils
"""

import os
import pathlib
import pprint
import random
from collections import defaultdict
from typing import List, Optional, Dict

import lxml
import lxml.etree
import lxml.html
import graphviz
import yaml

DOT_NAMING_OPTION_HIERARCHICAL = "hierarchical"
DOT_NAMING_OPTION_SEQUENTIAL = "sequential"


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
        r += (0.85 + 0.30 * random.random()) * step + int(random.random() * 256 * 0.2)
        g += (0.85 + 0.30 * random.random()) * step + int(random.random() * 256 * 0.2)
        b += (0.85 + 0.30 * random.random()) * step + int(random.random() * 256 * 0.2)
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append("{:0>2X}{:0>2X}{:0>2X}".format(r, g, b))
    return ret


def html_to_dot_sequential_name(
    root: lxml.html.HtmlElement, graph_name: str, with_text: bool = False
) -> graphviz.Digraph:
    """
    todo(unittest)
    The names of the nodes are defined by `{tag}-{seq - 1}`, where:
        tag: the html tag of the node
        seq: the sequential order of that tag
            ex: if it is the 2nd `table` to be found in the process, it's name will be `table-00001`
    """
    graph = graphviz.Digraph(name=graph_name)
    tag_counts = defaultdict(int)

    def add_node(html_node: lxml.html.HtmlElement):
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

    add_node(root)
    return graph


def html_to_dot_hierarchical_name(
    root: lxml.html.HtmlElement, graph_name: str, with_text=False
) -> graphviz.Digraph:
    """
    todo(unittest)
    The names of the nodes are defined by `{tag}-{index-path-to-node}`, where:
        tag: the html tag of the node
        index-path-to-node: the sequential order of indices that should be called from the root to arrive at the node
            ex: todo(doc)
    """
    graph = graphviz.Digraph(name=graph_name)

    def add_node(
        node: lxml.html.HtmlElement, parent_suffix: Optional[str], brotherhood_index: Optional[int],
    ):
        """Recursive call on this function. Depth-first search through the entire tree."""
        tag = node.tag
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

        if len(node) > 0:
            for child_index, child in enumerate(node.iterchildren()):
                child_name = add_node(child, node_suffix, child_index)
                graph.edge(node_name, child_name)
        elif with_text:
            child_name = "-".join([node_name, "txt"])
            child_path = "-".join([node_suffix, "txt"])
            graph.node(child_name, node.text, path=child_path)
            graph.edge(node_name, child_name)
        return node_name

    add_node(root, None, None)
    return graph


def html_to_dot(
    root, graph_name="html-graph", name_option=DOT_NAMING_OPTION_HIERARCHICAL, with_text=False,
) -> graphviz.Digraph:
    """
    todo(unittest)
    Args:
        root:
        graph_name:
        name_option: hierarchical or sequential naming strategy
        with_text: include tags without children as a node with the text content of the tag
    Returns:
        directed graph representation of an html
    """
    if name_option == DOT_NAMING_OPTION_SEQUENTIAL:
        return html_to_dot_sequential_name(root, graph_name=graph_name, with_text=with_text)
    elif name_option == DOT_NAMING_OPTION_HIERARCHICAL:
        return html_to_dot_hierarchical_name(root, graph_name=graph_name, with_text=with_text)
    else:
        raise Exception("No name option `{}`".format(name_option))


class FormatPrinter(pprint.PrettyPrinter):
    """A custom pretty printer specifier for debug purposes."""

    def __init__(self, formats: Dict[type, str]):
        super(FormatPrinter, self).__init__()
        self.formats = formats

    def format(self, obj, ctx, max_lvl, lvl):
        obj_type = type(obj)
        if obj_type in self.formats:
            type_format = self.formats[obj_type]
            return "{{0:{}}}".format(type_format).format(obj), 1, 0
        return pprint.PrettyPrinter.format(self, obj, ctx, max_lvl, lvl)


project_path = pathlib.Path(os.path.realpath(__file__)).parent.parent.absolute()


def get_config_dict() -> dict:
    config = project_path.joinpath("config.yml").absolute()
    with config.open("r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    return config_dict


def get_config_outputs_parent_dir() -> pathlib.Path:
    config_dict = get_config_dict()
    outputs_parent_dir_path = pathlib.Path(config_dict["outputs-parent-dir"])
    if outputs_parent_dir_path.is_absolute():
        return outputs_parent_dir_path
    return project_path.joinpath(outputs_parent_dir_path).absolute()
