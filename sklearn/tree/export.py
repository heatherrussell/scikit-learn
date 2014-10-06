"""
This module defines export functions for decision trees.
"""

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Trevor Stephens <trev.stephens@gmail.com>
# Licence: BSD 3 clause

import warnings

import numpy as np

from ..externals import six

from . import _tree


def export_graphviz(decision_tree, out_file="tree.dot", feature_names=None,
                    max_depth=None, plot_options=None):
    """Export a decision tree in DOT format.

    This function generates a GraphViz representation of the decision tree,
    which is then written into `out_file`. Once exported, graphical renderings
    can be generated using, for example::

        $ dot -Tps tree.dot -o tree.ps      (PostScript format)
        $ dot -Tpng tree.dot -o tree.png    (PNG format)

    Parameters
    ----------
    decision_tree : decision tree classifier
        The decision tree to be exported to GraphViz.

    out_file : file object or string, optional (default="tree.dot")
        Handle or name of the output file.

    feature_names : list of strings, optional (default=None)
        Names of each of the features.

    max_depth : int, optional (default=None)
        The maximum depth of the representation. If None, the tree is fully
        generated.

    plot_options : list of strings, optional (default=None)
                   (which implies ['samples', 'metric', 'labels'])
        Plot the tree with aesthetic options. Available keywords include:
            - 'filled' : Paint nodes to indicate majority class
            - 'rounded' : Draw node boxes with rounded corners
            - 'helvetica' : Replace Times-Roman font with Helvetica
            - 'leaf' : Draw the leaf nodes at the bottom of the tree
            - 'rotate' : Orient the tree left to right (default is top-down)
            - 'yes' : Show Yes/No labels at first split
            - 'true' : Show True/False labels at first split
            - 'id' : Show the ID number on each node
            - 'metric' : Show the error metric at each node
            - 'class' : Show the majority class at each node
            - 'values' : Show the class breakdown at each node
                - default is to show this only for the leaf nodes
                - for regression this is equivalent to 'class'
            - 'samples' : Show the number of samples in each node
            - 'proportion' : Change the display of 'values' and/or 'samples'
                             to be proportions and percentages respectively
            - 'labels' : Show informative labels for 'samples', 'metric', etc
            - 'label' : Same as 'labels', but for root node only

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree

    >>> clf = tree.DecisionTreeClassifier()
    >>> iris = load_iris()

    >>> clf = clf.fit(iris.data, iris.target)
    >>> tree.export_graphviz(clf,
    ...     out_file='tree.dot')                # doctest: +SKIP
    """

    def parse_options():
        # Clean up plot_options and determine if subtree recursion is required
        final_options = set(['samples', 'metric', 'labels'])
        valid_options = set(['class', 'filled', 'helvetica', 'id', 'label',
                             'labels', 'leaf', 'metric', 'proportion',
                             'rotate', 'rounded', 'samples', 'true', 'values',
                             'yes'])
        # The set of options that require subtree recursion
        recurse_options = set(['filled', 'class', 'values'])

        if hasattr(plot_options, '__iter__'):
            bad_options = set(plot_options) - valid_options
            good_options = set(plot_options) & valid_options
            if len(good_options) == 0:
                warnings.warn("No valid plot options, using defaults.")
            elif len(bad_options) != 0:
                warnings.warn("Invalid plot options:" + str(bad_options))
            if len(good_options) != 0:
                final_options = good_options
        else:
            warnings.warn("Expected list-type object, using defaults.")

        return final_options, len(recurse_options & final_options) > 0

    def colour_brew(n):
        # For regression n == 1, but we need two colours
        n = max(2, n)

        # Initialize saturation & value; calculate chroma & value shift
        s, v = 0.75, 0.9
        c = s * v
        m = v - c

        colours = []
        # Loop through n equally spaced hues
        for h in np.arange(25, 385, 360./n).astype(int):

            # Calculate some intermediate values
            h_bar = h / 60.
            x = c * (1 - abs((h_bar % 2) - 1))

            # Initialize RGB with same hue & chroma as our colour
            if h_bar < 1:
                r, g, b = c, x, 0
            elif h_bar < 2:
                r, g, b = x, c, 0
            elif h_bar < 3:
                r, g, b = 0, c, x
            elif h_bar < 4:
                r, g, b = 0, x, c
            elif h_bar < 5:
                r, g, b = x, 0, c
            elif h_bar < 6:
                r, g, b = c, 0, x
            else:
                r, g, b = c, x, 0

            # Shift the initial RGB values to match value and store
            rgb = [(int(255 * (r + m))),
                   (int(255 * (g + m))),
                   (int(255 * (b + m)))]
            colours.append(rgb)

        return colours

    def get_colour(colours, value):
        # Find the appropriate colour & intensity for a node
        colour = [c for c in colours[list(value).index(value.max())]]
        sorted_values = sorted(value, reverse=True)
        alpha = int(255 * (sorted_values[0] - sorted_values[1]) /
                    (1 - sorted_values[1]))
        colour.append(alpha)
        return '#' + str(bytearray(colour)).encode('hex')

    def recurse_subtree(tree, node_id, values=None):
        # Gather the leaf node classifications for a subtree
        if values is None:
            values = []

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        if left_child != _tree.TREE_LEAF:
            recurse_subtree(tree, left_child, values=values)
            recurse_subtree(tree, right_child, values=values)
        else:
            if len(tree.value[node_id][0, :]) != 1:
                # Classification tree
                values.append(tree.value[node_id][0, :])
            else:
                # Regression tree
                values.append(tree.value[node_id][0, :] *
                              float(tree.n_node_samples[node_id]))
        return values

    def node_to_str(tree, node_id, criterion, value=None):
        if not isinstance(criterion, six.string_types):
            criterion = "impurity"

        if value is None:
            value = tree.value[node_id]
            if tree.n_outputs == 1:
                value = value[0, :]

        if feature_names is not None:
            feature = feature_names[tree.feature[node_id]]
        else:
            feature = "X[%s]" % tree.feature[node_id]

        # Should root note labels be drawn?
        root_labels = 'label' in plot_options and node_id == 0

        # Build up node string for plot options
        node_string = ''
        if 'id' in plot_options:
            if 'labels' in plot_options or root_labels:
                node_string += 'node '
            node_string += '#' + str(node_id) + '\\n'
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            node_string += '%s <= %s\\n' % (feature,
                                            round(tree.threshold[node_id], 4))
        if 'metric' in plot_options:
            if 'labels' in plot_options or root_labels:
                node_string += '%s = ' % criterion
            node_string += str(round(tree.impurity[node_id], 4)) + '\\n'
        if 'samples' in plot_options:
            if 'labels' in plot_options or root_labels:
                node_string += 'samples = '
            if 'proportion' in plot_options:
                percent = (100. * tree.n_node_samples[node_id] /
                           float(tree.n_node_samples[0]))
                node_string += str(round(percent, 1)) + '%' + '\\n'
            else:
                node_string += str(tree.n_node_samples[node_id]) + '\\n'
        if 'values' in plot_options:
            if 'labels' in plot_options or root_labels:
                node_string += 'value = '
            if 'proportion' in plot_options:
                value = value / value.sum()
                node_string += str([round(v, 2) for v in value]) + '\\n'
            else:
                node_string += str(value) + '\\n'
        elif tree.children_left[node_id] == _tree.TREE_LEAF:
            if 'labels' in plot_options or root_labels:
                node_string += 'value = '
            node_string += str(value) + '\\n'
        if 'class' in plot_options:
            if 'labels' in plot_options or root_labels:
                node_string += 'class = '
            node_string += str(list(value).index(value.max()))

        # Clean up
        if node_string[-2:] == '\\n':
            node_string = node_string[:-2]
        return node_string

    def recurse(tree, node_id, criterion, parent=None, depth=0, colours=None):
        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        if left_child == _tree.TREE_LEAF:
            ranks['leaves'].append(str(node_id))
        elif str(depth) not in ranks:
            ranks[str(depth)] = [str(node_id)]
        else:
            ranks[str(depth)].append(str(node_id))

        leaves = None
        if subtree_required:
            leaves = np.sum(np.array(recurse_subtree(tree, node_id)), axis=0)
        if 'filled' in plot_options and colours is None:
            colours = colour_brew(tree.n_classes[0])

        # Add node with description
        if max_depth is None or depth <= max_depth:

            out_file.write('%d [label="%s"'
                           % (node_id,
                              node_to_str(tree, node_id, criterion, leaves)))
            if 'filled' in plot_options:
                out_file.write(', fillcolor="%s"'
                               % (get_colour(colours, leaves / leaves.sum())))
            out_file.write('] ;\n')

            if parent is not None:
                # Add edge to parent
                out_file.write('%d -> %d' % (parent, node_id))
                root_labels = None
                if 'yes' in plot_options:
                    root_labels = ('Yes', 'No')
                if 'true' in plot_options:
                    root_labels = ('True', 'False')
                if parent == 0 and root_labels is not None:
                    out_file.write(' [labelangle=45, labeldistance=2.5')
                    if node_id == 1:
                        out_file.write(', headlabel="%s"]' % root_labels[0])
                    else:
                        out_file.write(', headlabel="%s"]' % root_labels[1])
                out_file.write(' ;\n')

            if left_child != _tree.TREE_LEAF:
                recurse(tree, left_child, criterion=criterion, parent=node_id,
                        depth=depth + 1, colours=colours)
                recurse(tree, right_child, criterion=criterion, parent=node_id,
                        depth=depth + 1, colours=colours)

        else:
            out_file.write('%d [label="(...)"' % node_id)
            if 'filled' in plot_options:
                out_file.write(', fillcolor="#C0C0C0"')
            out_file.write('] ;\n' % node_id)

            if parent is not None:
                # Add edge to parent
                out_file.write('%d -> %d ;\n' % (parent, node_id))

    own_file = False
    try:
        if isinstance(out_file, six.string_types):
            if six.PY3:
                out_file = open(out_file, "w", encoding="utf-8")
            else:
                out_file = open(out_file, "wb")
            own_file = True

        # The depth of each node for plotting with 'leaf' option
        ranks = {'leaves': []}

        if plot_options is None:
            plot_options = ['samples', 'metric', 'labels']
        plot_options, subtree_required = parse_options()

        out_file.write('digraph Tree {\n')

        # Specify node aesthetics
        out_file.write('node [shape=box')
        rounded_filled = set(['rounded', 'filled']) & plot_options
        if len(rounded_filled) > 0:
            out_file.write(', style="%s", color="black"'
                           % ",".join(rounded_filled))
        if 'helvetica' in plot_options:
            out_file.write(', fontname=helvetica')
        out_file.write('] ;\n')

        # Specify graph & edge aesthetics
        if 'leaf' in plot_options:
            out_file.write('graph [ranksep=equally, splines=polyline] ;\n')
        if 'helvetica' in plot_options:
            out_file.write('edge [fontname=helvetica] ;\n')
        if 'rotate' in plot_options:
            out_file.write('rankdir=LR ;\n')

        if isinstance(decision_tree, _tree.Tree):
            recurse(decision_tree, 0, criterion="impurity")
        else:
            recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)

        if 'leaf' in plot_options:
            # Draw each node of the same depth at the same level,
            # and all terminal/leaf nodes at the bottom of the tree
            for rank in ranks:
                out_file.write("{rank=same ; " +
                               ", ".join(r for r in ranks[rank]) + "}\n")
        out_file.write("}")

    finally:
        if own_file:
            out_file.close()
