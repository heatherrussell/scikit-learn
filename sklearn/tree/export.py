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

import numpy as np

from ..externals import six

from . import _tree


def export_graphviz(decision_tree, out_file="tree.dot", feature_names=None,
                    max_depth=None, pretty=False, simple=False):
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

    def colour_brew(n):

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

    def recurse_subtree(tree, node_id, values=[]):
        # Gather the leaf node classifications for a subtree
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        if left_child != _tree.TREE_LEAF:
            recurse_subtree(tree, left_child, values=values)
            recurse_subtree(tree, right_child, values=values)
        else:
            values.append(tree.value[node_id][0, :])
        return values

    def node_to_str(tree, node_id, criterion, value=None):
        if not isinstance(criterion, six.string_types):
            criterion = "impurity"

        if value is None or not simple:
            value = tree.value[node_id]
            if tree.n_outputs == 1:
                value = value[0, :]

        if feature_names is not None:
            feature = feature_names[tree.feature[node_id]]
        else:
            feature = "X[%s]" % tree.feature[node_id]

        if simple:
            percent = (100. * tree.n_node_samples[node_id] /
                       float(tree.n_node_samples[0]))
            if tree.children_left[node_id] == _tree.TREE_LEAF:
                return ("%d\\n%s\\n%.1f%%"
                        % (list(value).index(value.max()),
                           [round(v, 2) for v in value],
                           percent))
            return ("%s <= %.4f\\n%d\\n%s\\n%.1f%%"
                    % (feature,
                       tree.threshold[node_id],
                       list(value).index(value.max()),
                       [round(v, 2) for v in value],
                       percent))

        else:
            if tree.children_left[node_id] == _tree.TREE_LEAF:
                return ("%s = %.4f\\nsamples = %s\\nvalue = %s"
                        % (criterion,
                           tree.impurity[node_id],
                           tree.n_node_samples[node_id],
                           value))
            return ("%s <= %.4f\\n%s = %s\\nsamples = %s"
                    % (feature,
                       tree.threshold[node_id],
                       criterion,
                       tree.impurity[node_id],
                       tree.n_node_samples[node_id]))

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
        if pretty or simple:
            leaves = np.sum(np.array(recurse_subtree(tree, node_id, values=[])), axis=0)
            leaves = leaves / leaves.sum()
            if colours is None:
                colours = colour_brew(tree.n_classes[0])

        # Add node with description
        if max_depth is None or depth <= max_depth:

            if pretty:
                out_file.write('%d [label="%s", fillcolor="%s"] ;\n'
                               % (node_id,
                                  node_to_str(tree, node_id, criterion,
                                              leaves),
                                  get_colour(colours, leaves)))
            else:
                out_file.write('%d [label="%s"] ;\n'
                               % (node_id,
                                  node_to_str(tree, node_id, criterion,
                                              leaves)))

            if parent is not None:
                # Add edge to parent
                if pretty and parent == 0 and node_id == 1:
                    out_file.write('%d -> %d [headlabel="Yes", labelangle=45, '
                                   'labeldistance=2.5, headport=n] ;\n'
                                   % (parent, node_id))
                elif pretty and parent == 0:
                    out_file.write('%d -> %d [headlabel="No", labelangle=-45, '
                                   'labeldistance=2.5, headport=n] ;\n'
                                   % (parent, node_id))
                elif pretty:
                    out_file.write('%d -> %d [headport=n] ;\n'
                                   % (parent, node_id))
                else:
                    out_file.write('%d -> %d ;\n'
                                   % (parent, node_id))

            if left_child != _tree.TREE_LEAF:
                recurse(tree, left_child, criterion=criterion, parent=node_id,
                        depth=depth + 1, colours=colours)
                recurse(tree, right_child, criterion=criterion, parent=node_id,
                        depth=depth + 1, colours=colours)

        else:
            if pretty:
                out_file.write('%d [label="(...)", fillcolor="#C0C0C0"] ;\n'
                               % node_id)
            else:
                out_file.write('%d [label="(...)"] ;\n' % node_id)

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

        # The depth of each node for plotting with 'pretty'
        ranks = {'leaves': []}

        out_file.write("digraph Tree {\n")
        if pretty:
            out_file.write('graph [ranksep=equally, splines=polyline, '
                           'fontname=helvetica] ;\n')
            out_file.write('node [shape=box, style="rounded,filled", '
                           'fontname=helvetica, color="black"] ;\n')
            out_file.write('edge [fontname=helvetica] ;\n')
        else:
            out_file.write('node [shape=box] ;\n')
        if isinstance(decision_tree, _tree.Tree):
            recurse(decision_tree, 0, criterion="impurity")
        else:
            recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)
        if pretty:
            # Draw each node of the same depth at the same level,
            # and all terminal/leaf nodes at the bottom of the tree
            for rank in ranks:
                out_file.write("{rank=same ; " +
                               ", ".join(r for r in ranks[rank]) + "}\n")
        out_file.write("}")

    finally:
        if own_file:
            out_file.close()
