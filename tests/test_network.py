"""Tests for Bipartite Regional Scribal Network Graph."""

import pytest
from linear_a.network.scribal_graph import ScribalNetworkGraph


def test_scribal_network_construction():
    graph = ScribalNetworkGraph()
    rep = graph.analyze_network()

    assert rep.total_nodes > 15
    assert rep.total_edges > 15
    assert rep.total_sites >= 5
    assert rep.total_commodities >= 3
    assert len(rep.top_central_agents) > 0

    # Verify site nodes exist
    assert "Hagia_Triada" in graph.nodes
    assert graph.nodes["Hagia_Triada"].node_type == "SITE"

    # Verify commodity nodes exist
    assert "OLE" in graph.nodes or "FIC" in graph.nodes

    # Check export format
    cyto = graph.to_cytoscape_json()
    assert "elements" in cyto
    assert len(cyto["elements"]) == rep.total_nodes + rep.total_edges
