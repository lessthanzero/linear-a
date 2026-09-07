"""Bipartite Regional Scribal and Commodity Network Graph (LADP v1.0).

Constructs the socio-economic network connecting Minoan administrative agents,
regional archives (Hagia Triada, Knossos, Malia, Phaistos, Khania, Zakros, Tylissos),
and commodities (GRA, OLE, VIN, FIC) across Bronze Age Crete.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from linear_a.corpus.loader import load_all_tablets


@dataclass
class NetworkNode:
    """A node in the bipartite Minoan administrative network."""
    id: str
    label: str
    node_type: str  # "ENTITY", "SITE", "COMMODITY"
    degree: int = 0
    connected_nodes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkEdge:
    """A directed/undirected transaction edge between nodes."""
    source: str
    target: str
    edge_type: str  # "RECORDED_AT", "DEALS_IN", "TRANSFERS_TO"
    weight: float = 1.0
    tablets: List[str] = field(default_factory=list)


@dataclass
class ScribalNetworkReport:
    """Statistical topology of the Minoan palatial economic network."""
    total_nodes: int
    total_edges: int
    total_entities: int
    total_sites: int
    total_commodities: int
    top_central_agents: List[Dict[str, Any]]
    cross_site_agents: List[Dict[str, Any]]
    commodity_distribution: Dict[str, int]
    summary: str


class ScribalNetworkGraph:
    """Analyzes cross-site economic linkages and administrative structures."""

    def __init__(self):
        self.nodes: Dict[str, NetworkNode] = {}
        self.edges: List[NetworkEdge] = []
        self._build_graph()

    def _build_graph(self):
        tablets = load_all_tablets()
        edge_map: Dict[Tuple[str, str, str], NetworkEdge] = {}

        for t in tablets:
            t_id = t["id"]
            site = t.get("site", "Crete")
            comm_default = t.get("commodity")

            # 1. Ensure Site Node
            if site not in self.nodes:
                self.nodes[site] = NetworkNode(id=site, label=site, node_type="SITE")

            # 2. Iterate Line Items
            for it in t.get("items", []):
                agent = it.get("entry_header", "").strip()
                if not agent or agent in ("KU-RO", "KI-RO") or agent.isdigit():
                    continue

                comm = it.get("commodity") or comm_default or "UNSPECIFIED"
                qty = float(it.get("integer_amount", 0))

                # Ensure Agent Node
                if agent not in self.nodes:
                    self.nodes[agent] = NetworkNode(
                        id=agent,
                        label=agent,
                        node_type="ENTITY",
                        metadata={"attested_sites": set(), "commodities": set()},
                    )
                self.nodes[agent].metadata["attested_sites"].add(site)
                self.nodes[agent].metadata["commodities"].add(comm)

                # Ensure Commodity Node
                if comm not in self.nodes:
                    self.nodes[comm] = NetworkNode(id=comm, label=comm, node_type="COMMODITY")

                # Edge 1: Agent -> Site
                key_site = (agent, site, "RECORDED_AT")
                if key_site not in edge_map:
                    edge_map[key_site] = NetworkEdge(source=agent, target=site, edge_type="RECORDED_AT", weight=1.0)
                else:
                    edge_map[key_site].weight += 1.0
                if t_id not in edge_map[key_site].tablets:
                    edge_map[key_site].tablets.append(t_id)

                # Edge 2: Agent -> Commodity
                key_comm = (agent, comm, "DEALS_IN")
                if key_comm not in edge_map:
                    edge_map[key_comm] = NetworkEdge(source=agent, target=comm, edge_type="DEALS_IN", weight=qty)
                else:
                    edge_map[key_comm].weight += qty
                if t_id not in edge_map[key_comm].tablets:
                    edge_map[key_comm].tablets.append(t_id)

        self.edges = list(edge_map.values())

        # Update node degrees and connected lists
        for edge in self.edges:
            if edge.source in self.nodes:
                self.nodes[edge.source].degree += 1
                if edge.target not in self.nodes[edge.source].connected_nodes:
                    self.nodes[edge.source].connected_nodes.append(edge.target)
            if edge.target in self.nodes:
                self.nodes[edge.target].degree += 1
                if edge.source not in self.nodes[edge.target].connected_nodes:
                    self.nodes[edge.target].connected_nodes.append(edge.source)

    def analyze_network(self) -> ScribalNetworkReport:
        """Compute network statistics and identify cross-site administrative actors."""
        entities = [n for n in self.nodes.values() if n.node_type == "ENTITY"]
        sites = [n for n in self.nodes.values() if n.node_type == "SITE"]
        commodities = [n for n in self.nodes.values() if n.node_type == "COMMODITY"]

        # Rank agents by degree centrality
        entities_by_degree = sorted(entities, key=lambda e: e.degree, reverse=True)
        top_agents = [
            {
                "agent": e.label,
                "degree": e.degree,
                "sites": list(e.metadata.get("attested_sites", [])),
                "commodities": list(e.metadata.get("commodities", [])),
            }
            for e in entities_by_degree[:10]
        ]

        # Find agents appearing in multiple sites
        cross_site = []
        for e in entities:
            att_sites = list(e.metadata.get("attested_sites", []))
            if len(att_sites) > 1:
                cross_site.append({
                    "agent": e.label,
                    "sites_count": len(att_sites),
                    "sites": att_sites,
                    "commodities": list(e.metadata.get("commodities", [])),
                })
        cross_site.sort(key=lambda x: x["sites_count"], reverse=True)

        comm_dist = {}
        for edge in self.edges:
            if edge.edge_type == "DEALS_IN":
                comm_dist[edge.target] = comm_dist.get(edge.target, 0) + int(edge.weight)

        summary = (
            f"Minoan economic graph contains {len(entities)} administrative entities across "
            f"{len(sites)} sites dealing in {len(commodities)} commodities. "
            f"Cross-site economic agents identified: {len(cross_site)}."
        )

        return ScribalNetworkReport(
            total_nodes=len(self.nodes),
            total_edges=len(self.edges),
            total_entities=len(entities),
            total_sites=len(sites),
            total_commodities=len(commodities),
            top_central_agents=top_agents,
            cross_site_agents=cross_site,
            commodity_distribution=comm_dist,
            summary=summary,
        )

    def to_cytoscape_json(self) -> Dict[str, Any]:
        """Export nodes and edges in standard graph visualizer JSON format."""
        elements = []
        for n in self.nodes.values():
            color = "#059669" if n.node_type == "SITE" else ("#4f46e5" if n.node_type == "ENTITY" else "#d97706")
            elements.append({
                "data": {
                    "id": n.id,
                    "label": n.label,
                    "type": n.node_type,
                    "degree": n.degree,
                    "color": color,
                }
            })
        for idx, e in enumerate(self.edges):
            elements.append({
                "data": {
                    "id": f"edge_{idx}",
                    "source": e.source,
                    "target": e.target,
                    "type": e.edge_type,
                    "weight": e.weight,
                }
            })
        return {"elements": elements}
