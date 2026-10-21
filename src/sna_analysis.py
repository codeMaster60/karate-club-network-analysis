"""
Social Network Analysis of Zachary's Karate Club
=================================================

Computes every metric used in the lab and exports:
  - data/karate_edges.csv, data/karate_nodes.csv      (the dataset)
  - results/*.csv, results/summary.json               (all results)
  - results/figures/*.png                             (static figures)
  - docs/data.js                                      (data for the website)

Run from the repository root:
    python src/sna_analysis.py
"""

from __future__ import annotations

import json
import random
from itertools import islice
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS, FIGS, DOCS = (ROOT / "data", ROOT / "results",
                             ROOT / "results" / "figures", ROOT / "docs")
SEED = 42
LEADERS = {"Mr. Hi": 0, "Officer": 33}   # instructor and club administrator
METRICS = ["degree", "closeness", "betweenness", "eigenvector", "pagerank"]
METRIC_LABELS = {
    "degree": "Degree", "closeness": "Closeness", "betweenness": "Betweenness",
    "eigenvector": "Eigenvector", "pagerank": "PageRank",
}


# --------------------------------------------------------------------------- #
# 1. Data
# --------------------------------------------------------------------------- #
def load_graph() -> nx.Graph:
    """Zachary (1977): 34 members, 78 friendship ties, faction label per member.

    NetworkX numbers members 0..33; the original paper and the Network
    Repository file number them 1..34 (add 1 to convert).
    """
    return nx.karate_club_graph()


def export_dataset(G: nx.Graph) -> None:
    pd.DataFrame(
        [(u, v, d["weight"]) for u, v, d in G.edges(data=True)],
        columns=["source", "target", "weight"],
    ).to_csv(DATA / "karate_edges.csv", index=False)
    pd.DataFrame(
        [(n, G.nodes[n]["club"]) for n in G.nodes],
        columns=["node", "club"],
    ).to_csv(DATA / "karate_nodes.csv", index=False)


# --------------------------------------------------------------------------- #
# 2. Graph-level statistics
# --------------------------------------------------------------------------- #
def graph_stats(G: nx.Graph) -> dict:
    deg = dict(G.degree())
    tri = nx.triangles(G)
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "max_degree": max(deg.values()),
        "min_degree": min(deg.values()),
        "avg_degree": sum(deg.values()) / G.number_of_nodes(),
        "assortativity": nx.degree_assortativity_coefficient(G),
        "triangles_unique": sum(tri.values()) // 3,
        "triangles_per_node_sum": sum(tri.values()),
        "avg_clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
        "max_core_number": max(nx.core_number(G).values()),
        "max_clique_size": max(len(c) for c in nx.find_cliques(G)),
        "diameter": nx.diameter(G),
        "radius": nx.radius(G),
        "avg_shortest_path": nx.average_shortest_path_length(G),
        "connected_components": nx.number_connected_components(G),
        "bridges": [list(e) for e in nx.bridges(G)],
        "articulation_points": sorted(nx.articulation_points(G)),
    }


# --------------------------------------------------------------------------- #
# 3. Node-level centrality
# --------------------------------------------------------------------------- #
def centrality_table(G: nx.Graph) -> pd.DataFrame:
    df = pd.DataFrame({
        "node": list(G.nodes),
        "club": [G.nodes[n]["club"] for n in G.nodes],
        "degree_count": [G.degree(n) for n in G.nodes],
        "strength": [G.degree(n, weight="weight") for n in G.nodes],
    }).set_index("node")
    df["degree"] = pd.Series(nx.degree_centrality(G))
    df["closeness"] = pd.Series(nx.closeness_centrality(G))
    df["betweenness"] = pd.Series(nx.betweenness_centrality(G))
    df["eigenvector"] = pd.Series(nx.eigenvector_centrality(G, max_iter=1000))
    df["pagerank"] = pd.Series(nx.pagerank(G, alpha=0.85, weight=None))
    df["clustering"] = pd.Series(nx.clustering(G))
    df["core_number"] = pd.Series(nx.core_number(G))
    for m in METRICS:
        df[f"{m}_rank"] = df[m].rank(ascending=False, method="min").astype(int)
    return df


def directed_example() -> pd.DataFrame:
    """The karate graph is undirected, so in/out-degree is shown on a small
    'who follows whom' network instead."""
    D = nx.DiGraph([
        ("Asha", "Ravi"), ("Bala", "Ravi"), ("Chitra", "Ravi"),
        ("Deepa", "Ravi"), ("Ravi", "Asha"), ("Asha", "Bala"),
        ("Chitra", "Deepa"), ("Eshan", "Chitra"), ("Eshan", "Asha"),
    ])
    return pd.DataFrame({
        "indegree (popularity)": dict(D.in_degree()),
        "outdegree (activity)": dict(D.out_degree()),
        "pagerank": nx.pagerank(D),
    }).rename_axis("person").sort_values("indegree (popularity)", ascending=False), D


# --------------------------------------------------------------------------- #
# 4. Communities vs the real split
# --------------------------------------------------------------------------- #
def _labels(G, communities) -> list[int]:
    lab = {}
    for i, c in enumerate(communities):
        for n in c:
            lab[n] = i
    return [lab[n] for n in G.nodes]


def _purity_accuracy(G, communities) -> float:
    """Map each community to its majority faction, then score accuracy."""
    correct = 0
    for c in communities:
        clubs = [G.nodes[n]["club"] for n in c]
        correct += max(clubs.count("Mr. Hi"), clubs.count("Officer"))
    return correct / G.number_of_nodes()


def detect_communities(G: nx.Graph) -> dict:
    truth = [0 if G.nodes[n]["club"] == "Mr. Hi" else 1 for n in G.nodes]
    gn = nx.community.girvan_newman(G)
    methods = {
        "Girvan-Newman (2 groups)": next(islice(gn, 0, None)),
        "Louvain": nx.community.louvain_communities(G, seed=SEED),
        "Greedy modularity": nx.community.greedy_modularity_communities(G),
        "Label propagation": list(nx.community.label_propagation_communities(G)),
        "Ground truth (1977 split)": [
            {n for n in G.nodes if G.nodes[n]["club"] == k} for k in LEADERS
        ],
    }
    out = {}
    for name, comms in methods.items():
        comms = sorted((sorted(c) for c in comms), key=lambda c: (-len(c), c))
        labels = _labels(G, comms)
        out[name] = {
            "communities": comms,
            "n_communities": len(comms),
            "modularity": nx.community.modularity(G, comms),
            "accuracy_majority_map": _purity_accuracy(G, comms),
            "ARI": adjusted_rand_score(truth, labels),
            "NMI": normalized_mutual_info_score(truth, labels),
            "labels": labels,
        }
    return out


def leader_distance_prediction(G: nx.Graph) -> dict:
    """Predict each member's side by shortest-path distance to the two leaders
    (ties broken by the number of shared neighbours)."""
    d_hi = nx.single_source_shortest_path_length(G, LEADERS["Mr. Hi"])
    d_off = nx.single_source_shortest_path_length(G, LEADERS["Officer"])
    pred, wrong = {}, []
    for n in G.nodes:
        if d_hi[n] != d_off[n]:
            side = "Mr. Hi" if d_hi[n] < d_off[n] else "Officer"
        else:
            nh = len(set(G[n]) & (set(G[LEADERS["Mr. Hi"]]) | {LEADERS["Mr. Hi"]}))
            no = len(set(G[n]) & (set(G[LEADERS["Officer"]]) | {LEADERS["Officer"]}))
            side = "Mr. Hi" if nh >= no else "Officer"
        pred[n] = side
        if side != G.nodes[n]["club"]:
            wrong.append(n)
    return {"prediction": pred, "wrong": wrong,
            "accuracy": 1 - len(wrong) / G.number_of_nodes()}


def homophily(G: nx.Graph) -> dict:
    same = sum(G.nodes[u]["club"] == G.nodes[v]["club"] for u, v in G.edges)
    n_hi = sum(G.nodes[n]["club"] == "Mr. Hi" for n in G.nodes)
    n_off = G.number_of_nodes() - n_hi
    n = G.number_of_nodes()
    expected = (n_hi * (n_hi - 1) / 2 + n_off * (n_off - 1) / 2) / (n * (n - 1) / 2)
    return {
        "within_faction_edges": same,
        "cross_faction_edges": G.number_of_edges() - same,
        "within_fraction": same / G.number_of_edges(),
        "expected_if_random": expected,
        "attribute_assortativity": nx.attribute_assortativity_coefficient(G, "club"),
    }


# --------------------------------------------------------------------------- #
# 5. Robustness: remove top nodes and watch the network break
# --------------------------------------------------------------------------- #
def attack_simulation(G: nx.Graph, df: pd.DataFrame, k_max: int = 12,
                      random_runs: int = 200) -> dict:
    n = G.number_of_nodes()

    def curve(order):
        H, sizes = G.copy(), [1.0]
        for node in order[:k_max]:
            H.remove_node(node)
            sizes.append(max(len(c) for c in nx.connected_components(H)) / n)
        return sizes

    out = {m: {"order": df[m].sort_values(ascending=False).index[:k_max].tolist(),
               "lcc": None} for m in METRICS}
    for m in METRICS:
        out[m]["lcc"] = curve(out[m]["order"])
    rng = random.Random(SEED)
    rand = np.mean([curve(rng.sample(list(G.nodes), k_max))
                    for _ in range(random_runs)], axis=0)
    out["random"] = {"order": [], "lcc": rand.tolist()}
    return out


def top_edges(G: nx.Graph, k: int = 8) -> list[dict]:
    eb = nx.edge_betweenness_centrality(G)
    return [{"edge": list(e), "score": s,
             "cross_faction": G.nodes[e[0]]["club"] != G.nodes[e[1]]["club"]}
            for e, s in sorted(eb.items(), key=lambda x: -x[1])[:k]]


# --------------------------------------------------------------------------- #
# 6. Figures
# --------------------------------------------------------------------------- #
FACTION_COLORS = {"Mr. Hi": "#2f6fdb", "Officer": "#e0712c"}


def layout(G):
    return nx.kamada_kawai_layout(G)


def save_figures(G, df, pos, comms, attack):
    # 2x2 centrality grid + PageRank
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, m in zip(axes.flat, METRICS + ["club"]):
        if m == "club":
            colors = [FACTION_COLORS[G.nodes[n]["club"]] for n in G.nodes]
            nx.draw_networkx(G, pos, ax=ax, node_color=colors, node_size=320,
                             font_size=8, font_color="white", edge_color="#bbb")
            ax.set_title("Ground truth: the 1977 split")
        else:
            v = df[m].values
            s = (v - v.min()) / (v.max() - v.min())   # scale within this metric
            nx.draw_networkx(G, pos, ax=ax, node_color=v, cmap="viridis",
                             node_size=120 + 1100 * s, font_size=8,
                             font_color="white", edge_color="#ccc")
            ax.set_title(f"{METRIC_LABELS[m]} centrality")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGS / "centrality_grid.png", dpi=150)
    plt.close(fig)

    # Correlation heatmap
    import seaborn as sns
    corr = df[METRICS].rename(columns=METRIC_LABELS).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax)
    ax.set_title("Spearman correlation between centrality rankings")
    fig.tight_layout()
    fig.savefig(FIGS / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # Communities vs truth
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, name in zip(axes, ["Ground truth (1977 split)", "Louvain"]):
        labels = comms[name]["labels"]
        nx.draw_networkx(G, pos, ax=ax, node_color=labels, cmap="tab10",
                         vmin=0, vmax=9, node_size=320, font_size=8,
                         font_color="white", edge_color="#ccc")
        ax.set_title(f"{name}  (modularity {comms[name]['modularity']:.3f})")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGS / "communities_vs_truth.png", dpi=150)
    plt.close(fig)

    # Attack curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in METRICS + ["random"]:
        ax.plot(range(len(attack[m]["lcc"])), attack[m]["lcc"], marker="o",
                ms=3, ls="--" if m == "random" else "-",
                label="Random removal (avg)" if m == "random" else METRIC_LABELS[m])
    ax.set_xlabel("Members removed")
    ax.set_ylabel("Largest group still connected (fraction)")
    ax.set_title("Network robustness: targeted vs random removal")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIGS / "attack_simulation.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 7. Export everything
# --------------------------------------------------------------------------- #
def main():
    for p in (DATA, RESULTS, FIGS, DOCS):
        p.mkdir(parents=True, exist_ok=True)

    G = load_graph()
    export_dataset(G)
    stats = graph_stats(G)
    df = centrality_table(G)
    directed_df, D = directed_example()
    comms = detect_communities(G)
    leader_pred = leader_distance_prediction(G)
    homo = homophily(G)
    attack = attack_simulation(G, df)
    edges_top = top_edges(G)
    corr = df[METRICS].corr(method="spearman")
    pos = layout(G)

    df.to_csv(RESULTS / "centrality.csv", float_format="%.6f")
    corr.to_csv(RESULTS / "correlation_spearman.csv", float_format="%.4f")
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("communities", "labels")}
                  | {"method": name} for name, r in comms.items()]
                 ).set_index("method").to_csv(RESULTS / "communities.csv",
                                              float_format="%.4f")
    directed_df.to_csv(RESULTS / "directed_example.csv", float_format="%.4f")

    summary = {
        "graph_stats": stats, "homophily": homo,
        "leader_distance_prediction": {k: leader_pred[k] for k in ("accuracy", "wrong")},
        "communities": {k: {kk: vv for kk, vv in v.items() if kk != "labels"}
                        for k, v in comms.items()},
        "top_edges_betweenness": edges_top,
        "attack": attack,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    save_figures(G, df, pos, comms, attack)

    # ---- website data ----
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    norm = lambda a: ((a - a.min()) / (a.max() - a.min())).round(4)
    nx_, ny_ = norm(xs), norm(ys)
    nodes = []
    for i, n in enumerate(G.nodes):
        r = df.loc[n]
        nodes.append({
            "id": n, "club": r.club, "x": float(nx_[i]), "y": float(ny_[i]),
            "degreeCount": int(r.degree_count), "strength": int(r.strength),
            "clustering": round(float(r.clustering), 4),
            "core": int(r.core_number),
            "predicted": leader_pred["prediction"][n],
            "communities": {k: v["labels"][i] for k, v in comms.items()},
            **{m: round(float(r[m]), 5) for m in METRICS},
            **{f"{m}Rank": int(r[f"{m}_rank"]) for m in METRICS},
        })
    site = {
        "stats": {k: v for k, v in stats.items()},
        "nodes": nodes,
        "edges": [{"source": u, "target": v, "weight": d["weight"]}
                  for u, v, d in G.edges(data=True)],
        "correlation": {"metrics": METRICS,
                        "matrix": corr.round(3).values.tolist()},
        "communities": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                            for kk, vv in v.items() if kk != "labels"}
                        for k, v in comms.items()},
        "leaderPrediction": {"accuracy": leader_pred["accuracy"],
                             "wrong": leader_pred["wrong"]},
        "homophily": {k: round(v, 4) if isinstance(v, float) else v
                      for k, v in homo.items()},
        "attack": {k: {"order": v["order"], "lcc": [round(x, 4) for x in v["lcc"]]}
                   for k, v in attack.items()},
        "topEdges": [{**e, "score": round(e["score"], 4)} for e in edges_top],
        "directed": {
            "nodes": [{"id": p, "in": int(D.in_degree(p)), "out": int(D.out_degree(p)),
                       "pagerank": round(nx.pagerank(D)[p], 4)} for p in D.nodes],
            "edges": [{"source": u, "target": v} for u, v in D.edges],
        },
    }
    (DOCS / "data.js").write_text(
        "// Generated by src/sna_analysis.py -- do not edit by hand.\n"
        "window.SNA_DATA = " + json.dumps(site, default=float) + ";\n")

    # ---- console report ----
    print("== Graph statistics ==")
    for k in ("nodes", "edges", "density", "avg_degree", "assortativity",
              "avg_clustering", "transitivity", "diameter", "avg_shortest_path",
              "max_core_number", "max_clique_size", "triangles_unique"):
        print(f"  {k:22s} {stats[k]}")
    print("  articulation points   ", stats["articulation_points"])
    print("\n== Top 5 per metric ==")
    for m in METRICS:
        print(f"  {m:12s}", df[m].sort_values(ascending=False).head(5).round(3).to_dict())
    print("\n== Communities ==")
    for k, v in comms.items():
        print(f"  {k:28s} k={v['n_communities']} Q={v['modularity']:.3f} "
              f"acc={v['accuracy_majority_map']:.3f} ARI={v['ARI']:.3f} NMI={v['NMI']:.3f}")
    print("\n== Leader-distance prediction ==", leader_pred["accuracy"],
          "wrong:", leader_pred["wrong"])
    print("== Homophily ==", homo)
    print("== Attack (LCC after removing 1..5) ==")
    for k, v in attack.items():
        print(f"  {k:12s}", [round(x, 2) for x in v["lcc"][:7]], v["order"][:6])
    print("== Top edges ==", [(e["edge"], round(e["score"], 3), e["cross_faction"])
                             for e in edges_top[:5]])
    print("\nDirected example:\n", directed_df.round(3))


if __name__ == "__main__":
    main()
