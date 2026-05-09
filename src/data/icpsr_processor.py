"""ICPSR 22140 dataset processor.

Parses the three ICPSR 22140 TSV files, builds one NetworkX Graph/DiGraph per
study and per STD, extracts one-hot covariate vectors, and merges all studies
into a single index space for downstream use.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def _save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


class ICPSRProcessor:
    """Processes the ICPSR 22140 dataset into per-STD merged graph data.

    Args:
        tsv_file1: Node attributes TSV (DS0001).
        tsv_file2: First edge TSV (DS0002).
        tsv_file3: Second edge TSV (DS0003).
        pickle_filename: Cache path for the curated intermediate dataset.
    """

    STD_TO_DFKEY = {
        "Gonorrhea": "GONO",
        "Chlamydia": "CHLAM",
        "Syphilis": "SYPH",
        "HIV": "HIV",
        "Hepatitis": "HBV",
    }
    COVARIATE_HEADERS = [
        "LOCAL", "RACE", "ETHN", "SEX", "ORIENT", "BEHAV", "PRO",
        "PIMP", "JOHN", "DEALER", "DRUGMAN", "THIEF", "RETIRED",
        "HWIFE", "DISABLE", "UNEMP", "STREETS",
    ]

    def __init__(
        self,
        tsv_file1: str,
        tsv_file2: str,
        tsv_file3: str,
        pickle_filename: str,
    ) -> None:
        self.pickle_filename = pickle_filename
        self.curated_dataset = self._extract_curated_dataset(
            tsv_file1, tsv_file2, tsv_file3
        )
        self.merged_datasets = self._merge_all_std_datasets_into_one()

    def _generate_covariates(
        self,
        node_df: pd.DataFrame,
        rid: int,
        studynum: int,
    ) -> np.ndarray:
        covariates = []
        mask = (node_df["RID"] == rid) & (node_df["STUDYNUM"] == studynum)
        assert len(node_df.loc[mask, self.COVARIATE_HEADERS]) == 1
        for col in self.COVARIATE_HEADERS:
            values = sorted([int(x) for x in set(node_df[col].values)])
            one_hot = [0] * len(values)
            idx = values.index(node_df.loc[mask, col].iloc[0])
            one_hot[idx] = 1
            covariates += one_hot
        return np.array(covariates)

    def _extract_curated_dataset(
        self,
        tsv_file1: str,
        tsv_file2: str,
        tsv_file3: str,
    ) -> dict:
        if not os.path.isfile(self.pickle_filename):
            node_df = pd.read_csv(tsv_file1, sep="\t", dtype=str)
            df2 = pd.read_csv(tsv_file2, sep="\t", dtype=str)
            df3 = pd.read_csv(tsv_file3, sep="\t", dtype=str)

            assert set(df2.columns) == set(df3.columns)
            df3 = df3[df2.columns]

            assert df2.columns.equals(df3.columns)
            edge_df = pd.concat([df2, df3], ignore_index=True)

            columns_to_convert = (
                self.COVARIATE_HEADERS
                + ["STUDYNUM", "RID", "ID1", "ID2", "TIETYPE"]
                + [f"{dfkey}1" for dfkey in self.STD_TO_DFKEY.values()]
                + [f"{dfkey}2" for dfkey in self.STD_TO_DFKEY.values()]
            )
            for col in columns_to_convert:
                if col in node_df.columns:
                    node_df[col] = pd.to_numeric(node_df[col], errors="coerce").astype("Int64")
                if col in edge_df.columns:
                    edge_df[col] = pd.to_numeric(edge_df[col], errors="coerce").astype("Int64")

            curated_dataset: dict = {std: [] for std in self.STD_TO_DFKEY.keys()}
            for std, dfkey in self.STD_TO_DFKEY.items():
                std_edge_filter = edge_df[
                    edge_df[f"{dfkey}1"].isin({0, 1}) & edge_df[f"{dfkey}2"].isin({0, 1})
                ]
                graphs = {studynum: nx.Graph() for studynum in set(node_df["STUDYNUM"])}
                digraphs = {studynum: nx.DiGraph() for studynum in set(node_df["STUDYNUM"])}
                statuses: dict = {studynum: dict() for studynum in set(node_df["STUDYNUM"])}
                for _, row in std_edge_filter.iterrows():
                    studynum = row["STUDYNUM"]
                    u, v = row["ID1"], row["ID2"]
                    u_status, v_status = row[f"{dfkey}1"], row[f"{dfkey}2"]
                    if u not in statuses[studynum]:
                        statuses[studynum][u] = u_status
                    else:
                        statuses[studynum][u] = max(u_status, statuses[studynum][u])
                    if v not in statuses[studynum]:
                        statuses[studynum][v] = v_status
                    else:
                        statuses[studynum][v] = max(v_status, statuses[studynum][v])
                    graphs[studynum].add_edge(u, v)
                    digraphs[studynum].add_edge(u, v)

                for studynum in set(node_df["STUDYNUM"]):
                    G = graphs[studynum]
                    DG = digraphs[studynum]
                    assert G.number_of_nodes() == DG.number_of_nodes()
                    if G.number_of_nodes() == 0:
                        continue

                    individual_mapping: dict = {}
                    individual_covariates: dict = {}
                    individual_statuses: dict = {}
                    for u in G.nodes:
                        individual_mapping[u] = len(individual_mapping)
                        individual_covariates[individual_mapping[u]] = self._generate_covariates(
                            node_df, u, studynum
                        )
                        individual_statuses[individual_mapping[u]] = statuses[studynum][u]
                    G = nx.relabel_nodes(G, individual_mapping)
                    DG = nx.relabel_nodes(DG, individual_mapping)

                    G_roots = []
                    for u in G.nodes:
                        u_is_root = True
                        for v in G.neighbors(u):
                            if (v, u) in DG.edges:
                                u_is_root = False
                                break
                        if u_is_root:
                            G_roots.append(u)
                    DG_roots = [node for node in DG.nodes if DG.in_degree(node) == 0]

                    curated_dataset[std].append({
                        "studynum": studynum,
                        "covariates": individual_covariates,
                        "statuses": individual_statuses,
                        "graph": G,
                        "digraph": DG,
                        "graph_roots": G_roots,
                        "digraph_roots": DG_roots,
                    })

            _save_pickle(curated_dataset, self.pickle_filename)

        return _load_pickle(self.pickle_filename)

    def _merge_all_std_datasets_into_one(self) -> dict:
        merged_datasets: dict = {}
        for std in self.STD_TO_DFKEY.keys():
            sz = 0
            overall_covariates: dict = {}
            overall_statuses: dict = {}
            overall_graph = nx.Graph()
            overall_digraph = nx.DiGraph()
            overall_graph_roots: list = []
            overall_digraph_roots: list = []
            for std_dataset in self.curated_dataset[std]:
                covariates = std_dataset["covariates"]
                statuses = std_dataset["statuses"]
                G = std_dataset["graph"]
                DG = std_dataset["digraph"]
                G_roots = std_dataset["graph_roots"]
                DG_roots = std_dataset["digraph_roots"]
                n = G.number_of_nodes()
                assert DG.number_of_nodes() == n
                for idx in range(n):
                    overall_covariates[idx + sz] = covariates[idx]
                    overall_statuses[idx + sz] = statuses[idx]
                overall_graph.add_nodes_from([idx + sz for idx in G.nodes])
                overall_digraph.add_nodes_from([idx + sz for idx in DG.nodes])
                for u, v in G.edges():
                    overall_graph.add_edge(u + sz, v + sz)
                for u, v in DG.edges():
                    overall_digraph.add_edge(u + sz, v + sz)
                overall_graph_roots += [u + sz for u in G_roots]
                overall_digraph_roots += [u + sz for u in DG_roots]
                sz += n
            merged_datasets[std] = [
                overall_covariates,
                overall_statuses,
                overall_graph,
                overall_digraph,
                np.array(overall_graph_roots),
                np.array(overall_digraph_roots),
            ]
        return merged_datasets
