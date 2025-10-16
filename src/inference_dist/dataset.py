import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from .utils import *
import pandas as pd
import random
from collections import defaultdict

NodeId = str
QueryId = str
Label = int

@dataclass
class SurveyGraph:
    nodes: List
    neighbor: Dict[NodeId, List]  # Neighbor
    label: Dict[NodeId, Label]  # Neighbor
    y: Dict[NodeId, Dict[QueryId, int]]  # ground-truth answers (if offline)


@dataclass
class Dataset:
    def __init__(
        self,
        graph: "SurveyGraph",
        observed_dict: Dict,
        Xpool: List[QueryId],
        X_heldout: List[QueryId],
        Y_heldout: Dict,
        option_sizes: Dict[QueryId, int],
        codebook: Dict
    ):
        """
        Args:
          graph:       SurveyGraph object (with Nei, Z, Y dicts)
          Xpool:       list of query IDs available for selection
          option_sizes: dict mapping query ID -> number of options
        """
        self.graph = graph
        self.observed_dict = observed_dict
        self.Xpool = Xpool
        self.X_heldout = X_heldout
        self.Y_heldout = Y_heldout
        self.option_sizes = option_sizes
        self.codebook = codebook
        self.ground_truth = dict()
        self.asked_queries = []


    def __len__(self):
        """Number of nodes in the graph."""
        return len(self.graph.nodes)

    def num_queries(self):
        """Number of unique queries in the pool."""
        return len(self.Xpool)

    def option_size(self, qid: QueryId) -> int:
        """Return cardinality of a query."""
        return self.option_sizes.get(qid, 0)
    

    def update_observed(self, qid, V_sel):
        """
        Update observed_dict with a new observation.

        Args:
            qid: QueryId (str or int), which query we are updating
            v: NodeId (int), the node being observed
            ans: int, the observed answer label
        """
        for v in V_sel:
            self.observed_dict[qid][v] = self.graph.y[v][qid]
        return self.observed_dict[qid]
    
    def update_observed_estimated(self, qid, V_rest=dict()):
    
        for v in V_rest.keys():
            self.observed_dict[qid][v] = V_rest[v]
        return self.observed_dict[qid]


    def update_neighbors_info(
        self,
        K: int = 2,
        seed: Optional[int] = 42) -> None:
        """Update self.graph.neighbor = {cid: [neighbor_cid, ...]} using top-K
        """
        rng = random.Random(seed)
        graph_nodes = self.graph.nodes

        # Build {cid: {qid: ans}}
        by_case: Dict[str, Dict[str, Ans]] = defaultdict(dict)
        for qid, case_map in self.observed_dict.items():
            for cid, ans in case_map.items():
                by_case[cid][qid] = ans  # cid is str; ans is "A"/"B"

        if len(by_case) == 0:
            # random select K neighbor for each caseid
            neighbors: Dict[str, List[str]] = {}
            for cid in graph_nodes:
                pool = [n for n in graph_nodes if n != cid]
                rng.shuffle(pool)
                neighbors[cid] = sorted(pool[:K])  # sort only for stable display
            self.graph.neighbor = neighbors
            return self.graph.neighbor

        case_ids = list(by_case.keys())
        qids_by_case = {cid: set(qa.keys()) for cid, qa in by_case.items()}

        def sim(a: str, b: str) -> Optional[float]:
            shared = qids_by_case[a] & qids_by_case[b]
            m = len(shared)   
            qa, qb = by_case[a], by_case[b]
            matches = sum(qa[q] == qb[q] for q in shared)
            return matches / m

  
        for cid in case_ids:
            cand: List[Tuple[str, float]] = []
            for nid in case_ids:
                if nid == cid:
                    continue
                s = sim(cid, nid)
                if s is not None:
                    cand.append((nid, float(s)))

            cand.sort(key=lambda x: (-x[1], x[0]))
            chosen = cand[:K]
            self.graph.neighbor[cid] = [u for u, _ in chosen]
        
        return self.graph.neighbor

    
    @staticmethod
    def load_dataset(df_survey, df_heldout, neighbors_info, codebook, top_k_nei=20, verbose: bool = False):

        # Use node universe from neighbors (they define the graph)
        Xpool = [c for c in df_survey.columns if c != "caseid"]
        df_survey["caseid"] = df_survey["caseid"].astype(str)
        df_heldout["caseid"] = df_heldout["caseid"].astype(str)
        # assert list(neighbors_info.keys()) == list(df_survey["caseid"])
        all_nodes = df_survey["caseid"].tolist()

        X_heldout = [c for c in df_heldout.columns if c != "caseid"]
        Y_heldout: Dict[NodeId, Dict[QueryId, int]] = {}

        cand = ["A", "B"]
        for cid in all_nodes:
            row = df_heldout[df_heldout["caseid"] == cid]
            row = row.iloc[0]
            ans_dict = {}
            for qid in X_heldout:
                if qid in row:
                    ans_dict[qid] = cand[int(row[qid])-1]
            if ans_dict:
                Y_heldout[cid] = ans_dict
      
        # 1) Neighbors
        Nei: Dict[NodeId, List[NodeId]] = {}
        for cid, val in neighbors_info.items():
            cid = str(cid)
            if cid in all_nodes:
                Nei[cid] = val["neighbors"][:top_k_nei]
            # Nei[cid] = []

        # rng = random.Random(42)
        # noise_frac = 1  # add ~10% as noise
        # for cid, val in neighbors_info.items():
        #     cid = str(cid)
        #     if cid in all_nodes:
        #         base = [str(x) for x in val["neighbors"]][:top_k_nei]
        #         base = [n for n in base if n in all_nodes and n != cid]
        #         # choose noise from nodes not already neighbors and not self
        #         candidates = list(set(all_nodes) - set(base) - {cid})
        #         k_noise = min(len(candidates), max(1, int(round(len(base) * noise_frac))) if base else 1)
        #         noise = rng.sample(candidates, k_noise) if k_noise > 0 else []
        #         Nei[cid] = base + noise
        ##########################################
        Label = dict()
        for cid, val in neighbors_info.items():
            cid = str(cid)
            if cid in all_nodes:
                Label[cid] = val["community"]

        # 2) Question Options (store cardinality, not list of keys)
        option_sizes: Dict[QueryId, int] = {}
        for qid in Xpool:
            option_sizes[qid] = len(codebook[qid]['options'])
        for qid in X_heldout:
            option_sizes[qid] = len(codebook[qid]['options'])


        # 3) Ground-truth answers Y (optional)
        Y: Dict[NodeId, Dict[QueryId, int]] = {}
        for cid in all_nodes:
            row = df_survey[df_survey["caseid"] == cid]
            row = row.iloc[0]
            ans_dict = {}
            for qid in Xpool:
                if qid in row:
                    ans_dict[qid] = cand[int(row[qid])-1]
            if ans_dict:
                Y[cid] = ans_dict

        observed_dict = {qid: dict() for qid in Xpool}

        for qid in Xpool:
            codebook[qid]["question"] += "\nA. Support\nB. Oppose"
        for qid in X_heldout:
            codebook[qid]["question"] += "\nA. Support\nB. Oppose"


        graph = SurveyGraph(nodes=all_nodes, neighbor=Nei, label=Label, y=Y)
        dataset = Dataset(graph=graph, observed_dict=observed_dict, Xpool=Xpool, X_heldout=X_heldout, Y_heldout=Y_heldout, option_sizes=option_sizes, codebook=codebook)

        if verbose:
            print("=== Dataset Info ===")
            print(f"#Nodes: {len(all_nodes)}")
            print(f"#Neighbors entries: {len(Nei)}")
            print(f"#Questions (Xpool): {len(Xpool)}")
            print(f"#Ground-truth Y entries: {len(Y)}")
            print(f"Example Node: {all_nodes[0] if all_nodes else None}")
            print(f"Example Question: {Xpool[0] if Xpool else None}")
            print(f"#Example Y: {Y[all_nodes[0]]}")
            if Xpool:
                print(f"Options[{Xpool[0]}]: {option_sizes[Xpool[0]]}")

        return dataset
