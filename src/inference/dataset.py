import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from .utils import *
import pandas as pd

NodeId = str
QueryId = str
Label = int

@dataclass
class SurveyGraph:
    nodes: List
    neighbor: Dict[NodeId, List]  # Neighbor
    label: Dict[NodeId, Label]  # Neighbor
    y: Dict[NodeId, Dict[QueryId, int]]  # ground-truth answers (if offline)

# @dataclass
# class History:
#     # Each round stores: (imputed full response vector for all nodes, the asked query)
#     nodes: List[NodeId] 
#     transition: Dict[NodeId, str] = field(default_factory=dict)
#     asked_queries: List[QueryId] = field(default_factory=list)

#     def __post_init__(self):
#         self.transition = {nodeid: "" for nodeid in self.nodes}
#         self.asked_queries = []

#     def add_round(self, queryid, query, options, answers: Dict[str, int], neighbors: Dict[str, List]) -> None:
#         """
#         Append a new round with imputed answers and the asked query.

#         Args:
#             query: the asked query
#             options: dict[label_id -> str], label meaning lookup
#             answers: dict[nodeid -> answer]
#             neighbors: dict[nodeid -> dict[neighbor -> answer]]
#         """
#         print(f"\n[Add Round] Query: {queryid}")
#         print(f"Total nodes updated: {len(answers)}")

#         def _neighbors_answers(y, neighbors):
#             ans: Dict[NodeId, int] = {}
#             for u in neighbors:
#                 ans[u] = int(y[u])
#             return ans
        
#         for nodeid, ans in answers.items():
#             # compute majority key
#             neighbors_ans = _neighbors_answers(answers, neighbors[nodeid])
#             counter = Counter(neighbors_ans.values())
#             maj_key, maj_count = counter.most_common(1)[0]

#             option_lst = ["A", "B"]
#             # meaning lookup if provided
#             maj_meaning = f"({options[str(maj_key)]})"
#             neighbor_txt = f"<Neighbor>Majority Answer: {option_lst[maj_key-1]}{maj_meaning}"

#             # update transition
#             self.transition[nodeid] += f"<Question>{query}{neighbor_txt}<Answer>{option_lst[int(ans)-1]}"

#             # --- debug print ---
#             # print(f" Node {nodeid}: answer={ans}, "
#             #     f"neighbor_majority={maj_key}{maj_meaning}, "
#             #     f"maj_count={maj_count}, "
#             #     f"neighbors={neighbors[nodeid]}")

#         # record asked query
#         self.asked_queries.append(queryid)
#         print(f"Appended query '{queryid}' to asked_queries.\n")
#         print('Example: ', self.transition[self.nodes[0]])

#     def queries_so_far(self) -> List[QueryId]:
#         """Return all queries that have been asked so far."""
#         return self.asked_queries

#     def num_rounds(self) -> int:
#         """Return number of rounds recorded."""
#         return len(self.asked_queries)

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

    
    @staticmethod
    def load_dataset(df_survey, df_heldout, neighbors_info, codebook, verbose: bool = False):

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
            Nei[cid] = val["topk_ids"]
        
        Label = dict()
        for cid, val in neighbors_info.items():
            cid = str(cid)
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
