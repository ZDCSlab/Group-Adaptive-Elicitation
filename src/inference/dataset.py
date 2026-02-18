import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import pandas as pd

NodeId = str
QueryId = str
Label = int

@dataclass
class SurveyGraph:
    nodes: List
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
    def load_dataset(df_survey, df_heldout, codebook, verbose: bool = False):
        # --- 1. Preparation & String Casting ---
        # Ensure caseid is integer string (match GNN convention) to prevent key mismatches
        df_survey = df_survey.dropna(subset=["caseid"])
        df_heldout = df_heldout.dropna(subset=["caseid"])
        df_survey["caseid"] = df_survey["caseid"].astype(int).astype(str)
        df_heldout["caseid"] = df_heldout["caseid"].astype(int).astype(str)
        
        Xpool = [c for c in df_survey.columns if c != "caseid"]
        X_heldout = [c for c in df_heldout.columns if c != "caseid"]
        all_nodes = df_survey["caseid"].tolist()
        all_qids = list(set(Xpool + X_heldout))

        # --- 2. Standardization Logic (Robust A/B/C Conversion) ---
        for qid in all_qids:
            if qid not in codebook:
                continue
            
   
        # --- 3. Build Y (Ground Truth) and Y_heldout ---
        Y: Dict[str, Dict[str, str]] = {}
        df_survey_indexed = df_survey.set_index("caseid")
        for cid in all_nodes:
            if cid in df_survey_indexed.index:
                row_ser = df_survey_indexed.loc[cid]
                if isinstance(row_ser, pd.DataFrame):
                    row_ser = row_ser.iloc[0]
                
                ans = {qid: row_ser[qid] for qid in Xpool if pd.notna(row_ser[qid])}
                if ans:
                    Y[str(cid)] = ans

        Y_heldout: Dict[str, Dict[str, str]] = {}
        df_heldout_indexed = df_heldout.set_index("caseid")
        for cid in all_nodes:
            if cid in df_heldout_indexed.index:
                row_ser = df_heldout_indexed.loc[cid]
                if isinstance(row_ser, pd.DataFrame):
                    row_ser = row_ser.iloc[0]
                    
                ans = {qid: row_ser[qid] for qid in X_heldout if pd.notna(row_ser[qid])}
                if ans:
                    Y_heldout[str(cid)] = ans # Explicitly cast key to str

        # --- 4. Question Formatting ---
        option_sizes: Dict[str, int] = {}
        def format_options_text(options_dict):
            sorted_items = sorted(options_dict.items())
            return "\n" + "\n".join([f"{k}. {v}" for k, v in sorted_items])

        for qid in all_qids:
            if qid in codebook:
                option_sizes[qid] = len(codebook[qid]['options'])
                codebook[qid]["question"] += format_options_text(codebook[qid]["options"])

        # --- 5. Return Dataset ---
        observed_dict = {qid: dict() for qid in Xpool}
        graph = SurveyGraph(nodes=all_nodes, y=Y)
        dataset = Dataset(
            graph=graph, 
            observed_dict=observed_dict, 
            Xpool=Xpool, 
            X_heldout=X_heldout, 
            Y_heldout=Y_heldout, 
            option_sizes=option_sizes, 
            codebook=codebook
        )

        if verbose:
            # Check if ANY nodes have data
            print("=== Dataset Info ===")
            print(f"#Nodes: {len(all_nodes)}")
            print(f"#Questions (Xpool): {len(Xpool)}")
            print(f"#Ground-truth Y entries: {len(Y)}")
            print(f"Example Node: {all_nodes[0] if all_nodes else None}")
            print(f"Example Question: {Xpool[0] if Xpool else None}")
            print(f"Example Question: {dataset.codebook[Xpool[0]]['question']}")
            print(f"#Example Y: {Y[all_nodes[0]]}")
            if Xpool:
                print(f"Options[{Xpool[0]}]: {option_sizes[Xpool[0]]}")
    
        return dataset

    