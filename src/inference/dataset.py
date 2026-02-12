import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from typing import Dict, Iterable, List, Tuple
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
        # self.asked_queries = defaultdict(list)
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
        # Ensure caseid is string in both DFs immediately to prevent key mismatches
        df_survey["caseid"] = df_survey["caseid"].astype(str)
        df_heldout["caseid"] = df_heldout["caseid"].astype(str)
        
        Xpool = [c for c in df_survey.columns if c != "caseid"]
        X_heldout = [c for c in df_heldout.columns if c != "caseid"]
        all_nodes = df_survey["caseid"].tolist()
        all_qids = list(set(Xpool + X_heldout))

        # --- 2. Standardization Logic (Robust A/B/C Conversion) ---
        for qid in all_qids:
            if qid not in codebook:
                continue
                
            opts = codebook[qid]['options']
            
            # Force all existing keys to strings and handle float-like strings (e.g., "1.0" -> "1")
            def clean_key(k):
                s = str(k).strip()
                if s.endswith('.0'): s = s[:-2]
                return s

            # Create a standardized version of the options dict
            clean_opts = {clean_key(k): v for k, v in opts.items()}
            
            # Sort keys naturally (so '1' comes before '2', '10')
            old_keys_sorted = sorted(clean_opts.keys(), key=lambda x: int(x) if x.isdigit() else x)
            
            # Create mapping: {'1': 'A', '2': 'B'}
            mapping = {old_key: chr(65 + i) for i, old_key in enumerate(old_keys_sorted)}
            
            # Update codebook with new A/B/C keys
            codebook[qid]['options'] = {mapping[k]: clean_opts[k] for k in old_keys_sorted}
            
            # --- Update DataFrames ---
            # 1. Convert DF column to string
            # 2. Strip ".0" from floats converted to strings
            # 3. Map to A/B/C
            for df in [df_survey, df_heldout]:
                if qid in df.columns:
                    df[qid] = df[qid].astype(str).str.replace(r'\.0$', '', regex=True).map(mapping)

   
        # --- 3. Build Y (Ground Truth) and Y_heldout ---
        Y: Dict[str, Dict[str, str]] = {}
        # Use groupby or set_index for faster lookup than filtering df in a loop
        df_survey_indexed = df_survey.set_index("caseid")
        for cid in all_nodes:
            if cid in df_survey_indexed.index:
                row_ser = df_survey_indexed.loc[cid]
                # Handle case where loc might return a DataFrame if IDs aren't unique
                if isinstance(row_ser, pd.DataFrame):
                    row_ser = row_ser.iloc[0]
                
                ans = {qid: row_ser[qid] for qid in Xpool if pd.notna(row_ser[qid])}
                if ans:
                    Y[str(cid)] = ans # Explicitly cast key to str

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

    # @staticmethod
    # def load_dataset(df_survey, df_heldout, codebook, verbose: bool = False):

    #     # Use node universe from neighbors (they define the graph)
    #     Xpool = [c for c in df_survey.columns if c != "caseid"]
    #     df_survey["caseid"] = df_survey["caseid"].astype(str)
    #     df_heldout["caseid"] = df_heldout["caseid"].astype(str)
    #     # assert list(neighbors_info.keys()) == list(df_survey["caseid"])
    #     all_nodes = df_survey["caseid"].tolist()

    #     X_heldout = [c for c in df_heldout.columns if c != "caseid"]
    #     Y_heldout: Dict[NodeId, Dict[QueryId, int]] = {}

    #     # cand = ["A", "B"]
    #     for cid in all_nodes:
    #         row = df_heldout[df_heldout["caseid"] == cid]
    #         row = row.iloc[0]
    #         ans_dict = {}
    #         for qid in X_heldout:
    #             if qid in row:
    #                 ans_dict[qid] = row[qid]
    #         if ans_dict:
    #             Y_heldout[cid] = ans_dict
      
    #     # 2) Question Options (store cardinality, not list of keys)
    #     option_sizes: Dict[QueryId, int] = {}
    #     for qid in Xpool:
    #         option_sizes[qid] = len(codebook[qid]['options'])
    #     for qid in X_heldout:
    #         option_sizes[qid] = len(codebook[qid]['options'])

    #     # 3) Ground-truth answers Y (optional)
    #     Y: Dict[NodeId, Dict[QueryId, int]] = {}
    #     for cid in all_nodes:
    #         row = df_survey[df_survey["caseid"] == cid]
    #         row = row.iloc[0]
    #         ans_dict = {}
    #         for qid in Xpool:
    #             if qid in row:
    #                 ans_dict[qid] = row[qid]
    #         if ans_dict:
    #             Y[cid] = ans_dict

    #     observed_dict = {qid: dict() for qid in Xpool}

    #     # Helper to format the dictionary into "\nA. Value\nB. Value"
    #     def get_option_text(options_dict):
    #         # Sort items by key (A, B, C...) to ensure consistent order
    #         sorted_opts = sorted(options_dict.items())
    #         # Create list ["A. Support", "B. Oppose"]
    #         lines = [f"{k}. {v}" for k, v in sorted_opts]
    #         # Join with newlines and add a leading newline
    #         return "\n" + "\n".join(lines)

    #     # --- Apply to Xpool ---
    #     for qid in Xpool:
    #         opts = codebook[qid]["options"]
    #         codebook[qid]["question"] += get_option_text(opts)

    #     # --- Apply to X_heldout ---
    #     for qid in X_heldout:
    #         opts = codebook[qid]["options"]
    #         codebook[qid]["question"] += get_option_text(opts)

    #     graph = SurveyGraph(nodes=all_nodes, y=Y)
    #     dataset = Dataset(graph=graph, observed_dict=observed_dict, Xpool=Xpool, X_heldout=X_heldout, Y_heldout=Y_heldout, option_sizes=option_sizes, codebook=codebook)

    #     if verbose:
    #         print("=== Dataset Info ===")
    #         print(f"#Nodes: {len(all_nodes)}")
    #         print(f"#Questions (Xpool): {len(Xpool)}")
    #         print(f"#Ground-truth Y entries: {len(Y)}")
    #         print(f"Example Node: {all_nodes[0] if all_nodes else None}")
    #         print(f"Example Question: {Xpool[0] if Xpool else None}")
    #         print(f"Example Question: {dataset.codebook[Xpool[0]]['question']}")
    #         print(f"#Example Y: {Y[all_nodes[0]]}")
    #         if Xpool:
    #             print(f"Options[{Xpool[0]}]: {option_sizes[Xpool[0]]}")
    #     input()

    #     return dataset
    
