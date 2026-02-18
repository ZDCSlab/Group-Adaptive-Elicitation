import json
import pandas as pd
import re
import numpy as np
import argparse
import yaml
import os
from datasets import load_dataset
from utils import *


def preprocess_ces(cfg):
    codebook_path = cfg["dataset"]["codebook_path"]
    raw_data_path = cfg["dataset"]["raw_data_path"]
    output_path = cfg["dataset"]["all_responses_path"]
    year = cfg["dataset"]["year"]

    def clean_colname(col: str, year: str) -> str:
        col = re.sub(fr'_{year}$', '', col)
        col = re.sub(fr'^(CC|RC){year}_', '', col)
        return col

    def map_birthyear(year: int) -> str:
        if year <= 1945:
            return 1
        elif 1946 <= year <= 1964:
            return 2
        elif 1965 <= year <= 1980:
            return 3
        elif 1981 <= year <= 1996:
            return 4
        elif 1997 <= year <= 2012:
            return 5


    def filtering_by_year(df, id_list, year='', isIdentity=True):
        if isIdentity:
            selected_cols = ["caseid", "birthyr_24", "gender4_24"]
        else:
            selected_cols = ["caseid"]
        for id_val in id_list:
            for pat in [f"CC{year}_{id_val}", f"RC{year}_{id_val}", f"{id_val}_{year}"]:
                if pat in df.columns and id_val not in ["birthyr", "gender4"]:
                    selected_cols.append(pat)
        filtered_df = df[selected_cols]
        filtered_df = filtered_df.rename(columns=lambda x: clean_colname(x, year=year))
        if isIdentity:
            filtered_df = filtered_df.rename(columns={"birthyr_24": "birthyr", "gender4_24": "gender4"})
            filtered_df["birthyr"] = filtered_df["birthyr"].apply(map_birthyear)
        filtered_df.insert(0, "year", year)
        return filtered_df


    codebook = load_jsonl_as_dict(codebook_path)
    id_list = list(codebook.keys())
    df = pd.read_stata(raw_data_path)

    filtered_df = filtering_by_year(df, id_list=id_list, year=year, isIdentity=True)
    codebook_ids = set(codebook.keys())
    keep_cols = [c for c in filtered_df.columns if c in codebook_ids or c == 'caseid']
    filtered_df = filtered_df[keep_cols]

    check_nan_stats(filtered_df)
    # For ces, we filter by valid values instead of filling with majority value
    # fill_nan_with_majority(filtered_df)

    for col in filtered_df.columns:
        if col == 'caseid':
            continue
        filtered_df[col] = filtered_df[col].apply(number_to_letter)

    filtered_df = filter_by_valid_values(filtered_df, list(set(codebook_ids) - set(['region'])), codebook)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    filtered_df.to_csv(output_path, index=False)

    print("Preprocessing CES dataset complete.")
    print(f"  Dataset shape: {filtered_df.shape}")
    print(f"  Output saved to: {output_path}")



def preprocess_opinionqa(cfg):
    codebook_path = cfg["dataset"]["codebook_path"]
    all_question_keys_path = cfg["dataset"]["all_question_keys_path"]
    raw_data_dir = cfg["dataset"]["raw_data_path"]
    output_path = cfg["dataset"]["all_responses_path"]

    codebook = load_jsonl_as_dict(codebook_path)
    codebook_ids = set(codebook.keys())
    
    IDENTITY_COL = cfg["dataset"]["identity_col"]

   

    DEMOGRAPHIC_COLUMNS = list(cfg["demographics"]["columns"].values()) + [cfg["region"]["column"]]
    EXCLUDED_QUESTIONS = ["MOTHER_W50", "FATHER_W50"]
    MISSING_CODES = ["Refused", "DK/REF"]


    def opinionqa_pipeline(
        all_question_keys_path: str,
        data_dir: str = raw_data_dir,
        user_thresh_pct: float = 0.3,
        question_thresh_pct: float = 0.7,
        KEY_COL: str = "QKEY",
    ):
        """Run the full preprocessing pipeline."""
        # 1. Load raw data
        # A. Load the Question List
        print(f"Loading question list from: {all_question_keys_path}")
        with open(all_question_keys_path, 'r') as f:
            q_list = json.load(f)

        df, _, _ = load_merge_fast(data_dir, key_col=KEY_COL)
        if df is None:
            return
        df.replace(MISSING_CODES, np.nan, inplace=True)
        print(f"Original shape: {df.shape}")

        # 2. Filter columns by question list
        df_filtered = filter_columns_by_questions(df, q_list, KEY_COL)
        print(f"Filtered shape: {df_filtered.shape}, kept {df_filtered.shape[1]} columns.")

        # 3. Filter users and questions by completeness
        dense_df = filter_users_then_questions(
            df_filtered,
            user_thresh_pct=user_thresh_pct,
            question_thresh_pct=question_thresh_pct,
        )
        
        print(f"Final density: {dense_df.count().sum() / dense_df.size:.2%}")

        # 4. Demographics: get user list and existing demo columns
        user_lst = dense_df[KEY_COL].values.tolist()
        df_demographics, existing_demo = extract_demographics(df, user_lst, DEMOGRAPHIC_COLUMNS)
        user_lst = df_demographics[KEY_COL].values.tolist()

        # 5. Question list (exclude specific questions)
        question_lst = [c for c in dense_df.columns if c != KEY_COL and c not in EXCLUDED_QUESTIONS]
        print(f"Users: {len(user_lst)}, Questions: {len(question_lst)}, Demographics: {len(existing_demo)}")

        # 6. Build final dataframe (QKEY + questions + demographics)
        df_final = build_final_dataframe(df, user_lst, question_lst, DEMOGRAPHIC_COLUMNS)
        print(f"Original df shape: {df.shape}")

        # 7. Missing data report
        report_missing(df_final)

        # 8. Impute by demographics
        cols_to_fix = [c for c in question_lst if c in df_final.columns]
        grouping_keys = [c for c in DEMOGRAPHIC_COLUMNS if c in df_final.columns]
        print(f"Grouping by {len(grouping_keys)} demographic columns to fill {len(cols_to_fix)} questions...")
        df_imputed = impute_by_demographics(df_final, cols_to_fix, grouping_keys)

        original_na = df_final[cols_to_fix].isnull().sum().sum()
        remaining_na = df_imputed[cols_to_fix].isnull().sum().sum()
        print(f"\nOriginal missing: {original_na}, Filled: {original_na - remaining_na}, Remaining: {remaining_na}")
        if remaining_na > 0:
            print("Remaining NaNs: some demographic groups had no valid answers for some questions.")

        # 9. Save clean features
        df_clean = clean_features(df_imputed, key_col="QKEY")

        return df_clean

  
    # 1. Run the pipeline
    df_origin = opinionqa_pipeline(all_question_keys_path=all_question_keys_path,
        data_dir=raw_data_dir,
        user_thresh_pct=0.3,
        question_thresh_pct=0.7,
        KEY_COL="QKEY")

    print(f"Original df shape: {df_origin.shape}")

    # 2. Load codebook mappings and apply to dataframe
    col_mappings = load_codebook_mappings(codebook_path)
    df_mapped = map_dataframe_to_options(df_origin, col_mappings)
    df_mapped.rename(columns={"QKEY": cfg["dataset"]["identity_col"]}, inplace=True)


    # 6. Subset and save
    final_cols = [IDENTITY_COL] + list(codebook_ids)
    df_out = df_mapped[final_cols]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)

    print("Preprocessing OpinionQA dataset complete.")
    print(f"  Dataset shape: {df_out.shape}")
    print(f"  Output saved to: {output_path}")



def preprocess_twin(cfg):
    codebook_path = cfg["dataset"]["codebook_path"]
    raw_data_path = cfg["dataset"]["raw_data_path"]
    output_path = cfg["dataset"]["all_responses_path"]

    ds = load_dataset(raw_data_path, 'full_persona')
    N_PERSON = len(ds['data'])
    data = [json.loads(ds['data'][i]['persona_json']) for i in range(N_PERSON)]
    N_BLOCKS = len(data[0])

    first_person = data[0]
    cnt_questions = 0
    for idx, block in enumerate(first_person):
        print(f"Block {idx}: {block['BlockName']} | number of questions: {len(block['Questions'])}")
        cnt_questions += len(block['Questions'])
    print(f"Total number of questions: {cnt_questions}")

    question_dict = {}

    for block_id, block in enumerate(first_person):
        questions = block['Questions']
        for question in questions:
                qid = question['QuestionID']
                qtext = question['QuestionText']
                qtype = question['QuestionType']
                qoptions = question['Options'] if 'Options' in question else None
                if qoptions is None:
                    continue
                question_dict[f'BLOCK{block_id}' + qid + "_W1"] = {
                    'text': qtext,
                    'type': qtype,
                    'options': qoptions
                }
    print(f"Total number of questions with options: {len(question_dict)}")

    for question in question_dict.values():
        print("Text: ", question['text'])
        print("Type: ", question['type'])
        print("Options: ", question['options'])
        print("=====")

    qkeys = list(question_dict.keys())
    assert len(qkeys) == len(set(qkeys))

    all_rows = []
    row_ids = []
    multiple_choice_cnt = 0
    qkeys = list(question_dict.keys())

    for i in range(N_PERSON):
        person = json.loads(ds["data"][i]["persona_json"])
        answers_for_person = {}
        for block_id, block in enumerate(person):
            for q in block.get("Questions", []):
                qid = f'BLOCK{block_id}' + q.get("QuestionID") + "_W1"
                if qid in question_dict:
                    options = question_dict[qid]["options"]
                    person_answer_ind = q['Answers'].get('SelectedByPosition')
                    person_answer_text = q['Answers'].get('SelectedText')
                    if person_answer_ind is None or person_answer_text is None:
                        import pdb; pdb.set_trace()
                    if isinstance(person_answer_ind, list):
                        if len(person_answer_ind) == 1:
                            person_answer_ind = person_answer_ind[0]
                            person_answer_text = person_answer_text[0]
                        else:
                            multiple_choice_cnt += 1
                            person_answer_ind = person_answer_ind[0]
                            person_answer_text = person_answer_text[0]
                    if not (
                        isinstance(person_answer_ind, int)
                        and 1 <= person_answer_ind <= len(options)
                        and options[person_answer_ind-1] == person_answer_text
                    ):
                        import pdb; pdb.set_trace()
                    answers_for_person[qid] = person_answer_ind
        answers_for_person['WEIGHT_W1'] = 1.0

        row = {qid: answers_for_person.get(qid, pd.NA) for qid in qkeys}
        row['WEIGHT_W1'] = answers_for_person.get('WEIGHT_W1', pd.NA)
        all_rows.append(row)
        row_ids.append(10000001 + i)


    df = pd.DataFrame(all_rows, index=row_ids)
    df.index.name = "caseid"

    codebook = load_jsonl_as_dict(codebook_path)
    codebook_ids = set(codebook.keys())
    keep_cols = [c for c in df.columns if c in codebook_ids or c == 'caseid']
    df = df[keep_cols]

    check_nan_stats(df)
    # For twin, we fill NaN with majority value because there are only 2 options for each question, and we want to keep all the rows
    fill_nan_with_majority(df)

    for col in df.columns:
        if col == 'caseid':
            continue
        df[col] = df[col].apply(number_to_letter)
    df = filter_by_valid_values(df, keep_cols, codebook)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=True)

    print("Preprocessing Twin dataset complete.")
    print(f"  Dataset shape: {df.shape}")
    print(f"  Output saved to: {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/twin.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"Loading config from: {args.config}")
    print(f"Dataset: {cfg['dataset']['name']}")

    if cfg["dataset"]["name"] == "ces":
        preprocess_ces(cfg)
    elif cfg["dataset"]["name"] == "opinionqa":
        preprocess_opinionqa(cfg)
    elif cfg["dataset"]["name"] == "twin":
        preprocess_twin(cfg)
    else:
        raise ValueError(f"Dataset {cfg['dataset']['name']} not supported")
