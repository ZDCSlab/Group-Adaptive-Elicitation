import json
import pandas as pd
from datasets import load_dataset
from utils import number_to_letter, load_jsonl_as_dict_of_dict

# set the paths
codebook_path = "./twin/codebook.jsonl"
input_path = "./twin/raw_data/Twin-2K-500"
output_path = "./twin/raw_data/twin_responses.csv"

# load the dataset
ds = load_dataset(input_path, 'full_persona')
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

codebook = load_jsonl_as_dict_of_dict(codebook_path)
codebook_ids = set(codebook.keys())
keep_cols = [c for c in df.columns if c in codebook_ids or c == 'caseid']
df = df[keep_cols]


for col in df.columns:
    if col == 'caseid':
        continue
    df[col] = df[col].apply(number_to_letter)

df.to_csv(output_path, index=True)

print("Preprocessing Twin dataset complete.")
print(f"  Dataset shape: {df.shape}")
print(f"  Output saved to: {output_path}")