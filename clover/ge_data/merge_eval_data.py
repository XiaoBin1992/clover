import json

question_file_path = 'data/mt_bench/question_spe.jsonl'
answer_file_path = 'mt_bench/ess-vicuna-7b-fp16-baseline-spe-temperature-0.0.jsonl'

with open(question_file_path, 'r', encoding='utf-8') as q_file:
    questions = [json.loads(line) for line in q_file]

with open(answer_file_path, 'r', encoding='utf-8') as a_file:
    answers = [json.loads(line) for line in a_file]

combined_data_reformatted = []
question_answer_map = {answer['question_id']: answer for answer in answers}

for question in questions:
    question_id = question['question_id']
    entry = {
        "id": f"question_{question_id}",
        "conversations": []
    }
    if question_id in question_answer_map:
        answer_data = question_answer_map[question_id]
        for i in range(len(question["turns"])):
            entry["conversations"].append({
                "from": "human",
                "value": question["turns"][i]
            })
            entry["conversations"].append({
                "from": "gpt",
                "value": answer_data["choices"][0]["turns"][i]
            })
        combined_data_reformatted.append(entry)

final_output_reformatted_path = 'datasets/Spe_eval_merge/Eval_data_split.json'
with open(final_output_reformatted_path, 'w', encoding='utf-8') as final_output_reformatted_file:
    json.dump(combined_data_reformatted, final_output_reformatted_file, ensure_ascii=False, indent=4)

print(final_output_reformatted_path)
