import json
import argparse

def process_jsonl_segment(file_segment):
    total_cc_num = 0
    total_ct_num = 0
    
    for line in file_segment:
        try:
            record = json.loads(line)
            total_cc_num += sum(record['choices'][0]['new_tokens'])
            total_ct_num += sum(record['choices'][0]['idxs'])
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line: {line} Error: {str(e)}")
    
    if total_ct_num == 0:
        return None  # avoid division by zero
    
    return total_cc_num, total_ct_num


def process_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_segments = [[], [], [], [], [], []]
        
        for i, line in enumerate(file):
            if i < 80:
                file_segments[0].append(line)
            elif i < 160:
                file_segments[1].append(line)
            elif i < 240:
                file_segments[2].append(line)
            elif i < 320:
                file_segments[3].append(line)
            elif i < 400:
                file_segments[4].append(line)
            elif i < 480:
                file_segments[5].append(line)
            else:
                break  # 超出所需的行数
        
        cc_num_segment_1, ct_num_segment_1 = process_jsonl_segment(file_segments[0])
        cc_num_segment_2, ct_num_segment_2 = process_jsonl_segment(file_segments[1])
        cc_num_segment_3, ct_num_segment_3 = process_jsonl_segment(file_segments[2])
        cc_num_segment_4, ct_num_segment_4 = process_jsonl_segment(file_segments[3])
        cc_num_segment_5, ct_num_segment_5 = process_jsonl_segment(file_segments[4])
        cc_num_segment_6, ct_num_segment_6 = process_jsonl_segment(file_segments[5])
    
    return cc_num_segment_1 / ct_num_segment_1, cc_num_segment_1, ct_num_segment_1, \
            cc_num_segment_2 / ct_num_segment_2, cc_num_segment_2, ct_num_segment_2, \
            cc_num_segment_3 / ct_num_segment_3, cc_num_segment_3, ct_num_segment_3, \
            cc_num_segment_4 / ct_num_segment_4, cc_num_segment_4, ct_num_segment_4, \
            cc_num_segment_5 / ct_num_segment_5, cc_num_segment_5, ct_num_segment_5, \
            cc_num_segment_6 / ct_num_segment_6, cc_num_segment_6, ct_num_segment_6
            


def main():
    parser = argparse.ArgumentParser(description="Process a .jsonl file to find the ratio of total cc_num to total ct_num.")
    parser.add_argument('file_path', type=str, help='The path to the .jsonl file')

    args = parser.parse_args()
    file_path = args.file_path

    result1, cc1, ct1, result2, cc2, ct2, result3, cc3, ct3, result4, cc4, ct4, result5, cc5, ct5, result6, cc6, ct6, = process_jsonl(file_path)
    
    print('The result of multi-turns: ', result1)
    print('cc: ', cc1)
    print('ct: ', ct1)
    
    print('The result of translation: ', result2)
    print('cc: ', cc2)
    print('ct: ', ct2)
    
    print('The result of summarization: ', result3)
    print('cc: ', cc3)
    print('ct: ', ct3)
    
    print('The result of qa: ', result4)
    print('cc: ', cc4)
    print('ct: ', ct4)
    
    print('The result of math_reasoning: ', result5)
    print('cc: ', cc5)
    print('ct: ', ct5)
    
    print('The result of rag: ', result6)
    print('cc: ', cc6)
    print('ct: ', ct6)
    

if __name__ == '__main__':
    main()
