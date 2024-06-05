import json

# file_path = 'Input JSON file path'
# file_path = '/home/ca2023/ys/ko-lm-evaluation-harness/results/all/beomi/gemma-mling-7b/5_shot.json'
file_path = '/home/ca2023/ys/ko-lm-evaluation-harness/results/all/x2bee/POLAR-14B-v0.2/5_shot.json'
# file_path = '/home/ca2023/ys/ko-lm-evaluation-harness/lm_eval/eval_result/BitAndBytes/8bit/5_shot.json'
# file_path = '/home/ca2023/ys/ko-lm-evaluation-harness/lm_eval/eval_result/BitAndBytes/4bit/5_shot.json'
# file_path = '/home/ca2023/ys/ko-lm-evaluation-harness/lm_eval/eval_result/base/beomi/gemma-mling-7b/5_shot.json'

# JSON 데이터 파일에서 불러오기
with open(file_path, 'r') as json_file:
    data = json.load(json_file)

# 필요한 정보 추출
results = data['results']
output = []

for benchmark, metrics in results.items():
    if 'f1' in metrics:
        score = round(metrics['f1'], 4)
        output.append(f"{benchmark}(f1): {score}")
    elif 'macro_f1' in metrics:
        score = round(metrics['macro_f1'], 4)
        output.append(f"{benchmark}(macro_f1): {score}")
    elif 'acc' in metrics:
        score = round(metrics['acc'], 4)
        output.append(f"{benchmark}(acc): {score}")

# 출력
output_text = "\n".join(output)
print(output_text)