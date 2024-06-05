# (nohup) ./run_all.sh <model_name> <cuda_device> <quantization_module> <bit_width> (&)

export CUDA_VISIBLE_DEVICES=$2
export TOKENIZERS_PARALLELISM=false

RESULT_DIR='results/all'
TASKS='kobest_hellaswag,kobest_copa,kobest_boolq,kobest_sentineg,kohatespeech,kohatespeech_apeach,kohatespeech_gen_bias,korunsmile,nsmc,pawsx_ko,klue_nli,klue_sts,klue_ynat'
GPU_NO=0

MODEL=$1
QUANTIZATION_MODULE=$3  # 양자화 모듈 인자 추가
BIT_WIDTH=$4  # 양자화 비트수 인자 추가

MODEL_PATH=$(echo $MODEL | awk -F/ '{print $(NF-1) "/" $NF}')

echo "mkdir -p $RESULT_DIR/$MODEL_PATH/$CURRENT_TRAINED_TOKENS"
mkdir -p $RESULT_DIR/$MODEL_PATH/$CURRENT_TRAINED_TOKENS

# 양자화 모듈과 비트 너비 인자가 설정되지 않았는지 확인
if [ -z "$QUANTIZATION_MODULE" ] || [ -z "$BIT_WIDTH" ]; then
  # 0-shot
  python main.py \
  --model hf-causal-experimental \
  --model_args pretrained=$MODEL,use_accelerate=true,trust_remote_code=true \
  --tasks $TASKS \
  --num_fewshot 0 \
  --no_cache \
  --batch_size 8 \
  --output_path $RESULT_DIR/$MODEL_PATH/0_shot.json

  # 5-shot
  python main.py \
  --model hf-causal-experimental \
  --model_args pretrained=$MODEL,use_accelerate=true,trust_remote_code=true \
  --tasks $TASKS \
  --num_fewshot 5 \
  --no_cache \
  --batch_size 4 \
  --output_path $RESULT_DIR/$MODEL_PATH/5_shot.json

  # 10-shot
  python main.py \
  --model hf-causal-experimental \
  --model_args pretrained=$MODEL,use_accelerate=true,trust_remote_code=true \
  --tasks $TASKS \
  --num_fewshot 10 \
  --no_cache \
  --batch_size 2 \
  --output_path $RESULT_DIR/$MODEL_PATH/10_shot.json

  # 50-shot
  python main.py \
  --model hf-causal-experimental \
  --model_args pretrained=$MODEL,use_accelerate=true,trust_remote_code=true \
  --tasks $TASKS \
  --num_fewshot 50 \
  --no_cache \
  --batch_size 1 \
  --output_path $RESULT_DIR/$MODEL_PATH/50_shot.json
else
# 양자화 모듈과 비트 너비 인자가 설정된 경우
  # 0-shot
  python main.py \
  --model hf-causal-experimental \
  --model_args pretrained=$MODEL,use_accelerate=true,trust_remote_code=true \
  --tasks $TASKS \
  --num_fewshot 0 \
  --no_cache \
  --batch_size 8 \
  --output_path $RESULT_DIR/$MODEL_PATH/0_shot.json \
  --quantization_module $QUANTIZATION_MODULE \
  --bit_width $BIT_WIDTH

  # 5-shot
  python main.py \
  --model hf-causal-experimental \
  --model_args pretrained=$MODEL,use_accelerate=true,trust_remote_code=true \
  --tasks $TASKS \
  --num_fewshot 5 \
  --no_cache \
  --batch_size 4 \
  --output_path $RESULT_DIR/$MODEL_PATH/5_shot.json \
  --quantization_module $QUANTIZATION_MODULE \
  --bit_width $BIT_WIDTH

  # 10-shot
  python main.py \
  --model hf-causal-experimental \
  --model_args pretrained=$MODEL,use_accelerate=true,trust_remote_code=true \
  --tasks $TASKS \
  --num_fewshot 10 \
  --no_cache \
  --batch_size 2 \
  --output_path $RESULT_DIR/$MODEL_PATH/10_shot.json \
  --quantization_module $QUANTIZATION_MODULE \
  --bit_width $BIT_WIDTH

  # 50-shot
  python main.py \
  --model hf-causal-experimental \
  --model_args pretrained=$MODEL,use_accelerate=true,trust_remote_code=true \
  --tasks $TASKS \
  --num_fewshot 50 \
  --no_cache \
  --batch_size 1 \
  --output_path $RESULT_DIR/$MODEL_PATH/50_shot.json \
  --quantization_module $QUANTIZATION_MODULE \
  --bit_width $BIT_WIDTH
fi
