import math
import torch
import torch.nn.functional as F
import transformers
import peft
from typing import List, Mapping, NewType, Optional, Tuple, Union
from tqdm import tqdm

from transformers import BatchEncoding, BitsAndBytesConfig, QuantoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import find_executable_batch_size

from lm_eval import utils
from lm_eval.base import BaseLM

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]

_DeviceMapping = NewType("DeviceMapping", Mapping[str, Union[int, str, torch.device]])


def _get_accelerate_args(
    device_map_option: Optional[str] = "auto",
    max_memory_per_gpu: Optional[Union[int, str]] = None,
    max_cpu_memory: Optional[Union[int, str]] = None,
    offload_folder: Optional[str] = "./offload",
) -> dict:
    """`AutoModel.from_pretrained`에서 `accelerate`를 적용하는 데 필요한 kwargs를 반환합니다."""
    max_memory = {}
    if max_memory_per_gpu is not None:
        max_memory_per_gpu_map = {
            device_idx: max_memory_per_gpu
            for device_idx in range(torch.cuda.device_count())
        }
        max_memory.update(max_memory_per_gpu_map)
    if max_cpu_memory is not None:
        max_memory["cpu"] = max_cpu_memory

    args = {}
    if max_memory:
        args["max_memory"] = max_memory
    args["device_map"] = device_map_option
    args["offload_folder"] = offload_folder
    return args


def _get_dtype(
    dtype: Union[str, torch.dtype], config: Optional[transformers.AutoConfig] = None
) -> torch.dtype:
    """가능한 경우 `dtype`를 `str`에서 torch.dtype로 변환합니다."""
    if dtype is None and config is not None:
        _torch_dtype = config.torch_dtype
    elif isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


class HuggingFaceAutoLM(BaseLM):
    AUTO_CONFIG_CLASS: transformers.AutoConfig = transformers.AutoConfig
    AUTO_TOKENIZER_CLASS: transformers.AutoTokenizer = transformers.AutoTokenizer
    AUTO_MODEL_CLASS: transformers.AutoModel = None
    AUTO_PEFT_CLASS: peft.PeftModel = None

    # `max_length`가 제공되지 않거나 
    # 모델 또는 토크나이저에서 최대 길이 설정을 찾을 수 없는 경우의 기본 최대 시퀀스 길이 설정.
    _DEFAULT_MAX_LENGTH: int = 2048

    def __init__(
        self,
        pretrained: str,
        tokenizer: Optional[str] = None,
        subfolder: Optional[str] = None,
        revision: Optional[str] = "main",
        batch_size: Optional[Union[int, str]] = 1,
        max_gen_toks: Optional[int] = 256,
        max_length: Optional[int] = None,
        add_special_tokens: Optional[bool] = None,
        use_accelerate: Optional[bool] = False,
        device_map_option: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[int, str]] = "cuda",
        peft: str = None,
        load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        quantization_module: Optional[str] = None,  # 양자화 모듈 추가
        bit_width: Optional[str] = None,  # 양자화 비트 추가
    ):
        """평가를 위해 HuggingFace `AutoModel`과 `AutoTokenizer`를 초기화합니다.
        Args:
            pretrained (str):
                로드할 사전 학습된 모델의 HuggingFace Hub 모델 ID 이름 또는 경로.
                이는 HuggingFace `transformers` API의 `from_pretrained`의 
                `pretrained_model_name_or_path` 인수에 해당합니다.
            add_special_tokens (bool, optional, defaults to True):
                입력 시퀀스에 특수 토큰을 추가할지 여부를 결정합니다. `None`인 경우,
                기본값은 seq2seq 모델(T5 등)에 대해 `True`로 설정되고,
                인과 모델에 대해서는 `False`로 설정됩니다.
                경고: 현재 `add_special_tokens=True` 옵션을 사용하여 인과 모델을
                평가하는 것은 지원되지 않습니다.
            > 대형 모델 로드를 위한 `accelerate` 인수
            use_accelerate (bool, optional, defaults to False):
                `accelerate` 라이브러리를 사용하여 대형 모델을 여러 장치에 걸쳐 로드할지 여부를 
                결정합니다.
            device_map_option (str, optional, defaults to "auto"):
                `accelerate`로 모델을 로드할 때 사용할 디바이스 맵 옵션.
                옵션:
                    "auto", "balanced", "balanced_low_0", "sequential"
                이러한 옵션에 대한 자세한 내용은 `accelerate` 문서를 참조하십시오:
                https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.device_map
            max_memory_per_gpu (Union[int, str], optional, defaults to None):
                각 GPU에 사용할 수 있는 최대 메모리를 바이트 단위의 `int`로 지정하거나
                f"{significand}{unit_symbol}" 형식으로 지정합니다. 여기서 {unit_symbol}은 
                ["GB", "MB", "GIB", "MIB"] 중 하나입니다. 다음 문서의 "Parameters for big model inference" 
                섹션에서 `max_memory` 인수를 참조하십시오:
                https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.max_memory
            max_cpu_memory (Union[int, str], optional, defaults to None):
                바이트 단위의 `int`로 지정된 최대 사용 가능 CPU RAM 또는 f"{significand}{unit_symbol}" 
                형식으로 지정합니다. 여기서 {unit_symbol}은 ["GB", "MB", "GIB", "MIB"] 중 하나입니다. 
                다음 문서의 "Parameters for big model inference" 섹션에서 `max_memory` 인수를 
                참조하십시오:
                https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.max_memory
            offload_folder (str, optional, defaults to "./offload"):
                `device_map`에 "disk" 값이 포함된 경우 가중치를 오프로드할 폴더입니다.
            dtype (Union[str, torch.dtype], optional, defaults to None):
                지정된 경우 모델 가중치를 `dtype`으로 변환합니다. 문자열은 `torch.dtype` 객체로 
                변환됩니다 (예: `float16` -> `torch.float16`). `dtype="auto"`를 사용하여 모델의 
                가중치에서 타입을 유도할 수 있습니다.
            peft (str, optional, defaults to None):
                Huggingface에서 로드할 어댑터 가중치의 경로. 일반적으로 
                `adapter_config.json` 및 `adapter_model.bin` 파일을 포함하는 디렉터리가 
                여기에 포함됩니다. [PEFT](https://github.com/huggingface/peft)와 호환됩니다.
            load_in_8bit (bool, optional, defaults to False):
                True로 설정하면 로드된 모델을 혼합 8비트 양자화된 모델로 변환합니다. 다음을 참조하십시오:
                https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.load_in_8bit
            trust_remote_code (bool, optional, defaults to False):
                True로 설정하면 모델을 로드할 때 원격 코드를 신뢰합니다.
        """
        super().__init__()

        assert isinstance(pretrained, str)
        assert isinstance(device, str)
        assert isinstance(batch_size, (int, str))
        if (
            add_special_tokens is not None
            and self.AUTO_MODEL_CLASS is transformers.AutoModelForCausalLM
        ):
            # TODO: 특수 토큰을 사용하여 인과 모델을 평가하는 것을 지원합니다. 현재,
            # 이 작업은 인과 언어 모델(causal LMs)을 위한 `_loglikelihood_tokens()` 메서드가
            # 문맥과 레이블/연속성을 별도로 토큰화하고 특수 토큰 없이
            # 연결한 후 입력으로 처리하는 가정으로 인해 불가능합니다.
            assert (
                not add_special_tokens
            ), "현재 `add_special_tokens=True` 옵션을 사용하여 causal models을 평가하는 것은 지원되지 않습니다."

        # 자동 배치 크기 감지를 설정합니다
        if batch_size == "auto":
            self._batch_size = batch_size
        else:
            self._batch_size = int(batch_size)

        self._max_gen_toks = max_gen_toks
        self._max_length = max_length
        self._config = self.AUTO_CONFIG_CLASS.from_pretrained(
            pretrained,
            trust_remote_code=trust_remote_code,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
        )

        self._add_special_tokens = add_special_tokens
        self.tokenizer = self._create_auto_tokenizer(
            pretrained=pretrained,
            revision=revision,
            subfolder=subfolder,
            tokenizer=tokenizer,
        )
        self.tokenizer.model_max_length = self.max_length

        if quantization_module and bit_width:
            self.quantization_config = self._get_quantization_config(quantization_module, bit_width)  # 양자화 설정 생성
        else:
            self.quantization_config = None

        model_kwargs = {}
        if use_accelerate:
            model_kwargs = _get_accelerate_args(
                device_map_option,
                max_memory_per_gpu,
                max_cpu_memory,
                offload_folder,
            )

        model_kwargs["load_in_8bit"] = load_in_8bit

        # 모델 로드
        self.model = self._create_auto_model(
            pretrained=pretrained,
            trust_remote_code=trust_remote_code,
            revision=revision,
            subfolder=subfolder,
            torch_dtype=_get_dtype(dtype, self._config),
            quantization_config=self.quantization_config,  # 양자화 설정 전달
            **model_kwargs,
        )

        # 참고: peft_path는 사전 훈련된 모델 경로와 다를 수 있습니다
        if peft is not None:
            self.model = self._create_auto_model_peft(
                model=self.model,
                peft=peft,
                revision=revision,
                subfolder=subfolder,
                torch_dtype=_get_dtype(dtype, self._config),
                **model_kwargs,
            )
        self.model.eval()
        torch.set_grad_enabled(False)

        self._device = device
        if use_accelerate and "lm_head" in self.model.hf_device_map:
            # `accelerate`는 `lm_head` 가중치를 사용자가 지정한 장치와 다른 장치에 배치할 수 있으므로 `lm_head`의 장치와 동일하게 `self._device`를 강제 설정
            self._device = self.model.hf_device_map["lm_head"]
        if not use_accelerate:
            self.model.to(self._device)

    # 양자화 설정 생성
    def _get_quantization_config(self, quantization_module, bit_width):
        if quantization_module == "bnb" and bit_width: # BitAndBytes
            from transformers import BitsAndBytesConfig
            if bit_width == "int8":
                return BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    llm_int8_enable_fp32_cpu_offload=False
                )
            elif bit_width == "int4":
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    # bnb_4bit_quant_type="fp4",
                    bnb_4bit_compute_dtype=torch.float16
                )
            else:
                raise ValueError(f"Invalid bit width '{bit_width}' for BitAndBytes quantization module.")
        else:
            raise ValueError(f"Invalid quantization module '{quantization_module}'.")

    def _create_auto_model(
        self,
        *,
        pretrained: str,
        revision: str,
        subfolder: str,
        device_map: Optional[Union[str, _DeviceMapping]] = None,
        max_memory: Optional[dict] = None,
        offload_folder: Optional[str] = None,
        load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        quantization_config: Optional[dict] = None,  # 양자화 설정 추가
    ) -> transformers.AutoModel:
        """사전 학습된 모델 구성에서 사전 학습된 pytorch 모델을 반환합니다."""
        model = self.AUTO_MODEL_CLASS.from_pretrained(
            pretrained,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            load_in_8bit=load_in_8bit,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,  # 양자화 설정 적용
            # force_download=force_download,  # 강제 다운로드 옵션
            # cache_dir=cache_dir  # 캐시 비활성화
        )
        return model

    def _create_auto_model_peft(
        self,
        *,
        model: transformers.PreTrainedModel,
        peft: str,
        revision: str,
        subfolder: str,
        device_map: Optional[Union[str, _DeviceMapping]] = None,
        max_memory: Optional[dict] = None,
        offload_folder: Optional[str] = None,
        load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
    ):
        model = self.AUTO_PEFT_CLASS.from_pretrained(
            model,
            peft,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            load_in_8bit=load_in_8bit,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        return model

    def _create_auto_tokenizer(
        self,
        *,
        pretrained: str,
        revision: str,
        subfolder: str,
        tokenizer: Optional[str] = None,
    ) -> transformers.PreTrainedTokenizer:
        """사전 학습된 토크나이저 구성을 기반으로 사전 학습된 토크나이저를 반환합니다."""
        tokenizer = self.AUTO_TOKENIZER_CLASS.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @property
    def add_special_tokens(self) -> bool:
        """인코딩된 텍스트에 특수 토큰을 포함할지 여부를 결정합니다. 
        이는 모델이 특수 토큰으로 학습되었는지 여부에 따라 결정되어야 합니다.
        TODO: HuggingFace가 임의의 모델이 특수 토큰으로 학습되었는지 확인하는 방법을 지원하면 
        이 조건문을 제거하십시오.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        elif self.AUTO_MODEL_CLASS is transformers.AutoModelForCausalLM:
            return False
        elif self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM:
            return True
        else:
            raise ValueError(
                "Could not determine `add_special_tokens` value from the model "
                "class. Set to `True` or `False` depending on whether the model "
                "was pre-trained with special tokens."
            )

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def max_length(self) -> int:
        """모델의 최대 시퀀스 길이를 반환합니다.
        참고: 다양한 모델 구성은 서로 다른 최대 시퀀스 길이 속성 이름을 가집니다.
            - n_positions: (CTRLConfig)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        참고: 상대 위치 인코딩된 모델의 경우 생성자에서 `max_length`를 통해 
        모델의 최대 시퀀스 길이를 지정해야 합니다.
        """
        if self._max_length is not None:
            return self._max_length
        # Try to get the sequence length from the model config.
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self._config, attr):
                return getattr(self._config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def batch_size(self) -> int:
        # TODO: Add adaptive batch size.
        return self._batch_size  # * gpus

    @property
    def device(self) -> Union[int, str, torch.device]:
        return self._device

    def tok_encode(self, string: str) -> TokenSequence:
        # TODO: Merge `tok_encode_batch` here.
        return self.tokenizer.encode(string, add_special_tokens=self.add_special_tokens)

    def tok_encode_batch(self, strings: List[str]) -> TokenSequence:
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

    def tok_decode(self, tokens: torch.LongTensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def greedy_until(
        self, requests: List[Tuple[str, Union[List[str], str]]]
    ) -> List[str]:
        def _collate(x):
            tokens = self.tok_encode(x[0])
            return len(tokens), x[0]

        results = []
        reorder = utils.Reorderer(requests, _collate)

        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")

            @find_executable_batch_size(
                starting_batch_size=512
            )  # if OOM, then halves batch_size and tries again
            def forward_batch(batch_size):
                test_batch = torch.ones(
                    (batch_size, self.max_length), device=self.device
                ).long()
                for _ in range(5):
                    _ = F.log_softmax(self._model_call(test_batch), dim=-1).cpu()
                return batch_size

            batch_size = forward_batch()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        for chunk in utils.chunks(
            tqdm(reorder.get_reordered(), disable=False),
            self.batch_size if self.batch_size != "auto" else adaptive_batch_size,
        ):
            context = [c[0] for c in chunk]
            request_args = chunk[0][1]
            stop = request_args.get("until", None)
            stop_sequences = stop if isinstance(stop, list) else [stop]
            max_generation_length = request_args.get("max_length", None)

            assert (
                isinstance(max_generation_length, int) or max_generation_length is None
            )
            assert isinstance(stop_sequences, list) or stop_sequences is None

            # TODO: Find a better way to handle stop sequences for 0-shot.
            if stop_sequences is None:
                until = [self.eot_token]
            else:
                until = stop_sequences + [self.eot_token]

            if max_generation_length is None:
                max_tokens = self.max_gen_toks
            else:
                max_tokens = max_generation_length

            token_context = self.tok_encode_batch(context)

            responses = self._model_generate(
                inputs=token_context,
                max_tokens=max_tokens,
                stop=until,
            )
            responses = self.tok_decode(responses.tolist())

            for response in responses:
                # Ensure the generated responses do not contain the stop sequences.
                for term in until:
                    response = response.split(term)[0]
                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until), response)
                results.append(response)
        return reorder.get_original(results)


class AutoCausalLM(HuggingFaceAutoLM):
    """Causal language modeling.
    You can find a set of supported models in the HF documentation:
    https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForCausalLM
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
    AUTO_PEFT_CLASS = peft.PeftModel

    def _create_auto_tokenizer(
        self,
        *,
        pretrained: str,
        revision: str,
        subfolder: str,
        tokenizer: Optional[str] = None,
    ) -> transformers.PreTrainedTokenizer:
        tokenizer = super()._create_auto_tokenizer(
            pretrained=pretrained,
            revision=revision,
            subfolder=subfolder,
            tokenizer=tokenizer,
        )
        tokenizer.padding_side = "left"
        return tokenizer

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        return self.model(inputs)["logits"]

    def _model_generate(
        self,
        inputs: transformers.BatchEncoding,
        max_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> TokenSequence:
        # Ensure that the context does not encroach into the `space`
        # for the generation.
        input_ids = inputs["input_ids"][:, self.max_gen_toks - self.max_length :]
        attention_mask = inputs["attention_mask"][
            :, self.max_gen_toks - self.max_length :
        ]
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, input_ids.shape[1], input_ids.shape[0]
        )

        generations = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # GPT style models require the `generate` `max_length` arg to include the
            # context length, so we instead set `max_new_tokens` which is the number
            # of new tokens to generate, excluding the current number of tokens.
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=False,
        )
        return utils.select_continuation_from_batch_left_padding(
            generations, max_context_size=inputs["input_ids"].size(1)
        )


class AutoSeq2SeqLM(HuggingFaceAutoLM):
    """Seq2Seq language modeling.
    You can find a set of supported models in the following documentation:
    https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForSeq2SeqLM
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM
    AUTO_PEFT_CLASS = peft.PeftModel

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        TODO: Currently only works for relative position encoded Seq2Seq models.
        """
        if self._max_length is not None:
            return self._max_length
        return self._DEFAULT_MAX_LENGTH

    def loglikelihood(
        self, requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        new_requests = []
        for chunk in utils.chunks(requests, self.batch_size):
            context, continuation = zip(*chunk)

            # Fill empty contexts with the EOT token.
            context = [
                f"{self.eot_token}" if len(text) == 0 else text for text in context
            ]
            context_enc = self.tok_encode_batch(context)
            for key in context_enc:
                context_enc[key] = context_enc[key][:, -self.max_length :]

            # Remove leading whitespace introduced by the default
            # `text_target_separator` since the context and continuation
            # will not be concatenated as a single (decoder) input.
            continuation = [text.lstrip() for text in continuation]
            continuation_enc = self.tok_encode_batch(list(continuation))
            for key in continuation_enc:
                continuation_enc[key] = continuation_enc[key][:, -self.max_length :]

            new_requests.append(
                ((context, continuation), context_enc, continuation_enc)
            )
        return self._loglikelihood_tokens(new_requests)

    def loglikelihood_rolling(self, requests: List[Tuple[str, str]]) -> List[float]:
        loglikelihoods = []
        for (string,) in tqdm(requests):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            contexts, conts = utils.split_and_pad_windows(
                rolling_token_windows,
                pad_token_id=self.eot_token_id,
                max_seq_len=self.max_length,
            )
            # Manually create BatchEncoding tensors with attention masks as
            # expected by `self._model_call` in `self._loglikelihood_tokens`.
            contexts_enc = torch.Tensor(contexts).long()
            contexts_enc = transformers.tokenization_utils_base.BatchEncoding(
                {
                    "input_ids": contexts_enc,
                    "attention_mask": (contexts_enc != self.eot_token_id).long(),
                }
            )
            conts_enc = torch.Tensor(conts).long()
            conts_enc = transformers.tokenization_utils_base.BatchEncoding(
                {
                    "input_ids": conts_enc,
                    "attention_mask": (conts_enc != self.eot_token_id).long(),
                }
            )
            # TODO: Extract out this call so it only gets called once and also
            # somehow figure out partial caching for.
            rolling_token_windows_request = [
                ((contexts, conts), contexts_enc, conts_enc)
            ]
            string_nll = self._loglikelihood_tokens(
                rolling_token_windows_request, disable_tqdm=True
            )
            string_nll = [x[0] for x in string_nll]  # discard is_greedy
            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
        return loglikelihoods

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], TokenSequence, TokenSequence]],
        disable_tqdm: Optional[bool] = False,
    ) -> List[Tuple[float, bool]]:
        results = []
        for chunk in tqdm(
            requests, total=math.ceil(len(requests)), disable=disable_tqdm
        ):
            cache_keys, inputs_tokens, targets_tokens = chunk
            inputs_tokens = inputs_tokens.to(self.device)
            targets_tokens = targets_tokens.to(self.device)
            outputs = self._model_call(inputs=inputs_tokens, labels=targets_tokens)
            log_softmaxes = F.log_softmax(outputs.logits, dim=-1)

            output_iterator = zip(
                zip(cache_keys[0], cache_keys[1]),
                log_softmaxes,
                targets_tokens["input_ids"],
                targets_tokens["attention_mask"],
            )
            for cache_key, log_softmax, target_tokens, target_mask in output_iterator:
                length = target_mask.sum()
                log_softmax = log_softmax[:length]
                target_tokens = target_tokens[:length]
                greedy_tokens = log_softmax.argmax(dim=-1)
                max_equal = (greedy_tokens == target_tokens).all()
                target_logits = torch.gather(
                    log_softmax, 1, target_tokens.unsqueeze(-1)
                ).squeeze(-1)
                answer = (float(target_logits.sum()), bool(max_equal))
                results.append(answer)
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
        return results

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        return self.model(**inputs, labels=labels["input_ids"])

    def _model_generate(
        self,
        inputs: transformers.BatchEncoding,
        max_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> TokenSequence:
        input_ids = inputs["input_ids"][:, -self.max_length :].to(self.device)
        attention_mask = inputs["attention_mask"][:, -self.max_length :].to(self.device)

        # Generate one token to calculate the number of start tokens prepended to decoder_input_ids
        # (leaving this here in case the below assumption is violated in the future)
        # one_tok_gen = self.model.generate(
        #    input_ids=torch.zeros((1, 1), dtype=torch.int),
        #    min_length=2,
        #    max_new_tokens=1,
        # ).squeeze()
        # initial_decoder_input_length = len(one_tok_gen) - 1

        # Assume that there will always only be one token in the decoder inputs, assumption holds for existing HF models
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, 1, input_ids.shape[0]
        )

        generations = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=False,
        )
        return generations


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ):
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :][
            :, -self.sequence_id_len :
        ]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )
