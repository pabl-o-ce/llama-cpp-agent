from typing import List, Dict, Union
from dataclasses import dataclass
from copy import deepcopy

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2StreamingGenerator

from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings, LlmStructuredOutputType
from llama_cpp_agent.providers.provider_base import LlmProvider, LlmProviderId, LlmSamplingSettings

import traceback
from exllamav2.generator.filters import ExLlamaV2Filter, ExLlamaV2PrefixFilter
from lmformatenforcer import JsonSchemaParser, RegexParser
from lmformatenforcer.integrations.exllamav2 import (
    ExLlamaV2TokenEnforcerFilter,
    build_token_enforcer_tokenizer_data,
)
from loguru import logger
from typing import List
from functools import lru_cache


class OutlinesTokenizerWrapper:
    """Wrapper for Outlines tokenizer"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        id_to_piece = self.tokenizer.get_id_to_piece_list()
        self.vocabulary = {piece: idx for idx, piece in enumerate(id_to_piece)}
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = id_to_piece[self.tokenizer.eos_token_id]
        self.special_tokens = list(self.tokenizer.extended_id_to_piece.keys())

    def convert_token_to_string(self, token):
        return token

    def decode(self, tokens):
        s = ""
        id_to_piece = self.tokenizer.get_id_to_piece_list()
        for t in tokens:
            s += id_to_piece[t]
        return s


class ExLlamaV2EbnfFilter(ExLlamaV2Filter):
    """Filter class for context-free grammar via outlines"""

    def __init__(self, model, tokenizer, grammar):
        from outlines.fsm.fsm import CFGFSM

        super().__init__(model, tokenizer)

        self.wrapped_tokenizer = OutlinesTokenizerWrapper(tokenizer)
        self.fsm = CFGFSM(grammar, self.wrapped_tokenizer)
        self.state = self.fsm.first_state

    def begin(self, prefix_str=""):
        self.state = self.fsm.first_state

    def feed(self, token):
        self.state = self.fsm.next_state(self.state, token.item())

    def next(self):
        return self.fsm.allowed_token_ids(self.state), set()


@lru_cache(10)
def _get_lmfe_tokenizer_data(tokenizer: ExLlamaV2Tokenizer):
    return build_token_enforcer_tokenizer_data(tokenizer)


def clear_grammar_func_cache():
    """Flush tokenizer_data cache to avoid holding references to
    tokenizers after unloading a model"""

    _get_lmfe_tokenizer_data.cache_clear()


class ExLlamaV2Grammar:
    """ExLlamaV2 class for various grammar filters/parsers."""

    filters: List[ExLlamaV2Filter]

    def __init__(self):
        self.filters = []

    def add_json_schema_filter(
        self,
        json_schema: dict,
        model: ExLlamaV2,
        tokenizer: ExLlamaV2Tokenizer,
    ):
        """Adds an ExllamaV2 filter based on a JSON schema."""

        # Create the parser
        try:
            schema_parser = JsonSchemaParser(json_schema)
        except Exception:
            traceback.print_exc()
            logger.error(
                "Skipping because the JSON schema couldn't be parsed. "
                "Please read the above error for more information."
            )

            return

        # Allow JSON objects or JSON arrays at the top level
        json_prefixes = ["[", "{"]

        lmfilter = ExLlamaV2TokenEnforcerFilter(
            schema_parser, _get_lmfe_tokenizer_data(tokenizer)
        )
        prefix_filter = ExLlamaV2PrefixFilter(model, tokenizer, json_prefixes)

        # Append the filters
        self.filters.extend([lmfilter, prefix_filter])

    def add_regex_filter(
        self,
        pattern: str,
        tokenizer: ExLlamaV2Tokenizer,
    ):
        """Adds an ExllamaV2 filter based on regular expressions."""

        # Create the parser
        try:
            pattern_parser = RegexParser(pattern)
        except Exception:
            traceback.print_exc()
            logger.error(
                "Skipping because the regex pattern couldn't be parsed. "
                "Please read the above error for more information."
            )

            return

        lmfilter = ExLlamaV2TokenEnforcerFilter(
            pattern_parser, _get_lmfe_tokenizer_data(tokenizer)
        )

        # Append the filters
        self.filters.append(lmfilter)

    def add_ebnf_filter(
        self,
        ebnf_string: str,
        model: ExLlamaV2,
        tokenizer: ExLlamaV2Tokenizer,
    ):
        """
        Add an EBNF grammar filter.
        Possibly replace outlines with an in-house solution in the future.
        """

        try:
            ebnf_filter = ExLlamaV2EbnfFilter(model, tokenizer, ebnf_string)
        except ImportError:
            logger.error(
                "Skipping EBNF parsing because Outlines is not installed.\n"
                "Please run the following command in your environment "
                "to install extra packages:\n"
                "pip install -U .[extras]"
            )

            return

        self.filters.append(ebnf_filter)

@dataclass
class ExLlamaV2SamplingSettings(LlmSamplingSettings):
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    max_tokens: int = -1
    stream: bool = False
    additional_stop_sequences: List[str] = None
    repetition_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    def get_provider_identifier(self) -> LlmProviderId:
        return LlmProviderId.exllama_v2

    def get_additional_stop_sequences(self) -> List[str]:
        if self.additional_stop_sequences is None:
            self.additional_stop_sequences = []
        return self.additional_stop_sequences

    def add_additional_stop_sequences(self, sequences: List[str]):
        if self.additional_stop_sequences is None:
            self.additional_stop_sequences = []
        self.additional_stop_sequences.extend(sequences)

    def is_streaming(self):
        return self.stream

    @staticmethod
    def load_from_dict(settings: dict) -> "ExLlamaV2SamplingSettings":
        return ExLlamaV2SamplingSettings(**settings)

    def as_dict(self) -> dict:
        return self.__dict__

class ExLlamaV2Provider(LlmProvider):
    def __init__(self, model_path: str, max_seq_len: int = 32768):
        config = ExLlamaV2Config(model_path)
        config.arch_compat_overrides()
        self.model = ExLlamaV2(config)
        self.cache = ExLlamaV2Cache(self.model, max_seq_len=max_seq_len, lazy=True)
        self.model.load_autosplit(self.cache)
        self.tokenizer = ExLlamaV2Tokenizer(config)
        self.generator = ExLlamaV2StreamingGenerator(self.model, self.cache, self.tokenizer)
        self.grammar_cache = {}

    def is_using_json_schema_constraints(self):
        return True

    def get_provider_identifier(self) -> LlmProviderId:
        return LlmProviderId.exllama_v2

    def get_provider_default_settings(self) -> ExLlamaV2SamplingSettings:
        return ExLlamaV2SamplingSettings()

    def create_completion(
        self,
        prompt: str,
        structured_output_settings: LlmStructuredOutputSettings,
        settings: ExLlamaV2SamplingSettings,
        bos_token: str,
    ):
        grammar = None
        if structured_output_settings.output_type != LlmStructuredOutputType.no_structured_output:
            grammar = self._get_or_create_grammar(structured_output_settings)

        settings_dict = deepcopy(settings.as_dict())
        stop_sequences = settings_dict.pop("additional_stop_sequences", [])
        
        self.generator.set_stop_conditions(stop_sequences)
        output = self.generator.generate_simple(
            prompt,
            max_new_tokens=settings_dict["max_tokens"],
            temperature=settings_dict["temperature"],
            top_k=settings_dict["top_k"],
            top_p=settings_dict["top_p"],
            min_p=settings_dict["min_p"],
            token_repetition_penalty=settings_dict["repetition_penalty"],
            grammar=grammar,
        )

        return {"choices": [{"text": output}]}

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        structured_output_settings: LlmStructuredOutputSettings,
        settings: ExLlamaV2SamplingSettings
    ):
        prompt = self._format_chat_messages(messages)
        return self.create_completion(prompt, structured_output_settings, settings, "")

    def tokenize(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt)

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        formatted_messages = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                formatted_messages.append(f"System: {content}")
            elif role == "user":
                formatted_messages.append(f"Human: {content}")
            elif role == "assistant":
                formatted_messages.append(f"Assistant: {content}")
        return "\n".join(formatted_messages) + "\nAssistant:"

    def _get_or_create_grammar(self, structured_output_settings: LlmStructuredOutputSettings):
        grammar_key = structured_output_settings.get_gbnf_grammar()
        if grammar_key not in self.grammar_cache:
            grammar = ExLlamaV2Grammar()
            if structured_output_settings.output_type == LlmStructuredOutputType.json_schema:
                grammar.add_json_schema_filter(
                    structured_output_settings.json_schema,
                    self.model,
                    self.tokenizer
                )
            elif structured_output_settings.output_type == LlmStructuredOutputType.regex:
                grammar.add_regex_filter(
                    structured_output_settings.regex_pattern,
                    self.tokenizer
                )
            self.grammar_cache[grammar_key] = grammar
        return self.grammar_cache[grammar_key]
