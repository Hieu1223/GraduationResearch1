from llm.pado_prompt_builder import MessageFormat, InferencePrompt,PromptBuilder,Trait
from ollama import chat
import re
from pathlib import Path
import json

class TraitResult:
    __slots__ = (
        "trait", "index", "text",
        "raw_response", "prediction", "ground_truth", "correct", "mode",
    )

    def __init__(
        self,
        trait: Trait,
        index: int,
        text: str,
        raw_response: str,
        prediction: str,
        ground_truth: str,
        mode: str,
    ) -> None:
        self.trait        = trait
        self.index        = index
        self.text         = text
        self.raw_response = raw_response
        self.prediction   = prediction
        self.ground_truth = ground_truth
        self.correct      = prediction == ground_truth
        self.mode         = mode

    def to_dict(self) -> dict:
        return {
            "mode":         self.mode,
            "trait":        self.trait.value,
            "index":        self.index,
            "text":         self.text,
            "raw_response": self.raw_response,
            "prediction":   self.prediction,
            "ground_truth": self.ground_truth,
            "correct":      self.correct,
        }




class PipelineBase:
    def __init__(self,model_name: str,message_format: MessageFormat,prompt_builder: PromptBuilder,mode:str, temperature: float, repeat_penalty: float, max_new_tokens: int):
        self.model_name = model_name
        self.message_format = message_format
        self.prompt_builder = prompt_builder
        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
        self.max_new_tokens = max_new_tokens
        self.mode = mode

    
    def infer(self, prompt: InferencePrompt) -> str:
        messages = prompt.to_messages(self.message_format)
        response = chat(
            model=self.model_name,
            messages=messages,
            options={
                "temperature":    self.temperature,
                "repeat_penalty": self.repeat_penalty,
                "num_predict":    self.max_new_tokens,
            },
        )
        return response.message.content
    


    
    @staticmethod
    def _extract_label(response: str) -> str:
        """Pull HIGH / LOW out of a judge response; falls back on tail scan."""
        match = re.search(
            r"Final\s+Judgement.*?(HIGH|LOW)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).upper()

        tail = response[-150:].upper()
        last_high = tail.rfind("HIGH")
        last_low  = tail.rfind("LOW")

        if last_high == last_low == -1:
            return "UNKNOWN"
        return "HIGH" if last_high > last_low else "LOW"
    
    @staticmethod        
    def _save(result: TraitResult, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)