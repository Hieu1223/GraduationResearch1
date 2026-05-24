import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from tqdm import tqdm

from llm.pado_prompt_builder import InferencePrompt, MessageFormat, Trait
from llm.oneoff_prompt_builder import OneOffPromptBuilder
from llm.pipeline_base import PipelineBase, TraitResult

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class OneOffPipeline(PipelineBase):
    """
    Single-LLM-call pipeline for Big Five classification.
    One prompt builder, one system file, one user file per mode.
    Only {trait} and {text} are injected at runtime — everything else
    (examples, instructions) is baked into the prompt files.
    Threading mirrors OCEANPipeline: one thread per trait, sequential samples.
    """

    def __init__(
        self,
        model_name: str,
        prompt_builder: OneOffPromptBuilder,
        mode: str,
        system_path: str,
        user_path: str,
        results_dir: str = "results",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        repeat_penalty: float = 1.3,
        message_format: MessageFormat = MessageFormat.HF,
    ) -> None:
        self.model_name     = model_name
        self.prompt_builder = prompt_builder
        self.mode           = mode
        self.system_path    = system_path
        self.user_path      = user_path
        self.results_dir    = Path(results_dir)
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.repeat_penalty = repeat_penalty
        self.message_format = message_format



    # ------------------------------------------------------------------
    # Label extraction
    # ------------------------------------------------------------------


    def run_one(self, index: int, text: str, ground_truth: str, trait: Trait) -> TraitResult:
        prompt   = self.prompt_builder.build(self.system_path, self.user_path, trait, text)
        response = self.infer(prompt)
        pred     = self._extract_label(response)

        return TraitResult(
            trait=trait,
            index=index,
            text=text,
            raw_response=response,
            prediction=pred,
            ground_truth=ground_truth,
            mode=self.mode,
        )

    # ------------------------------------------------------------------
    # Worker: one trait, all samples sequentially
    # ------------------------------------------------------------------

    def _trait_worker(self, trait: Trait, samples: list[tuple[int, str, str]]) -> list[TraitResult]:
        errors: list[TraitResult] = []
        trait_dir = self.results_dir / self.mode / trait.value
        trait_dir.mkdir(parents=True, exist_ok=True)

        bar = tqdm(samples, desc=f"{self.mode}/{trait.value}", leave=True)
        for idx, text, ground_truth in bar:
            result_path = trait_dir / f"{idx}.json"

            if result_path.exists():
                bar.set_postfix(status="skip")
                continue

            bar.set_postfix(idx=idx)
            result = self.run_one(idx, text, ground_truth, trait)
            OneOffPipeline._save(result, result_path)

            if not result.correct:
                errors.append(result)

        return errors

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataset,
        ground_truth_fn: Callable[[any], dict[Trait, str]] = lambda x: x,
        traits: Optional[list[Trait]] = None,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> list[TraitResult]:
        active_traits = traits if traits is not None else list(Trait)

        samples_per_trait: dict[Trait, list] = {t: [] for t in active_traits}
        processed = 0

        for idx, (text, raw_label) in enumerate(dataset):
            if idx < offset:
                continue
            if limit is not None and processed >= limit:
                break
            ground_truths = ground_truth_fn(raw_label)
            for trait in active_traits:
                if trait in ground_truths:
                    samples_per_trait[trait].append((idx, text, ground_truths[trait]))
            processed += 1

        all_errors: list[TraitResult] = []
        with ThreadPoolExecutor(max_workers=len(active_traits)) as ex:
            futures = {
                ex.submit(self._trait_worker, trait, samples_per_trait[trait]): trait
                for trait in active_traits
            }
            for fut in futures:
                all_errors.extend(fut.result())

        return all_errors

