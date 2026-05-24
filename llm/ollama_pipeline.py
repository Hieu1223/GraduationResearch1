import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Optional

from tqdm import tqdm

from llm.pado_prompt_builder import InduceLevel, InferencePrompt, MessageFormat, PromptBuilder, PromptType, Trait
from llm.pipeline_base import PipelineBase


class TraitResult:
    __slots__ = (
        "trait", "index", "text",
        "high_explanation", "low_explanation",
        "verdict_response", "prediction", "ground_truth", "correct",
    )

    def __init__(
        self,
        trait: Trait,
        index: int,
        text: str,
        high_explanation: str,
        low_explanation: str,
        verdict_response: str,
        prediction: str,
        ground_truth: str,
    ) -> None:
        self.trait             = trait
        self.index             = index
        self.text              = text
        self.high_explanation  = high_explanation
        self.low_explanation   = low_explanation
        self.verdict_response  = verdict_response
        self.prediction        = prediction
        self.ground_truth      = ground_truth
        self.correct           = prediction == ground_truth

    def to_dict(self) -> dict:
        return {
            "trait":            self.trait.value,
            "index":            self.index,
            "text":             self.text,
            "high_explanation": self.high_explanation,
            "low_explanation":  self.low_explanation,
            "verdict_response": self.verdict_response,
            "prediction":       self.prediction,
            "ground_truth":     self.ground_truth,
            "correct":          self.correct,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class OCEANPipeline(PipelineBase):
    """
    Runs 5 trait workers in parallel. Each worker processes its assigned
    trait sequentially sample-by-sample through the full dataset.

    Per trait per sample:
      - HIGH explain
      - LOW explain
      - Judge verdict
    """

    def __init__(
        self,
        model_name: str,
        prompt_builder: PromptBuilder,
        results_dir: str = "results",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        repeat_penalty: float = 1.3,
        message_format: MessageFormat = MessageFormat.HF,
    ) -> None:
        self.model_name     = model_name
        self.prompt_builder = prompt_builder
        self.results_dir    = Path(results_dir)
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.repeat_penalty = repeat_penalty
        self.message_format = message_format



    @staticmethod
    def _extract_label(response: str) -> str:
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

    # ------------------------------------------------------------------
    # Single sample, single trait
    # ------------------------------------------------------------------

    def run_one(
        self,
        index: int,
        text: str,
        ground_truth: str,
        trait: Trait,
    ) -> TraitResult:
        high_prompt = self.prompt_builder.build_explain(
             trait=trait, text=text, induce=InduceLevel.HIGH,
        )
        low_prompt = self.prompt_builder.build_explain(
            trait=trait, text=text, induce=InduceLevel.LOW,
        )
        judge_prompt_builder = lambda h, l: self.prompt_builder.build_judge(
            trait=trait, text=text,
            explanation1=h, explanation2=l,
        )

        high_explanation = self.infer(high_prompt)
        low_explanation  = self.infer(low_prompt)
        verdict_response = self.infer(judge_prompt_builder(high_explanation, low_explanation))
        prediction       = self._extract_label(verdict_response)

        return TraitResult(
            trait=trait,
            index=index,
            text=text,
            high_explanation=high_explanation,
            low_explanation=low_explanation,
            verdict_response=verdict_response,
            prediction=prediction,
            ground_truth=ground_truth,
        )

    # ------------------------------------------------------------------
    # Worker: one trait, all samples sequentially
    # ------------------------------------------------------------------

    def _trait_worker(
        self,
        trait: Trait,
        samples: list[tuple[int, str, str]],   # (index, text, ground_truth)
    ) -> list[TraitResult]:
        """Runs sequentially through all samples for one trait."""
        errors: list[TraitResult] = []
        trait_dir = self.results_dir / trait.value
        trait_dir.mkdir(parents=True, exist_ok=True)

        bar = tqdm(samples, desc=trait.value, position=list(Trait).index(trait), leave=True)
        for idx, text, ground_truth in bar:
            result_path = trait_dir / f"{idx}.json"

            if result_path.exists():
                bar.set_postfix(status="skip")
                continue

            bar.set_postfix(idx=idx)
            result = self.run_one(idx, text, ground_truth, trait)
            self._save(result, result_path)

            if not result.correct:
                errors.append(result)

        return errors

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataset: Iterable[tuple[str, dict[Trait, str]]],
        ground_truth_fn: Callable[[any], dict[Trait, str]] = lambda x: x,
        traits: Optional[list[Trait]] = None,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> list[TraitResult]:
        """
        Spawns one thread per trait. Each thread walks through the sample
        list sequentially.

        Args:
            dataset:         Iterable of (text, raw_label) pairs.
            ground_truth_fn: Maps raw_label → {Trait: "HIGH"|"LOW"}.
                             Defaults to identity for ProcessedEssayDataset.
            traits:          Subset of traits to run. None = all five.
            offset:          Skip the first N samples.
            limit:           Max samples to process after offset.

        Saves:
            results/<trait>/<index>.json

        Returns:
            All wrong TraitResults across all traits.
        """
        active_traits = traits if traits is not None else list(Trait)

        # Build the flat sample list once, shared across all trait workers
        samples_per_trait: dict[Trait, list[tuple[int, str, str]]] = {t: [] for t in active_traits}

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

        # Launch one worker thread per trait
        all_errors: list[TraitResult] = []
        with ThreadPoolExecutor(max_workers=len(active_traits)) as ex:
            futures = {
                ex.submit(self._trait_worker, trait, samples_per_trait[trait]): trait
                for trait in active_traits
            }
            for fut in futures:
                all_errors.extend(fut.result())

        return all_errors

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _save(result: TraitResult, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

