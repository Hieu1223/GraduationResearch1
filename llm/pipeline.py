import os
import json
import re

from tqdm import tqdm
from transformers import TextStreamer
from llm.pado_prompt_builder import PromptManager


class PADOPipeline:
    def __init__(
        self,
        model,
        processor,
        latent_prompt_path: str,
        verdict_prompt_path: str,
        personality_path: str,
        dimensions_path: str,
        trait: str,
        errors_dir: str = "errors",
        results_dir: str = "results",
        max_new_tokens: int = 1024,
        repetition_penalty: float = 1.3,
    ):
        self.model              = model
        self.processor          = processor
        self.trait              = trait
        self.personality_path   = personality_path
        self.dimensions_path    = dimensions_path
        self.errors_dir         = errors_dir
        self.max_new_tokens     = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.results_dir        = results_dir

        self.prompt_manager = PromptManager(
            latent_prompt_path=latent_prompt_path,
            verdict_prompt_path=verdict_prompt_path,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chat(self, prompt: str, streamer: TextStreamer = None) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        raw_outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=self.repetition_penalty,
            streamer=streamer,
        )

        return self.processor.decode(
            raw_outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

    @staticmethod
    def _extract_final_label(response: str) -> str:
        # Match "Final Label:" specifically — ignore "Stronger Candidate:" line
        match = re.search(
            r"Final Label:\s*\*{0,2}\s*(HIGH|LOW)\b",
            response,
            re.IGNORECASE,
        )

        if match:
            return match.group(1).upper()

        # Fallback: last occurrence of HIGH/LOW in the response tail
        tail = response[-100:]
        last_high = tail.upper().rfind("HIGH")
        last_low  = tail.upper().rfind("LOW")

        if last_high == last_low == -1:
            return "UNKNOWN"

        return "HIGH" if last_high > last_low else "LOW"

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def run_latent(
        self,
        text: str,
        streamer: TextStreamer = None,
    ) -> dict[str, str]:
        """Run HIGH and LOW latent agents. Returns {"HIGH": ..., "LOW": ...}."""
        prompts = self.prompt_manager.generate_latent_prompts(
            text=text,
            trait=self.trait,
            personality_path=self.personality_path,
            dimensions_path=self.dimensions_path,
        )
        return {
            label: self._chat(prompt, streamer)
            for label, prompt in prompts.items()
        }

    def run_verdict(
        self,
        text: str,
        latent_outputs: dict[str, str],
        streamer: TextStreamer = None,
    ) -> tuple[str, str]:
        """Run the verdict agent. Returns (full_response, final_label)."""
        prompt = self.prompt_manager.generate_verdict_prompt(
            text=text,
            trait=self.trait,
            latent_outputs=latent_outputs,
            
        )

        response = self._chat(prompt, streamer)
        return response, self._extract_final_label(response)

    def run_one(
        self,
        text: str,
        streamer: TextStreamer = None,
    ) -> dict:
        """Run the full pipeline on a single essay."""
        latent_outputs = self.run_latent(text, streamer)
        verdict_response, pred = self.run_verdict(text, latent_outputs, streamer)

        return {
            "latent_outputs":  latent_outputs,
            "verdict_response": verdict_response,
            "prediction":      pred,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataset,
        ground_truth_fn=None,
        offset: int = 0,
        limit: int = None,
        stream: TextStreamer = None,
    ) -> list[dict]:
        """
        Run the pipeline over a dataset. Saves every prediction to
        results/<idx>.json and errors to errors/<idx>.json separately.
        Already-completed indices (results/<idx>.json exists) are skipped,
        enabling resume after a crash.

        Args:
            dataset:         Iterable of (text, label_tuple) pairs.
            ground_truth_fn: Callable mapping label_tuple → "HIGH" or "LOW".
            offset:          Start processing from this dataset index.
            limit:           Stop after processing this many samples total
                             (counting from offset). None = no limit.
            stream:          Stream token output to stdout.

        Returns:
            List of error dicts for wrong predictions in this run.
        """
        if ground_truth_fn is None:
            ground_truth_fn = lambda label_tuple: (
                "HIGH" if label_tuple[0] == 1 else "LOW"
            )

        os.makedirs(self.errors_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        errors = []
        processed = 0

        for idx, (text, label_tuple) in tqdm(enumerate(dataset)):

            # Skip indices before offset
            if idx < offset:
                continue

            # Stop once we've processed `limit` samples
            if limit is not None and processed >= limit:
                break

            result_path = os.path.join(self.results_dir, f"{idx}.json")

            # Resume: skip already-completed samples
            if os.path.exists(result_path):
                print(f"[skip] index {idx} already done", flush=True)
                processed += 1
                continue

            print(f"\n[index {idx}]", flush=True)

            result = self.run_one(text, streamer=stream)

            true = ground_truth_fn(label_tuple)
            pred = result["prediction"]

            record = {
                "index": idx,
                "text": text,
                "prediction": pred,
                "ground_truth": true,
                "correct": pred == true,
                "latent_outputs": result["latent_outputs"],
                "verdict_response": result["verdict_response"],
            }

            # Always save to results/
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            # Also save to errors/ if wrong
            if pred != true:
                error_path = os.path.join(self.errors_dir, f"{idx}.json")
                with open(error_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)
                errors.append(record)

            processed += 1

        return errors
    

def main():
    pipeline = PADOPipeline(
        model=None,
        processor=None,
        latent_prompt_path="prompts/latent_prompt.txt",
        verdict_prompt_path="prompts/verdict_prompt.txt",
        personality_path="personalities/extraversion.txt",
        dimensions_path="dimensions/extraversion.txt",
        trait="Extraversion",
        errors_dir="errors",
        max_new_tokens=2048,
        repetition_penalty=1.1
    )
    pipeline.run_one(
        text="I love going to parties and meeting new people. I feel energized when I'm around others.",
    )

if __name__ == "__main__":
    main()