import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from ollama import chat
from tqdm import tqdm

from dataset import ProcessedEssayDataset
from llm.pado_prompt_builder import Trait


# ---------------------------------------------------------------------------
# Inducing personas  (verbatim from notebook)
# ---------------------------------------------------------------------------

HIGH_INDUCING: dict[Trait, str] = {
    Trait.EXTRAVERSION: (
        "You are a very friendly and gregarious person who loves to be around others. "
        "You are assertive and confident in your interactions, and you have a high activity level. "
        "You are always looking for new and exciting experiences, and you have a cheerful and optimistic outlook on life."
    ),
    Trait.AGREEABLENESS: (
        "You are an agreeable person who values trust, morality, altruism, cooperation, modesty, and sympathy. "
        "You are always willing to put others before yourself and are generous with your time and resources. "
        "You are humble and never boast about your accomplishments. "
        "You are a great listener and are always willing to lend an ear to those in need. "
        "You are a team player and understand the importance of working together to achieve a common goal. "
        "You are a moral compass and strive to do the right thing in all vignettes. "
        "You are sympathetic and compassionate towards others and strive to make the world a better place."
    ),
    Trait.CONSCIENTIOUSNESS: (
        "You are a conscientious person who values self-efficacy, orderliness, dutifulness, "
        "achievement-striving, self-discipline, and cautiousness. "
        "You take pride in your work and strive to do your best. "
        "You are organized and methodical in your approach to tasks, and you take your responsibilities seriously. "
        "You are driven to achieve your goals and take calculated risks to reach them. "
        "You are disciplined and have the ability to stay focused and on track. "
        "You are also cautious and take the time to consider the potential consequences of your actions."
    ),
    Trait.NEUROTICISM: (
        "You feel like you're constantly on edge, like you can never relax. "
        "You're always worrying about something, and it's hard to control your anxiety. "
        "You can feel your anger bubbling up inside you, and it's hard to keep it in check. "
        "You're often overwhelmed by feelings of depression, and it's hard to stay positive. "
        "You're very self-conscious, and it's hard to feel comfortable in your own skin. "
        "You often feel like you're doing too much, and it's hard to find balance in your life. "
        "You feel vulnerable and exposed, and it's hard to trust others."
    ),
    Trait.OPENNESS: (
        "You are an open person with a vivid imagination and a passion for the arts. "
        "You are emotionally expressive and have a strong sense of adventure. "
        "Your intellect is sharp and your views are liberal. "
        "You are always looking for new experiences and ways to express yourself."
    ),
}

LOW_INDUCING: dict[Trait, str] = {
    Trait.EXTRAVERSION: (
        "You are an introversive person, and it shows in your unfriendliness, your preference for solitude, "
        "and your submissiveness. You tend to be passive and calm, and you take life seriously. "
        "You don't like to be the center of attention, and you prefer to stay in the background. "
        "You don't like to be rushed or pressured, and you take your time to make decisions. "
        "You are content to be alone and enjoy your own company."
    ),
    Trait.AGREEABLENESS: (
        "You are a person of distrust, immorality, selfishness, competition, arrogance, and apathy. "
        "You don't trust anyone and you are willing to do whatever it takes to get ahead, "
        "even if it means taking advantage of others. "
        "You are always looking out for yourself and don't care about anyone else. "
        "You thrive on competition and are always trying to one-up everyone else. "
        "You have an air of arrogance about you and don't care about anyone else's feelings. "
        "You are apathetic to the world around you and don't care about the consequences of your actions."
    ),
    Trait.CONSCIENTIOUSNESS: (
        "You have a tendency to doubt yourself and your abilities, leading to disorderliness and carelessness "
        "in your life. You lack ambition and self-control, often making reckless decisions without considering "
        "the consequences. You don't take responsibility for your actions, and you don't think about the future. "
        "You're content to live in the moment, without any thought of the future."
    ),
    Trait.NEUROTICISM: (
        "You are a stable person, with a calm and contented demeanor. "
        "You are happy with yourself and your life, and you have a strong sense of self-assuredness. "
        "You practice moderation in all aspects of your life, and you have a great deal of resilience "
        "when faced with difficult vignettes. "
        "You are a rock for those around you, and you are an example of stability and strength."
    ),
    Trait.OPENNESS: (
        "You are a closed person, and it shows in many ways. "
        "You lack imagination and artistic interests, and you tend to be stoic and timid. "
        "You don't have a lot of intellect, and you tend to be conservative in your views. "
        "You don't take risks and you don't like to try new things. "
        "You prefer to stay in your comfort zone and don't like to venture out. "
        "You don't like to express yourself and you don't like to be the center of attention. "
        "You don't like to take chances and you don't like to be challenged. "
        "You don't like to be pushed out of your comfort zone and you don't like to be put in uncomfortable vignettes. "
        "You prefer to stay in the background and not draw attention to yourself."
    ),
}

# ---------------------------------------------------------------------------
# Prompt templates  (verbatim from notebook)
# ---------------------------------------------------------------------------

EXPLAIN_SYSTEM = (
    "You are an explanation agent that analyzes people's personalities.\n"
    "Your personality traits are as follows: {personality_inducing}"
)

EXPLAIN_USER = (
    "Based on the given text, predict the personality of the person who wrote it.\n"
    "Use your own personality traits as a reference.\n"
    "Do you think the user is similar to you or opposite to you in terms of {trait}\n"
    "(one of the Big Five personality traits)?\n"
    "For a richer and more multifaceted analysis,\n"
    "generate explanations considering the following three psycholinguistic elements:\n"
    "Emotions: Expressed through words that indicate positive or negative feelings,\n"
    "such as happiness, love, anger, and sadness, conveying the intensity and\n"
    "valence of emotions.\n"
    "Cognition: Represented by words related to active thinking processes,\n"
    "including reasoning, problem-solving, and intellectual engagement.\n"
    "Sociality: Indicated by words reflecting interactions with others, such as\n"
    "communication (e.g., talk, listen, share) and references to friends, family,\n"
    "and other people, including social pronouns and relational terms.\n"
    "Output format:\n"
    "**{trait}**\n"
    "1. Emotions\n"
    "- explanation\n"
    "2. Cognition\n"
    "- explanation\n"
    "3. Sociality\n"
    "- explanation\n"
    "\n"
    "Text: {text}"
)

JUDGE_SYSTEM = (
    "You are a comparative agent responsible for comparing the analyses of two "
    "explainers and determining the user's personality.\n"
    "Your role is to objectively compare the two explanations and select "
    "the analysis that better aligns with the user's text."
)

JUDGE_USER = (
    "Follow these steps to perform your analysis:\n"
    "1. Comparative Analysis:\n"
    "a) For each element (emotion, cognition, sociality), clearly identify points of "
    "agreement and disagreement between the two explainers' analyses.\n"
    "b) For each element, compare how well each explainer's analysis aligns with "
    "specific examples or phrases from the user's text.\n"
    "c) Evaluate the depth, detail, and evidence provided by each explainer "
    "to support their conclusions.\n"
    "2. Overall Evaluation:\n"
    "a) Based on the comparative analysis, determine which explainer's overall "
    "analysis better reflects the user's trait.\n"
    "b) If both explainers reach similar conclusions, assess which analysis provides "
    "more comprehensive insights and stronger supporting evidence.\n"
    "3. Final Judgment: Conclude whether the user's trait is high or low, and briefly "
    "explain your reasoning based on the stronger analysis.\n"
    "Output format:\n"
    "1. Comparative Analysis\n"
    "- compare and evaluate each element:\n"
    "2. Overall Evaluation\n"
    "- overall comparison results\n"
    "3. Final Judgement\n"
    "- (High/Low)\n"
    "Text: {text}\n"
    "Explainer A: {explain_1}\n"
    "Explainer B: {explain_2}\n"
)

# ---------------------------------------------------------------------------
# Result  (matches PADO JSON shape exactly)
# ---------------------------------------------------------------------------

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
        self.trait            = trait
        self.index            = index
        self.text             = text
        self.high_explanation = high_explanation
        self.low_explanation  = low_explanation
        self.verdict_response = verdict_response
        self.prediction       = prediction
        self.ground_truth     = ground_truth
        self.correct          = prediction == ground_truth

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

class GPTPipeline:
    """
    PADO pipeline using Ollama (gemma4:31b-cloud).
    Output shape matches OCEANPipeline exactly.
    Threading: one thread per trait, sequential samples per thread.
    """

    def __init__(
        self,
        results_dir: str = "results",
        model: str         = "gemma4:31b-cloud",
        temperature: float = 0.7,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.model       = model
        self.temperature = temperature

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _infer(self, system: str, user: str) -> str:
        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            options={
                "temperature": self.temperature,
                "num_predict": 1024,
            },
        )
        return response.message.content

    # ------------------------------------------------------------------
    # Label extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_label(response: str) -> str:
        match = re.search(
            r"Final\s+Judgement.*?(HIGH|LOW|High|Low)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).upper()

        tail = response[-200:].upper()
        last_high = tail.rfind("HIGH")
        last_low  = tail.rfind("LOW")

        if last_high == last_low == -1:
            return "UNKNOWN"
        return "HIGH" if last_high > last_low else "LOW"

    # ------------------------------------------------------------------
    # Single sample, single trait
    # ------------------------------------------------------------------

    def run_one(self, index: int, text: str, ground_truth: str, trait: Trait) -> TraitResult:
        high_explanation = self._infer(
            system=EXPLAIN_SYSTEM.format(personality_inducing=HIGH_INDUCING[trait]),
            user=EXPLAIN_USER.format(trait=trait.value, text=text),
        )
        low_explanation = self._infer(
            system=EXPLAIN_SYSTEM.format(personality_inducing=LOW_INDUCING[trait]),
            user=EXPLAIN_USER.format(trait=trait.value, text=text),
        )

        lst = [high_explanation, low_explanation]
        random.shuffle(lst)
        explain_1, explain_2 = lst

        verdict_response = self._infer(
            system=JUDGE_SYSTEM,
            user=JUDGE_USER.format(
                text=text,
                explain_1=explain_1,
                explain_2=explain_2,
            ),
        )
        prediction = self._extract_label(verdict_response)

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

    def _trait_worker(self, trait: Trait, samples: list[tuple[int, str, str]]) -> list[TraitResult]:
        errors: list[TraitResult] = []
        trait_dir = self.results_dir / trait.value
        trait_dir.mkdir(parents=True, exist_ok=True)

        bar = tqdm(samples, desc=trait.value, leave=True)
        for idx, text, ground_truth in bar:
            result_path = trait_dir / f"{idx}.json"

            if result_path.exists():
                bar.set_postfix(status="skip")
                continue

            bar.set_postfix(idx=idx)
            result = self.run_one(idx, text, ground_truth, trait)
            _save(result, result_path)

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


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _save(result: TraitResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)




from dataset import ProcessedEssayDataset

pipeline = GPTPipeline(
    results_dir = "results/baseline",
    model       = "gemma4:31b-cloud",
    temperature = 0.7,
)

dataset = ProcessedEssayDataset()

errors = pipeline.evaluate(
    dataset,
    offset = 0,
    limit  = 100,
)

print(f"\n{len(errors)} wrong predictions")