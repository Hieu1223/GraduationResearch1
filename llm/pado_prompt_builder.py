import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enums  (shared across both builders)
# ---------------------------------------------------------------------------

class Trait(Enum):
    EXTRAVERSION      = "Extraversion"
    AGREEABLENESS     = "Agreeableness"
    CONSCIENTIOUSNESS = "Conscientiousness"
    NEUROTICISM       = "Neuroticism"
    OPENNESS          = "Openness"


class InduceLevel(Enum):
    HIGH = "high"
    LOW  = "low"


class PromptType(Enum):
    EXPLAIN   = "explain"
    JUDGE     = "judge"
    COT       = "cot"
    ONE_SHOT  = "one_shot"
    FEW_SHOT  = "few_shot"


class MessageFormat(Enum):
    OPENAI = "openai"
    HF     = "hf"


# ---------------------------------------------------------------------------
# InferencePrompt  (shared)
# ---------------------------------------------------------------------------

@dataclass
class InferencePrompt:
    """Fully-rendered system + user prompt pair, ready for inference."""
    system: str
    user: str

    def to_messages(self, fmt: MessageFormat = MessageFormat.OPENAI) -> List[dict]:
        if fmt is MessageFormat.OPENAI:
            return [
                {"role": "system", "content": [{"type": "text", "text": self.system}]},
                {"role": "user",   "content": [{"type": "text", "text": self.user}]},
            ]
        if fmt is MessageFormat.HF:
            return [
                {"role": "system", "content": self.system},
                {"role": "user",   "content": self.user},
            ]
        raise ValueError(f"Unsupported MessageFormat: {fmt}")


# ---------------------------------------------------------------------------
# PromptBuilder  (PADO pipeline only: EXPLAIN + JUDGE)
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Builds prompts for the two-stage PADO pipeline (EXPLAIN and JUDGE).

    Template placeholders
    ---------------------
    pado_system.txt        {personality_inducing}
    pado_user.txt          {trait}, {text}, {personality_inducing}
    judge_system.txt       (none)
    judge_user_<trait>.txt {trait}, {text}, {explain_1}, {explain_2}
    """

    def __init__(
        self,
        pado_system_path: str                  = "prompts/pado/pado_system.txt",
        pado_user_path: str                    = "prompts/pado/pado_user.txt",
        judge_system_path: str                 = "prompts/pado/judge_system.txt",
        judge_user_extraversion_path: str      = "prompts/pado/judge_user_extraversion.txt",
        judge_user_agreeableness_path: str     = "prompts/pado/judge_user_agreeableness.txt",
        judge_user_conscientiousness_path: str = "prompts/pado/judge_user_conscientiousness.txt",
        judge_user_neuroticism_path: str       = "prompts/pado/judge_user_neuroticism.txt",
        judge_user_openness_path: str          = "prompts/pado/judge_user_openness.txt",
        high_extraversion_path: str            = "prompts/pado/induce_high_extraversion.txt",
        high_agreeableness_path: str           = "prompts/pado/induce_high_agreeableness.txt",
        high_conscientiousness_path: str       = "prompts/pado/induce_high_conscientiousness.txt",
        high_neuroticism_path: str             = "prompts/pado/induce_high_neuroticism.txt",
        high_openness_path: str                = "prompts/pado/induce_high_openness.txt",
        low_extraversion_path: str             = "prompts/pado/induce_low_extraversion.txt",
        low_agreeableness_path: str            = "prompts/pado/induce_low_agreeableness.txt",
        low_conscientiousness_path: str        = "prompts/pado/induce_low_conscientiousness.txt",
        low_neuroticism_path: str              = "prompts/pado/induce_low_neuroticism.txt",
        low_openness_path: str                 = "prompts/pado/induce_low_openness.txt",
    ) -> None:
        self._pado_system_tmpl  = self._read(pado_system_path)
        self._pado_user_tmpl    = self._read(pado_user_path)

        self._judge_system_tmpl = self._read(judge_system_path)
        self._judge_user_tmpls: dict[Trait, str] = {
            Trait.EXTRAVERSION:      self._read(judge_user_extraversion_path),
            Trait.AGREEABLENESS:     self._read(judge_user_agreeableness_path),
            Trait.CONSCIENTIOUSNESS: self._read(judge_user_conscientiousness_path),
            Trait.NEUROTICISM:       self._read(judge_user_neuroticism_path),
            Trait.OPENNESS:          self._read(judge_user_openness_path),
        }

        self._induce: dict[tuple[InduceLevel, Trait], str] = {
            (InduceLevel.HIGH, Trait.EXTRAVERSION):      self._read(high_extraversion_path),
            (InduceLevel.HIGH, Trait.AGREEABLENESS):     self._read(high_agreeableness_path),
            (InduceLevel.HIGH, Trait.CONSCIENTIOUSNESS): self._read(high_conscientiousness_path),
            (InduceLevel.HIGH, Trait.NEUROTICISM):       self._read(high_neuroticism_path),
            (InduceLevel.HIGH, Trait.OPENNESS):          self._read(high_openness_path),
            (InduceLevel.LOW,  Trait.EXTRAVERSION):      self._read(low_extraversion_path),
            (InduceLevel.LOW,  Trait.AGREEABLENESS):     self._read(low_agreeableness_path),
            (InduceLevel.LOW,  Trait.CONSCIENTIOUSNESS): self._read(low_conscientiousness_path),
            (InduceLevel.LOW,  Trait.NEUROTICISM):       self._read(low_neuroticism_path),
            (InduceLevel.LOW,  Trait.OPENNESS):          self._read(low_openness_path),
        }

    def build_explain(self, trait: Trait, text: str, induce: InduceLevel = InduceLevel.HIGH) -> InferencePrompt:
        return self._build_explain(trait, text, induce)

    def build_judge(self, trait: Trait, text: str, explanation1: str, explanation2: str) -> InferencePrompt:
        return self._build_judge(trait, text, explanation1, explanation2)

    def _build_explain(self, trait: Trait, text: str, induce: InduceLevel) -> InferencePrompt:
        inducing_text = self._induce[(induce, trait)]
        system = self._pado_system_tmpl.format(personality_inducing=inducing_text)
        user   = self._pado_user_tmpl.format(
            trait=trait.value,
            text=text,
            personality_inducing=inducing_text,
        )
        return InferencePrompt(system=system, user=user)

    def _build_judge(self, trait: Trait, text: str, explanation1: str, explanation2: str) -> InferencePrompt:
        explain_a, explain_b = random.sample([explanation1, explanation2], k=2)
        system = self._judge_system_tmpl
        user   = self._judge_user_tmpls[trait].format(
            trait=trait.value,
            text=text,
            explain_1=explain_a,
            explain_2=explain_b,
        )
        return InferencePrompt(system=system, user=user)

    @staticmethod
    def _read(path: str) -> str:
        return Path(path).read_text(encoding="utf-8")