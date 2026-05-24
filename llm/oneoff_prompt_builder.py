from pathlib import Path

from llm.pado_prompt_builder import InferencePrompt, Trait


class OneOffPromptBuilder:
    def build(self, system_path: str, user_path: str, trait: Trait, text: str) -> InferencePrompt:
        return InferencePrompt(
            system=Path(system_path).read_text(encoding="utf-8"),
            user=Path(user_path).read_text(encoding="utf-8").format(
                trait=trait.value,
                text=text,
            ),
        )