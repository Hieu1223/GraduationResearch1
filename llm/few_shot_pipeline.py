from llm.cot_oneshot_pipeline import OneOffPipeline, OneOffPromptBuilder,TraitResult
from llm.pado_prompt_builder import MessageFormat, Trait


class MultiUserPromptPipeline(OneOffPipeline):
    """
    Variant of OneOffPipeline that allows different user prompt files
    per trait.

    Example:
        user_paths = {
            Trait.OPENNESS: "prompts/open_user.txt",
            Trait.CONSCIENTIOUSNESS: "prompts/con_user.txt",
            ...
        }

        pipeline = MultiUserPromptPipeline(
            model_name="llama3",
            prompt_builder=builder,
            mode="zero_shot",
            system_path="prompts/system.txt",
            user_paths=user_paths,
        )
    """

    def __init__(
        self,
        model_name: str,
        prompt_builder: OneOffPromptBuilder,
        mode: str,
        system_path: str,
        user_paths: dict[Trait, str],
        results_dir: str = "results",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        repeat_penalty: float = 1.3,
        message_format: MessageFormat = MessageFormat.HF,
    ) -> None:
        super().__init__(
            model_name=model_name,
            prompt_builder=prompt_builder,
            mode=mode,
            system_path=system_path,
            user_path="",  # unused
            results_dir=results_dir,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
            message_format=message_format,
        )

        self.user_paths = user_paths

    # ------------------------------------------------------------------
    # Resolve trait-specific user prompt
    # ------------------------------------------------------------------

    def _get_user_path(self, trait: Trait) -> str:
        if trait not in self.user_paths:
            raise ValueError(f"No user prompt path configured for trait: {trait}")
        return self.user_paths[trait]

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
        user_path = self._get_user_path(trait)

        prompt = self.prompt_builder.build(
            self.system_path,
            user_path,
            trait,
            text,
        )

        response = self.infer(prompt)
        pred = self._extract_label(response)

        return TraitResult(
            trait=trait,
            index=index,
            text=text,
            raw_response=response,
            prediction=pred,
            ground_truth=ground_truth,
            mode=self.mode,
        )