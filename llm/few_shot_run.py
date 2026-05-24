from dataset import ProcessedEssayDataset

from llm.oneoff_prompt_builder import OneOffPromptBuilder
from llm.few_shot_pipeline import MultiUserPromptPipeline

from llm.pado_prompt_builder import MessageFormat, Trait


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME     = "gemma4:31b-cloud"
MAX_NEW_TOKENS = 1024
TEMPERATURE    = 0.7
REPEAT_PENALTY = 1.3
MESSAGE_FORMAT = MessageFormat.HF

OFFSET = 0
LIMIT  = 100


# ---------------------------------------------------------------------------
# Prompt file map
# ---------------------------------------------------------------------------

PROMPT_PATHS = {
    "few_shot": {
        "system": "prompts/few_shot/few_shot_system.txt",

        "users": {
            Trait.OPENNESS:
                "prompts/few_shot/openness.txt",

            Trait.CONSCIENTIOUSNESS:
                "prompts/few_shot/conscientiousness.txt",

            Trait.EXTRAVERSION:
                "prompts/few_shot/extraversion.txt",

            Trait.AGREEABLENESS:
                "prompts/few_shot/agreeableness.txt",

            Trait.NEUROTICISM:
                "prompts/few_shot/neuroticism.txt",
        },
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    builder = OneOffPromptBuilder()

    dataset = ProcessedEssayDataset()

    shared = dict(
        model_name=MODEL_NAME,
        prompt_builder=builder,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        repeat_penalty=REPEAT_PENALTY,
        message_format=MESSAGE_FORMAT,
    )

    eval_kwargs = dict(
        offset=OFFSET,
        limit=LIMIT,
    )

    # -------------------------------------------------------------------
    # Few-shot
    # -------------------------------------------------------------------

    print(
        f"\n{'='*60}\n"
        f"Mode: few_shot\n"
        f"{'='*60}",
        flush=True,
    )

    few_shot_errors = MultiUserPromptPipeline(
        mode="few_shot",
        system_path=PROMPT_PATHS["few_shot"]["system"],
        user_paths=PROMPT_PATHS["few_shot"]["users"],
        results_dir="results/few_shot",
        **shared,
    ).evaluate(
        dataset,
        **eval_kwargs,
    )

    print(
        f"[few_shot] {len(few_shot_errors)} wrong predictions",
        flush=True,
    )

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------

    print(
        f"\n{'='*60}\n"
        f"Summary\n"
        f"{'='*60}"
    )

    print(f"  few_shot : {len(few_shot_errors)} errors")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()