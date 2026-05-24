from dataset import ProcessedEssayDataset
from llm.oneoff_prompt_builder import OneOffPromptBuilder
from llm.cot_oneshot_pipeline import OneOffPipeline
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
    "cot": {
        "system": "prompts/cot/cot_system.txt",
        "user":   "prompts/cot/cot_user.txt",
    },
    "one_shot": {
        "system": "prompts/one_shot/one_shot_system.txt",
        "user":   "prompts/one_shot/one_shot_user.txt",
    },
    "few_shot": {
        "system": "prompts/few_shot/few_shot_system.txt",
        "user":   "prompts/few_shot/few_shot_user.txt",
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
    eval_kwargs = dict(offset=OFFSET, limit=LIMIT)
    

    # --- One-shot ---
    print(f"\n{'='*60}\nMode: one_shot\n{'='*60}", flush=True)
    one_shot_errors = OneOffPipeline(
        mode="one_shot",
        system_path=PROMPT_PATHS["one_shot"]["system"],
        user_path=PROMPT_PATHS["one_shot"]["user"],
        results_dir="results/one_shot",
        **shared,
    ).evaluate(dataset, **eval_kwargs)
    print(f"[one_shot] {len(one_shot_errors)} wrong predictions", flush=True)




if __name__ == "__main__":
    main()