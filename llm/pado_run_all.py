from llm.pado_prompt_builder import PromptBuilder
from llm.ollama_pipeline import OCEANPipeline

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    from dataset import ProcessedEssayDataset

    builder  = PromptBuilder()
    pipeline = OCEANPipeline(
        model_name="gemma4:31b-cloud",
        prompt_builder=builder,
        results_dir="results",
        max_new_tokens=2048,
        temperature=0.7,
    )

    dataset = ProcessedEssayDataset()

    errors = pipeline.evaluate(
        dataset,
        offset=0,
        limit=100,
    )
    print(f"\n{len(errors)} wrong predictions", flush=True)


if __name__ == "__main__":
    main()