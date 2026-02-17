import wandb
from loguru import logger

def final_excision():
    api = wandb.Api()
    project_path = "antonvice/StegaDNA-Alpha"
    
    # These are the 4 runs that tell the WHOLE story of StegaDNA v4.
    TO_KEEP = ["3t97y74k", "tm7gm7or", "kfi8qma4", "v6u54ag4"]
    
    logger.info("Executing the Final Excision. Streamlining dashboard to 4 Milestone Runs...")
    
    runs = api.runs(project_path)
    for run in runs:
        if run.id not in TO_KEEP:
            logger.warning(f"Purging non-milestone run: {run.name} ({run.id})")
            run.delete()
        else:
            logger.success(f"Preserving Milestone: {run.name} ({run.id})")

    logger.success("Dashboard finalized. All clutter removed.")

if __name__ == "__main__":
    final_excision()
