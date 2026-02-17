import wandb
from loguru import logger

def manual_cleanup():
    api = wandb.Api()
    project_path = "antonvice/StegaDNA-Alpha"
    
    # Run IDs to KEEP (The Winners)
    TO_KEEP = [
        "3t97y74k", # The Gold Breakthrough (v4 Arch, 0.24 BER)
        "tm7gm7or", # Phase 3: Polish Failure (v3 Arch)
        "kfi8qma4", # Phase 2: Resurrection Best (0.31 BER)
        "oif2stba", # Phase 2: Resurrection Early
        "65d9rzkb", # Phase 1: v2-balanced-polish
    ]
    
    # Run IDs to DELETE (The Fragments/Clutter)
    TO_DELETE = [
        "9hluicwx", # Fragment
        "7hks4d3o", # Fragment
        "np0t0i9l", # Fragment
        "lakn6oiz", # Fragment
        "8karqnks", # Fragment
        "9xvz55rj", # Fragment
        "mwob5lwp", # Fragment
    ]
    
    logger.info("Starting manual W&B excision...")
    
    for run_id in TO_DELETE:
        try:
            run = api.run(f"{project_path}/{run_id}")
            logger.warning(f"Exterminating redundant run: {run.name} ({run.id})")
            run.delete()
        except Exception as e:
            logger.error(f"Could not delete {run_id}: {e}")

    logger.success("WandB Dashboard sanitized. Only the historical milestones remain.")

if __name__ == "__main__":
    manual_cleanup()
