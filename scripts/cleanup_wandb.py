import wandb
from loguru import logger

def cleanup_wandb():
    api = wandb.Api()
    project_path = "antonvice/StegaDNA-Alpha"
    runs = api.runs(project_path)
    
    logger.info(f"Analyzing {len(runs)} runs in {project_path}...")
    
    # 1. Delete "Trash" runs (crashed, fewer than 5 steps, or N/A metrics)
    deleted_count = 0
    for run in runs:
        # Get step count safely
        steps = run.summary.get("_step", 0)
        
        # 1. Delete "Trash" runs (failed/killed/crashed with few steps)
        # We define trash as any non-finished run with < 50 steps
        if run.state != "finished" and steps < 50:
            logger.warning(f"Deleting non-finished/tiny run: {run.name} ({run.id}) | Steps: {steps}")
            run.delete()
            deleted_count += 1
            continue
            
        # 2. Delete runs with no bit metrics (failed warmup)
        if "epoch/avg_bit_error_rate" not in run.summary and steps < 100:
             logger.warning(f"Deleting empty/warmup-only run: {run.name} ({run.id}) | Steps: {steps}")
             run.delete()
             deleted_count += 1
             continue

    logger.success(f"Cleanup complete. Deleted {deleted_count} redundant runs.")

if __name__ == "__main__":
    cleanup_wandb()
