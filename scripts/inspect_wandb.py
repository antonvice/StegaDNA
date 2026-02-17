import wandb
from loguru import logger

def inspect_runs():
    api = wandb.Api()
    project_path = "antonvice/StegaDNA-Alpha"
    runs = api.runs(project_path)
    
    print(f"{'ID':<12} | {'Name':<25} | {'State':<10} | {'Steps':<8} | {'BER':<8} | {'PSNR':<8}")
    print("-" * 80)
    
    for run in runs:
        steps = run.summary.get("_step", 0)
        ber = run.summary.get("epoch/avg_bit_error_rate", "N/A")
        if isinstance(ber, float): ber = f"{ber:.4f}"
        
        psnr = run.summary.get("epoch/avg_psnr", "N/A")
        if isinstance(psnr, float): psnr = f"{psnr:.2f}"
        
        print(f"{run.id:<12} | {run.name[:25]:<25} | {run.state:<10} | {steps:<8} | {ber:<8} | {psnr:<8}")

if __name__ == "__main__":
    inspect_runs()
