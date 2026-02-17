import os
import glob
import re
import shutil
from loguru import logger

def cleanup_checkpoints():
    checkpoint_root = "model/checkpoints"
    if not os.path.exists(checkpoint_root):
        logger.error(f"Root checkpoint directory {checkpoint_root} not found.")
        return

    # 1. Create Archive for Absolute Best Models
    archive_dir = "model/best_archive"
    os.makedirs(archive_dir, exist_ok=True)

    # Walk through tag folders
    tags = [d for d in os.listdir(checkpoint_root) if os.path.isdir(os.path.join(checkpoint_root, d))]
    
    for tag in tags:
        tag_path = os.path.join(checkpoint_root, tag)
        logger.info(f"Processing tag: {tag}...")
        
        pth_files = glob.glob(os.path.join(tag_path, "*.pth"))
        if not pth_files:
            continue
            
        # Parse BER from filenames: stegadna_e221_ber0.2463.pth or best_model.pth
        models = []
        for f in pth_files:
            basename = os.path.basename(f)
            if basename == "best_model.pth":
                # We already have a specific best model pointer, prioritize it
                models.append({"path": f, "ber": -1.0}) # Mark as extremely good
                continue
                
            match = re.search(r"_ber(\d+\.\d+)", basename)
            if match:
                ber = float(match.group(1))
                models.append({"path": f, "ber": ber})
            else:
                # Fallback for files without BER in name
                models.append({"path": f, "ber": 1.0})

        # Sort by BER (lowest is better)
        models.sort(key=lambda x: x["ber"])
        
        if not models:
            continue
            
        # The best one
        best = models[0]
        logger.success(f"Best model for {tag}: {os.path.basename(best['path'])} (BER: {best['ber']})")
        
        # Copy to central archive
        archive_name = f"stegadna_{tag}_best.pth"
        shutil.copy2(best["path"], os.path.join(archive_dir, archive_name))
        
        # Keep only the best in the tag folder and delete the rest
        # (Exclude best_model.pth if it exists as it might be a symlink or duplicate we want to keep)
        kept_count = 0
        deleted_count = 0
        
        for m in models:
            m_path = m["path"]
            m_name = os.path.basename(m_path)
            
            # Policy: Keep the absolute best one, plus 'best_model.pth' if it exists separately
            if m_path == best["path"] or m_name == "best_model.pth":
                kept_count += 1
                continue
            else:
                os.remove(m_path)
                deleted_count += 1
        
        logger.info(f"Tag {tag}: Kept {kept_count}, Deleted {deleted_count} redundant checkpoints.")

if __name__ == "__main__":
    cleanup_checkpoints()
