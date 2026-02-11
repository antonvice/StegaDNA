import os
import glob

def cleanup():
    # 1. v2-stealth milestones
    v2_dir = "model/checkpoints/v2-stealth"
    v2_milestones = ["e201", "e236"]
    if os.path.exists(v2_dir):
        files = glob.glob(os.path.join(v2_dir, "*.pth"))
        for f in files:
            is_milestone = any(m in f for m in v2_milestones)
            if not is_milestone:
                try:
                    os.remove(f)
                    print(f"Removed: {f}")
                except Exception as e:
                    print(f"Error removing {f}: {e}")

    # 2. v3-analog milestones
    v3_dir = "model/checkpoints/v3-analog"
    # Keep start (237), mid-progress (250, 275), and current peak (300)
    v3_milestones = ["e237", "e250", "e275", "e300"]
    if os.path.exists(v3_dir):
        files = glob.glob(os.path.join(v3_dir, "*.pth"))
        for f in files:
            # Check for milestones and don't delete e30x etc if they are happening now
            is_milestone = any(f"_e{m}_" in f for m in v3_milestones)
            epoch_num = -1
            try:
                # Extract epoch num: stegadna_e300_ber...
                epoch_part = f.split("_e")[1].split("_")[0]
                epoch_num = int(epoch_part)
            except:
                pass
            
            # Keep milestones AND anything newer than 300 (live training)
            if not is_milestone and epoch_num < 300:
                try:
                    os.remove(f)
                    print(f"Removed: {f}")
                except Exception as e:
                    print(f"Error removing {f}: {e}")

if __name__ == "__main__":
    cleanup()
