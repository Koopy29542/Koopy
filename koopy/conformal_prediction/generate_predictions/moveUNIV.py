import os
import shutil

def organize_by_code(base_dir):
    # get all immediate subdirectories (e.g. ['001', '003'])
    codes = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    for fname in os.listdir(base_dir):
        src_path = os.path.join(base_dir, fname)
        # skip directories
        if not os.path.isfile(src_path):
            continue

        for code in codes:
            # if the code appears anywhere in the filename
            if code in fname:
                dest_dir = os.path.join(base_dir, code)
                # make sure the subfolder exists
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, fname)
                # copy instead of move
                shutil.copy2(src_path, dest_path)
                print(f"Copied {fname} → {code}/")
                break  # stop checking other codes once copied

if __name__ == "__main__":
    base_directory = "../lobby2/univ/test"
    organize_by_code(base_directory)
