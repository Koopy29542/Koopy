import os
import shutil

def organize_by_code(base_dir):

    codes = [d for d in os.listdir(base_dir)
             if os.path.isdir(os.path.join(base_dir, d))]

    for fname in os.listdir(base_dir):
        src_path = os.path.join(base_dir, fname)
        if not os.path.isfile(src_path):
            continue

        for code in codes:
            if fname.startswith(code):
                dest_dir = os.path.join(base_dir, code)
                dest_path = os.path.join(dest_dir, fname)
                shutil.move(src_path, dest_path)
                print(f"Moved {fname} → {code}/")
                break

if __name__ == "__main__":
    base_directory = "../lobby2/univ/test"
    organize_by_code(base_directory)
