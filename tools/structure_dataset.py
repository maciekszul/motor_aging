from pathlib import Path
import shutil
import re
import argparse
import sys

def restructure_dataset(base_dir):
    """
    Restructures a dataset by moving nested participant directories to the 
    root level while preserving their internal sub-folder hierarchy.

    Parameters
    ----------
    base_dir: str or pathlib.Path
        The root directory of the dataset to be restructured.
    ----------

    """

    base_dir = Path(base_dir)

    for folder_path in base_dir.rglob("*"):
        if folder_path.is_dir():
            folder_name = folder_path.name
            
            is_sub_prefix = folder_name.startswith("sub-")
            is_cc_id = bool(re.match(r"^CC\d+", folder_name))
            
            if is_sub_prefix or is_cc_id:
                sub_id = f"sub-{folder_name}" if (is_cc_id and not is_sub_prefix) else folder_name
                
                try:
                    relative_parts = folder_path.relative_to(base_dir)
                except ValueError:
                    continue
                
                parts_list = list(relative_parts.parts)
                
                if len(parts_list) == 2 and parts_list[0] == sub_id:
                    continue
                
                parts_list.pop() 
                if not parts_list:
                    continue
                
                new_sub_path = base_dir / sub_id
                for part in parts_list:
                    new_sub_path = new_sub_path / part
                
                if folder_path.resolve() == new_sub_path.resolve():
                    continue

                print(f"Connect: {folder_name} -> Goal: {new_sub_path}")
                
                merge_and_move(folder_path, new_sub_path)

def merge_and_move(src_path: Path, dst_path: Path):
    """
    Merge same subject folders from subfolders (meg, anat, cc700-scored)
    """
    dst_path.mkdir(parents=True, exist_ok=True)
    
    for item in src_path.iterdir():
        target_item = dst_path / item.name
        
        if item.is_dir():
            if target_item.exists() and target_item.is_dir():
                merge_and_move(item, target_item)
            else:
                shutil.move(str(item), str(target_item))
        else:
            if target_item.exists():
                base_name = target_item.stem
                suffix = target_item.suffix
                counter = 1
                while target_item.exists():
                    target_item = target_item.parent / f"{base_name}_dup{counter}{suffix}"
                    counter += 1
                print(f"    Renamed: {target_item.name}")
            
            shutil.move(str(item), str(target_item))
            
    try:
        src_path.rmdir()
    except OSError:
        pass

def main():
    parser = argparse.ArgumentParser(description="Rearrange dataset folder structure based on participant numbers.")
    parser.add_argument("directory", type=str, help="Path to the dataset directory")
    
    args = parser.parse_args()
    restructure_dataset(args.directory)

if __name__ == "__main__":
    main()