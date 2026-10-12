import zipfile
from pathlib import Path
from typing import List


data_dir = Path("car_image_dataset/data1a")

def extract_dataset():
    with zipfile.ZipFile("archive.zip", "r") as zip_ref:
        zip_ref.extractall(data_dir)
        
     
        
train_damage_dir = Path(data_dir, "training/00-damage")
train_whole_dir = Path(data_dir, "training/01-whole")
validation_damage_dir = Path(data_dir, "validation/00-damage")
validation_whole_dir = Path(data_dir, "validation/01-whole")

path_list = [train_whole_dir, train_damage_dir, validation_whole_dir, validation_damage_dir]

def clean_dataset(path_list: List[Path]) -> None:
    """
    Standardize file extensions to uppercase '.JPEG' within the given directories.

    Iterate through each directory in path_list and rename files whose
    extension is not exactly '.JPEG'.

    Args:
        path_list (List[Path]): A list of directory paths containing image files.

    Returns:
        None: Renames files in place without returning a value.

    Raises:
        FileNotFoundError: If a specified directory or file does not exist.
        PermissionError: If the process lacks permission to rename a file.
        OSError: If a file cannot be renamed due to an operating-system error.

    Notes:
        - Only the file extension is changed; image data is not converted.
        - The function currently processes all directory entries, not just images.
        - Renaming extensions does not guarantee that files contain valid JPEG data.
    """
    for path in path_list:
        image_path_list = list(path.iterdir())
        for image_path in image_path_list:
            suf = image_path.suffix
            if suf != ".JPEG":
                Path.rename(image_path, path / image_path.with_suffix(".JPEG").name)
                

extract_dataset()
clean_dataset(path_list)