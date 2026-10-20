import zipfile
from pathlib import Path
from typing import List
import json
import pandas as pd



def extract_dataset(path: Path, zip_file: Path) -> None:
    """
    Extract a ZIP archive into the specified directory.

    Args:
        path (Path): The destination directory for the extracted files.
        zip_file (Path): The path to the ZIP archive to extract.

    Returns:
        None
    """
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(path)
        
        
class DataCleaning:
    """Clean COCO annotations and prepare labels for car damage classification."""

    def __init__(self, file_path: str) -> None:
        """
        Load COCO annotations from a JSON file.

        Args:
            file_path (str): Path to the COCO JSON annotation file.
        """
        self.file = open(file_path, "r", encoding="utf-8")
        try:
            self.data = json.load(self.file)
        finally:
            self.file.close()

    def clean_data(self) -> pd.DataFrame:
        """
        Convert COCO annotations into binary image-level labels.

        Returns:
            pd.DataFrame: A DataFrame indexed by image filename, with six
            damage-category columns containing binary labels (0 or 1).
        """
        categories = self.data["categories"]
        images = self.data["images"]
        annotations = self.data["annotations"]

        categories_df = pd.DataFrame(categories).rename(columns={"name": "category"})
        images_df = pd.DataFrame(images)
        annotations_df = pd.DataFrame(annotations)[["image_id", "category_id"]]

        df = (annotations_df.merge(
            categories_df,
            left_on="category_id",
            right_on="id"
        )).merge(
            images_df,
            left_on="image_id",
            right_on="id"
        )[["category", "file_name"]]

        labels_df = df.pivot_table(
            index="file_name",
            columns="category",
            aggfunc="size",
            fill_value=0
        )
        labels_df = (labels_df > 0).astype(int)

        categories = [
            "dent",
            "scratch",
            "crack",
            "glass shatter",
            "lamp broken",
            "tire flat"
        ]
        labels_df = labels_df.reindex(columns=categories, fill_value=0)

        return labels_df

    def save_data(self, filepath: str) -> None:
        """
        Save the cleaned image-level labels to a CSV file.

        Args:
            filepath (str): Destination path for the CSV file.

        Returns:
            None
        """
        self.clean_data().to_csv(filepath)