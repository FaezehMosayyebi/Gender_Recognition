from pathlib import Path
import imghdr
import os
import sys


def file_format_checker(data_dir: str) -> None:
    image_extensions = [".png", ".jpg"]  # add there all your images file extensions

    img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
    for filepath in Path(data_dir).rglob("*"):
        if filepath.suffix.lower() in image_extensions:
            img_type = imghdr.what(filepath)
            if img_type is None:
                print(f"{filepath} is not an image")
            elif img_type not in img_type_accepted_by_tf:
                print(f"{filepath} is a {img_type}, not accepted by TensorFlow")


def path_valiadtor(path: str) -> bool:
    if not os.path.exists(path):
        print(f"{path} is not valid")
        sys.exit(1)
    else:
        return True


def directory_maker(path: str) -> None:
    if not os.path.exists(path):
        os.mkdir(path)
        print(f"A directory made at {path}")
