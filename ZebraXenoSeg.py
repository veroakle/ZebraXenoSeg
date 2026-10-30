"""Segment TIFF images found directly inside the folders given on the command line.

Example:
    python ZebraXenoSeg.py /data/experiment1

The selected folder receives a Segmented subfolder containing one binary PNG
mask and one outlined overlay per TIFF image.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

IMAGE_SIZE = 1152
SCRIPT_FOLDER = Path(__file__).resolve().parent
TORCHSCRIPT_MODEL = SCRIPT_FOLDER / "ZebraXenoSeg_ResNet_v1.0.pt"


def main() -> None:
    # Use the supplied folder, or ./SampleData/ when no argument is provided.
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./SampleData/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not TORCHSCRIPT_MODEL.is_file():
        raise FileNotFoundError(f"TorchScript model not found: {TORCHSCRIPT_MODEL}")
    traced_model = torch.jit.load(str(TORCHSCRIPT_MODEL), map_location=device)
    traced_model.to(device).eval()
    # Only inspect the specified folder itself (no recursive search).
    tiff_files = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    output_folder = folder / "Segmented"
    output_folder.mkdir(exist_ok=True)

    for image_path in tiff_files:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Could not read: {image_path}")
            continue

        resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
        value_range = resized.max() - resized.min()
        if value_range > 0:
            resized = (resized - resized.min()) / value_range
        else:
            resized.fill(0)

        tensor = torch.from_numpy(resized)[None, None].to(device)
        with torch.inference_mode():
            mask = (
                traced_model(tensor)["out"]
                .argmax(1)[0]
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

        mask = cv2.resize(
            mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        output_path = output_folder / f"{image_path.stem}.png"
        cv2.imwrite(str(output_path), mask)
        print(f"Saved: {output_path}")

        foreground = np.where(mask > 0, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness=2)
        overlay_path = output_folder / f"{image_path.stem}_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)
        print(f"Saved: {overlay_path}")


if __name__ == "__main__":
    main()
