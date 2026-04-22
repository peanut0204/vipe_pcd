import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from tqdm import tqdm

import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/360-videos-new",
        help="path to video data_root to process",
    )
    parser.add_argument(
        "--output_root", type=str, default="./filter_results/dual_fisheye_detection"
    )
    parser.add_argument(
        "--video_list",
        type=str,
        # default="metadata/video_list_benchmark_500.csv",
        default="video_list.csv",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    args = parser.parse_args()
    return args


def dual_fisheye_detection(
    video_path, num_frames=4, radius_tol_ratio=0.05, thd_tol_ratio=0.05
):
    info = utils.get_video_info(video_path)
    total_frames = info["total_frames"]

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    # (num_frames, H, W, 3)
    frames_rgb = utils.read_video(
        video_path,
        indices,
        return_float=False,
        resize_kwargs={"width": 960, "height": 480},
    )

    num_frames, height, width = frames_rgb.shape[:3]

    dual_circle_count = 0
    for i in range(num_frames):
        gray = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2GRAY)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        # Use Hough Circle detection
        circles = cv2.HoughCircles(
            blurred,
            # gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=width // 2,  # Minimum distance between circles
            param1=50,
            param2=30,
            minRadius=int(
                height * (0.5 - radius_tol_ratio)
            ),  # Minimum radius (30% of frame height)
            maxRadius=int(
                height * (0.5 + radius_tol_ratio)
            ),  # Maximum radius (60% of frame height)
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            # Check if we found exactly 2 circles
            if len(circles) == 2:
                # Verify the circles are side by side
                x1, y1, r1 = circles[0]
                x2, y2, r2 = circles[1]

                tol_px = height * thd_tol_ratio
                # Almost same height
                height_cond = abs(y1 - y2) < tol_px
                # # Distance is almost 0.5 * width
                width_cond = abs(abs(x1 - x2) - 0.5 * width) < tol_px
                # Both radius should be close to 0.5 * height
                radius_cond = (abs(r1 - 0.5 * height) < tol_px) and (
                    abs(r2 - 0.5 * height) < tol_px
                )

                if height_cond and width_cond and radius_cond:
                    dual_circle_count += 1

    pred_score = dual_circle_count / num_frames
    # print(f"{video_path}:\n\t{pred_score:.4f}")
    return pred_score


# # True
# dual_fisheye_detection("/share/ma/scratch/tao/360video-split/Travel_Events/Y3Sfs4JK6Pg/0000052.mp4")
# dual_fisheye_detection("/share/ma/scratch/tao/360video-split/Science_Technology/72_74aYFUpA/0000011.mp4")
# dual_fisheye_detection("/share/ma/scratch/tao/360video-split/People_Blogs/F6RzeCwDGAY/0000010.mp4")

# # False
# dual_fisheye_detection("/share/ma/scratch/tao/360video-split/Autos_Vehicles/-0J6OPxTq0A/0000000.mp4")
# dual_fisheye_detection("/share/ma/scratch/tao/360video-split/Autos_Vehicles/0lWcnLkuhpo/0000003.mp4")
# dual_fisheye_detection("/share/ma/scratch/tao/360video-split/Autos_Vehicles/0pExDTf1sUM/0000001.mp4")
# dual_fisheye_detection("/share/ma/scratch/tao/360video-split/Autos_Vehicles/0SNHBmQXSQ0/0000011.mp4")
# dual_fisheye_detection("/share/ma/scratch/tao/360video-split/Music/tjYUvwW7LNE/0000036.mp4")
# dual_fisheye_detection("/share/ma/scratch/tao/360video-split/People_Blogs/DsmJdlmtiaQ/0000008.mp4")
# dual_fisheye_detection("/share/ma/scratch/tao/360video-split/Travel_Events/Zk1fHPhaABo/0000024.mp4")
# dual_fisheye_detection("/share/ma/scratch/tao/360video-split/Travel_Events/YvOS82C5gks/0000040.mp4")
# dual_fisheye_detection("/share/ma/scratch/tao/360video-split/People_Blogs/yIH3m7VP_2g/0000006.mp4")


def main():
    args = parse_args()
    filelist = utils.read_video_list(args.video_list, args.start, args.end)
    print(f"Found {len(filelist)} total videos to process.")

    undo_list = []
    for file in filelist:
        category, video_id, clip_id = (
            file.split(",")[0],
            file.split(",")[1],
            file.split(",")[2].split(".")[0],
        )
        file_path = Path(
            args.data_root, category, video_id, clip_id
        ).with_suffix(".mp4")

        save_path = Path(args.output_root) / file_path.relative_to(
            args.data_root
        )
        save_path = save_path.with_suffix(".txt")
        if not save_path.exists():
            undo_list.append(file)

    print(f"Found {len(undo_list)} undo videos to process.")
    filelist = undo_list

    for file in tqdm(filelist):
        category, video_id, clip_id = (
            file.split(",")[0],
            file.split(",")[1],
            file.split(",")[2].split(".")[0],
        )

        file_path = Path(
            args.data_root, category, video_id, clip_id
        ).with_suffix(".mp4")

        if not file_path.exists():
            print(f"Video {file_path} does not exist.")
            continue

        save_path = Path(args.output_root) / file_path.relative_to(
            args.data_root
        )
        save_path = save_path.with_suffix(".txt")
        if save_path.exists():
            print(f"Skip {save_path} since it already exists")
            continue

        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = dual_fisheye_detection(file_path, num_frames=4)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        else:
            with open(save_path, "w") as f:
                f.write(f"{result:.4f}")


if __name__ == "__main__":
    main()
