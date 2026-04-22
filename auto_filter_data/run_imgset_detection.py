import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
        "--output_root", type=str, default="./filter_results/imgset_detection"
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


def image_set_detection(video_path, keyframe_rate):
    kf_indices, total_frames, kf_fps, fps = \
        utils.compute_keyframes(video_path, keyframe_rate)

    # (num_frames, H, W, 3)
    frames = utils.read_video(
        video_path,
        kf_indices,
        return_float=False,
        resize_kwargs={"width": 960, "height": 480},
    )

    num_frames, _, _ = frames.shape[:3]

    similarity = []
    for i in range(num_frames - 1):
        fm1 = frames[i]
        fm2 = frames[i + 1]

        mse = ((fm1 - fm2) ** 2).mean()
        similarity.append(mse)

    min_similarity = np.min(similarity)
    # print(f"{video_path}:\n\t{min_similarity:.6f}")
    # print(f"\n\t{similarity}")
    return min_similarity


# # # True
# image_set_detection("/share/ma/scratch/tao/360video-split/Music/zyXDgwOoeO0/0000002.mp4")
# image_set_detection("/share/ma/scratch/tao/360video-split/People_Blogs/tgE-xr5aaHE/0000007.mp4")
# image_set_detection("/share/ma/scratch/tao/360video-split/Science_Technology/yXXp1jvpfrQ/0000002.mp4")
# image_set_detection("/share/ma/scratch/tao/360video-split/Howto_Style/X-gNB07NhS4/0000004.mp4")

# # # False
# image_set_detection("/share/ma/scratch/tao/360video-split/Sports/isat8KYEX7A/0000217.mp4")
# image_set_detection("/share/ma/scratch/tao/360video-split/Travel_Events/JWJYReemJ7s/0000012.mp4")
# image_set_detection("/share/ma/scratch/tao/360video-split/Sports/sbJxb12tEZo/0000017.mp4")
# image_set_detection("/share/ma/scratch/tao/360video-split/Entertainment/GZL0cIXQSFI/0000182.mp4")
# image_set_detection("/share/ma/scratch/tao/360video-split/Autos_Vehicles/FpjE76w44Os/0000122.mp4")
# image_set_detection("/share/ma/scratch/tao/360video-split/Travel_Events/V5bfini3g6c/0000007.mp4")
# image_set_detection("/share/ma/scratch/tao/360video-split/People_Blogs/kV1Xt1vDwmE/0000030.mp4")
# image_set_detection("/share/ma/scratch/tao/360video-split/Entertainment/g38EPjD0mAM/0000008.mp4")
# stop()


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
            result = image_set_detection(file_path, keyframe_rate=1)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        else:
            with open(save_path, "w") as f:
                f.write(f"{result:.4f}")


if __name__ == "__main__":
    main()
