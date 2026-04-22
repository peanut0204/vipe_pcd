import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm

import utils


REGION_HEIGHT_RATIOS = [
    0.01,
    0.025,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/360-videos-new",
        help="path to video data_root to process",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./filter_results/static_region_detection",
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


def static_region_detection(video_path, keyframe_rate, region_height_ratios):
    kf_indices, total_frames, kf_fps, fps = utils.compute_keyframes(
        video_path, keyframe_rate
    )

    # (num_frames, H, W, 3)
    frames_rgb = utils.read_video(
        video_path,
        kf_indices,
        return_float=False,
        resize_kwargs={"width": 960, "height": 480},
    )

    num_frames, height = frames_rgb.shape[:2]

    result = {
        "mse_top": [],
        "mse_bottom": [],
    }

    for region_height_ratio in region_height_ratios:
        region_height = int(region_height_ratio * height)

        mse_top_all = []
        mse_bottom_all = []
        for i in range(num_frames - 1):
            fm1 = frames_rgb[i]
            fm2 = frames_rgb[i + 1]

            mse_top = ((fm1[:region_height] - fm2[:region_height]) ** 2).mean()
            mse_bottom = (
                (fm1[-region_height:] - fm2[-region_height:]) ** 2
            ).mean()

            mse_top_all.append(mse_top)
            mse_bottom_all.append(mse_bottom)

        result["mse_top"].append(np.mean(mse_top_all))
        result["mse_bottom"].append(np.mean(mse_bottom_all))

    # print(f"{video_path}")
    # print(f"\t{result['mse_top']}\n\t{result['mse_bottom']}")
    return result


# # True
# # Bottom
# print("*" * 5 + "[BOTTOM]" + "*" * 5)
# static_region_detection("/share/ma/scratch/tao/360video-split/People_Blogs/VXojp7t0d7o/0000009.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/Travel_Events/kcJp9mudE8o/0000006.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/Travel_Events/kWxQ54JBwF4/0000025.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/Film_Animation/gsnxau8cyVc/0000061.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/Entertainment/yKPhfl4xo4E/0000017.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/Pets_Animals/YYCg-Rs7VNo/0000006.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)

# # Top
# print("\n\n" + "*" * 5 + "[TOP]" + "*" * 5)
# static_region_detection("/share/ma/scratch/tao/360video-split/Entertainment/jzADvPNVRyM/0000705.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/Entertainment/jzADvPNVRyM/0000024.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/People_Blogs/DXEex5cwEf8/0000005.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/Science_Technology/0bly6qaU0V4/0000016.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/Entertainment/jzADvPNVRyM/0000027.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)

# # False
# print("\n\n" + "*" * 5 + "[NORMAL]" + "*" * 5)
# static_region_detection("/share/ma/scratch/tao/360video-split/Travel_Events/Tzdhw5tj9C0/0000018.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/Sports/S7b3EQWlE2U/0000076.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/Nonprofits_Activism/sJqxCKQ1v9o/0000006.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/People_Blogs/kPG3ZBqbzYI/0000015.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/People_Blogs/tcksMIus4U4/0000002.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/Travel_Events/8gBBurvGvtA/0000035.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/Autos_Vehicles/1tq6ANaFlTA/0000056.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/People_Blogs/UMd1xeDE4N4/0000010.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/People_Blogs/9NADgdGvx7E/0000007.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/Sports/jmRHIdjHfvM/0000002.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/Travel_Events/CF_buV7GmiQ/0000006.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/Education/0uUfcVQ-Uz8/0000009.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/Travel_Events/aUDb5fQr58I/0000074.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/Music/BGQNiKEXrlE/0000003.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/Sports/vPOYWt7JQBo/0000002.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/Comedy/VmEWNRGxfps/0000046.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/News_Politics/rsm5x0TMaGo/0000046.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/People_Blogs/0wwHP5egDuk/0000008.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
# static_region_detection("/share/ma/scratch/tao/360video-split/Travel_Events/1f-WnJjM3As/0000006.mp4", keyframe_rate=1, region_height_ratios=REGION_HEIGHT_RATIOS)
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
            result = static_region_detection(
                file_path,
                keyframe_rate=1,
                region_height_ratios=REGION_HEIGHT_RATIOS,
            )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        else:
            result = np.stack([result["mse_top"], result["mse_bottom"]])
            with open(save_path, "w") as f:
                f.write(",".join(map(str, REGION_HEIGHT_RATIOS)) + "\n")
                for row in result:
                    f.write(",".join(map(lambda x: f"{x:.4f}", row)) + "\n")


if __name__ == "__main__":
    main()
