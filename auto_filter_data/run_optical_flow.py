import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from tqdm import tqdm

import utils

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


DEBUG = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/360-videos-new",
        help="path to video data_root to process",
    )
    parser.add_argument(
        "--output_root", type=str, default="./filter_results/info_extraction"
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


def visualize(output_path, imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = torchvision.transforms.functional.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    plt.savefig(output_path, dpi=500)
    plt.close()


def read_keyframes(video_path, kf_fps):
    kf_indices, _, _, _ = utils.compute_keyframes(video_path, kf_fps)
    frames = utils.read_video(
        video_path,
        indices=kf_indices,
        return_float=False,
        channel_first=True,
    )
    frames = torch.from_numpy(frames)
    return frames


def create_pairs(frames):
    img1_batch = frames[:-1]
    img2_batch = frames[1:]
    return img1_batch, img2_batch


def preprocess(batch, size):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ConvertImageDtype(torch.float32),
            torchvision.transforms.Normalize(
                mean=0.5, std=0.5
            ),  # map [0, 1] into [-1, 1]
            torchvision.transforms.Resize(size=size, antialias=False),
        ]
    )
    batch = transforms(batch)
    return batch


def load_raft(device):
    model = torchvision.models.optical_flow.raft_large(
        pretrained=True, progress=False
    ).to(device)
    model = model.eval()
    return model


@torch.no_grad()
def compute_flow(batch, model):
    img1_batch, img2_batch = batch
    list_of_flows = model(img1_batch, img2_batch)
    predicted_flows = list_of_flows[-1]
    return predicted_flows


class FlowDataset(torch.utils.data.Dataset):
    def __init__(self, video_paths, kf_fps, im_size=(128, 256), max_frames=-1):
        self.paths = video_paths
        self.kf_fps = kf_fps
        self.im_size = im_size
        self.max_frames = max_frames

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        video_path = self.paths[idx]
        frames = read_keyframes(video_path, self.kf_fps)
        frames = frames[
            : len(frames) if self.max_frames == -1 else self.max_frames
        ]
        img1, img2 = create_pairs(frames)
        img1 = preprocess(img1, self.im_size)
        img2 = preprocess(img2, self.im_size)
        return img1, img2


# def collate_fn(batch):
#     indices = torch.cat([idx for _, _, idx in batch], dim=0)
#     img1_batch = torch.cat([img1 for img1, _, _ in batch], dim=0)
#     img2_batch = torch.cat([img2 for _, img2, _ in batch], dim=0)
#     return img1_batch, img2_batch, indices


def undistort_flow_magnitude(flows_mag):
    # flows_mag: (N x H x W)
    height, width = flows_mag.shape[-2:]
    latitude = torch.deg2rad(torch.linspace(90, -90, height))
    latitude = latitude.unsqueeze(-1).expand(-1, width)
    weight = torch.cos(latitude).to(flows_mag.device)
    # weight = torch.ones_like(flows_mag[0], dtype=torch.float32)
    # for h in range(height):
    # weight[h] = torch.cos(latitude[h])
    return weight.unsqueeze(0) * flows_mag


def compute_stats(flows_mag):
    flows_mag = flows_mag.view(flows_mag.shape[0], -1)
    mag_min = flows_mag.min(dim=-1).values.mean()
    mag_max = flows_mag.max(dim=-1).values.mean()
    mag_avg = flows_mag.mean()
    mag_q05 = flows_mag.quantile(0.05, dim=-1).mean()
    mag_q10 = flows_mag.quantile(0.10, dim=-1).mean()
    mag_q25 = flows_mag.quantile(0.25, dim=-1).mean()
    mag_q50 = flows_mag.quantile(0.50, dim=-1).mean()
    mag_q75 = flows_mag.quantile(0.75, dim=-1).mean()
    mag_q90 = flows_mag.quantile(0.90, dim=-1).mean()
    mag_q95 = flows_mag.quantile(0.95, dim=-1).mean()
    return {
        "min": mag_min.item(),
        "max": mag_max.item(),
        "avg": mag_avg.item(),
        "q05": mag_q05.item(),
        "q10": mag_q10.item(),
        "q25": mag_q25.item(),
        "q50": mag_q50.item(),
        "q75": mag_q75.item(),
        "q90": mag_q90.item(),
        "q95": mag_q95.item(),
    }


def verify_optical_flow(image_a, image_b, flow):
    """Flow: a -> b"""
    # Ensure inputs are in the correct format
    assert image_a.dim() == 4, "image_a must be 4D tensor (B x C x H x W)"
    assert image_b.dim() == 4, "image_b must be 4D tensor (B x C x H x W)"
    assert flow.dim() == 4, "flow must be 4D tensor (B x 2 x H x W)"

    batch_size, _, height, width = image_a.shape

    # Create base grid (H x W)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=image_b.device),
        torch.linspace(-1, 1, width, device=image_b.device),
        indexing="ij",
    )

    # Normalize flow to match grid sampling coordinate system
    normalized_flow_x = flow[:, 0, :, :] / width * 2  # [-1, 1]
    normalized_flow_y = flow[:, 1, :, :] / height * 2  # [-1, 1]

    # Add flow to base grid (B x H x W)
    grid_x = (
        grid_x.view(1, height, width).expand(batch_size, -1, -1)
        + normalized_flow_x
    )
    grid_y = (
        grid_y.view(1, height, width).expand(batch_size, -1, -1)
        + normalized_flow_y
    )

    # Combine into sampling grid (B x H x W x 2)
    sampling_grid = torch.stack([grid_x, grid_y], dim=-1)

    # Warp image b to a using grid sampling
    warped_image = torch.nn.functional.grid_sample(
        image_b,
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    # Create a mask for valid pixels
    # Pixels are valid if they are within the image bounds after flow mapping
    # (B x 1 x H x W)
    valid_mask = (
        ((grid_x >= -1) & (grid_x <= 1) & (grid_y >= -1) & (grid_y <= 1))
        .float()
        .unsqueeze(1)
    )

    # Compute masked MSE
    # Only compute error for pixels that can be mapped
    diff = (warped_image - image_a) ** 2
    masked_diff = diff * valid_mask

    # Compute MSE only for valid pixels
    valid_pixel_count = valid_mask.sum()
    mse = masked_diff.sum() / (valid_pixel_count + 1e-8)
    return mse, valid_mask, warped_image


if __name__ == "__main__":
    args = parse_args()
    filelist = utils.read_video_list(args.video_list, args.start, args.end)
    filelist = [
        Path(args.data_root) / Path(*x.split(",")).with_suffix(".mp4")
        for x in filelist
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_raft(device)

    # ******* Hyperparameters *******
    kf_fps = 1
    n_videos_per_batch = 1
    num_workers = 8
    pin_memory = True

    # MSE: (128, 256): 0.130
    # MSE: (256, 512): 0.106
    # video_paths = [Path("pexelscom_pavel_danilyuk_basketball_hd.mp4")]
    im_size = (256, 512)
    # im_size = (128, 256)
    max_frames = 10
    # *******************************

    dataset = FlowDataset(
        filelist,
        kf_fps,
        im_size=im_size,
        max_frames=max_frames,
    )
    print(f"Run RAFT for {len(dataset)} videos")

    for i, file_path in tqdm(enumerate(filelist), total=len(filelist)):
        save_path = Path(args.output_root) / file_path.relative_to(
            args.data_root
        )
        save_path = save_path.with_suffix(".json")
        if save_path.exists():
            print(f"Skip {save_path} since it already exists")
            continue

        save_path.parent.mkdir(parents=True, exist_ok=True)

        # (N, 3, H, W)
        try:
            img1, img2 = dataset[i]
        except Exception as e:
            print(f"Failed to read {file_path}")
            continue

        if img1.shape[0] == 0:
            print(f"Skip {file_path} since it has no frames")
            continue

        img1, img2 = (
            img1.to(device, non_blocking=True),
            img2.to(device, non_blocking=True),
        )
        try:
            # (N, 2, H, W)
            flows = compute_flow((img1, img2), model)
        except Exception as e:
            print(f"Failed to run RAFT for {file_path}, use zeros instead")
            # (N, 2, H, W)
            flows = torch.zeros_like(img1)[:, :2]

        # (N, H, W)
        flows_mag = torch.linalg.vector_norm(flows, dim=1)
        flows_mag = undistort_flow_magnitude(flows_mag)
        stats = compute_stats(flows_mag)

        with open(save_path, "w") as f:
            json.dump(stats, f, indent=2)

        if DEBUG:
            mse, valid_mask, warped_img_2_to_1 = verify_optical_flow(
                img1, img2, flows
            )
            print(f"MSE: {mse.item()}")
            normalize_fn = lambda x: (x + 1) / 2
            for j, (img1_j, img2_1_j, img2_j, flow_j) in enumerate(
                zip(img1, warped_img_2_to_1, img2, flows)
            ):
                visualize(
                    save_path.parent / f"{save_path.stem}_{j}.png",
                    [
                        [normalize_fn(img1_j)],
                        [normalize_fn(img2_1_j)],
                        [normalize_fn(img2_j)],
                        [torchvision.utils.flow_to_image(flow_j)],
                    ],
                )
