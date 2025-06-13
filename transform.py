import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional
import numpy as np
import torch
from torch import Tensor
from compressai.transforms.functional import (
    rgb2ycbcr,
    ycbcr2rgb,
    yuv_420_to_444,
    yuv_444_to_420,
)
import subprocess
from PIL import Image
import ffmpeg
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

Frame = Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, ...]]
from rawvideo import RawVideoSequence, VideoFormat

RAWVIDEO_EXTENSIONS = (".yuv",)  # read raw yuv videos for now

def run_cmdline(cmdline: List[Any], logpath: Optional[Path] = None, dry_run: bool = False) -> None:
    cmdline = list(map(str, cmdline))
    print(f"--> Running: {' '.join(cmdline)}", file=sys.stderr)

    if dry_run:
        return

    if logpath is None:
        out = subprocess.check_output(cmdline).decode()
        if out:
            print(out)
        return

    p = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with logpath.open("w") as f:
        if p.stdout is not None:
            for bline in p.stdout:
                line = bline.decode()
                f.write(line)
    p.wait()

def convert_video(rootpath: str):
    rootpath = Path(rootpath)
    if rootpath.suffix in RAWVIDEO_EXTENSIONS:
        return rootpath
    else:
        probe = ffmpeg.probe(str(rootpath)) # obtain the video information
        height, width = probe['streams'][0]['height'], probe['streams'][0]['width']
        bitdepth = probe['streams'][0]['bits_per_raw_sample']
        framerate = int(int(probe['streams'][0]['r_frame_rate'].split('/')[0]) / int(probe['streams'][0]['r_frame_rate'].split('/')[1]))
        filename = f"{rootpath.stem}_{width}x{height}_{framerate}fps_420_{bitdepth}bit_YUV.yuv"
        convert_rootpath = rootpath.parent / filename
        convert_cmd = ["ffmpeg", "-y", "-i", str(rootpath), "-c:v", "rawvideo", "-pixel_format", "yuv420p", f"{str(convert_rootpath)}"]
        run_cmdline(convert_cmd)
        return convert_rootpath

def convert_tensor_to_video(frames, outputdir, filepath, **args: Any):
    max_val = 2**args["bitdepth"] - 1
    num_frames = len(frames)
    bit_n = len(str(num_frames))
    print("Now we are saving a series of png images.")
    for i, frame in enumerate(frames):
        frame = (frame * max_val).clamp(0, max_val).round().squeeze(0).permute(1,2,0).cpu().type(torch.uint8).numpy()
        frame = Image.fromarray(frame).convert('RGB')
        index = str(i).zfill(bit_n)
        frame.save(str(outputdir / f'{index}.png'))

    print("After saving images, we are coverting images to avi video.")
    merge_cmd = ["ffmpeg", "-y", "-r", args["frame_rate"], "-i", f"{str(outputdir)}/%{bit_n}d.png", "-q", 0, f"{str(outputdir)}/{filepath.stem}.yuv"]
    run_cmdline(merge_cmd)
    for f in outputdir.glob("*.png"):
        f.unlink()


# TODO (racapef) duplicate from bench
def to_tensors(
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
    max_value: int = 1,
    device: str = "cpu",
) -> Frame:
    return tuple(
        torch.from_numpy(np.true_divide(c, max_value, dtype=np.float32)).to(device)
        for c in frame
    )

def convert_yuv420_to_rgb(
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray], device: torch.device, max_val: int
) -> Tensor:
    # yuv420 [0, 2**bitdepth-1] to rgb 444 [0, 1] only for now
    out = to_tensors(frame, device=str(device), max_value=max_val)
    out = yuv_420_to_444(
        tuple(c.unsqueeze(0).unsqueeze(0) for c in out), mode="bicubic"  # type: ignore
    )
    return ycbcr2rgb(out)  # type: ignore


def convert_rgb_to_yuv420(frame: Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # yuv420 [0, 2**bitdepth-1] to rgb 444 [0, 1] only for now
    return yuv_444_to_420(rgb2ycbcr(frame), mode="avg_pool")
def pad(x: Tensor, p: int = 2 ** (4 + 2)) -> Tuple[Tensor, Tuple[int, ...]]:
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    padding = (padding_left, padding_right, padding_top, padding_bottom)
    x = F.pad(x, padding, mode="replicate")
    return x, padding


def crop(x: Tensor, padding: Tuple[int, ...]) -> Tensor:
    return F.pad(x, tuple(-p for p in padding))
def compute_metrics_for_frame(
    org_frame: Tensor,
    rec_frame: Tensor,
    device: str = "cpu",
    max_val: int = 255,
) :
    out: Dict[str, Any] = {}

    org_frame = (org_frame * max_val).clamp(0, max_val).round()
    rec_frame = (rec_frame * max_val).clamp(0, max_val).round()
    mse_rgb = (org_frame - rec_frame).pow(2).mean()
    psnr_rgb = 20 * np.log10(max_val) - 10 * torch.log10(mse_rgb)

    ms_ssim_rgb = ms_ssim(org_frame, rec_frame, data_range=max_val)
    print("psnr:",psnr_rgb)
    print("msssim:", ms_ssim_rgb)
sequence1 = "/root/yezhuang/work/video/hall_176x144_15fps_420_8bit_YUV.yuv"
sequence2 = "/root/yezhuang/work/video/300frames/hall/qp_24/rec_hall_176x144_15fps_420_8bit_YUV_24.yuv"
org_seq = RawVideoSequence.from_file(str(sequence1))
rec_seq = RawVideoSequence.from_file(str(sequence2))
max_val = 255
device = torch.device('cuda')

for i in range(1,301):

    x_cur = convert_yuv420_to_rgb(org_seq[i-1], device, max_val)
    x_cur, padding1 = pad(x_cur)
    cur_frame = (crop(x_cur, padding1) * max_val).clamp(0, max_val).round().squeeze(0).permute(1, 2, 0).cpu().type(torch.uint8).numpy()

    # 使用 PIL 保存图像
    Image.fromarray(cur_frame).save(f'./img_path/hall_qcif_{i:05d}.png')
for i in range(1,151) :
    x_rec = convert_yuv420_to_rgb(rec_seq[i-1], device, max_val)
    x_rec, padding2 = pad(x_rec)
    x_rec = x_rec.clamp(0, 1)
    rec_frame = (crop(x_rec, padding1) * max_val).clamp(0, max_val).round().squeeze(0).permute(1, 2, 0).cpu().type(torch.uint8).numpy()
    index = i * 2 - 1
    # 使用 PIL 保存图像
    Image.fromarray(rec_frame).save(f'./de_key_path/hall_qcif_{index:05d}.png')

    #compute_metrics_for_frame(crop(x_cur, padding1), crop(x_rec, padding2), device, max_val)
print("ok")
    #break


