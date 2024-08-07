import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

# Configuration for CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# Enable mixed precision
torch.autocast(device_type='cuda', dtype=torch.bfloat16).__enter__()

# Enable TensorFloat-32 for improved performance on suitable hardware
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Paths and configurations
SAM2_CHECKPOINT = "D:\\personal-projects\\py-repos\\segment-anything-2\\checkpoints\\sam2_l.pt"
MODEL_CFG = "D:\\personal-projects\\py-repos\\segment-anything-2\\sam2_configs\\sam2_hiera_l.yaml"
VIDEO_DIR = "D:\\personal-projects\\py-repos\\segment-anything-2\\videos"

# Get sorted frame names
frame_names = sorted(
    [p for p in os.listdir(VIDEO_DIR) if os.path.splitext(p)[-1].lower() in [".jpg", ".png", ".jpeg"]],
    key=lambda x: int(os.path.splitext(x)[0])
)

# Initialize the predictor
predictor = build_sam2_video_predictor(MODEL_CFG, SAM2_CHECKPOINT)

def show_mask(mask, ax, ob_id=None, random_color=False):
    """Show mask on the given axis."""
    color = np.concatenate([np.random.random(3), [0.6]], axis=0) if random_color else np.array([*plt.get_cmap('tab10')(0 if ob_id is None else ob_id)[:3], 0.6])
    mask_image = mask.reshape(*mask.shape[-2:], 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    """Show points on the given axis."""
    ax.scatter(coords[labels == 1, 0], coords[labels == 1, 1], color='green', s=marker_size, marker='*', edgecolors='white', linewidths=1.20)
    ax.scatter(coords[labels == 0, 0], coords[labels == 0, 1], color='red', s=marker_size, marker='*', edgecolors='white', linewidths=1.20)

def main():
    frame_idx = 0

    # Display initial frame
    plt.figure(figsize=(12, 8))
    plt.title(f"Frame {frame_idx}")
    plt.imshow(Image.open(os.path.join(VIDEO_DIR, frame_names[frame_idx])))
    plt.show()
    input("Press Enter to continue")

    # Initialize inference state
    inference_state = predictor.init_state(video_path=VIDEO_DIR)
    predictor.reset_state(inference_state)

    # Annotation details
    ann_frame_idx = 0
    ann_object_id = 1
    points = np.array([[601, 360], [630, 360]], dtype=np.float32)
    labels = np.array([1, 1], dtype=np.int32)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_object_id,
        points=points,
        labels=labels
    )

    # Display annotated frame
    plt.figure(figsize=(12, 8))
    plt.title(f"Frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(VIDEO_DIR, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), ob_id=out_obj_ids[0])
    plt.show()
    input("Press Enter to continue")

    # Process and save video segments
    video_segments = {
        out_frame_idx: {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state)
    }

    vis_frame_stride = 1
    plt.close('all')

    # Visualize and save frames
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"Frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(VIDEO_DIR, frame_names[out_frame_idx])), animated=True)
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), ob_id=out_obj_id)
        plt.savefig(f"output/s{out_frame_idx}.png")

if __name__ == "__main__":
    main()
