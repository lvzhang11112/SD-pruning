from diffusers1 import DiffusionPipeline, DDPMPipeline, DDIMPipeline, DDIMScheduler, DDPMScheduler
from diffusers1.models import UNet2DModel
import torch_pruning as tp
import torch
import torchvision
from torchvision import transforms
import torchvision
from tqdm import tqdm
import os
from glob import glob
from PIL import Image
import accelerate
import utils
import torch.nn.functional as F
import argparse
import os
import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=None, help="path to an image folder")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--pruning_ratio", type=float, default=0.3)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--pruner", type=str, default='taylor',
                    choices=['taylor', 'random', 'magnitude', 'reinit', 'diff-pruning', 'slide-pruning'])

parser.add_argument("--thr", type=float, default=0.05, help="threshold for diff-pruning")

args = parser.parse_args()

batch_size = args.batch_size
dataset = args.dataset

if __name__ == '__main__':
    import os
    import glob
    from PIL import Image
    from torchvision import transforms
    from torch.utils.data import Dataset

    # loading images for gradient-based pruning
    if args.pruner in ['taylor', 'diff-pruning', 'slide-pruning']:
        dataset = utils.get_dataset(args.dataset)
        print(f"Dataset size: {len(dataset)}")
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True
        )
        import torch_pruning as tp

        clean_images = next(iter(train_dataloader))
        if isinstance(clean_images, (list, tuple)):
            clean_images = clean_images[0]
        clean_images = clean_images.to(args.device)
        noise = torch.randn(clean_images.shape).to(clean_images.device)

    # Loading pretrained model
    print("Loading pretrained model from {}".format(args.model_path))
    pipeline = DDPMPipeline.from_pretrained(args.model_path).to(args.device)
    scheduler = pipeline.scheduler
    model = pipeline.unet.eval()
    if 'cifar' in args.model_path:
        example_inputs = {'sample': torch.randn(1, 3, 32, 32).to(args.device),
                          'timestep': torch.ones((1,)).long().to(args.device)}
    else:
        example_inputs = {'sample': torch.randn(1, 3, 256, 256).to(args.device),
                          'timestep': torch.ones((1,)).long().to(args.device)}

    if args.pruning_ratio > 0:
        if args.pruner == 'taylor':
            imp = tp.importance.TaylorImportance(multivariable=True)  # standard first-order taylor expansion
        elif args.pruner == 'random' or args.pruner == 'reinit':
            imp = tp.importance.RandomImportance()
        elif args.pruner == 'magnitude':
            imp = tp.importance.MagnitudeImportance()
        elif args.pruner == 'diff-pruning':
            imp = tp.importance.TaylorImportance(
                multivariable=False)  # a modified version, estimating the accumulated error of weight removal
        elif args.pruner == 'slide-pruning':
            imp = tp.importance.TaylorImportance(
                multivariable=False)  # a modified version, estimating the accumulated error of weight removal
        else:
            raise NotImplementedError

        ignored_layers = [model.conv_out]
        channel_groups = {}
        # from diffusers.models.attention import
        # for m in model.modules():
        #    if isinstance(m, AttentionBlock):
        #        channel_groups[m.query] = m.num_heads
        #        channel_groups[m.key] = m.num_heads
        #        channel_groups[m.value] = m.num_heads

        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=1,
            channel_groups=channel_groups,
            ch_sparsity=args.pruning_ratio,
            ignored_layers=ignored_layers,
        )

        base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
        model.zero_grad()
        model.eval()
        import random

        if args.pruner in ['taylor', 'diff-pruning']:
            loss_max = 0
            print("Accumulating gradients for pruning...")
            for step_k in tqdm(range(1000)):
                timesteps = (step_k * torch.ones((args.batch_size,), device=clean_images.device)).long()
                noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
                model_output = model(noisy_images, timesteps).sample
                loss = torch.nn.functional.mse_loss(model_output, noise)
                loss.backward()

                if args.pruner == 'diff-pruning':
                    if loss > loss_max: loss_max = loss
                    if loss < loss_max * args.thr: break  # taylor expansion over pruned timesteps ( L_t / L_max > thr )
        if args.pruner in ['slide-pruning']:
            print("Accumulating gradients for pruning with window-based thresholds...")

            total_steps = 1000
            window_size = 100
            num_windows = total_steps // window_size
            all_step_losses = []

            # 第一次遍历记录所有 step 的 loss（前向 + 计算 loss）
            print("Pre-scanning loss for all timesteps...")
            # Scanning Loss Phase
            for step_k in tqdm(range(total_steps), desc="Scanning Loss"):
                timesteps = torch.full((args.batch_size,), step_k, device=clean_images.device, dtype=torch.long)
                noisy_images = scheduler.add_noise(clean_images, noise, timesteps)

                with torch.no_grad():
                    model_output = model(noisy_images, timesteps).sample

                    # loss_diff（MSE）
                    loss_diff_val = F.mse_loss(model_output, noise).item()

                    # loss_disp（Dispersive）
                    pooled_out = F.adaptive_avg_pool2d(model_output, 8)
                    Z = pooled_out.view(pooled_out.size(0), -1)
                    D = torch.cdist(Z, Z, p=2).pow(2)
                    tau = args.tau if hasattr(args, 'tau') else 0.5
                    loss_disp_val = torch.log(torch.mean(torch.exp(-D / tau))).item()

                    lambda_disp = args.lambda_disp if hasattr(args, 'lambda_disp') else 0.3
                    score = loss_diff_val + lambda_disp * loss_disp_val

                all_step_losses.append((step_k, score))
            fixed_thr = args.thr  

            for win_id in range(num_windows):
                window_start = win_id * window_size
                window_end = window_start + window_size
                window_losses = all_step_losses[window_start:window_end]

                # 差分 Δloss
                deltas = [abs(window_losses[i + 1][1] - window_losses[i][1]) for i in range(len(window_losses) - 1)]
                max_delta = max(deltas) if deltas else 0

                valid_backward_steps = 0
                backward_indices = set()

                print(f"\n[Window {win_id + 1}/{num_windows}] Steps {window_start}-{window_end - 1}")

                if win_id < 3:
                    # 前3窗口使用动态阈值
                    window_threshold = max_delta * args.thr
                    print(f" - Δmax = {max_delta:.6f}, Δthr = {window_threshold:.6f} (threshold-based)")

                    for idx in range(len(window_losses)):
                        prev_loss = window_losses[idx - 1][1] if idx > 0 else window_losses[idx][1]
                        delta = abs(window_losses[idx][1] - prev_loss)
                        if delta >= window_threshold:
                            backward_indices.add(idx)

                else:
                    # 后7窗口使用 Top 20% 策略
                    topk = max(int(len(deltas) * 0.2), 1)
                    topk_indices = sorted(range(len(deltas)), key=lambda i: deltas[i], reverse=True)[:topk]
                    backward_indices = {i + 1 for i in topk_indices}  
                    print(f" - Δmax = {max_delta:.6f}, Top-{topk} steps selected (top-k based)")

                for idx in sorted(backward_indices):
                    step_k, _ = window_losses[idx]
                    timesteps = torch.full((args.batch_size,), step_k, device=clean_images.device, dtype=torch.long)
                    noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
                    model_output = model(noisy_images, timesteps).sample
                    loss = torch.nn.functional.mse_loss(model_output, noise)
                    loss.backward()
                    valid_backward_steps += 1

                print(f" - Valid backward steps: {valid_backward_steps}")

        for g in pruner.step(interactive=True):
            g.prune()

        # Update static attributes
        from diffusers.models.resnet import Upsample2D, Downsample2D

        for m in model.modules():
            if isinstance(m, (Upsample2D, Downsample2D)):
                m.channels = m.conv.in_channels
                m.out_channels == m.conv.out_channels

        macs, params = tp.utils.count_ops_and_params(model, example_inputs)
        print(model)
        print("#Params: {:.4f} M => {:.4f} M".format(base_params / 1e6, params / 1e6))
        print("#MACS: {:.4f} G => {:.4f} G".format(base_macs / 1e9, macs / 1e9))
        model.zero_grad()
        del pruner

        if args.pruner == 'reinit':
            def reset_parameters(model):
                for m in model.modules():
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()


            reset_parameters(model)

    pipeline.save_pretrained(args.save_path)
    if args.pruning_ratio > 0:
        os.makedirs(os.path.join(args.save_path, "pruned"), exist_ok=True)
        torch.save(model, os.path.join(args.save_path, "pruned", "unet_pruned.pth"))

    # Sampling images from the pruned model
    pipeline = DDIMPipeline(
        unet=model,
        scheduler=DDIMScheduler.from_pretrained(args.save_path, subfolder="scheduler")
    )
    with torch.no_grad():
        generator = torch.Generator(device=pipeline.device).manual_seed(0)
        pipeline.to("cuda")
        images = pipeline(num_inference_steps=100, batch_size=args.batch_size, generator=generator,
                          output_type="numpy").images
        os.makedirs(os.path.join(args.save_path, 'vis'), exist_ok=True)
        torchvision.utils.save_image(torch.from_numpy(images).permute([0, 3, 1, 2]),
                                     "{}/vis/after_pruning.png".format(args.save_path))

