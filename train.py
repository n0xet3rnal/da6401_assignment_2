"""Training entrypoint
"""

import argparse
import os

import torch
import wandb
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


def _should_log_batch(step_idx: int, total_steps: int, interval: int) -> bool:
	if total_steps <= 0:
		return False
	if step_idx == 1 or step_idx == total_steps:
		return True
	return (step_idx % max(1, interval)) == 0


def _dice_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	num_classes = logits.shape[1]
	probs = torch.softmax(logits, dim=1)
	target_oh = torch.nn.functional.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
	intersection = (probs * target_oh).sum(dim=(0, 2, 3))
	denominator = probs.sum(dim=(0, 2, 3)) + target_oh.sum(dim=(0, 2, 3))
	dice = (2.0 * intersection + eps) / (denominator + eps)
	return dice.mean()


def _segmentation_pixel_accuracy(logits: torch.Tensor, target: torch.Tensor) -> tuple[int, int]:
	pred = torch.argmax(logits, dim=1)
	correct = int((pred == target).sum().item())
	total = int(target.numel())
	return correct, total


def _default_num_workers() -> int:
	cpu = os.cpu_count() or 2
	# Aggressive but usually stable on laptops/workstations.
	return max(2, min(8, cpu - 2))


def _make_loaders(root_dir: str, batch_size: int, task: str, num_workers: int, pin_memory: bool, image_size: int):
	train_ds = OxfordIIITPetDataset(root_dir=root_dir, split="train", task=task, image_size=image_size)
	val_ds = OxfordIIITPetDataset(root_dir=root_dir, split="val", task=task, image_size=image_size)
	loader_kwargs = {
		"num_workers": num_workers,
		"pin_memory": pin_memory,
	}
	if num_workers > 0:
		loader_kwargs["persistent_workers"] = True
		loader_kwargs["prefetch_factor"] = 4

	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
	return train_loader, val_loader


def _amp_enabled(args, device: str) -> bool:
	return bool(args.amp and device == "cuda")


def _resolve_pretrained_backbone_path(path: str) -> str:
	if path is None:
		return None
	path = path.strip()
	if path == "":
		return None
	return path if os.path.isfile(path) else None


def _activation_backbone_features(model):
	backbone = getattr(model, "vgg", None)
	if backbone is None:
		backbone = getattr(model, "encoder", None)
	if backbone is None:
		return None
	return getattr(backbone, "features", None)


def _effective_dropout(dropout_p: float, min_dropout_p: float, task_name: str) -> float:
	eff = max(dropout_p, min_dropout_p)
	if eff > dropout_p:
		print(
			f"[{task_name}] Forcing dropout from {dropout_p:.3f} to {eff:.3f} (min_dropout_p={min_dropout_p:.3f})",
			flush=True,
		)
	return eff


def train_task1(args, device):
	effective_dropout_p = _effective_dropout(args.dropout_p, args.min_dropout_p, "Task 1")
	train_loader, val_loader = _make_loaders(
		args.data_root,
		args.batch_size,
		task="classification",
		num_workers=args.num_workers,
		pin_memory=args.pin_memory,
		image_size=args.image_size,
	)
	model = VGG11Classifier(
		num_classes=37,
		dropout_p=effective_dropout_p,
		use_batchnorm=args.use_batchnorm,
	).to(device)
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode="min",
		factor=args.lr_factor,
		patience=args.lr_patience,
		min_lr=args.min_lr,
	)
	use_amp = _amp_enabled(args, device)
	scaler = torch.amp.GradScaler(device, enabled=use_amp)
	log_interval = max(1, args.log_interval)
	features = _activation_backbone_features(model)
	conv_layers = [m for m in features if isinstance(m, torch.nn.Conv2d)] if features is not None else []
	conv3 = conv_layers[2] if len(conv_layers) > 2 else None

	print(f"[Task 1] Classification on {device} | train_batches={len(train_loader)} val_batches={len(val_loader)}", flush=True)

	best_metric = -1.0
	for epoch in range(args.epochs):
		print(f"[Task 1] Epoch {epoch + 1}/{args.epochs} started", flush=True)
		activation_tensor = None
		activation_logged = False
		hook = None

		def _hook_conv3(_, __, out):
			nonlocal activation_tensor
			activation_tensor = out

		if conv3 is not None:
			hook = conv3.register_forward_hook(_hook_conv3)

		model.train()
		train_loss = 0.0
		train_correct = 0
		train_total = 0
		for step, batch in enumerate(train_loader, start=1):
			images = batch["image"].to(device, non_blocking=args.pin_memory)
			labels = batch["label"].to(device, non_blocking=args.pin_memory)

			optimizer.zero_grad(set_to_none=True)
			with torch.amp.autocast(device_type=device, enabled=use_amp):
				logits = model(images)
				loss = criterion(logits, labels)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			train_loss += loss.item() * images.size(0)
			train_correct += (logits.argmax(dim=1) == labels).sum().item()
			train_total += images.size(0)
			if step == 1 and not activation_logged and activation_tensor is not None:
				activation_array = activation_tensor.detach().flatten().cpu().numpy()
				wandb.log({"conv3_activations": wandb.Histogram(activation_array), "epoch": epoch})
				activation_logged = True
				if hook is not None:
					hook.remove()
					hook = None
			if _should_log_batch(step, len(train_loader), log_interval):
				print(
					f"[Task 1][Epoch {epoch + 1}] batch {step}/{len(train_loader)} loss={loss.item():.4f}",
					flush=True,
				)

		if hook is not None:
			hook.remove()

		model.eval()
		val_loss = 0.0
		val_correct = 0
		val_total = 0
		with torch.no_grad():
			for batch in val_loader:
				images = batch["image"].to(device, non_blocking=args.pin_memory)
				labels = batch["label"].to(device, non_blocking=args.pin_memory)
				with torch.amp.autocast(device_type=device, enabled=use_amp):
					logits = model(images)
					loss = criterion(logits, labels)
				val_loss += loss.item() * images.size(0)
				val_correct += (logits.argmax(dim=1) == labels).sum().item()
				val_total += images.size(0)

		train_loss /= max(1, train_total)
		val_loss /= max(1, val_total)
		train_acc = train_correct / max(1, train_total)
		val_acc = val_correct / max(1, val_total)

		log_dict = {
			"epoch": epoch,
			"train/loss": train_loss,
			"train/acc": train_acc,
			"val/loss": val_loss,
			"val/acc": val_acc,
		}

		wandb.log(log_dict)
		print(
			f"[Task 1] Epoch {epoch + 1} done | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
			f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={optimizer.param_groups[0]['lr']:.2e}",
			flush=True,
		)
		scheduler.step(val_loss)

		if val_acc > best_metric:
			best_metric = val_acc
			torch.save(model.state_dict(), "checkpoints/classifier.pth")
			print(f"[Task 1] New best val_acc={val_acc:.4f} -> saved checkpoints/classifier.pth", flush=True)

	if hook is not None:
		hook.remove()


def train_task2(args, device):
	effective_dropout_p = _effective_dropout(args.dropout_p, args.min_dropout_p, "Task 2")
	pretrained_backbone = _resolve_pretrained_backbone_path(args.pretrained_vgg_path)
	if pretrained_backbone is None and args.freeze_backbone != "none":
		print(
			f"[Task 2] Warning: freeze_backbone={args.freeze_backbone} but pretrained backbone "
			f"checkpoint not found at '{args.pretrained_vgg_path}'. Freezing random features can hurt performance.",
			flush=True,
		)
	train_loader, val_loader = _make_loaders(
		args.data_root,
		args.batch_size,
		task="localization",
		num_workers=args.num_workers,
		pin_memory=args.pin_memory,
		image_size=args.image_size,
	)
	model = VGG11Localizer(
		dropout_p=effective_dropout_p,
		use_batchnorm=args.use_batchnorm,
		pretrained_vgg_path=pretrained_backbone,
		freeze_backbone=args.freeze_backbone,
	).to(device)
	mse_loss = torch.nn.MSELoss()
	iou_loss = IoULoss(reduction="mean")
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode="min",
		factor=args.lr_factor,
		patience=args.lr_patience,
		min_lr=args.min_lr,
	)
	use_amp = _amp_enabled(args, device)
	scaler = torch.amp.GradScaler(device, enabled=use_amp)
	log_interval = max(1, args.log_interval)

	print(f"[Task 2] Localization on {device} | train_batches={len(train_loader)} val_batches={len(val_loader)}", flush=True)

	best_metric = -1e9
	for epoch in range(args.epochs):
		print(f"[Task 2] Epoch {epoch + 1}/{args.epochs} started", flush=True)
		model.train()
		train_loss_total = 0.0
		train_iou_total = 0.0
		train_total = 0
		for step, batch in enumerate(train_loader, start=1):
			images = batch["image"].to(device, non_blocking=args.pin_memory)
			bboxes = batch["bbox"].to(device, non_blocking=args.pin_memory)

			optimizer.zero_grad(set_to_none=True)
			with torch.amp.autocast(device_type=device, enabled=use_amp):
				preds = model(images)
				loss_mse = mse_loss(preds, bboxes)
				loss_iou = iou_loss(preds, bboxes)
				loss = loss_mse + loss_iou
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			bs = images.size(0)
			train_loss_total += loss.item() * bs
			train_iou_total += (1.0 - loss_iou.item()) * bs
			train_total += bs
			if _should_log_batch(step, len(train_loader), log_interval):
				print(
					f"[Task 2][Epoch {epoch + 1}] batch {step}/{len(train_loader)} loss={loss.item():.4f} iou={(1.0-loss_iou.item()):.4f}",
					flush=True,
				)

		model.eval()
		val_loss_total = 0.0
		val_iou_total = 0.0
		val_total = 0
		with torch.no_grad():
			for batch in val_loader:
				images = batch["image"].to(device, non_blocking=args.pin_memory)
				bboxes = batch["bbox"].to(device, non_blocking=args.pin_memory)
				with torch.amp.autocast(device_type=device, enabled=use_amp):
					preds = model(images)
					loss_mse = mse_loss(preds, bboxes)
					loss_iou = iou_loss(preds, bboxes)
					loss = loss_mse + loss_iou

				bs = images.size(0)
				val_loss_total += loss.item() * bs
				val_iou_total += (1.0 - loss_iou.item()) * bs
				val_total += bs

		train_loss = train_loss_total / max(1, train_total)
		train_iou = train_iou_total / max(1, train_total)
		val_loss = val_loss_total / max(1, val_total)
		val_iou = val_iou_total / max(1, val_total)

		log_dict = {
			"epoch": epoch,
			"localization/train_loss": train_loss,
			"localization/val_loss": val_loss,
			"localization/val_iou": val_iou,
		}

		wandb.log(log_dict)
		print(
			f"[Task 2] Epoch {epoch + 1} done | train_loss={train_loss:.4f} train_iou={train_iou:.4f} "
			f"val_loss={val_loss:.4f} val_iou={val_iou:.4f} lr={optimizer.param_groups[0]['lr']:.2e}",
			flush=True,
		)
		scheduler.step(val_loss)

		if val_iou > best_metric:
			best_metric = val_iou
			torch.save(model.state_dict(), "checkpoints/localizer.pth")
			print(f"[Task 2] New best val_iou={val_iou:.4f} -> saved checkpoints/localizer.pth", flush=True)


def train_task3(args, device):
	effective_dropout_p = _effective_dropout(args.dropout_p, args.min_dropout_p, "Task 3")
	pretrained_backbone = _resolve_pretrained_backbone_path(args.pretrained_vgg_path)
	if pretrained_backbone is None and args.freeze_backbone != "none":
		print(
			f"[Task 3] Warning: freeze_backbone={args.freeze_backbone} but pretrained backbone "
			f"checkpoint not found at '{args.pretrained_vgg_path}'. Freezing random features can hurt performance.",
			flush=True,
		)
	train_loader, val_loader = _make_loaders(
		args.data_root,
		args.batch_size,
		task="segmentation",
		num_workers=args.num_workers,
		pin_memory=args.pin_memory,
		image_size=args.image_size,
	)
	model = VGG11UNet(
		num_classes=3,
		dropout_p=effective_dropout_p,
		use_batchnorm=args.use_batchnorm,
		pretrained_vgg_path=pretrained_backbone,
		freeze_backbone=args.freeze_backbone,
	).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode="min",
		factor=args.lr_factor,
		patience=args.lr_patience,
		min_lr=args.min_lr,
	)
	use_amp = _amp_enabled(args, device)
	scaler = torch.amp.GradScaler(device, enabled=use_amp)
	log_interval = max(1, args.log_interval)

	print(f"[Task 3] Segmentation on {device} | train_batches={len(train_loader)} val_batches={len(val_loader)}", flush=True)

	best_metric = -1.0
	for epoch in range(args.epochs):
		print(f"[Task 3] Epoch {epoch + 1}/{args.epochs} started", flush=True)
		model.train()
		train_loss_total = 0.0
		train_dice_total = 0.0
		train_total = 0
		for step, batch in enumerate(train_loader, start=1):
			images = batch["image"].to(device, non_blocking=args.pin_memory)
			masks = batch["mask"].to(device, non_blocking=args.pin_memory)

			optimizer.zero_grad(set_to_none=True)
			with torch.amp.autocast(device_type=device, enabled=use_amp):
				logits = model(images)
				loss = model.loss_fn(logits, masks)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			bs = images.size(0)
			train_loss_total += loss.item() * bs
			train_dice_total += _dice_score(logits.detach(), masks).item() * bs
			train_total += bs
			if _should_log_batch(step, len(train_loader), log_interval):
				print(
					f"[Task 3][Epoch {epoch + 1}] batch {step}/{len(train_loader)} loss={loss.item():.4f}",
					flush=True,
				)

		model.eval()
		val_loss_total = 0.0
		val_dice_total = 0.0
		val_correct_total = 0
		val_pixel_total = 0
		val_total = 0
		with torch.no_grad():
			for batch in val_loader:
				images = batch["image"].to(device, non_blocking=args.pin_memory)
				masks = batch["mask"].to(device, non_blocking=args.pin_memory)
				with torch.amp.autocast(device_type=device, enabled=use_amp):
					logits = model(images)
					loss = model.loss_fn(logits, masks)
				correct, pixels = _segmentation_pixel_accuracy(logits, masks)

				bs = images.size(0)
				val_loss_total += loss.item() * bs
				val_dice_total += _dice_score(logits, masks).item() * bs
				val_correct_total += correct
				val_pixel_total += pixels
				val_total += bs

		train_loss = train_loss_total / max(1, train_total)
		train_dice = train_dice_total / max(1, train_total)
		val_loss = val_loss_total / max(1, val_total)
		val_dice = val_dice_total / max(1, val_total)
		val_pixel_acc = val_correct_total / max(1, val_pixel_total)
		log_dict = {
			"epoch": epoch,
			"segmentation/train_loss": train_loss,
			"segmentation/val_loss": val_loss,
			"segmentation/dice_score": val_dice,
			"segmentation/pixel_accuracy": val_pixel_acc,
		}

		wandb.log(log_dict)
		print(
			f"[Task 3] Epoch {epoch + 1} done | train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
			f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_pixel_acc={val_pixel_acc:.4f} "
			f"lr={optimizer.param_groups[0]['lr']:.2e}",
			flush=True,
		)
		scheduler.step(val_loss)

		if val_dice > best_metric:
			best_metric = val_dice
			torch.save(model.state_dict(), "checkpoints/unet.pth")
			print(f"[Task 3] New best val_dice={val_dice:.4f} -> saved checkpoints/unet.pth", flush=True)


def main():
	parser = argparse.ArgumentParser(description="Train multi-task models for Oxford-IIIT Pet")
	parser.add_argument("--task", type=int, required=True, choices=[1, 2, 3], help="Task ID: 1(classification), 2(localization), 3(segmentation)")
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--min_lr", type=float, default=1e-6)
	parser.add_argument("--lr_factor", type=float, default=0.5)
	parser.add_argument("--lr_patience", type=int, default=3)
	parser.add_argument("--weight_decay", type=float, default=1e-4)
	parser.add_argument("--image_size", type=int, default=224)
	parser.add_argument("--num_workers", type=int, default=_default_num_workers())
	parser.add_argument("--log_interval", type=int, default=20)
	parser.add_argument("--pin_memory", action="store_true")
	parser.add_argument("--no_pin_memory", action="store_true")
	parser.add_argument("--amp", action="store_true")
	parser.add_argument("--no_amp", action="store_true")
	parser.add_argument("--data_root", type=str, default="data")
	parser.add_argument("--dropout_p", type=float, default=0.5)
	parser.add_argument("--min_dropout_p", type=float, default=0.2)
	parser.add_argument("--freeze_backbone", type=str, choices=["none", "all", "partial"], default="none")
	parser.add_argument("--pretrained_vgg_path", type=str, default="checkpoints/classifier.pth")
	parser.add_argument("--run_name", type=str, default=None)
	parser.add_argument("--wandb_project", type=str, default="da6401_assignment2")
	parser.add_argument("--use_batchnorm", action="store_true")
	parser.add_argument("--no_batchnorm", action="store_true")
	parser.set_defaults(use_batchnorm=True)
	parser.set_defaults(amp=True)
	args = parser.parse_args()
	if args.no_batchnorm:
		args.use_batchnorm = False
	if args.no_pin_memory:
		args.pin_memory = False
	if args.no_amp:
		args.amp = False

	device = "cuda" if torch.cuda.is_available() else "cpu"
	if device == "cuda":
		torch.backends.cudnn.benchmark = True
	if not args.pin_memory:
		args.pin_memory = device == "cuda"

	default_run_name = f"task_{args.task}"
	wandb.init(project=args.wandb_project, name=(args.run_name if args.run_name is not None else default_run_name), config=vars(args))
	if args.task == 1:
		train_task1(args, device)
	elif args.task == 2:
		train_task2(args, device)
	else:
		train_task3(args, device)
	wandb.finish()


if __name__ == "__main__":
	main()