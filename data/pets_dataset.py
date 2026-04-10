"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


def _parse_tag(xml_text: str, tag: str) -> float:
    start_token = f"<{tag}>"
    end_token = f"</{tag}>"
    start = xml_text.find(start_token)
    end = xml_text.find(end_token)
    if start == -1 or end == -1:
        return 0.0
    value_str = xml_text[start + len(start_token) : end].strip()
    return float(value_str)


def _breed_from_image_id(image_id: str) -> str:
    return image_id.rsplit("_", 1)[0]


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""

    def __init__(
        self,
        root_dir: str = "data",
        split: str = "train",
        task: str = "multitask",
        image_size: int = 224,
        val_ratio: float = 0.1,
        normalize: bool = True,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of {'train', 'val', 'test'}")
        if task not in {"classification", "localization", "segmentation", "multitask"}:
            raise ValueError("task must be one of {'classification', 'localization', 'segmentation', 'multitask'}")

        self.root_dir = root_dir.rstrip("/")
        self.split = split
        self.task = task
        self.image_size = image_size
        self.image_dir = f"{self.root_dir}/images"
        self.annotation_dir = f"{self.root_dir}/annotations"
        self.trimap_dir = f"{self.annotation_dir}/trimaps"
        self.xml_dir = f"{self.annotation_dir}/xmls"

        self.need_label = task in {"classification", "multitask"}
        self.need_bbox = task in {"localization", "multitask"}
        self.need_mask = task in {"segmentation", "multitask"}

        self.label_to_index = self._build_label_index()
        self.image_ids = self._build_split_ids(val_ratio=val_ratio)

        tfs = [A.Resize(height=image_size, width=image_size)]

        # Strong augmentation only for classification training.
        # For localization/segmentation, targets are generated in image coordinates,
        # so keep geometric transforms minimal unless target transforms are added.
        if self.split == "train" and self.task == "classification":
            tfs.extend(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
                ]
            )

        if normalize:
            tfs.append(
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=1.0,
                )
            )
        self.transforms = A.Compose(tfs)

    def _read_split_file(self, split_file: str):
        ids = []
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token = line.split()[0]
                ids.append(token)
        return ids

    def _build_label_index(self):
        list_file = f"{self.annotation_dir}/list.txt"
        breeds = []
        with open(list_file, "r") as f:
            for line in f:
                line = line.strip()
                if (not line) or line.startswith("#"):
                    continue
                image_id = line.split()[0]
                breeds.append(_breed_from_image_id(image_id))
        unique_breeds = sorted(set(breeds))
        return {breed: i for i, breed in enumerate(unique_breeds)}

    def _build_split_ids(self, val_ratio: float):
        trainval_file = f"{self.annotation_dir}/trainval.txt"
        test_file = f"{self.annotation_dir}/test.txt"

        trainval_ids = self._read_split_file(trainval_file)
        test_ids = sorted(self._read_split_file(test_file))

        if self.split == "test":
            selected = test_ids
        else:
            # Deterministic stratified split by breed to avoid class-missing validation sets.
            by_breed = {}
            for image_id in trainval_ids:
                breed = _breed_from_image_id(image_id)
                if breed not in by_breed:
                    by_breed[breed] = []
                by_breed[breed].append(image_id)

            rng = np.random.RandomState(42)
            train_selected = []
            val_selected = []

            for breed in sorted(by_breed.keys()):
                ids = by_breed[breed]
                ids = list(ids)
                rng.shuffle(ids)

                n_total = len(ids)
                n_val = max(1, int(n_total * val_ratio))
                if n_val >= n_total:
                    n_val = max(1, n_total - 1)

                val_selected.extend(ids[:n_val])
                train_selected.extend(ids[n_val:])

            selected = train_selected if self.split == "train" else val_selected

        available_xml = set()
        if self.need_bbox:
            for name in sorted(os.listdir(self.xml_dir)):
                if name.startswith("."):
                    continue
                if name.lower().endswith(".xml"):
                    available_xml.add(name.rsplit(".", 1)[0])

        available_trimaps = set()
        if self.need_mask:
            for name in sorted(os.listdir(self.trimap_dir)):
                if name.startswith(".") or name.startswith("._"):
                    continue
                if name.lower().endswith(".png"):
                    available_trimaps.add(name.rsplit(".", 1)[0])

        filtered = []
        dropped_bbox = 0
        dropped_mask = 0
        for image_id in selected:
            ok = True
            if self.need_bbox and image_id not in available_xml:
                ok = False
                dropped_bbox += 1
            if self.need_mask and image_id not in available_trimaps:
                ok = False
                dropped_mask += 1
            if ok:
                filtered.append(image_id)

        dropped_total = len(selected) - len(filtered)
        if dropped_total > 0:
            print(
                f"[OxfordIIITPetDataset] split={self.split}, task={self.task}: "
                f"dropped {dropped_total} samples (missing_xml={dropped_bbox}, missing_trimap={dropped_mask})"
            )
        if len(filtered) == 0:
            print(f"[OxfordIIITPetDataset] Warning: no samples available for split={self.split}, task={self.task}")

        return filtered

    def __len__(self):
        return len(self.image_ids)

    def _read_image(self, image_id: str):
        jpg_path = f"{self.image_dir}/{image_id}.jpg"
        png_path = f"{self.image_dir}/{image_id}.png"
        try:
            image = plt.imread(jpg_path)
        except Exception:
            image = plt.imread(png_path)

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]

        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        return image

    def _read_bbox_xywh(self, image_id: str):
        xml_path = f"{self.xml_dir}/{image_id}.xml"
        with open(xml_path, "r") as f:
            xml_text = f.read()

        xmin = _parse_tag(xml_text, "xmin")
        ymin = _parse_tag(xml_text, "ymin")
        xmax = _parse_tag(xml_text, "xmax")
        ymax = _parse_tag(xml_text, "ymax")

        width = max(0.0, xmax - xmin)
        height = max(0.0, ymax - ymin)
        x_center = xmin + width / 2.0
        y_center = ymin + height / 2.0
        return np.array([x_center, y_center, width, height], dtype=np.float32)

    def _read_mask(self, image_id: str):
        mask_path = f"{self.trimap_dir}/{image_id}.png"
        mask = plt.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]

        mask = mask.astype(np.float32)
        if mask.max() <= 1.0:
            mask = np.rint(mask * 255.0)
        mask = mask.astype(np.int64)

        # Original trimap values are {1,2,3}; remap to {0,1,2}.
        mask = np.clip(mask - 1, 0, 2)
        return mask

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]

        image = self._read_image(image_id)
        orig_h, orig_w = image.shape[0], image.shape[1]

        sample = {}

        if self.need_mask:
            mask = self._read_mask(image_id)
            transformed = self.transforms(image=image, mask=mask)
            image_t = torch.from_numpy(transformed["image"]).permute(2, 0, 1).float()
            mask_t = torch.from_numpy(transformed["mask"]).long()
            sample["mask"] = mask_t
        else:
            transformed = self.transforms(image=image)
            image_t = torch.from_numpy(transformed["image"]).permute(2, 0, 1).float()

        sample["image"] = image_t

        if self.need_label:
            label_name = _breed_from_image_id(image_id)
            label = self.label_to_index[label_name]
            sample["label"] = torch.tensor(label, dtype=torch.long)

        if self.need_bbox:
            bbox = self._read_bbox_xywh(image_id)
            scale_x = self.image_size / float(orig_w)
            scale_y = self.image_size / float(orig_h)
            bbox_scaled = np.array(
                [bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y],
                dtype=np.float32,
            )
            sample["bbox"] = torch.from_numpy(bbox_scaled)

        return sample