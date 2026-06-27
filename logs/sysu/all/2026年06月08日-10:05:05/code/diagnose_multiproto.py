import argparse
import json
import os
import re
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from datasets.data_process import transform_test
from models.agw import AGW


class ImageList(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.paths[index]))
        return transform_test(image)


def parse_image_info(path):
    match = re.search(r"/cam(\d+)/(\d{4})/", path)
    if match is None:
        raise ValueError("Cannot parse camera and identity from {}".format(path))
    return int(match.group(2)), int(match.group(1))


def collect_paths(data_root, cameras):
    with open(os.path.join(data_root, "exp/test_id.txt"), "r") as file:
        identities = ["{:04d}".format(int(value)) for value in file.readline().strip().split(",")]

    paths = []
    for identity in identities:
        for camera in cameras:
            image_dir = os.path.join(data_root, "cam{}".format(camera), identity)
            if os.path.isdir(image_dir):
                paths.extend(
                    os.path.join(image_dir, name)
                    for name in sorted(os.listdir(image_dir))
                )
    labels, camera_ids = zip(*(parse_image_info(path) for path in paths))
    return paths, np.asarray(labels), np.asarray(camera_ids)


def build_model(checkpoint, device):
    args = SimpleNamespace(
        enable_rgmfd=1,
        rgmfd_reduction=16,
        rgmfd_gate_scale=0.5,
    )
    model = AGW(args).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["backbone"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def extract_features(model, paths, modality, device, batch_size, workers):
    loader = DataLoader(
        ImageList(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        drop_last=False,
    )
    features = []
    for images in loader:
        images = images.to(device)
        flipped = images.flip(-1)
        if modality == "rgb":
            _, feature = model(x1=images)
            _, feature_flip = model(x1=flipped)
        else:
            _, feature = model(x2=images)
            _, feature_flip = model(x2=flipped)
        features.append(F.normalize((feature + feature_flip) / 2, dim=1).cpu())
    return torch.cat(features)


def normalized_mean(features):
    return F.normalize(features.mean(dim=0, keepdim=True), dim=1)[0]


def spherical_kmeans_two(features, iterations=20):
    if features.shape[0] < 2:
        return normalized_mean(features).unsqueeze(0)

    center_a = features[0]
    center_b = features[torch.argmin(features @ center_a)]
    centers = F.normalize(torch.stack((center_a, center_b)), dim=1)

    for _ in range(iterations):
        assignment = torch.argmax(features @ centers.T, dim=1)
        new_centers = []
        for cluster in range(2):
            selected = features[assignment == cluster]
            if selected.shape[0] == 0:
                new_centers.append(centers[cluster])
            else:
                new_centers.append(normalized_mean(selected))
        new_centers = torch.stack(new_centers)
        if torch.allclose(new_centers, centers, atol=1e-5):
            break
        centers = new_centers
    return centers


def build_prototypes(features, labels, cameras, mode):
    prototypes = []
    prototype_labels = []
    for identity in sorted(np.unique(labels)):
        selected = features[torch.from_numpy(labels == identity)]
        if mode == "single":
            centers = normalized_mean(selected).unsqueeze(0)
        elif mode == "k2":
            centers = spherical_kmeans_two(selected)
        elif mode == "camera":
            centers = torch.stack([
                normalized_mean(features[torch.from_numpy((labels == identity) & (cameras == camera))])
                for camera in sorted(np.unique(cameras[labels == identity]))
            ])
        else:
            raise ValueError("Unknown prototype mode {}".format(mode))
        prototypes.append(centers)
        prototype_labels.extend([identity] * centers.shape[0])
    return torch.cat(prototypes), torch.tensor(prototype_labels)


def retrieval_diagnostics(query_features, query_labels, prototypes, prototype_labels):
    similarity = query_features @ prototypes.T
    positive_mask = torch.from_numpy(query_labels[:, None] == prototype_labels.numpy()[None, :])
    valid_queries = positive_mask.any(dim=1)
    similarity = similarity[valid_queries]
    positive_mask = positive_mask[valid_queries]
    query_labels = query_labels[valid_queries.numpy()]
    positive_similarity = similarity.masked_fill(~positive_mask, -1.0).max(dim=1).values
    negative_similarity = similarity.masked_fill(positive_mask, -1.0).max(dim=1).values
    predicted_labels = prototype_labels[similarity.argmax(dim=1)]
    margins = positive_similarity - negative_similarity
    positive_distances = 1.0 - positive_similarity
    return {
        "valid_query_ratio": float(valid_queries.float().mean()),
        "prototype_top1": float((predicted_labels == torch.from_numpy(query_labels)).float().mean()),
        "positive_distance_mean": float(positive_distances.mean()),
        "positive_distance_p90": float(torch.quantile(positive_distances, 0.9)),
        "hard_margin_mean": float(margins.mean()),
        "positive_margin_ratio": float((margins > 0).float().mean()),
    }


def rgb_dispersion(features, labels, cameras):
    cross_camera_distances = []
    single_residuals = []
    k2_residuals = []
    for identity in sorted(np.unique(labels)):
        identity_mask = labels == identity
        identity_features = features[torch.from_numpy(identity_mask)]
        single = normalized_mean(identity_features)
        k2 = spherical_kmeans_two(identity_features)
        single_residuals.extend((1.0 - identity_features @ single).tolist())
        k2_residuals.extend((1.0 - (identity_features @ k2.T).max(dim=1).values).tolist())

        camera_centers = []
        for camera in sorted(np.unique(cameras[identity_mask])):
            camera_centers.append(
                normalized_mean(features[torch.from_numpy(identity_mask & (cameras == camera))])
            )
        camera_centers = torch.stack(camera_centers)
        if camera_centers.shape[0] > 1:
            distances = 1.0 - camera_centers @ camera_centers.T
            upper = torch.triu_indices(distances.shape[0], distances.shape[1], offset=1)
            cross_camera_distances.extend(distances[upper[0], upper[1]].tolist())

    return {
        "cross_camera_centroid_distance_mean": float(np.mean(cross_camera_distances)),
        "cross_camera_centroid_distance_p90": float(np.quantile(cross_camera_distances, 0.9)),
        "single_prototype_residual_mean": float(np.mean(single_residuals)),
        "k2_prototype_residual_mean": float(np.mean(k2_residuals)),
        "k2_residual_reduction": float(
            1.0 - np.mean(k2_residuals) / max(np.mean(single_residuals), 1e-12)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--search-mode", choices=["all", "indoor"], required=True)
    parser.add_argument("--data-root", default="/root/data/SYSU-MM01")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rgb_cameras = [1, 2, 4, 5] if args.search_mode == "all" else [1, 2]
    rgb_paths, rgb_labels, rgb_cams = collect_paths(args.data_root, rgb_cameras)
    ir_paths, ir_labels, _ = collect_paths(args.data_root, [3, 6])

    model = build_model(args.checkpoint, device)
    rgb_features = extract_features(
        model, rgb_paths, "rgb", device, args.batch_size, args.workers
    )
    ir_features = extract_features(
        model, ir_paths, "ir", device, args.batch_size, args.workers
    )

    result = {
        "search_mode": args.search_mode,
        "checkpoint": args.checkpoint,
        "rgb_samples": len(rgb_paths),
        "ir_samples": len(ir_paths),
        "rgb_dispersion": rgb_dispersion(rgb_features, rgb_labels, rgb_cams),
        "retrieval": {},
    }
    for mode in ("single", "k2", "camera"):
        prototypes, prototype_labels = build_prototypes(
            rgb_features, rgb_labels, rgb_cams, mode
        )
        result["retrieval"][mode] = retrieval_diagnostics(
            ir_features, ir_labels, prototypes, prototype_labels
        )

    text = json.dumps(result, indent=2, ensure_ascii=True)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as file:
            file.write(text + "\n")


if __name__ == "__main__":
    main()
