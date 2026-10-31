"""Dataset loading and Dirichlet non-IID partitioning across N gossip nodes.

Reused near-verbatim from ../DBT-CFL-PTT/experiments/data.py (dirichlet_partition,
preload_client_tensors, preload_test_tensors are byte-identical logic) per
PCTA_Experimental_Methodology.md B.2. Dataset choice differs: PCTA_Experimental_Methodology.md
B.2 locks CIFAR-10 + Fashion-MNIST (not the sibling project's MNIST substitution),
deliberately avoiding repeating the exact same benchmark pair across the two sibling papers,
and matching the original PCTA report's own Table 8.3 dataset recommendation.
"""
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

# Public release: the only change from the frozen code (see manifests/) is this line.
DATA_ROOT = __import__("os").environ.get("PCTA_DATA_ROOT", "data_cache")


def load_dataset(name):
    if name == "cifar10":
        tfm = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=tfm)
        test = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=tfm)
        num_classes, in_channels, input_size = 10, 3, 32
    elif name == "fashion_mnist":
        tfm = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
        train = torchvision.datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tfm)
        test = torchvision.datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tfm)
        num_classes, in_channels, input_size = 10, 1, 28
    else:
        raise ValueError(name)
    return train, test, num_classes, in_channels, input_size


def dirichlet_partition(dataset, num_clients, alpha, num_classes, seed=0, max_per_client=600,
                        min_per_client=32, max_attempts=200):
    """Revision v2: re-draw until every client holds >= min_per_client samples
    (standard practice, e.g. NIID-Bench's min_require_size). v1 had no minimum:
    at alpha=0.03, 10 of 100 nodes held NO data and 30 held < one batch --
    empty honest nodes send zero updates, which every detector flags and which
    inflated the reported honest FPR at severe Non-IID."""
    best = None
    for attempt in range(max_attempts):
        parts = _dirichlet_partition_once(dataset, num_clients, alpha, num_classes,
                                          seed + 100003 * attempt, max_per_client)
        if min(len(p) for p in parts) >= min_per_client:
            return parts
        if best is None or min(len(p) for p in parts) > min(len(p) for p in best):
            best = parts
    # Extreme alpha (e.g. 0.01) essentially never satisfies the minimum by
    # re-drawing alone. Fallback that preserves the skew: top up each short
    # client with samples of ITS OWN dominant class (a random class if empty),
    # taken from whichever clients hold the most of that class.
    targets = np.array(dataset.targets)
    rng = np.random.default_rng(seed + 7)
    parts = [list(p) for p in best]
    for i in range(num_clients):
        need = min_per_client - len(parts[i])
        if need <= 0:
            continue
        cls = (int(np.bincount(targets[parts[i]], minlength=num_classes).argmax()) if parts[i]
               else int(rng.integers(num_classes)))
        while need > 0:
            donor = max((j for j in range(num_clients) if j != i),
                        key=lambda j: int(np.sum(targets[parts[j]] == cls)) if len(parts[j]) > min_per_client else -1)
            pool = [x for x in parts[donor] if targets[x] == cls]
            take = pool[:min(need, len(parts[donor]) - min_per_client, len(pool))]
            if not take:
                raise RuntimeError(f"cannot top up client {i} with class {cls}")
            taken = set(take)
            parts[donor] = [x for x in parts[donor] if x not in taken]
            parts[i].extend(take)
            need -= len(take)
    return parts


def _dirichlet_partition_once(dataset, num_clients, alpha, num_classes, seed, max_per_client):
    targets = np.array(dataset.targets)
    rng = np.random.default_rng(seed)
    idx_by_class = [np.where(targets == c)[0] for c in range(num_classes)]
    for idxs in idx_by_class:
        rng.shuffle(idxs)
    client_idxs = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idxs = idx_by_class[c]
        if len(idxs) == 0:
            continue
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        split_points = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        splits = np.split(idxs, split_points)
        for i, s in enumerate(splits):
            client_idxs[i].extend(s.tolist())
    for i in range(num_clients):
        rng.shuffle(client_idxs[i])
        if max_per_client is not None and len(client_idxs[i]) > max_per_client:
            client_idxs[i] = client_idxs[i][:max_per_client]
    return client_idxs


def preload_client_tensors(dataset, client_idxs, device):
    """Load each client's images/labels fully onto device as tensors (small datasets, fits in VRAM)."""
    clients = []
    for idxs in client_idxs:
        if len(idxs) == 0:
            clients.append((torch.empty(0), torch.empty(0, dtype=torch.long)))
            continue
        xs = torch.stack([dataset[i][0] for i in idxs]).to(device)
        ys = torch.tensor([dataset[i][1] for i in idxs], dtype=torch.long).to(device)
        clients.append((xs, ys))
    return clients


def preload_test_tensors(dataset, device, max_n=3000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(dataset)
    idxs = rng.choice(n, size=min(max_n, n), replace=False)
    xs = torch.stack([dataset[int(i)][0] for i in idxs]).to(device)
    ys = torch.tensor([dataset[int(i)][1] for i in idxs], dtype=torch.long).to(device)
    return xs, ys
