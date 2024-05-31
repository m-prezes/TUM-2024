import json
import random
from pathlib import Path

import numpy as np
import torch


def save_results(model, train_losses, test_losses, test_accuracies, path):
    Path(path).mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), f"{path}/model.pth")

    with open(f"{path}/train_losses.json", "w") as f:
        json.dump(train_losses, f)
    with open(f"{path}/test_losses.json", "w") as f:
        json.dump(test_losses, f)
    with open(f"{path}/test_accuracies.json", "w") as f:
        json.dump(test_accuracies, f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
