import yaml
import numpy as np
import matplotlib.pyplot as plt
from  data_loading import get_dataloaders
from collections import Counter


cfg = yaml.safe_load(open("configs/config.yaml"))
tr, va, te, meta = get_dataloaders(cfg)

num_classes = meta["num_classes"]

def class_counts(torch_dataset, num_classes):
    # TinyDataset -> hf_ds accessible via .hf_ds, labels via colonne "label"
    labels = np.array(torch_dataset.hf_ds["label"], dtype=int)
    counts = np.bincount(labels, minlength=num_classes)
    return counts
counts_train = class_counts(tr.dataset, num_classes)
counts_val   = class_counts(va.dataset, num_classes)
counts_test  = class_counts(te.dataset, num_classes)
print("Train min/max:", counts_train.min(), counts_train.max())
print("Val   min/max:", counts_val.min(), counts_val.max())
print("Test  min/max:", counts_test.min(), counts_test.max())
for i in range(20):
    print(f"class {i:3d}: train={counts_train[i]:4d}  val={counts_val[i]:3d}  test={counts_test[i]:3d}")
plt.figure()
plt.bar(np.arange(num_classes), counts_train)
plt.title("Tiny ImageNet — distribution des classes")
plt.xlabel("Classe")
plt.ylabel("Nombre de labels")
plt.show()
plt.savefig("class_distribution.png")
tr, va, te, meta = get_dataloaders(cfg)
hf_train = tr.dataset.hf_ds
sizes = Counter()
modes = Counter()
# échantillon
N = min(2000, len(hf_train))
for i in range(N):
    img = hf_train[i]["image"]  
    sizes[img.size] += 1        
    modes[img.mode] += 1        
print("Tailles (W,H) les plus fréquentes:", sizes.most_common(5))
print("Modes (canaux) les plus fréquents:", modes.most_common(5))
