import torch
from torch.quantization import quantize_dynamic
from transformers import ViTForImageClassification, ViTImageProcessor

# load the model
DEVICE = torch.device("cpu")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=21,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load("best_model_fold3.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()


quantized_model = quantize_dynamic(
    model,                         # kuantize edilecek model
    {torch.nn.Linear},             # kuantize edilecek modüller
    dtype=torch.qint8              # çevrilecek vei tipi
)

# save the model
torch.save(quantized_model, "vit_ucmerced_quantized.pth")

print("Kuantizasyon tamamlandı. Kaydedilen dosya: vit_ucmerced_quantized.pth")
