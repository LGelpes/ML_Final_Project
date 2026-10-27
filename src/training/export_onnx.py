import argparse
from pathlib import Path
import torch
from src.config import BASE_DIR, load_config
from src.models import SUPPORTED_MODELS, build_model


def export_to_onnx(model, onnx_path: str | Path, image_size: int = 32):
    """Exporta un modelo PyTorch a formato ONNX."""
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    model.cpu().eval()
    dummy_input = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    model.to(device)
    print(f"Modelo ONNX guardado en: {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=[*SUPPORTED_MODELS, "all"])
    parser.add_argument("--image-size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config()
    targets = SUPPORTED_MODELS if args.model == "all" else [args.model]

    for name in targets:
        pt_path = BASE_DIR / f"weights/{name}.pt"
        onnx_path = BASE_DIR / f"weights/{name}.onnx"
        model = build_model(name, image_size=args.image_size, pretrained=False)
        if pt_path.exists():
            model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
        export_to_onnx(model, onnx_path, image_size=args.image_size)
