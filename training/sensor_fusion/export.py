"""
Model export module for sensor fusion LSTM.
Handles exporting PyTorch models to ONNX and TFLite for deployment.
"""

import argparse
import os
import torch
import numpy as np
from typing import Optional

# Conditional imports for validation and TFLite
try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    
try:
    import onnx_tf
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


def export_to_onnx(model_path: str, output_path: str, seq_len: int = 50, input_features: int = 9) -> str:
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model_path: Path to the PyTorch checkpoint (.pth/.pt)
        output_path: Path where the ONNX model will be saved
        seq_len: Sequence length for the dummy input
        input_features: Number of input features
        
    Returns:
        The path to the saved ONNX model
    """
    from .model import create_model  # Dynamic import to avoid circular dependency
    
    print(f"Loading PyTorch model from {model_path}...")
    
    # Check if the path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Reconstruct model (assuming create_model uses defaults or can run parameter-free)
    model = create_model()
    
    # Handle state dict from standard PyTorch or Lightning saves
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    # Create dummy input (batch_size=1, seq_len, input_features)
    dummy_input = torch.randn(1, seq_len, input_features)
    
    print(f"Exporting to ONNX: {output_path}...")
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print("ONNX export completed successfully.")
    return output_path


def export_to_tflite(onnx_path: str, output_path: str, quantize: bool = True) -> str:
    """
    Export an ONNX model to TFLite format.
    
    Args:
        onnx_path: Path to the ONNX model
        output_path: Path where the TFLite model will be saved
        quantize: Whether to apply INT8 post-training quantization
        
    Returns:
        The path to the saved TFLite model
    """
    if not HAS_TF:
        print("ERROR: TensorFlow and onnx-tf are required for TFLite export.")
        print("Please install them using: pip install tensorflow onnx-tf")
        return ""
        
    print(f"Loading ONNX model from {onnx_path}...")
    try:
        from onnx_tf.backend import prepare
        import onnx
        
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        
        # Save temporarily as SavedModel
        saved_model_path = output_path.replace('.tflite', '_saved_model')
        tf_rep.export_graph(saved_model_path)
        
        print("Converting SavedModel to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        
        if quantize:
            print("Applying INT8 post-training quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"TFLite export completed successfully: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Failed to convert ONNX to TFLite. Error: {str(e)}")
        print("Note: ONNX to TFLite conversion for LSTM can be complex.")
        print("Alternative approach: try using 'onnx2tf' CLI tool instead.")
        return ""


def validate_onnx(onnx_path: str, seq_len: int = 50, input_features: int = 9) -> bool:
    """
    Validate the exported ONNX model.
    
    Args:
        onnx_path: Path to the ONNX model
        seq_len: Sequence length for the dummy input
        input_features: Number of input features
        
    Returns:
        True if validation passes, False otherwise
    """
    if not HAS_ONNX:
        print("WARNING: onnx and onnxruntime are not installed. Skipping validation.")
        return False
        
    print(f"Validating ONNX model: {onnx_path}")
    
    try:
        # Load and verify ONNX structure
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Run inference with ONNX Runtime
        ort_session = ort.InferenceSession(onnx_path)
        
        # Get input name and shape
        input_name = ort_session.get_inputs()[0].name
        
        # Create dummy input (must match float32 type)
        dummy_input = np.random.randn(1, seq_len, input_features).astype(np.float32)
        
        # Run inference
        outputs = ort_session.run(None, {input_name: dummy_input})
        
        # Check output shape (batch_size, 6)
        output_shape = outputs[0].shape
        print(f"ONNX Model output shape: {output_shape}")
        
        if output_shape[1:] != (6,):
            print(f"Validation failed: Expected output shape ending in (6,), got {output_shape}")
            return False
            
        print("ONNX model validated successfully!")
        return True
        
    except Exception as e:
        print(f"Validation failed with error: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch LSTM model to ONNX and TFLite")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained PyTorch model (.pth)")
    parser.add_argument("--output_dir", type=str, default="mobile_app/inference/", help="Directory to save exported models")
    parser.add_argument("--format", type=str, choices=["onnx", "tflite", "both"], default="onnx", help="Export format")
    parser.add_argument("--quantize", action="store_true", default=True, help="Apply INT8 quantization for TFLite (default: True)")
    parser.add_argument("--no-quantize", dest="quantize", action="store_false", help="Disable INT8 quantization")
    parser.add_argument("--seq_len", type=int, default=50, help="Sequence length of the input data")
    parser.add_argument("--input_features", type=int, default=9, help="Number of input features")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate base filename from model_path
    base_name = os.path.splitext(os.path.basename(args.model_path))[0]
    
    onnx_path = os.path.join(args.output_dir, f"{base_name}.onnx")
    tflite_path = os.path.join(args.output_dir, f"{base_name}.tflite")
    
    exported_onnx_path = None
    if args.format in ["onnx", "both"]:
        exported_onnx_path = export_to_onnx(
            args.model_path, 
            onnx_path, 
            seq_len=args.seq_len, 
            input_features=args.input_features
        )
        
        if exported_onnx_path:
            validate_onnx(exported_onnx_path, seq_len=args.seq_len, input_features=args.input_features)
            
    if args.format in ["tflite", "both"]:
        if args.format == "tflite" and not exported_onnx_path:
            print("TFLite conversion requires ONNX model first. Exporting to ONNX temporary file...")
            exported_onnx_path = export_to_onnx(
                args.model_path, 
                onnx_path, 
                seq_len=args.seq_len, 
                input_features=args.input_features
            )
            
        if exported_onnx_path:
            export_to_tflite(exported_onnx_path, tflite_path, quantize=args.quantize)


if __name__ == "__main__":
    main()
