try:
    from .onnx_model import ONNXModel
except ImportError:
    print("ONNXModel not available")
