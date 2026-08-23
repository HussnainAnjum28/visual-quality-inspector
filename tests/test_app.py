"""
Unit tests for the Visual Quality Inspector system.
Run with: pytest tests/test_app.py -v
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Add project root to path so we can import from app/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

MODEL_PATH = "models/resnet18_best.pth"
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ------------------ FIXTURES ------------------

@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def checkpoint(device):
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
    return torch.load(MODEL_PATH, map_location=device)


@pytest.fixture(scope="module")
def loaded_model(checkpoint, device):
    class_names = checkpoint['class_names']
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


@pytest.fixture(scope="module")
def transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


@pytest.fixture
def dummy_image():
    """Creates a random RGB test image (200x200, like our dataset)."""
    array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    return Image.fromarray(array)


# ------------------ TESTS: MODEL LOADING ------------------

class TestModelLoading:

    def test_model_file_exists(self):
        assert os.path.exists(MODEL_PATH), "Trained model checkpoint is missing"

    def test_checkpoint_has_required_keys(self, checkpoint):
        required_keys = ['model_state_dict', 'class_names', 'architecture']
        for key in required_keys:
            assert key in checkpoint, f"Checkpoint missing required key: {key}"

    def test_class_names_count(self, checkpoint):
        assert len(checkpoint['class_names']) == 6, "Expected 6 defect classes"

    def test_model_loads_without_error(self, loaded_model):
        assert loaded_model is not None
        assert isinstance(loaded_model, nn.Module)

    def test_model_is_in_eval_mode(self, loaded_model):
        assert not loaded_model.training


# ------------------ TESTS: PREPROCESSING ------------------

class TestPreprocessing:

    def test_transform_output_shape(self, transform, dummy_image):
        tensor = transform(dummy_image)
        assert tensor.shape == (3, IMG_SIZE, IMG_SIZE), "Transform did not produce expected shape"

    def test_transform_output_is_tensor(self, transform, dummy_image):
        tensor = transform(dummy_image)
        assert isinstance(tensor, torch.Tensor)

    def test_transform_handles_grayscale_image(self, transform):
        # Some uploaded images might be single-channel; ensure conversion to RGB works
        gray_array = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        gray_image = Image.fromarray(gray_array).convert("L").convert("RGB")
        tensor = transform(gray_image)
        assert tensor.shape == (3, IMG_SIZE, IMG_SIZE)

    def test_normalization_range(self, transform, dummy_image):
        tensor = transform(dummy_image)
        # After ImageNet normalization, values should generally fall in a reasonable range
        assert tensor.min() > -5 and tensor.max() < 5


# ------------------ TESTS: PREDICTION ------------------

class TestPrediction:

    def test_prediction_output_shape(self, loaded_model, transform, dummy_image, device):
        tensor = transform(dummy_image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = loaded_model(tensor)
        assert output.shape == (1, 6), "Model output should have shape (1, num_classes)"

    def test_prediction_probabilities_sum_to_one(self, loaded_model, transform, dummy_image, device):
        tensor = transform(dummy_image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = loaded_model(tensor)
            probs = torch.softmax(output, dim=1)
        total = probs.sum().item()
        assert abs(total - 1.0) < 1e-4, "Softmax probabilities should sum to 1"

    def test_predicted_class_is_valid_index(self, loaded_model, transform, dummy_image, device, checkpoint):
        tensor = transform(dummy_image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = loaded_model(tensor)
            pred_idx = torch.argmax(output, dim=1).item()
        assert 0 <= pred_idx < len(checkpoint['class_names'])

    def test_confidence_is_between_zero_and_one(self, loaded_model, transform, dummy_image, device):
        tensor = transform(dummy_image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = loaded_model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            confidence = torch.max(probs).item()
        assert 0.0 <= confidence <= 1.0


# ------------------ TESTS: ERROR HANDLING ------------------

class TestErrorHandling:

    def test_missing_model_path_raises_error(self):
        with pytest.raises(FileNotFoundError):
            torch.load("models/nonexistent_model.pth")

    def test_invalid_image_mode_conversion(self):
        # Ensure RGBA images can still be converted to RGB without crashing
        rgba_array = np.random.randint(0, 255, (200, 200, 4), dtype=np.uint8)
        rgba_image = Image.fromarray(rgba_array, mode="RGBA")
        converted = rgba_image.convert("RGB")
        assert converted.mode == "RGB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])