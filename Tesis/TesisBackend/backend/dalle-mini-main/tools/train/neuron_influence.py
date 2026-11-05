"""
Neuron Influence Tracker
=======================
Tracks which training images most influence each neuron via gradient analysis.

Key Concepts:
1. Gradient Magnitude as Influence Metric
2. Per-batch Accumulation
3. Hierarchical Parameter Naming
"""
from collections import defaultdict
import flax
import jax
import jax.numpy as jnp
import json
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

""" 
for cada batch de entrenamiento:
   1. Extraer IDs de imágenes (filename)
   2. Calcular gradientes (∇W)
   3. Para cada parámetro W_j:
      a. Calcular ||∇W_j||
      b. Asociar (image_id, ||∇W_j||) al parámetro


if gradiente_es_grande:
    "Esta imagen hizo que la neurona cambiara significativamente"
elif gradiente_es_pequeño:
    "La neurona casi no se ajustó por esta imagen"
 """

class NeuronInfluenceTracker:
    """
    Tracks neuron-to-image influence during training.
    
    Attributes:
        influence_map: {
            'parameter_path': [
                (image_id, gradient_magnitude),
                ...
            ],
            ...
        }
    """
    def __init__(self):
        """Initializes the tracker with a nested defaultdict to accumulate scores."""
        self.influence_map = defaultdict(lambda: defaultdict(float))

    def _flatten_grads(self, grads, prefix=''):
        """Recursively flatten gradients with parameter paths."""
        if isinstance(grads, (dict, flax.core.FrozenDict)):
            items = grads.items() if isinstance(grads, dict) else grads.unfreeze().items()
            for k, v in items:
                yield from self._flatten_grads(v, f'{prefix}{k}.')
        elif isinstance(grads, (list, tuple)):
            for i, v in enumerate(grads):
                yield from self._flatten_grads(v, f'{prefix}{i}.')
        else:
            yield (prefix[:-1], grads)  # Remove trailing dot

    def update(self, batch_filenames, grads_per_example):
        """
        Update influence map by accumulating gradient magnitudes for each image.
        
        Args:
            batch_filenames: List[str] - Image IDs from current batch
            grads_per_example: PyTree - Gradients with a leading batch dimension.
        """
        for param_path, grad_batch in self._flatten_grads(grads_per_example):
            # Calculate the norm for each slice along the batch dimension
            grad_magnitudes = jax.vmap(jnp.linalg.norm)(grad_batch)
            
            # Accumulate the influence for each image
            for img_id, magnitude in zip(batch_filenames, grad_magnitudes):
                self.influence_map[param_path][img_id] += float(magnitude)

    def save(self, path):
        """Save influence map as JSON."""
        Path(path).parent.mkdir(exist_ok=True)
        # Convert defaultdict to dict for clean JSON serialization
        serializable_map = {k: dict(v) for k, v in self.influence_map.items()}
        with open(path, 'w') as f:
            json.dump(serializable_map, f, indent=2)

    @classmethod
    def load(cls, path):
        """Load saved influence map."""
        tracker = cls()
        try:
            with open(path) as f:
                loaded_map = json.load(f)
            
            # Reconstruct the nested defaultdict structure for future updates
            new_map = defaultdict(lambda: defaultdict(float))
            for param_path, influences in loaded_map.items():
                new_map[param_path] = defaultdict(float, influences)
            tracker.influence_map = new_map
        except FileNotFoundError:
            print(f"Warning: Influence map file not found at {path}. Starting with an empty tracker.")
        return tracker

    def get_top_images(self, param_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Returns the top_k images that most influenced a specific parameter, based on accumulated influence.
        
        Args:
            param_path: Path of the parameter (e.g., 'model.decoder.layers.0.kernel')
            top_k: Number of images to return
        
        Returns:
            List of (image_id, accumulated_gradient_magnitude) tuples, sorted descending.
        """
        influences_for_param = self.influence_map.get(param_path)
        if not influences_for_param:
            return []
        
        # Sort the dictionary items by value (the accumulated influence)
        sorted_influences = sorted(
            influences_for_param.items(),
            key=lambda item: item[1],
            reverse=True
        )
        return sorted_influences[:top_k]

    def plot_top_images(self, param_path: str, dataset_dir: str, top_k: int = 3):
        """
        Muestra las imágenes más influyentes para un parámetro.
        
        Args:
            param_path: Ruta del parámetro (ej: 'model.encoder.1.kernel')
            dataset_dir: Directorio base donde buscar las imágenes
            top_k: Número de imágenes a mostrar
        """
        top_images = self.get_top_images(param_path, top_k)
        if not top_images:
            print(f"No hay datos para {param_path}")
            return
        
        plt.figure(figsize=(15, 5))
        for i, (img_id, magnitude) in enumerate(top_images, 1):
            img_path = Path(dataset_dir) / img_id
            try:
                img = np.array(Image.open(img_path))
                plt.subplot(1, top_k, i)
                plt.imshow(img)
                plt.title(f"{img_id}\nInfluencia: {magnitude:.2f}")
                plt.axis('off')
            except FileNotFoundError:
                print(f"¡Imagen no encontrada: {img_path}!")
        
        plt.suptitle(f"Top {top_k} imágenes para {param_path}", y=1.05)
        plt.tight_layout()
        plt.show()
