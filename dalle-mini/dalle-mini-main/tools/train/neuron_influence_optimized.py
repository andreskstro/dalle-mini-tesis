"""
Memory-Optimized Neuron Influence Tracker for Excitation Backpropagation
========================================================================
Optimized version that reduces memory usage by:
1. Sequential processing instead of vmap
2. Immediate disk writing
3. Active memory cleanup
4. Focus on specific EB layers only
5. Per-neuron influence calculation (not per-parameter)
"""
from collections import defaultdict
import flax
import jax
import jax.numpy as jnp
import json
import gc
import os
from pathlib import Path
from typing import List, Tuple, Optional, Set, Dict
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import re
import logging
import time

logger = logging.getLogger(__name__)

class MemoryOptimizedNeuronInfluenceTracker:
    """
    Memory-optimized version of NeuronInfluenceTracker that:
    - Processes gradients sequentially to avoid OOM
    - Writes influence data immediately to disk
    - Actively manages memory cleanup
    - Focuses on specific EB layers only
    - Calculates per-neuron influences (L2 norm per output neuron)
    """
    
    # Default EB layers to track (can be overridden via constructor)
    EB_LAYERS_DEFAULT = {
        'GLU_0.Dense_0.kernel',  # fc1 (entrada) - 4096 neurons
        'GLU_0.Dense_2.kernel',  # fc2 (salida) - 4096 neurons  
        'lm_head.kernel'         # output head - 16392 neurons
    }
    
    def __init__(self, 
                 output_file: str = 'eb_neuron_influences.json',
                 batch_write_size: int = 100,
                 use_bfloat16: bool = True,
                 target_layers: Optional[Set[str]] = None):
        """
        Args:
            output_file: Path to save influence data
            batch_write_size: Write to disk every N updates
            use_bfloat16: Use bfloat16 for gradient calculations to save memory
            target_layers: Optional set of layer suffixes to track (e.g., {'lm_head.kernel'}).
                           If None, tracks the default EB layers.
        """
        self.output_file = output_file
        self.batch_write_size = batch_write_size
        self.use_bfloat16 = use_bfloat16
        # Selected layers to track
        self.target_layers: Set[str] = set(target_layers) if target_layers else set(self.EB_LAYERS_DEFAULT)
        
        # In-memory buffer for batched writes - now stores per-image data
        self.influence_buffer = []  # List of influence records
        self.update_count = 0
        # Epoch-scoped writing state
        self._epoch_active = False
        self._default_output_file = output_file
        self._prev_output_file = None
        
        # Load existing data if available
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing influence data from disk if available."""
        if not os.path.exists(self.output_file):
            return
            
        try:
            # Check if file is empty or contains valid JSON
            if os.path.getsize(self.output_file) == 0:
                print(f"Warning: {self.output_file} is empty, starting fresh")
                return
                
            with open(self.output_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    print(f"Warning: {self.output_file} is empty, starting fresh")
                    return
                
                existing_data = json.loads(content)
                
            # Existing data is already in the correct format, no need to load into buffer
            print(f"Loaded existing influence data from {self.output_file}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not load existing data: {e}. Starting fresh.")
        except Exception as e:
            print(f"Warning: Unexpected error loading data: {e}. Starting fresh.")
    
    def _is_eb_layer(self, param_path: str) -> bool:
        """Check if this parameter is one of the target layers we want to track."""
        return any(eb_layer in param_path for eb_layer in self.target_layers)
    
    def _get_layer_name(self, param_path: str) -> str:
        """Extract the standardized layer name for EB compatibility."""
        # For decoder layers, we want to preserve the layer number
        # e.g., "model.decoder.layers.3.GLU_0.Dense_0.kernel" -> "model.decoder.layers.3.GLU_0.Dense_0.kernel"
        if 'decoder.layers' in param_path and ('GLU_0.Dense_0.kernel' in param_path or 'GLU_0.Dense_2.kernel' in param_path):
            return param_path
        elif 'lm_head.kernel' in param_path:
            return 'lm_head.kernel'
        else:
            return param_path  # Fallback
    
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
    
    def _convert_to_bfloat16_if_needed(self, tensor):
        """Convert tensor to bfloat16 if enabled to save memory."""
        if self.use_bfloat16 and tensor.dtype == jnp.float32:
            return tensor.astype(jnp.bfloat16)
        return tensor
    
    def _compute_per_neuron_influence(self, grad_tensor):
        """
        Compute per-neuron influence (L2 norm per output neuron).
        
        For weight matrices of shape (input_dim, output_dim), we compute
        the L2 norm along the input dimension for each output neuron.
        
        Args:
            grad_tensor: Gradient tensor of shape (input_dim, output_dim)
            
        Returns:
            List of floats: L2 norm for each output neuron
        """
        # Convert to bfloat16 if needed
        grad_tensor = self._convert_to_bfloat16_if_needed(grad_tensor)
        
        # Compute L2 norm along input dimension (axis=0) for each output neuron
        # grad_tensor shape: (input_dim, output_dim)
        # Result shape: (output_dim,) - one value per output neuron
        per_neuron_norms = jnp.linalg.norm(grad_tensor, axis=0)
        
        # Move to CPU and convert to Python list
        per_neuron_norms_cpu = jax.device_get(per_neuron_norms)
        
        # Force garbage collection to free GPU memory
        del grad_tensor, per_neuron_norms
        
        return per_neuron_norms_cpu.tolist()

    def update_sequential(self, batch_filenames: List[str], grads_per_example):
        """
        Update influence map by processing gradients sequentially to avoid OOM.
        Now calculates per-neuron influences for EB layers only.
        
        Args:
            batch_filenames: List[str] - Image IDs from current batch
            grads_per_example: PyTree or List[PyTree] - Either gradients with a leading batch dimension
                              or a list of individual gradient trees
        """
        batch_size = len(batch_filenames)
        _t0 = time.perf_counter()
        eb_layers_info = ",".join(sorted(self.target_layers)) if self.target_layers else "(default)"
        logger.info(f"[EB] ENTER update_sequential: batch_size={batch_size}, target_layers={eb_layers_info}")
        _processed_examples = 0
        _processed_params = 0
        
        # Handle both formats: list of gradient trees or single tree with batch dimension
        if isinstance(grads_per_example, list):
            logger.info("[EB] START update_sequential: iterate list of per-example grads")
            # grads_per_example is already a list of individual gradient trees
            if len(grads_per_example) != batch_size:
                print(f"Warning: Mismatch between batch_filenames ({batch_size}) and grads_per_example ({len(grads_per_example)})")
                # Use the minimum to avoid index errors
                actual_batch_size = min(batch_size, len(grads_per_example))
            else:
                actual_batch_size = batch_size
                
            # Process each example sequentially
            for example_idx in range(actual_batch_size):
                img_id = batch_filenames[example_idx]
                example_grads = grads_per_example[example_idx]
                
                # Collect influences for this image
                layer_influences = {}
                
                # Process each parameter
                for param_path, grad_tensor in self._flatten_grads(example_grads):
                    if not self._is_eb_layer(param_path):
                        continue
                    
                    # Get standardized layer name
                    layer_name = self._get_layer_name(param_path)
                    
                    # Compute per-neuron influences
                    neuron_influences = self._compute_per_neuron_influence(grad_tensor)
                    layer_influences[layer_name] = neuron_influences
                    _processed_params += 1
                
                # Store influence record for this image
                if layer_influences:  # Only store if we found EB layers
                    influence_record = {
                        "image_id": img_id,
                        "layer_influences": layer_influences
                    }
                    self.influence_buffer.append(influence_record)
                
                # Clean up example gradients from memory
                del example_grads
                
                # Force garbage collection periodically
                if example_idx % 10 == 0:
                    gc.collect()
                _processed_examples += 1
        else:
            logger.info("[EB] START update_sequential: iterate tree with batch dimension")
            # grads_per_example is a single tree with batch dimension
            # Process each example sequentially instead of using vmap
            for example_idx in range(batch_size):
                img_id = batch_filenames[example_idx]
                
                # Extract gradients for this single example
                example_grads = jax.tree_util.tree_map(
                    lambda x: x[example_idx], grads_per_example
                )
                
                # Collect influences for this image
                layer_influences = {}
                
                # Process each parameter
                for param_path, grad_tensor in self._flatten_grads(example_grads):
                    if not self._is_eb_layer(param_path):
                        continue
                    
                    # Get standardized layer name
                    layer_name = self._get_layer_name(param_path)
                    
                    # Compute per-neuron influences
                    neuron_influences = self._compute_per_neuron_influence(grad_tensor)
                    layer_influences[layer_name] = neuron_influences
                    _processed_params += 1
                
                # Store influence record for this image
                if layer_influences:  # Only store if we found EB layers
                    influence_record = {
                        "image_id": img_id,
                        "layer_influences": layer_influences
                    }
                    self.influence_buffer.append(influence_record)
                
                # Clean up example gradients from memory
                del example_grads
                
                # Force garbage collection periodically
                if example_idx % 10 == 0:
                    gc.collect()
                _processed_examples += 1
        
        self.update_count += 1
        
        # Write to disk periodically to avoid memory buildup
        if self.update_count % self.batch_write_size == 0:
            self._write_to_disk()
            self._clear_buffer()

        _t1 = time.perf_counter()
        logger.info(
            f"[EB] EXIT update_sequential: processed_examples={_processed_examples}, processed_params={_processed_params}, "
            f"buffer_size={len(self.influence_buffer)}, took={(_t1 - _t0):.3f}s"
        )

    def _write_to_disk(self):
        """Write current buffer to disk, appending to existing data."""
        if not self.influence_buffer:
            return
            
        try:
            _t0 = time.perf_counter()
            _buf = len(self.influence_buffer)
            logger.info(f"[EB] ENTER _write_to_disk: file={self.output_file}, buffer_size={_buf}")
            # Load existing data
            existing_data = []
            if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
                with open(self.output_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        existing_data = json.loads(content)
            
            # Append new data
            existing_data.extend(self.influence_buffer)
            
            # Write back to file
            with open(self.output_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
            _t1 = time.perf_counter()
            logger.info(f"[EB] EXIT _write_to_disk: wrote={_buf}, file={self.output_file}, took={(_t1 - _t0):.3f}s")
            print(f"Wrote {len(self.influence_buffer)} influence records to {self.output_file}")
            
        except Exception as e:
            print(f"Error writing to disk: {e}")
    
    def _clear_buffer(self):
        """Clear the in-memory buffer and force garbage collection."""
        self.influence_buffer.clear()
        gc.collect()

    def save(self, path: Optional[str] = None):
        """
        Final save - write any remaining buffer data to disk.
        If 'path' is provided, writes to that file (and updates current output_file for this call).
        """
        if path:
            self.output_file = path
        
        if self.influence_buffer:
            self._write_to_disk()
            self._clear_buffer()
        
        print(f"Final influence data saved to {self.output_file}")

    def start_epoch(self, epoch_index: int, output_dir: Optional[str] = None, prefix: str = "eb_neuron_influences_epoch"):
        """
        Begin an epoch-scoped recording session so that all periodic writes go to an epoch-specific file.
        This ensures a single JSON per tracked epoch.

        Args:
            epoch_index: Zero-based epoch index
            output_dir: Optional directory to store the epoch file; if None, uses directory of default output file
            prefix: Filename prefix for the epoch JSON
        """
        if self._epoch_active:
            return  # already active; avoid nested calls
        self._epoch_active = True
        self._prev_output_file = self.output_file
        # Determine destination directory
        if output_dir is None:
            base_dir = os.path.dirname(self._default_output_file) or "."
        else:
            base_dir = output_dir
        os.makedirs(base_dir, exist_ok=True)
        epoch_fname = f"{prefix}_{epoch_index + 1}.json"
        self.output_file = os.path.join(base_dir, epoch_fname)

    def finish_epoch(self):
        """
        Finish an epoch-scoped recording session: flush remaining buffer, then restore default output file.
        """
        try:
            if self.influence_buffer:
                self._write_to_disk()
                self._clear_buffer()
        finally:
            if self._prev_output_file is not None:
                self.output_file = self._prev_output_file
                self._prev_output_file = None
            self._epoch_active = False

    def get_top_images(self, layer_name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Returns the top_k images that most influenced a specific layer.
        Loads data from disk if not in buffer.
        """
        # Check buffer first
        layer_influences = {}
        for record in self.influence_buffer:
            if layer_name in record['layer_influences']:
                layer_influences[record['image_id']] = record['layer_influences'][layer_name]
        
        # Load from disk
        if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
            try:
                with open(self.output_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
                        for record in data:
                            if layer_name in record['layer_influences']:
                                layer_influences[record['image_id']] = record['layer_influences'][layer_name]
            except (json.JSONDecodeError, ValueError):
                print(f"Warning: Could not read influence data for {layer_name}")
        
        if not layer_influences:
            return []
        
        # Sort by influence magnitude
        sorted_influences = sorted(
            layer_influences.items(),
            key=lambda item: sum(item[1]),  # Sum of neuron influences
            reverse=True
        )
        return sorted_influences[:top_k]

    def plot_top_images(self, layer_name: str, dataset_dir: str, top_k: int = 3):
        """Plot the most influential images for a layer."""
        top_images = self.get_top_images(layer_name, top_k)
        if not top_images:
            print(f"No data for {layer_name}")
            return
        
        plt.figure(figsize=(15, 5))
        for i, (img_id, neuron_influences) in enumerate(top_images, 1):
            img_path = Path(dataset_dir) / img_id
            try:
                img = np.array(Image.open(img_path))
                plt.subplot(1, top_k, i)
                plt.imshow(img)
                plt.title(f"{img_id}\nInfluence: {sum(neuron_influences):.2f}")
                plt.axis('off')
            except FileNotFoundError:
                print(f"Image not found: {img_path}")
        
        plt.suptitle(f"Top {top_k} images for {layer_name}", y=1.05)
        plt.tight_layout()
        plt.show()


# Convenience function to create tracker for EB layers
def create_eb_neuron_tracker(output_file: str = 'eb_neuron_influences.json', target_layers: Optional[Set[str]] = None) -> MemoryOptimizedNeuronInfluenceTracker:
    """Create tracker specifically for EB neuron influence tracking.

    Args:
        output_file: Path for JSON output file.
        target_layers: Optional set of layer suffixes to track (e.g., {'lm_head.kernel'}). If None, uses defaults.
    """
    return MemoryOptimizedNeuronInfluenceTracker(
        output_file=output_file,
        batch_write_size=50,  # Write more frequently for EB tracking
        use_bfloat16=True,
        target_layers=target_layers,
    )
