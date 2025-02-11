import math
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew, kurtosis
import librosa
import time
import json
from typing import List, Dict, Tuple, Union, Callable, Any  # Type hints
import warnings  # For warnings

# --- FastVectorRF Library (Single File) ---
class Entity:
    """
    Represents a data entity (waveform) with transformations/properties.

    Attributes:
        name (str): Entity name.
        data (np.ndarray): Primary data (waveform).
        position (float): 1D position.
        scale (float): Scaling factor.
        visible (bool): Visibility flag.
        physics_enabled (bool): Physics simulation flag.
        velocity (float): Velocity.
        acceleration (float): Acceleration.
        data_original (np.ndarray): Original data copy.
    """
    def __init__(self, data: np.ndarray, name: str = "Unnamed"):
        self.name: str = name
        self.data: np.ndarray = data
        self.position: float = 0.0
        self.scale: float = 1.0
        self.visible: bool = True
        self.physics_enabled: bool = False
        self.velocity: float = 0.0
        self.acceleration: float = 0.0
        self.data_original: np.ndarray = data.copy() if isinstance(data, np.ndarray) else None

    def apply_transform(self):
        """Applies transformations (scaling) to entity's data."""
        if isinstance(self.data, np.ndarray):
            self.data = self.data_original * self.scale

    def decimate(self, keep_ratio: float = 0.5):
        """Reduces data points for performance (large datasets)."""
        if not isinstance(self.data, np.ndarray) or len(self.data) < 2:
            return
        keep_count = max(2, int(len(self.data) * keep_ratio))
        stride = max(1, len(self.data) // keep_count)
        self.data = self.data_original[::stride]

    def restore_full_data(self):
        """Restores entity's data to original, full resolution."""
        if self.data_original is not None:
            self.data = self.data_original.copy()

    def update_physics(self, dt: float = 0.016):
        """Simulates simple 1D physics (position, velocity, acceleration)."""
        if self.physics_enabled:
            self.velocity += self.acceleration * dt
            self.position += self.velocity * dt

class Group:
    """
    Manages multiple Entities (group-level operations).

    Attributes:
        name (str): Group name.
        entities (List[Entity]): List of Entity objects.
        alignment (str): Alignment type ("center", "flex-start", "flex-end").
        lod_level (str): Level of Detail ("Fine", "Coarse").
    """
    def __init__(self, name: str = "Group"):
        self.name: str = name
        self.entities: List[Entity] = []
        self.alignment: str = "center"
        self.lod_level: str = "Fine"

    def add_entity(self, entity: Entity):
        """Adds an Entity to the group."""
        self.entities.append(entity)

    def remove_entity(self, entity: Entity):
        """Removes an Entity from the group."""
        if entity in self.entities:
            self.entities.remove(entity)

    def auto_align(self, spacing: float = 1.0):
        """Automatically positions entities in 1D based on alignment."""
        total = len(self.entities)
        if total == 0:
            return
        if self.alignment == "center":
            start = -((total - 1) * spacing) / 2.0
            for i, ent in enumerate(self.entities):
                ent.position = start + i * spacing
        elif self.alignment == "flex-start":
            for i, ent in enumerate(self.entities):
                ent.position = i * spacing
        elif self.alignment == "flex-end":
            start = -(total - 1) * spacing
            for i, ent in enumerate(self.entities):
                ent.position = start + i * spacing

    def context_aware_rescale(self):
        """Rescales entities based on the maximum amplitude within the group."""
        max_amp_global = 1e-9
        for ent in self.entities:
            if isinstance(ent.data, np.ndarray) and len(ent.data) > 0:
                max_amp_global = max(max_amp_global, np.max(np.abs(ent.data_original)))
        if max_amp_global < 1e-9:
            return
        for ent in self.entities:
            if isinstance(ent.data, np.ndarray) and len(ent.data) > 0:
                local_max = np.max(np.abs(ent.data_original))
                if local_max > 1e-9:
                    ent.scale = 1.0 / (max_amp_global / local_max)  # Corrected scaling
                ent.apply_transform()

    def cull_invisible(self, threshold: float = 0.001):
        """Sets entities invisible if their max amplitude is below threshold."""
        for ent in self.entities:
            if isinstance(ent.data, np.ndarray) and np.max(np.abs(ent.data)) < threshold:
                ent.visible = False
            else:
                ent.visible = True

    def set_lod(self, lod: str):
        """Sets Level of Detail for all entities in the group."""
        if lod not in ("Fine", "Coarse"):
            raise ValueError("LOD must be 'Fine' or 'Coarse'")
        self.lod_level = lod
        for ent in self.entities:
            if lod == "Coarse":
                ent.decimate(keep_ratio=0.2)
            else:
                ent.restore_full_data()

    def update_physics_all(self, dt: float = 0.016):
        """Updates physics for all entities in the group."""
        for ent in self.entities:
            ent.update_physics(dt)

    def get_visible_entities(self) -> List[Entity]:
        """Returns a list of only the visible entities."""
        return [ent for ent in self.entities if ent.visible]

class GrandUnifiedEngine:
    """
    Main engine class for waveform generation, classification, and visualization.

    Attributes:
        waveforms (List[np.ndarray]): List of waveform data.
        default_group (Group): Group object for managing waveform entities.
        normalized_tuple_sets_fine_lod (List[Tuple]): Normalized feature tuples (Fine LOD).
        normalized_tuple_sets_coarse_lod (List[Tuple]): Normalized feature tuples (Coarse LOD).
        custom_metrics (Dict[str, Callable]): Custom metric functions.
        current_visualization_mode (str): Current visualization mode.
        current_lod (str): Current Level of Detail.
        type_definitions (Dict): Loaded type definitions from JSON.
        user_type_definitions (Dict): Loaded user-defined type definitions.
    """
    def __init__(self, user_types_file: str = "user_types.json", standard_types_file: str = "standard_types.json"):
        self.waveforms: List[np.ndarray] = []
        self.default_group: Group = Group(name="DefaultGroup")
        self.normalized_tuple_sets_fine_lod: List[Tuple] = []
        self.normalized_tuple_sets_coarse_lod: List[Tuple] = []
        self.custom_metrics: Dict[str, Callable] = {}
        self.current_visualization_mode: str = "2D Waveform"
        self.current_lod: str = "Fine"
        self.type_definitions: Dict = self._load_type_definitions(standard_types_file)
        self.user_type_definitions: Dict = self._load_user_type_definitions(user_types_file)
        self._merge_type_definitions()

    def _load_type_definitions(self, file_path: str) -> Dict:
        """Loads standard type definitions from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            warnings.warn(f"Error loading standard types from '{file_path}': {e}. Using default types.")
            return {"types": {}, "classifications": {}}

    def _load_user_type_definitions(self, file_path: str) -> Dict:
        """Loads user-defined type definitions from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            warnings.warn(f"Error loading user types from '{file_path}': {e}. No user types loaded.")
            return {"classifications": {}}

    def _merge_type_definitions(self):
        """Merges user-defined classifications into standard definitions."""
        if 'classifications' in self.user_type_definitions:
            for name, definition in self.user_type_definitions['classifications'].items():
                if name in self.type_definitions['classifications']:
                    warnings.warn(f"User classification '{name}' overrides standard classification.")
                self.type_definitions['classifications'][name] = definition

    def register_metric(self, name: str, func: Callable[[np.ndarray], Union[float, str]]):
        """Registers a custom metric function for feature extraction."""
        if not isinstance(name, str) or not name:
            raise ValueError("Metric name must be a non-empty string.")
        if not callable(func):
            raise TypeError("Metric function must be callable.")
        self.custom_metrics[name] = func

    def generate_waveform(self, wave_type: str) -> Dict[str, str]:
        """Generates a waveform of the specified type."""
        wave_data = self._generate_waveform_data(wave_type)
        if wave_data is None:
            return {"status": f"Error: Unsupported waveform type '{wave_type}'."}

        self.waveforms.append(wave_data)
        entity = Entity(wave_data, name=f"{wave_type} Wave {len(self.waveforms)}")
        self.default_group.add_entity(entity)
        return {"status": f"Generated {wave_type} waveform (total: {len(self.waveforms)})."}

    def _generate_waveform_data(self, wave_type: str) -> Union[np.ndarray, None]:
        """Internal method to generate waveform data."""
        num_samples = 500 # Consistent sample count
        time = np.linspace(0, 1, num_samples, endpoint=False)
        try:
            if wave_type == "sine":
                return np.sin(2 * np.pi * 5 * time)
            elif wave_type == "square":
                return np.sign(np.sin(2 * np.pi * 3 * time))
            elif wave_type == "noise":
                return 0.5 * np.random.randn(num_samples)
            elif wave_type == "complex":
                return (np.sin(2 * np.pi * 5 * time) + 0.5 * np.sin(2 * np.pi * 12 * time) + 0.2 * np.random.randn(num_samples))
            elif wave_type == "time_dilation_example":
                return np.sin(2 * np.pi * 5 * np.linspace(0, 0.5, num_samples, endpoint=False))
            else:
                return None
        except Exception as e:
            warnings.warn(f"Error generating waveform '{wave_type}': {e}")
            return None

    def clear_waveforms(self) -> Dict[str, str]:
        """Clears all waveforms and resets data."""
        self.waveforms = []
        self.default_group.entities = []
        self.normalized_tuple_sets_fine_lod = []
        self.normalized_tuple_sets_coarse_lod = []
        return {"status": "Waveforms cleared."}

    def extract_features_fine_lod(self, waveform: np.ndarray) -> Tuple:
        """Extracts features for Fine Level of Detail."""
        try:
            yf = fft(waveform)
            xf = fftfreq(len(waveform))
            dominant_frequency = np.abs(xf[np.argmax(np.abs(yf))])
            rms_amplitude = np.sqrt(np.mean(waveform**2))
            magnitude_spectrum = np.abs(yf)
            frequency_axis = np.abs(xf)
            spectral_centroid = np.sum(frequency_axis * magnitude_spectrum) / np.sum(magnitude_spectrum) if np.sum(magnitude_spectrum) > 0 else 0.0
            spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=waveform)[0]))  # Ensure float
            zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y=waveform)[0]))  # Ensure float
            waveform_skewness = float(skew(waveform)) # Ensure float
            waveform_kurtosis = float(kurtosis(waveform)) # Ensure float
            custom_results = {name: func(waveform) for name, func in self.custom_metrics.items()}
            return (dominant_frequency, rms_amplitude, spectral_centroid, spectral_flatness, zero_crossing_rate, waveform_skewness, waveform_kurtosis, custom_results)
        except Exception as e:
            warnings.warn(f"Error in fine LOD feature extraction: {e}")
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {})

    def extract_features_coarse_lod(self, waveform: np.ndarray) -> Tuple:
        """Extracts features for Coarse Level of Detail."""
        try:
            yf = fft(waveform)
            xf = fftfreq(len(waveform))
            dominant_frequency = np.abs(xf[np.argmax(np.abs(yf))])
            rms_amplitude = np.sqrt(np.mean(waveform**2))
            custom_results = {name: func(waveform) for name, func in self.custom_metrics.items()}
            return (dominant_frequency, rms_amplitude, custom_results)
        except Exception as e:
            warnings.warn(f"Error in coarse LOD feature extraction: {e}")
            return (0.0, 0.0, {})

    def _minmax_scale(self, values: List[float]) -> np.ndarray:
      """Applies min-max scaling to a list of values."""
      return MinMaxScaler().fit_transform(np.array(values).reshape(-1, 1)).flatten()

    def create_normalized_tuple_set(self, lod_level: str = "Fine") -> List[Tuple]:
        """Creates normalized feature tuple sets."""
        if not self.waveforms:
            return []
        try:
            if lod_level == "Fine":
                feature_sets = [self.extract_features_fine_lod(wf) for wf in self.waveforms]
            elif lod_level == "Coarse":
                feature_sets = [self.extract_features_coarse_lod(wf) for wf in self.waveforms]
            else:
                raise ValueError("Invalid LOD level. Must be 'Fine' or 'Coarse'.")

            feature_sets_numeric = [fs[:-1] for fs in feature_sets]  # Exclude custom metrics
            num_features = len(feature_sets_numeric[0])
            normalized_feature_columns = [self._minmax_scale([fs[i] for fs in feature_sets_numeric]) for i in range(num_features)]
            return [tuple(normalized_feature_columns[j][i] for j in range(num_features)) for i in range(len(self.waveforms))]

        except Exception as e:
            warnings.warn(f"Error creating normalized tuple set: {e}")
            return []


    def classify_signal_type(self, waveform: np.ndarray) -> str:
        """Classifies the signal type based on spectral flatness."""
        try:
            flatness = np.mean(librosa.feature.spectral_flatness(y=waveform)[0])
            if flatness < 0.2:  return "Sine-like/Tonal"
            elif flatness > 0.8: return "Noise-like"
            else:               return "Complex/Mixed"
        except Exception as e:
            warnings.warn(f"Error classifying signal type: {e}")
            return "Unknown"

    def classify_waveforms(self) -> Dict[str, Union[str, List[Tuple]]]:
        """Classifies waveforms and updates normalized tuple sets."""
        if not self.waveforms:
            return {"status": "No waveforms to classify."}
        self.normalized_tuple_sets_fine_lod = self.create_normalized_tuple_set("Fine")
        self.normalized_tuple_sets_coarse_lod = self.create_normalized_tuple_set("Coarse")
        return {"status": "Waveforms classified.", "fine_lod_tuples": self.normalized_tuple_sets_fine_lod, "coarse_lod_tuples": self.normalized_tuple_sets_coarse_lod}


    def set_lod_level(self, lod_level: str) -> Dict[str, str]:
        """Sets the Level of Detail (LOD)."""
        if lod_level not in ("Fine", "Coarse"):
            return {"status": f"Error: Invalid LOD '{lod_level}'. Must be 'Fine' or 'Coarse'."}
        self.current_lod = lod_level
        self.default_group.set_lod(lod_level)
        return {"status": f"LOD set to {lod_level}."}

    def auto_layout_rescale(self) -> Dict[str, str]:
        """Automatically layouts and rescales visible waveforms."""
        self.default_group.auto_align(spacing=1.0)
        self.default_group.context_aware_rescale()
        self.default_group.cull_invisible()
        return {"status": "Layout and rescale applied."}

    def animate_all(self, dt: float = 0.016) -> Dict[str, str]:
        """Performs one step of physics animation."""
        self.default_group.update_physics_all(dt)
        return {"status": "Physics animation step completed."}

    def get_visualization_data(self, mode: str = None, overlay: str = None, time_dilation_factor: float = 1.0) -> Dict:
        """
        Prepares data for visualization based on the selected mode and overlays.

        Args:
            mode: Visualization mode ("2D Waveform", "Heatmap", "4D Visualization").
            overlay: Overlay type ("show_scale", "show_class_key").
            time_dilation_factor: Factor to scale the time axis (2D Waveform).

        Returns:
            Dictionary containing visualization data or error message.
        """
        mode = mode or self.current_visualization_mode
        visible_entities = self.default_group.get_visible_entities()
        visible_waveforms = [ent.data for ent in visible_entities if isinstance(ent.data, np.ndarray)]

        if not visible_waveforms:
            return {"mode": mode, "error": "No visible waveforms to display."}

        try:
            if mode == "2D Waveform":
                traces = []
                for ent in visible_entities:
                    wf = ent.data
                    t_local = np.linspace(0, 1 * time_dilation_factor, len(wf), endpoint=False).tolist()  # Apply dilation
                    traces.append({"x": t_local, "y": wf.tolist(), "name": ent.name, "type": "line"})
                return {"mode": "2D Waveform", "data": traces}

            elif mode == "Heatmap":
                wf = visible_waveforms[-1]
                if len(wf) < 2:
                    return {"mode": "Heatmap", "error": "Waveform too short for heatmap."}
                spec_data = gaussian_filter(np.log10(np.abs(fft(wf)) + 1e-9), sigma=2)  # Gaussian smoothing
                return {"mode": "Heatmap", "data": spec_data.tolist()}  # Return heatmap data

            elif mode == "4D Visualization":
                if self.current_lod == "Fine":
                    tuple_set = self.normalized_tuple_sets_fine_lod
                    feature_names = ["Norm. Dominant Freq", "Norm. RMS Amp", "Norm. Spec. Centroid"]
                else:
                    tuple_set = self.normalized_tuple_sets_coarse_lod
                    feature_names = ["Norm. Dominant Freq", "Norm. RMS Amp", "Placeholder"]

                if not tuple_set:
                    return {"mode": "4D Visualization", "data": None, "error": "Please classify waveforms first."}

                visible_tuples, visible_labels = [], []
                for i, wf in enumerate(self.waveforms):
                    if self.default_group.entities[i].visible:
                        visible_tuples.append(tuple_set[i])
                        visible_labels.append(self.classify_signal_type(wf))

                if not visible_tuples:
                    return {"mode": "4D Visualization", "data": None, "error": "No visible waveforms for 4D."}

                return {
                    "mode": "4D Visualization",
                    "x": [t[0] for t in visible_tuples],
                    "y": [t[1] for t in visible_tuples],
                    "z": [t[2] if len(t) > 2 else 0 for t in visible_tuples],
                    "labels": visible_labels,
                    "features": feature_names
                }
            else:
                return {"mode": mode, "error": f"Invalid visualization mode: {mode}."}

        except Exception as e:
            error_msg = f"Error generating visualization data for mode '{mode}': {e}"
            warnings.warn(error_msg)
            return {"mode": mode, "error": error_msg}
