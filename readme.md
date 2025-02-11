# Grand Unified Visualization & Scientific Engine

A **single** Python/Flask application that integrates:

- **Waveform generation** (sine, square, noise, complex)  
- **Feature extraction & classification** (dominant frequency, RMS amplitude, spectral centroid, etc.)  
- **Grouping, LOD, decimation, and culling**  
- **Physics-based animation stubs**  
- **Multiple visualization modes** (2D Waveform, Heatmap, 4D “3D scatter” Visualization)  
- **Overlays and labeling** (scale bars, classification keys)  

Everything is in one file, **`GrandUnifiedApp.py`**, for ease of demonstration.

---

## Features at a Glance

1. **Waveforms**:  
   Generate any number of waveforms (sine, square, noise, complex).  
   Each waveform is stored as an “entity” in a default “group,” allowing further alignment, rescaling, or decimation.  

2. **Classification & Feature Extraction**:  
   - Fine LOD: Extract dominant frequency, RMS amplitude, spectral centroid, spectral flatness, zero-crossing rate, skewness, kurtosis, plus optional custom metrics.  
   - Coarse LOD: Dominant frequency, RMS amplitude, plus custom metrics.  
   - A rudimentary classification by spectral flatness (“Sine-like/Tonal,” “Noise-like,” or “Complex/Mixed”).  

3. **LOD & Decimation**:  
   - “Fine” mode: Keep all waveform data.  
   - “Coarse” mode: Automatically decimates each waveform to ~20% of its samples.  
   - Optionally revert to full data when returning to “Fine.”  

4. **Physics / Animation**:  
   - Each waveform entity has basic velocity/acceleration for a toy physics step.  
   - A single “Animate” button calls the physics update function.  

5. **Visualization Modes** (served via Flask + Plotly):
   - **2D Waveform**: Time vs. amplitude for all visible waveforms.  
   - **Heatmap**: Log-spectrum of the **last** visible waveform, smoothed with a Gaussian filter.  
   - **4D Visualization**: 3D scatter using normalized features (like Dominant Freq, RMS Amp, etc.).  

6. **Overlays**:
   - A dropdown with “None,” “show_scale,” and “show_class_key.”  
   - For 2D waveforms: a small annotation about amplitude scale.  
   - For Heatmap: a text label for frequency domain.  
   - For 4D scatter: a classification legend.  

---

## Requirements

You need Python 3.7+ and a few libraries:
- **Flask** (for the web server)
- **Plotly** (loaded via CDN in the HTML template)
- **numpy**, **scipy**, **scikit-learn**, **librosa**, **requests** (if you plan to test endpoints externally)
- **matplotlib** (optional, only if you want deeper data inspection, not required for this example)

Install the Python dependencies with:

```bash
pip install flask scipy scikit-learn librosa

(numpy will usually be installed as a dependency of these packages.)

Getting Started
	1.	Clone or Download this repository.
	2.	Make sure your Python environment is active.
	3.	Run the main script:

python GrandUnifiedApp.py

	4.	The server will start on http://0.0.0.0:5000 by default (or localhost:5000).
	5.	Open a web browser and navigate to http://localhost:5000.

File Overview
	•	GrandUnifiedApp.py
The single script containing everything:
	•	GrandUnifiedEngine: Core “engine” class for generating, storing, classifying waveforms, and returning Plotly data.
	•	Entity & Group classes: Demonstrate grouping, alignment, LOD decimation, culling, etc.
	•	Flask routes: /api/generate, /api/clear, /api/classify, /api/layout_rescale, /api/set_lod, /api/animate, /api/visualization.
	•	HTML_TEMPLATE: A simple front-end with Plotly.

Usage Instructions
	1.	Generate waveforms
Use the top-row buttons in the UI (“Generate Sine,” “Generate Square,” etc.) to add waveforms to the system.
	2.	Clear All
Removes all stored waveforms.
	3.	Classify
Runs feature extraction for “Fine” and “Coarse” sets, storing them in the engine.
	•	This also enables “4D Visualization” since it depends on the classification results.
	4.	AutoLayout & Rescale
Calls a simple alignment (positions waveforms in 1D) plus amplitude normalization.
	•	Entities with amplitude below a threshold become invisible.
	5.	LOD
	•	Click “LOD: Coarse” to decimate waveforms to ~20% of their samples.
	•	“LOD: Fine” returns them to full resolution.
	6.	Animate Physics
This performs one step of a very basic “position += velocity * dt” update.
	•	Entities must have physics_enabled=True to move. Currently, it’s an example stub.
	7.	Visualization Mode
	•	2D Waveform: Plots time vs. amplitude for each visible waveform.
	•	Heatmap: Shows the log-spectrum (FFT) of the last visible waveform, smoothed.
	•	4D Visualization: 3D scatter plot. The first 3 normalized features become X, Y, Z. Points are labeled by classification.
	8.	Overlay
	•	“None”: No extra annotations.
	•	“Show Scale Bar”: 2D waveforms get a small “Amplitude Scale ~1.0” note, Heatmap/4D get textual hints.
	•	“Show Classification Key”: 4D scatter can show a legend mapping color => classification type.
	9.	Plotly
	•	The front-end uses Plotly’s scattergl (for 2D waveforms) and scatter3d (for 4D), plus heatmap.
	•	The server returns JSON describing the traces, and the front-end calls Plotly.newPlot(...).

Example Flow
	1.	Generate Waveforms: Click “Generate Sine” twice, “Generate Noise” once.
	2.	Classify: Click “Classify.”
	3.	AutoLayout & Rescale: Notice the amplitude alignment.
	4.	Visualization:
	•	Switch to “2D Waveform” => You’ll see multiple lines.
	•	Switch to “Heatmap” => You’ll see a single heatmap for the last visible waveform.
	•	Switch to “4D Visualization” => Each waveform becomes a point in 3D, with color based on classification.
	5.	Overlays:
	•	Select “Show Scale Bar” => A small annotation about amplitude or frequency domain might appear.
	•	Select “Show Classification Key” => In 4D Visualization, you’ll see color => label info.

Extending the System
	•	Custom Metrics:
Uncomment or add code to register_metric("crest_factor", crest_factor) inside the main block to incorporate extra feature calculations in the classification.
	•	Physics:
For real motion, set entity.physics_enabled=True and specify velocities or accelerations. Then repeatedly call “Animate Physics.”
	•	4D → 3D:
The “4D Visualization” is actually a 3D scatter. In principle, you could have more advanced 4D slicing or partial rendering.
	•	Advanced Layout:
Currently, layout is only 1D. You can easily expand to 2D grid layout or something akin to CSS Flexbox.
	•	Scaling up:
If you want to handle thousands of waveforms or large data, consider an asynchronous server, or use a more specialized visualization approach.

Contributing
	1.	Fork this repository
	2.	Create a feature branch (git checkout -b feature/YourFeature)
	3.	Commit your changes (git commit -am 'Add new feature')
	4.	Push to the branch (git push origin feature/YourFeature)
	5.	Create a new Pull Request

License

This demo code is provided under a permissive license (e.g. MIT). See the LICENSE file for details.

