# Visual Recognizability in Image Geolocation: An Occlusion and Counterfactual Analysis of StreetCLIP

This repository contains the data and code accompanying the bachelor thesis *"Visual Recognizability in Image Geolocation: An Occlusion and Counterfactual Analysis of StreetCLIP"* (Vrije Universiteit Amsterdam, 2026).

The thesis investigates the interpretability of [StreetCLIP](https://huggingface.co/geolocal/StreetCLIP), a vision-language model for image geolocation, using two perturbation-based attribution methods: occlusion sensitivity and counterfactual patch masking. The analysis was conducted across 25,000 street-level images spanning 50 countries, collected via the Google Street View Static API.

---

## Repository Structure

```
├── coords.csv          # Coordinates and camera headings used in experiments
├── occlusion.py        # Occlusion sensitivity analysis
├── counterfactual.py   # Counterfactual patch masking analysis
└── README.md
```

---

## Data

### `coords.csv`

Contains the geographic coordinates and camera headings for all images used in the experiments. Images themselves are not included due to Google Street View Terms of Service restrictions on redistribution. The images can be re-downloaded using the coordinates and the Google Street View Static API.

| Column | Description |
|---|---|
| `country` | Ground truth country label |
| `lat` | Latitude of the snapped Street View location |
| `lon` | Longitude of the snapped Street View location |
| `heading` | Camera heading in degrees (0–359) |

To re-download the images, use the following API endpoint with your own API key:

```
https://maps.googleapis.com/maps/api/streetview?size=336x336&location={lat},{lon}&heading={heading}&pitch=0&fov=90&key=YOUR_API_KEY
```

---

## Code

### `occlusion.py`

Runs occlusion sensitivity analysis on one picture by covering one of 49 patches at the time in order to measure the importance of that patch to the final prediction.


---

### `counterfactual.py`

Runs counterfactual analysis on one picture by covering one patch at the time (in descending order of importance) in order to record the minimum number of patches needed to switch the prediction.

---

## Model

This project uses [StreetCLIP](https://huggingface.co/geolocal/StreetCLIP) by Haas et al. (2023). The model is loaded automatically via the HuggingFace `transformers` library:

```python
from transformers import CLIPProcessor, CLIPModel
model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
```
