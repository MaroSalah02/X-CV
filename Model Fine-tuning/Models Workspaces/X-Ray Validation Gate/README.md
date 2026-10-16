# X-ray validation gate

The region and fracture models always return an answer, even for a photo of a book or a person.
This gate runs first in the Pipeline. When it decides the upload is not an X-ray the app can
analyze, the scan stops with a clear message.

- **Model:** EfficientNet-B0 (ImageNet weights), binary output = P(X-ray), 224px, pad-to-square.
  Checkpoint: `Pipeline/Inference_Modules/models/xray_validation/xray_gate.pt` (16 MB), which
  stores its threshold (0.5) and preprocessing.
- **Pipeline:** `Pipeline/Inference_Modules/xray_validation.py`. `pipeline.py` sends an
  `xray_check` step, and a rejected upload ends with `{"step": "pipeline", "status": "rejected"}`.
- **Training:** `train_xray_gate.py`, run as the private Kaggle kernel
  `seifeldinmoheb/xcv-xray-validation-gate` (T4, about 50 min). For a CPU smoke test set
  `XCV_SMOKE=1`, `XCV_INPUT_ROOT` and `XCV_OUT_DIR`.

## Data
| | X-rays | Not X-rays |
|---|---|---|
| **Training** | FracAtlas, MURA, paediatric chest, knee (25k) | ImageNet-mini, book covers, LFW faces, scanned documents, brain MRI, chest CT, breast ultrasound, generated blank / flat images (43k) |
| **Held-out test** (never trained on) | GRAZPEDWRI-DX wrists, UNIFESP body parts, TB chest X-rays, dental panoramics | Caltech-256 objects, UTKFace (people of all ages), Intel scenes, IAM handwriting, a second MRI set, blank / flat images |

Training adds simulated phone photos of X-ray films (colour cast, uneven lighting, border,
perspective, blur, JPEG). Non-X-rays are randomly converted to grayscale, so the model cannot
rely on colour.

## Held-out results (threshold 0.5)
| Source | Correct |
|---|---|
| GRAZPEDWRI-DX wrist X-rays accepted | 99.9% |
| TB chest X-rays accepted | 99.5% |
| Simulated phone photos of X-rays accepted | 99.6–100% |
| UNIFESP body-part X-rays accepted | 96.9% (see below) |
| Caltech-256 photos rejected (colour / grayscale) | 99.8% / 99.4% |
| UTKFace people rejected (colour / grayscale) | 100% / 100% |
| Scenes, handwriting, MRI, blank / flat images rejected | 99.9–100% |

Overall, 99.87% of the 10,800 held-out non-X-rays are rejected.

## Known limitations
- **Rejected on purpose or acceptably:** DXA bone-density scans, CT scout images, skull X-rays
  and dental panoramics. These make up most of the UNIFESP rejects, and about 40% of dental
  panoramics are rejected. None of them is one of the 9 regions the app supports, so the
  message says "not an X-ray that X-CV can analyze".
- **Occasionally accepted:** shiny metal objects on a white background (tuning fork, tweezers),
  white mattresses and single handwritten letters. This is about 0.1% of held-out photos.
- Phone photos of X-ray films were only tested as simulations, not real photos.
