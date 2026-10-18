# Interactive Wireless Multipath Signal Propagation and Degradation Simulator

A Streamlit web app that shows how a signal arriving over several paths
(direct + reflected) is shaped by path length, delay, attenuation, phase and
reflection, using

    y(t) = sum_i  A_i * cos(2*pi*f*(t - tau_i) + phi_i),   tau_i = L_i / c

Results are **simulated observations**, not laboratory measurements. The 2D
scene uses simple image-method reflections; it is not a full electromagnetic solver.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `signal_generator.py` | Sine / digital waveforms, fractional delay |
| `propagation_model.py` | Paths, geometry, combining, sweeps, responses |
| `signal_analysis.py` | Amplitude, RMS, power, dB, SNR, distortion, interpretation |
| `visualizations.py` | Plotly figures |
| `experiments.py` | Presets A-H |
| `report_export.py` | CSV / summary export |
| `tests/test_model.py` | 39 validation tests |

## Run locally
    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py

## Run the tests (no Streamlit/Plotly needed)
    python -m unittest discover -s tests -v

## Put it on GitHub
1. Create an empty repository on github.com (e.g. `multipath-simulator`).
2. In this folder:

        git init
        git add .
        git commit -m "Multipath simulator"
        git branch -M main
        git remote add origin https://github.com/<your-username>/multipath-simulator.git
        git push -u origin main

   Or use "Add file > Upload files" on GitHub and drag in the folder contents.

## Deploy on Streamlit Community Cloud (free)
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app**, choose your repository, branch `main`, main file path `app.py`.
3. Click **Deploy**. You get a public HTTPS URL like `https://<name>.streamlit.app`.
4. Open that URL on your phone; the layout stacks on narrow screens.

## Troubleshooting
- **ModuleNotFoundError on deploy**: the package is missing from `requirements.txt`; add it, commit, push.
- **App shows old version**: use "Reboot app" in the Streamlit Cloud menu.
- **Wrong main file**: in app settings, set main file path to `app.py`.
- **Slow with many samples**: lower sampling resolution or duration in the sidebar.
- **`pip` can't find packages**: check your internet/proxy and Python version (3.9+).
