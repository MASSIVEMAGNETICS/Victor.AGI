# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`

## Experimental Model
The repository includes `paradigm_shatterer_model.py`, a small prototype
featuring fractal-inspired fusion blocks intended to bypass typical scaling
limits. Run it with `python paradigm_shatterer_model.py` to see a dummy
forward pass.
