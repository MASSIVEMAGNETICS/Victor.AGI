# Run the Victor.AGI app

This repository contains a Vite frontend and experimental Python model prototypes.

## Run Locally

**Prerequisites:** Node.js

1. Install dependencies: `npm install`
2. Start the development server: `npm run dev`

## Local-first credential boundary

The browser frontend does not require a provider API key. Do not put provider credentials in `.env.local` or inject them into the Vite client bundle. Any future provider integration must run behind an explicitly approved local/server-side adapter and must never expose credentials to browser code.

This credential-boundary change does not add a local inference backend. The frontend-to-local-model integration remains a separate implementation requirement.

## Experimental Model

The repository includes `paradigm_shatterer_model.py`, a small prototype featuring fractal-inspired fusion blocks intended to explore alternative model architectures. Run it with `python paradigm_shatterer_model.py` to see its dummy forward pass.
