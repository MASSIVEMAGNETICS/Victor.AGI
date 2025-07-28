# Victor HyperFractal Command Deck

This folder contains a basic frontend and backend skeleton for the interactive Victor GUI.

## Frontend
- React 18 with TailwindCSS and Three.js.
- Framer Motion handles small interactions.
- Start with `npm install` and `npm run dev` inside `frontend/`.

## Backend
- FastAPI WebSocket endpoint at `/ws/victor` returning random load data every 200ms.
- Run using `uvicorn main:app --reload` in the `backend/` folder.
