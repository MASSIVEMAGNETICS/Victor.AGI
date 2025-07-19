# VictorTensor AI Simulation

This project is a sophisticated AI simulation environment that provides a unique glimpse into the inner workings of an advanced artificial intelligence. It combines a visually rich frontend with a powerful backend that features a custom-built GPT model with a novel "Flower of Life" neural architecture.

## Key Features

- **Interactive AI Simulation:** Observe the AI as it cycles through different states like "THINKING," "EVOLVING," and "IDLE."
- **Code Evolution:** Witness the AI's code self-mutate and evolve over time, with a detailed history of all changes.
- **3D Neural Visualization:** Explore a stunning 3D representation of the AI's "Flower of Life" neural mesh, providing a unique visual insight into its structure.
- **Advanced AI Models:** The backend is powered by a custom GPT implementation featuring a Mixture of Experts (MoE) layer and a unique "VictorTensor" engine.
- **Modular and Extensible:** The project is designed with a clear separation of concerns, making it easy to understand, modify, and extend.

## Project Structure

The project is organized into the following main directories:

- `src/`: Contains the frontend source code, built with React and TypeScript.
- `components/`: Reusable React components for the UI.
- `core/`: The heart of the AI, featuring the "Flower of Life" neural mesh and custom deep learning blocks.
- `bark_victortensor/`: A specialized GPT model with self-evolving capabilities.
- `text_gpt/`: A general-purpose GPT model for text generation.
- `public/`: Static assets, including the `output.json` file that feeds the 3D visualization.

## Getting Started

### Prerequisites

- Node.js
- Python 3

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/victor-tensor-ai.git
   cd victor-tensor-ai
   ```

2. **Install frontend dependencies:**
   ```bash
   npm install
   ```

3. **Install backend dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your environment:**
   - Create a `.env.local` file in the root directory.
   - Add your Gemini API key to the file:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```

### Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`

## How It Works

The simulation is driven by a continuous loop of "thinking" and "evolving." The AI's core logic resides in the `core/flower_of_life.py` script, which generates a complex neural mesh. This mesh is then visualized in the frontend using the `FlowerOfLife` component.

The AI's "thoughts" are powered by the custom GPT models in the `bark_victortensor/` and `text_gpt/` directories. These models are designed to be highly efficient and feature advanced techniques like Mixture of Experts.

The AI's ability to "evolve" is a key feature of this simulation. The `meta_evolution` module in `bark_victortensor/model.py` allows the AI to modify its own code, leading to a constantly changing and improving system.

## Contributing

Contributions are welcome! If you have any ideas for new features, improvements, or bug fixes, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
