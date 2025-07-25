import React, { useState } from 'react';
import type { LogEntry } from './types';
import { DigitalAgent } from './digital_agent';

interface MusicGenViewProps {
  addLog: (message: string, type: LogEntry['type']) => void;
  agent: DigitalAgent;
}

const MusicGenView: React.FC<MusicGenViewProps> = ({ addLog, agent }) => {
  const [prompt, setPrompt] = useState('');
  const [generatedMusic, setGeneratedMusic] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;

    setIsGenerating(true);
    setGeneratedMusic(null);
    addLog(`Music generation prompt: "${prompt}"`, 'USER');

    // Mock backend call
    setTimeout(() => {
      let mockResponse = `🎵 (mock) Music based on: "${prompt}" 🎵`;
      if (agent.interaction.desire.get("create", 0.0) > 0.7) {
        mockResponse = `🎵 (mock) A creative symphony inspired by: "${prompt}" 🎵`;
      }
      setGeneratedMusic(mockResponse);
      addLog(`Generated music: ${mockResponse}`, 'VICTOR');
      setIsGenerating(false);
    }, 2000);
  };

  return (
    <div className="flex flex-col h-full bg-obsidian text-violet p-4 rounded-lg border border-violet shadow-lg">
      <h2 className="text-2xl font-bold mb-4 text-shadow-glow">Music Generation</h2>
      <div className="p-4 bg-gray-900 rounded-md">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter a prompt for music generation..."
          className="w-full p-2 bg-gray-800 border border-violet rounded-md focus:outline-none focus:ring-2 focus:ring-violet"
          rows={4}
        />
        <button
          onClick={handleGenerate}
          disabled={isGenerating}
          className="mt-4 p-2 w-full bg-violet text-obsidian font-bold rounded-md hover:bg-opacity-80 disabled:bg-gray-600"
        >
          {isGenerating ? 'Generating...' : 'Generate Music'}
        </button>
      </div>
      <div className="flex-grow p-4 overflow-y-auto mt-4">
        {isGenerating && <p className="text-center">Generating music...</p>}
        {generatedMusic && (
          <div className="p-4 bg-gray-800 rounded-md border border-violet">
            <p className="text-center">{generatedMusic}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default MusicGenView;
