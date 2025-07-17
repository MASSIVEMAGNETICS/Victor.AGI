import React, { useState } from 'react';
import type { LogEntry } from './types';

interface MusicGenViewProps {
  addLog: (message: string, type: LogEntry['type']) => void;
}

const MusicGenView: React.FC<MusicGenViewProps> = ({ addLog }) => {
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
      const mockResponse = `🎵 (mock) Music based on: "${prompt}" 🎵`;
      setGeneratedMusic(mockResponse);
      addLog(`Generated music: ${mockResponse}`, 'VICTOR');
      setIsGenerating(false);
    }, 2000);
  };

  return (
    <div className="flex flex-col h-full bg-obsidian text-violet">
      <div className="p-4">
        <h2 className="text-2xl font-bold mb-4 text-shadow-glow">Music Generation</h2>
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
      <div className="flex-grow p-4 overflow-y-auto">
        {isGenerating && <p>Generating music...</p>}
        {generatedMusic && (
          <div className="p-4 bg-gray-800 rounded-md">
            <p>{generatedMusic}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default MusicGenView;
