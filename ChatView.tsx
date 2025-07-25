import React, { useState, useEffect } from 'react';
import type { LogEntry } from './types';
import { DigitalAgent } from './digital_agent'; // Assuming digital_agent is in the same directory

interface ChatViewProps {
  addLog: (message: string, type: LogEntry['type']) => void;
  agent: DigitalAgent; // Pass the agent as a prop
}

const ChatView: React.FC<ChatViewProps> = ({ addLog, agent }) => {
  const [messages, setMessages] = useState<{ author: 'USER' | 'VICTOR', text: string }[]>([]);
  const [input, setInput] = useState('');

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { author: 'USER' as const, text: input };
    setMessages(prev => [...prev, userMessage]);
    addLog(`User: ${input}`, 'USER');
    setInput('');

    // Mock backend call
    setTimeout(() => {
      let mockResponse = `This is a mocked response to: "${input}"`;
      if (agent.interaction.desire.get("create", 0.0) > 0.7) {
        mockResponse = `I feel creative. Here's a poem about ${input}: ...`;
      }
      const victorMessage = { author: 'VICTOR' as const, text: mockResponse };
      setMessages(prev => [...prev, victorMessage]);
      addLog(`Victor: ${mockResponse}`, 'VICTOR');
    }, 1000);
  };

  return (
    <div className="flex flex-col h-full bg-obsidian text-lime p-4 rounded-lg border border-lime shadow-lg">
      <h2 className="text-2xl font-bold mb-4 text-shadow-glow">Chat with Victor</h2>
      <div className="flex-grow p-4 overflow-y-auto bg-gray-900 rounded-md">
        {messages.map((msg, index) => (
          <div key={index} className={`mb-4 flex ${msg.author === 'USER' ? 'justify-end' : 'justify-start'}`}>
            <div className={`p-3 rounded-lg ${msg.author === 'USER' ? 'bg-violet text-white' : 'bg-lime text-obsidian'}`}>
              <p><strong>{msg.author}:</strong> {msg.text}</p>
            </div>
          </div>
        ))}
      </div>
      <div className="p-4 flex mt-4">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          className="flex-grow p-2 bg-gray-800 border border-lime rounded-l-md focus:outline-none focus:ring-2 focus:ring-lime"
        />
        <button
          onClick={handleSend}
          className="p-2 bg-lime text-obsidian font-bold rounded-r-md hover:bg-opacity-80"
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatView;
