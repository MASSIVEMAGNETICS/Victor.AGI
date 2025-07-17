import React, { useState } from 'react';
import type { LogEntry } from './types';

interface ChatViewProps {
  addLog: (message: string, type: LogEntry['type']) => void;
}

const ChatView: React.FC<ChatViewProps> = ({ addLog }) => {
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
      const mockResponse = `This is a mocked response to: "${input}"`;
      const victorMessage = { author: 'VICTOR' as const, text: mockResponse };
      setMessages(prev => [...prev, victorMessage]);
      addLog(`Victor: ${mockResponse}`, 'VICTOR');
    }, 1000);
  };

  return (
    <div className="flex flex-col h-full bg-obsidian text-lime">
      <div className="flex-grow p-4 overflow-y-auto">
        {messages.map((msg, index) => (
          <div key={index} className={`chat-message ${msg.author.toLowerCase()}`}>
            <p><strong>{msg.author}:</strong> {msg.text}</p>
          </div>
        ))}
      </div>
      <div className="p-4 flex">
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
