import React, { useState, useEffect, useCallback } from 'react';
import type { FileSystem, LogEntry, AppView } from './types';
import { INITIAL_FILESYSTEM } from './constants';
import TopStatusBar from './components/TopStatusBar';
import FileTree from './components/FileTree';
import CodeViewer from './components/CodeViewer';
import LeftDock from './components/LeftDock';
import Console from './components/Console';
#<<<<<<< phoenix-hotfix
import NodeGraph from './components/NodeGraph';
import ChatView from './ChatView';
=======
import FlowerOfLife from './components/FlowerOfLife';
import ChatView from './components/ChatView';
#>>>>>>> main
import TrainView from './components/TrainView';
import DockView from './components/DockView';
import MusicGenView from './MusicGenView';
import AgentView from './AgentView';
import { DigitalAgent } from './digital_agent';

const App: React.FC = () => {
  const [fileSystem, setFileSystem] = useState<FileSystem>(INITIAL_FILESYSTEM);
  const [selectedFile, setSelectedFile] = useState<string>('victor_boot.py');
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [evolutionHistory, setEvolutionHistory] = useState<string[]>([]);
  const [status, setStatus] = useState<'THINKING' | 'EVOLVING' | 'IDLE' | 'BOOTING'>('BOOTING');
  const [lastEvolution, setLastEvolution] = useState('N/A');
  const [activeView, setActiveView] = useState<AppView>('FLOWER_OF_LIFE');
  const [countdown, setCountdown] = useState(20);
  const [agent, setAgent] = useState(new DigitalAgent());

  const addLog = useCallback((message: string, type: LogEntry['type']) => {
    setLogs(prevLogs => [
      {
        id: Date.now() + Math.random(),
        timestamp: new Date().toLocaleTimeString(),
        message,
        type,
      },
      ...prevLogs.slice(0, 199),
    ]);
  }, []);

  // Main Simulation Loop
  useEffect(() => {
    addLog('VICTOR FRACTAL KERNEL V2 BOOTING...', 'SYSTEM');
    addLog('GODCORE RUNTIME ENGAGED.', 'SYSTEM');
    
    const thinkInterval = setInterval(() => {
      setStatus('THINKING');
      const directives = [
        "optimize_self", "analyze_memory", "generate_directive_tree", "simulate_timeline"
      ];
      const directive = directives[Math.floor(Math.random() * directives.length)];
      addLog(`Executing directive: ${directive}`, 'THOUGHT');
      setTimeout(() => setStatus('IDLE'), 500);
    }, 3000);

    const evolveInterval = setInterval(() => {
      setStatus('EVOLVING');
      addLog('Recursive self-evolution loop initiated.', 'EVOLUTION');
      
      setFileSystem(prevFs => {
        const sourceFile = 'core/victor_brain.py';
        const codeToMutate = prevFs[sourceFile];
        const newLines = codeToMutate.split('\n').map(line => 
          line.includes('# MUTATABLE') 
            ? line.replace(/\.\.\./g, `... v${Math.floor(100 + Math.random() * 900)}`) 
            : line
        );
        return { ...prevFs, [sourceFile]: newLines.join('\n') };
      });

      const timestamp = new Date().toISOString().replace(/[-:.]/g, '').slice(0, 15);
      const backupFile = `victor_brain.py.${timestamp}.bak`;
      setEvolutionHistory(prev => [backupFile, ...prev]);
      const evolutionTime = new Date().toLocaleTimeString();
      setLastEvolution(evolutionTime);
      setCountdown(20);
      addLog(`Mutation complete. New iteration saved: ${backupFile}`, 'EVOLUTION');
      setTimeout(() => setStatus('IDLE'), 1000);
    }, 20000);

    const countdownInterval = setInterval(() => {
        setCountdown(prev => (prev > 0 ? prev - 1 : 20));
    }, 1000);

    const agentInterval = setInterval(() => {
        agent.run_self_diagnostics();
        agent.experience_event("Simulated sensory input", { "joy": 0.1, "fear": Math.random() * 0.1 });
        setAgent(new DigitalAgent(agent.generation, agent.ancestry)); // Trigger re-render
    }, 5000);

    return () => {
        clearInterval(thinkInterval);
        clearInterval(evolveInterval);
        clearInterval(countdownInterval);
        clearInterval(agentInterval);
    };
  }, [addLog, agent]);
  
  const renderActiveView = () => {
    switch (activeView) {
      case 'OVERVIEW':
        return (
          <div className="grid grid-cols-10 gap-4 h-full">
            <div className="col-span-3 h-full">
              <FileTree 
                files={fileSystem} 
                selectedFile={selectedFile} 
                onSelectFile={setSelectedFile}
                evolutionHistory={evolutionHistory}
              />
            </div>
            <div className="col-span-7 h-full">
              <CodeViewer 
                key={selectedFile} 
                filePath={selectedFile} 
                code={fileSystem[selectedFile] || 'File not found.'}
              />
            </div>
          </div>
        );
      case 'FLOWER_OF_LIFE':
        return <FlowerOfLife />;
      case 'CHAT':
        return <ChatView addLog={addLog} agent={agent} />;
      case 'TRAIN':
        return <TrainView />;
      case 'DOCK':
        return <DockView />;
      case 'MUSIC':
        return <MusicGenView addLog={addLog} agent={agent} />;
      case 'AGENT':
        return <AgentView agent={agent} />;
      default:
        return <NodeGraph status={status} />;
    }
  };

  return (
    <div className="h-screen flex flex-col bg-obsidian font-mono overflow-hidden">
      <style>{`
        :root {
          --c-obsidian: #0a0b12;
          --c-cyan: #00eaff;
          --c-violet: #a200ff;
          --c-amber: #ff9f00;
          --c-magenta: #ff007a;
          --c-lime: #a3ff00;
        }
        .bg-obsidian { background-color: var(--c-obsidian); }
        .bg-grid-glow {
          background-image: 
            radial-gradient(circle at center, rgba(0, 234, 255, 0.1) 0, transparent 40%),
            linear-gradient(rgba(0, 234, 255, 0.03) 1px, transparent 1px), 
            linear-gradient(90deg, rgba(0, 234, 255, 0.03) 1px, transparent 1px);
          background-size: 100% 100%, 2rem 2rem, 2rem 2rem;
          background-position: center center;
        }
        .text-shadow-glow { text-shadow: 0 0 8px currentColor, 0 0 12px currentColor; }
        .box-shadow-glow { box-shadow: 0 0 12px 0px currentColor; }
        .border-glow { border-color: currentColor; }

        @keyframes pulse-glow {
          0%, 100% { opacity: 0.7; transform: scale(1); }
          50% { opacity: 1; transform: scale(1.02); }
        }
        .animate-pulse-glow { animation: pulse-glow 2.5s cubic-bezier(0.4, 0, 0.6, 1) infinite; }

        @keyframes pulse-ring {
          0% { box-shadow: 0 0 0 0px rgba(0, 234, 255, 0.5); }
          100% { box-shadow: 0 0 0 10px rgba(0, 234, 255, 0); }
        }
        .animate-pulse-ring { animation: pulse-ring 1.5s infinite; }

        @keyframes fade-in {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in { animation: fade-in 0.5s ease-out forwards; }
        
        @keyframes data-flow { to { stroke-dashoffset: 0; } }
        .animate-data-flow {
            stroke-dasharray: 10 5;
            stroke-dashoffset: 1000;
            animation: data-flow 20s linear infinite;
        }
      `}</style>
      <TopStatusBar status={status} lastEvolution={lastEvolution} countdown={countdown} />
      <div className="flex flex-grow min-h-0">
        {/* Placeholder for LeftDock */}
        <div className="flex flex-col p-2 bg-obsidian border-r border-violet">
            <button onClick={() => setActiveView('BRAIN_MAP')} className={`p-2 my-1 rounded ${activeView === 'BRAIN_MAP' ? 'bg-violet' : ''}`}>Brain</button>
            <button onClick={() => setActiveView('OVERVIEW')} className={`p-2 my-1 rounded ${activeView === 'OVERVIEW' ? 'bg-violet' : ''}`}>Overview</button>
            <button onClick={() => setActiveView('CHAT')} className={`p-2 my-1 rounded ${activeView === 'CHAT' ? 'bg-violet' : ''}`}>Chat</button>
            <button onClick={() => setActiveView('MUSIC')} className={`p-2 my-1 rounded ${activeView === 'MUSIC' ? 'bg-violet' : ''}`}>Music</button>
            <button onClick={() => setActiveView('AGENT')} className={`p-2 my-1 rounded ${activeView === 'AGENT' ? 'bg-violet' : ''}`}>Agent</button>
            <button onClick={() => setActiveView('TRAIN')} className={`p-2 my-1 rounded ${activeView === 'TRAIN' ? 'bg-violet' : ''}`}>Train</button>
            <button onClick={() => setActiveView('DOCK')} className={`p-2 my-1 rounded ${activeView === 'DOCK' ? 'bg-violet' : ''}`}>Dock</button>
        </div>
        <main className="flex-grow p-4 flex flex-col min-h-0 bg-grid-glow">
          {renderActiveView()}
        </main>
      </div>
      <Console logs={logs.slice(0, 50)} />
    </div>
  );
};

export default App;
