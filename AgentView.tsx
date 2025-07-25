import React from 'react';
import type { DigitalAgent } from './digital_agent';

interface AgentViewProps {
  agent: DigitalAgent;
}

const TraitCategory: React.FC<{ title: string; traits: Record<string, any> }> = ({ title, traits }) => (
  <div className="bg-gray-900 p-4 rounded-lg mb-4 border border-violet">
    <h3 className="text-xl font-bold mb-2 text-violet text-shadow-glow">{title}</h3>
    <ul>
      {Object.entries(traits).map(([key, value]) => (
        <li key={key} className="flex justify-between">
          <span>{key}:</span>
          <span>{typeof value === 'number' ? value.toFixed(2) : JSON.stringify(value)}</span>
        </li>
      ))}
    </ul>
  </div>
);

const AgentView: React.FC<AgentViewProps> = ({ agent }) => {
  return (
    <div className="flex flex-col h-full bg-obsidian text-lime p-4 rounded-lg border border-lime shadow-lg overflow-y-auto">
      <h2 className="text-2xl font-bold mb-4 text-shadow-glow">Agent State</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <TraitCategory title="Cognitive Traits" traits={agent.cognitive as any} />
        <TraitCategory title="Operational Traits" traits={agent.operational as any} />
        <TraitCategory title="Interaction Traits" traits={agent.interaction as any} />
        <TraitCategory title="Emotional State" traits={agent.emotional as any} />
        <TraitCategory title="Autonomous Traits" traits={agent.autonomous as any} />
        <TraitCategory title="Self Modification" traits={agent.self_modification as any} />
        <TraitCategory title="State Diagnostics" traits={agent.state_diagnostics as any} />
        <TraitCategory title="Next-Gen Abilities" traits={agent.next_gen as any} />
      </div>
    </div>
  );
};

export default AgentView;
