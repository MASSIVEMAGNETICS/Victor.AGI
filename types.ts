export interface LogEntry {
  id: number;
  timestamp: string;
  message: string;
  type: 'THOUGHT' | 'EVOLUTION' | 'ERROR' | 'SYSTEM' | 'USER' | 'VICTOR';
}

export type FileSystem = {
  [path: string]: string;
};

export interface FileTreeNode {
  name: string;
  path: string;
  children?: FileTreeNode[];
}

#<<<<<<< phoenix-hotfix
export type AppView = 'OVERVIEW' | 'BRAIN_MAP' | 'CHAT' | 'TRAIN' | 'DOCK' | 'MUSIC' | 'AGENT';
=======
export type AppView = 'OVERVIEW' | 'FLOWER_OF_LIFE' | 'CHAT' | 'TRAIN' | 'DOCK';
#>>>>>>> main

export interface Node {
  id: string;
  label: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  modality: 'core' | 'memory' | 'cognition' | 'evolution' | 'io' | 'directive';
}

export interface Edge {
  source: string;
  target: string;
}
