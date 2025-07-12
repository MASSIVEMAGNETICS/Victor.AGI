import React, { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, Line } from '@react-three/drei';
import * as THREE from 'three';

const FlowerOfLife: React.FC = () => {
  const [nodes, setNodes] = useState({});
  const [adjacency, setAdjacency] = useState({});

  useEffect(() => {
    fetch('/output.json')
      .then((response) => response.json())
      .then((data) => {
        setNodes(data.nodes);
        setAdjacency(data.adjacency);
      });
  }, []);

  return (
    <Canvas>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      {Object.values(nodes).map((node: any) => (
        <mesh key={node.id} position={new THREE.Vector3(...node.coords)}>
          <sphereGeometry args={[0.1, 32, 32]} />
          <meshStandardMaterial color="white" />
        </mesh>
      ))}
      {Object.entries(adjacency).map(([sourceId, targetIds]: [string, any]) =>
        targetIds.map((targetId: any) => (
          <Line
            key={`${sourceId}-${targetId}`}
            points={[nodes[sourceId].coords, nodes[targetId].coords]}
            color="white"
            lineWidth={1}
          />
        ))
      )}
      <OrbitControls />
      <Stars />
    </Canvas>
  );
};

export default FlowerOfLife;
