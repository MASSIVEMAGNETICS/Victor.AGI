import { useEffect, useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { motion, useAnimationControls } from 'framer-motion';

function FractalCore({ load }: { load: number }) {
  const mesh = useRef<THREE.Mesh>(null!);
  useFrame(() => {
    mesh.current.rotation.x += 0.01;
    mesh.current.rotation.y += 0.02;
  });
  return (
    <mesh ref={mesh} scale={1 + load / 100 * 0.2}>
      <sphereGeometry args={[1, 64, 64]} />
      <meshStandardMaterial color="#00e7ff" />
    </mesh>
  );
}

function CommandConsole({ ws }: { ws: WebSocket | null }) {
  const [input, setInput] = useState('');
  const controls = useAnimationControls();

  const send = () => {
    ws?.send(input);
    setInput('');
    controls.start({ scale: [1, 1.2, 1], transition: { type: 'spring', duration: 0.6 } });
  };

  return (
    <div className="w-full p-2 bg-white/10 backdrop-blur text-white">
      <motion.input
        className="bg-transparent outline-none w-full"
        value={input}
        onChange={e => setInput(e.target.value)}
        onKeyDown={e => e.key === 'Enter' && send()}
        animate={controls}
        aria-label="command input"
      />
    </div>
  );
}

export default function App() {
  const [load, setLoad] = useState(0);
  const [ws, setWs] = useState<WebSocket | null>(null);

  useEffect(() => {
    const sock = new WebSocket('ws://localhost:8000/ws/victor');
    sock.onmessage = e => {
      const data = JSON.parse(e.data);
      setLoad(data.load);
    };
    setWs(sock);
    return () => sock.close();
  }, []);

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex-1 relative">
        <Canvas className="w-full h-full">
          <ambientLight intensity={0.5} />
          <FractalCore load={load} />
        </Canvas>
      </div>
      <CommandConsole ws={ws} />
    </div>
  );
}
