import React from "react";
import { Cpu, Layout, Terminal, Box, Lock, Code2, Globe, Brain } from "lucide-react";

export default function BentoFeatures() {
  const features = [
    { title: "AI Kernel", desc: "Core orchestrator managing agent lifecycles.", icon: <Cpu />, size: "col-span-2 row-span-2" },
    { title: "Dynamic VRAM Scheduler", desc: "Swaps 7B+ models on 16GB RAM laptops.", icon: <Layout />, size: "" },
    { title: "Dual Sandbox Verification", desc: "Polyglot execution across 13 languages.", icon: <Lock />, size: "" },
    { title: "Tool Calling", desc: "Native API and local filesystem tooling.", icon: <Terminal />, size: "col-span-2" },
    { title: "ChromaDB Memory", desc: "Persistent semantic vector retrieval.", icon: <Box />, size: "" },
    { title: "Code Interpreter", desc: "Real-time Python & JS AST patching.", icon: <Code2 />, size: "" },
    { title: "Internet Agent", desc: "Deep thematic search synthesis.", icon: <Globe />, size: "" },
    { title: "Reasoning Engine", desc: "DeepSeek R1 pedagogical chains.", icon: <Brain />, size: "" },
  ];

  return (
    <section className="section reveal-up">
      <div className="content-wrapper">
        <h2 style={{ fontSize: '3rem', textAlign: 'center', marginBottom: '1rem' }}>Architected for <span className="premium-gradient-text">Scale</span>.</h2>
        <p style={{ textAlign: 'center', color: 'var(--text-muted)', marginBottom: '3rem' }}>Every system component is engineered for maximum local performance.</p>
        
        <div className="bento-grid">
          {features.map((f, i) => (
            <div key={i} className={`bento-item reveal-up ${f.size}`} style={{ transitionDelay: `${i * 50}ms` }}>
              <div style={{ color: 'var(--accent-purple)', marginBottom: '1rem' }}>{f.icon}</div>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
