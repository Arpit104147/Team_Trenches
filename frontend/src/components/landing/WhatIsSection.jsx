import React from "react";

export default function WhatIsSection() {
  return (
    <section className="section reveal-up">
      <div className="content-wrapper split-layout">
        <div className="split-visual">
          <div className="architecture-mockup">
            <h3 style={{ marginBottom: '1rem', color: 'var(--accent-blue)' }}>AI Kernel Core</h3>
            <div style={{ display: 'flex', gap: '1rem', flexDirection: 'column' }}>
              <div style={{ background: 'var(--glass-bg)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>Routing Node [Phi-3.5]</div>
              <div style={{ background: 'var(--glass-bg)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--glass-border)', marginLeft: '2rem' }}>Execution Layer [Ornith]</div>
              <div style={{ background: 'var(--glass-bg)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--glass-border)', marginLeft: '2rem' }}>Reasoning Matrix [DeepSeek]</div>
            </div>
          </div>
        </div>
        <div className="split-text">
          <h2>DeepThink-AIOS is <span className="premium-gradient-text">not another chatbot.</span></h2>
          <p>
            It is an AI Operating System that intelligently coordinates multiple specialized agents, local LLMs, memory systems, retrieval engines, and execution environments into a unified intelligence platform. 
            <br/><br/>
            Engineered for developers, designers, and researchers who demand full control, absolute privacy, and unprecedented autonomous execution.
          </p>
        </div>
      </div>
    </section>
  );
}
