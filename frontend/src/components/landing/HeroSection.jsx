import React from "react";
import { ChevronRight } from "lucide-react";

export default function HeroSection({ onStart }) {
  return (
    <section className="hero">
      <div className="content-wrapper" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <div className="hero-badge reveal-up" style={{ marginBottom: '2rem' }}>
          Your Personal AI Operating System
        </div>
        <h1 className="reveal-up" style={{ transitionDelay: '100ms' }}>
          <span className="premium-gradient-text">Run intelligent agents.</span><br/>
          Think locally. Execute autonomously.
        </h1>
        <p className="reveal-up" style={{ transitionDelay: '200ms' }}>
          DeepThink-AIOS is a $100M-grade intelligence platform that intelligently coordinates multiple specialized agents, local LLMs, and execution environments into a unified operating system.
        </p>
        <div className="hero-actions reveal-up" style={{ transitionDelay: '300ms' }}>
          <button className="btn-primary" onClick={onStart}>
            Get Started <ChevronRight size={16} />
          </button>
          <button className="btn-secondary">Watch Demo</button>
          <a href="https://github.com/Bshdhorrhh/Team_Trenches" className="btn-secondary" target="_blank" rel="noreferrer">
            GitHub
          </a>
        </div>
      </div>
    </section>
  );
}
