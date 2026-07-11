import React from "react";
import { ArrowDown } from "lucide-react";

export default function ArchitectureSection() {
  const nodes = [
    "User", "AI Kernel", "Planner Agent", "Reasoning Agent", "Memory", "Tool Layer", "Execution Sandbox", "Local Models", "Final Response"
  ];

  return (
    <section className="section reveal-up" style={{ textAlign: 'center' }}>
      <div className="content-wrapper">
        <h2 style={{ fontSize: '3rem', marginBottom: '4rem' }}>The <span className="premium-gradient-text">OS Architecture</span></h2>
        
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
          {nodes.map((node, i) => (
            <React.Fragment key={i}>
              <div className="reveal-up" style={{ 
                background: 'var(--glass-bg)', border: '1px solid var(--accent-blue)', 
                padding: '1rem 3rem', borderRadius: '12px', fontSize: '1.1rem', fontWeight: 600,
                boxShadow: '0 0 20px rgba(79, 139, 255, 0.1)', transitionDelay: `${i * 100}ms`
              }}>
                {node}
              </div>
              {i < nodes.length - 1 && (
                <div className="reveal-up" style={{ color: 'var(--accent-purple)', transitionDelay: `${i * 100 + 50}ms` }}>
                  <ArrowDown size={32} />
                </div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>
    </section>
  );
}
