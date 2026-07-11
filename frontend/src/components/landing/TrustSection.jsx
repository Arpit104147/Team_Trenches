import React from "react";
import { Shield, Database, Cpu, Globe, Boxes, Zap } from "lucide-react";

export default function TrustSection() {
  const badges = [
    { icon: <Boxes size={14}/>, text: "Multi-Agent AI" },
    { icon: <Shield size={14}/>, text: "100% Local AI" },
    { icon: <Database size={14}/>, text: "Dynamic Memory" },
    { icon: <Zap size={14}/>, text: "Dual Sandbox" },
    { icon: <Globe size={14}/>, text: "Reasoning Engine" },
    { icon: <Cpu size={14}/>, text: "Multimodal" },
  ];

  return (
    <section className="section reveal-up" style={{ padding: '2rem 0 6rem 0', borderBottom: 'none' }}>
      <div className="content-wrapper">
        <div className="trust-bar">
          {badges.map((b, i) => (
            <div key={i} className="trust-badge reveal-up" style={{ transitionDelay: `${i * 50}ms` }}>
              {b.icon} {b.text}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
