"use client";

import React, { useState } from 'react';
import EarthDemo from '@/components/ui/demo';

export default function ChatUI() {
  const [messages, setMessages] = useState([
    { role: 'assistant', text: `You are given a task to integrate an existing React component in the codebase\nThe codebase should support:\n- shadcn project structure\n- Tailwind CSS\n- Typescript\n\nIf it doesn't, provide instructions on how to setup project via shadcn CLI, install Tailwind or Typescript.` }
  ]);
  const [input, setInput] = useState('');

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    setMessages([...messages, { role: 'user', text: input }]);
    setInput('');
  };

  return (
    <main className="relative flex h-screen w-full flex-col bg-black text-white font-sans overflow-hidden">
      {/* Earth Component as Background */}
      <div className="absolute inset-0 z-0">
        <EarthDemo />
      </div>

      {/* Chat UI Overlay */}
      <div className="relative z-10 flex h-full w-full flex-col p-4 sm:p-8">
        
        {/* Header */}
        <header className="mb-6 flex items-center justify-between rounded-2xl border border-white/10 bg-black/40 p-4 backdrop-blur-md">
          <h1 className="text-xl font-semibold tracking-tight">AI Earth Assistant</h1>
          <div className="flex items-center space-x-2">
            <span className="flex h-2 w-2 rounded-full bg-green-500"></span>
            <span className="text-sm text-gray-300">System Online</span>
          </div>
        </header>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto rounded-2xl border border-white/10 bg-black/40 p-6 backdrop-blur-md space-y-6">
          {messages.map((msg, idx) => (
            <div key={idx} className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] rounded-2xl p-4 text-sm leading-relaxed shadow-lg ${
                msg.role === 'user' 
                  ? 'bg-blue-600 text-white rounded-br-none' 
                  : 'bg-white/10 text-gray-100 rounded-bl-none border border-white/5'
              }`}>
                {msg.text.split('\n').map((line, i) => (
                  <React.Fragment key={i}>
                    {line}
                    <br />
                  </React.Fragment>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Chat Input */}
        <form onSubmit={handleSend} className="mt-6 flex gap-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type a message to the Earth Assistant..."
            className="flex-1 rounded-2xl border border-white/10 bg-black/40 p-4 text-sm text-white placeholder-gray-400 backdrop-blur-md focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
          <button
            type="submit"
            className="rounded-2xl bg-blue-600 px-8 py-4 text-sm font-semibold text-white shadow-lg transition-colors hover:bg-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-black"
          >
            Send
          </button>
        </form>
      </div>
    </main>
  );
}
