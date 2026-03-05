"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import MessageBubble from "./MessageBubble";
import ModeToggle from "./ModeToggle";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isComposing, setIsComposing] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [mode, setMode] = useState<"emotional" | "rational">("emotional");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = useCallback(async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: "user", content: input.trim() };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: updatedMessages.map((m) => ({
            role: m.role,
            content: m.content,
          })),
          mode,
        }),
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let assistantContent = "";

      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        const lines = text.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            if (data === "[DONE]") break;
            try {
              const parsed = JSON.parse(data);
              if (parsed.content) {
                assistantContent += parsed.content;
                // 過濾 <think>...</think> 區塊
                const display = assistantContent
                  .replace(/<think>[\s\S]*?<\/think>/g, "")
                  .replace(/<think>[\s\S]*/g, "")
                  .trim();
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    role: "assistant",
                    content: display,
                  };
                  return updated;
                });
              }
            } catch {
              // 跳過格式錯誤的 SSE 行
            }
          }
        }
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "連線錯誤，請確認後端是否已啟動。" },
      ]);
    } finally {
      setIsLoading(false);
    }
  }, [input, messages, mode, isLoading]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey && !isComposing) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b-[1.5px] border-border-strong px-6 py-3 flex items-baseline justify-between">
        <span className="text-[12px] tracking-wide uppercase">De-insight</span>
        <ModeToggle mode={mode} onModeChange={setMode} />
        <span className="text-[10px] tracking-widest uppercase text-muted">
          {new Date().toLocaleDateString("zh-TW")}
        </span>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-8">
        <div className="max-w-2xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="border-t border-border-strong pt-4">
              <span className="text-[10px] uppercase tracking-widest text-muted">
                / De-insight
              </span>
              <p className="mt-2 text-[13px] text-muted">
                輸入你的想法，開始對話。
              </p>
            </div>
          )}
          {messages.map((msg, i) => (
            <MessageBubble key={i} message={msg} />
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <footer className="border-t-[1.5px] border-border-strong px-6 py-4">
        <div className="max-w-2xl mx-auto flex items-center gap-4">
          <span className="text-[10px] uppercase tracking-widest text-muted shrink-0">
            /
          </span>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            onCompositionStart={() => setIsComposing(true)}
            onCompositionEnd={() => setIsComposing(false)}
            placeholder="輸入你的想法..."
            rows={1}
            className="w-full bg-transparent text-text text-[13px] placeholder:text-muted focus:outline-none resize-none"
            disabled={isLoading}
          />
        </div>
      </footer>
    </div>
  );
}
