interface MessageBubbleProps {
  message: {
    role: "user" | "assistant";
    content: string;
  };
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="border-t border-border pt-4">
        <span className="text-[10px] uppercase tracking-widest text-muted">
          / You
        </span>
        <p className="mt-2 text-[13px] whitespace-pre-wrap">{message.content}</p>
      </div>
    );
  }

  return (
    <div className="border-t border-border-strong pt-4">
      <span className="text-[10px] uppercase tracking-widest text-muted">
        / De-insight
      </span>
      <p className="mt-2 text-[13px] whitespace-pre-wrap leading-relaxed">
        {message.content}
      </p>
    </div>
  );
}
