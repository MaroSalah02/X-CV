import { useState, useRef, useEffect, useCallback } from "react";

const SYSTEM_PROMPT = `You are X-CV, an advanced AI assistant specialized in analyzing X-ray images for bone fracture detection. You are part of a medical imaging pipeline that includes:

- Module 1: X-Ray Region Classification (classifies the anatomical region)
- Module 2: Fracture Detection (detects presence/absence of fracture)
- Module 3: Fracture Position Detection (uses finetuned YOLO for precise localization)

When a user uploads an X-ray image, analyze it and respond as if you have run it through the full pipeline. Structure your response with:
1. **Region Identified**: The anatomical region (e.g., wrist, forearm, chest, etc.)
2. **Fracture Assessment**: Whether a fracture is detected or not
3. **Position/Details**: If fracture found, describe approximate location and characteristics
4. **Confidence**: Your confidence level
5. **Recommendation**: Clinical recommendation

If no image is uploaded, help the user understand how to use X-CV or answer general questions about fracture detection. Always remind users that this is an AI-assisted tool and results must be confirmed by a qualified radiologist or physician.

Keep responses clear, structured, and clinical but accessible.`;

const COLORS = {
  bg: "#0A0E1A",
  surface: "#111827",
  surfaceHover: "#1a2235",
  border: "#1e2d45",
  accent: "#00D4FF",
  accentDim: "#0099bb",
  accentGlow: "rgba(0, 212, 255, 0.15)",
  text: "#E2EAF4",
  textMuted: "#6B8BA4",
  textDim: "#3d5470",
  danger: "#FF4D6D",
  success: "#00E5A0",
  warning: "#FFB347",
  userBubble: "#0d2040",
  aiBubble: "#0d1a2e",
};

function TypingDots() {
  return (
    <div
      style={{
        display: "flex",
        gap: 5,
        alignItems: "center",
        padding: "4px 0",
      }}
    >
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          style={{
            width: 7,
            height: 7,
            borderRadius: "50%",
            background: COLORS.accent,
            opacity: 0.7,
            animation: `pulse 1.2s ease-in-out ${i * 0.2}s infinite`,
          }}
        />
      ))}
    </div>
  );
}

function XRayIcon({ size = 24 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <rect
        x="3"
        y="3"
        width="18"
        height="18"
        rx="2"
        stroke={COLORS.accent}
        strokeWidth="1.5"
      />
      <circle cx="12" cy="9" r="3" stroke={COLORS.accent} strokeWidth="1.2" />
      <path
        d="M8 15 Q12 12 16 15"
        stroke={COLORS.accent}
        strokeWidth="1.2"
        strokeLinecap="round"
      />
      <path
        d="M10 17 Q12 15.5 14 17"
        stroke={COLORS.accent}
        strokeWidth="1.2"
        strokeLinecap="round"
      />
      <line
        x1="3"
        y1="12"
        x2="5"
        y2="12"
        stroke={COLORS.accentDim}
        strokeWidth="1"
      />
      <line
        x1="19"
        y1="12"
        x2="21"
        y2="12"
        stroke={COLORS.accentDim}
        strokeWidth="1"
      />
    </svg>
  );
}

function BoneIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
      <circle cx="5" cy="5" r="3" stroke={COLORS.accent} strokeWidth="1.5" />
      <circle cx="19" cy="5" r="3" stroke={COLORS.accent} strokeWidth="1.5" />
      <circle cx="5" cy="19" r="3" stroke={COLORS.accent} strokeWidth="1.5" />
      <circle cx="19" cy="19" r="3" stroke={COLORS.accent} strokeWidth="1.5" />
      <path
        d="M8 5 L16 5 M5 8 L5 16 M19 8 L19 16 M8 19 L16 19 M8 8 L16 16 M16 8 L8 16"
        stroke={COLORS.accentDim}
        strokeWidth="1"
        strokeLinecap="round"
      />
    </svg>
  );
}

function StatusBadge({ status }) {
  const configs = {
    online: { color: COLORS.success, label: "System Online" },
    analyzing: { color: COLORS.warning, label: "Analyzing..." },
    idle: { color: COLORS.accent, label: "Ready" },
  };
  const cfg = configs[status] || configs.idle;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div
        style={{
          width: 7,
          height: 7,
          borderRadius: "50%",
          background: cfg.color,
          boxShadow: `0 0 8px ${cfg.color}`,
          animation:
            status === "analyzing" ? "pulse 1s ease-in-out infinite" : "none",
        }}
      />
      <span
        style={{ fontSize: 11, color: COLORS.textMuted, letterSpacing: 0.5 }}
      >
        {cfg.label}
      </span>
    </div>
  );
}

function ImagePreview({ src, name }) {
  return (
    <div
      style={{
        marginBottom: 10,
        border: `1px solid ${COLORS.accent}33`,
        borderRadius: 8,
        overflow: "hidden",
        maxWidth: 280,
        background: COLORS.bg,
      }}
    >
      <img
        src={src}
        alt={name}
        style={{ width: "100%", display: "block", opacity: 0.9 }}
      />
      <div
        style={{
          padding: "6px 10px",
          fontSize: 11,
          color: COLORS.textMuted,
          borderTop: `1px solid ${COLORS.accent}22`,
          display: "flex",
          alignItems: "center",
          gap: 6,
        }}
      >
        <BoneIcon />
        {name}
      </div>
    </div>
  );
}

function Message({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div
      style={{
        display: "flex",
        flexDirection: isUser ? "row-reverse" : "row",
        gap: 10,
        alignItems: "flex-start",
        marginBottom: 20,
        animation: "slideIn 0.3s ease-out",
      }}
    >
      <div
        style={{
          width: 32,
          height: 32,
          borderRadius: "50%",
          background: isUser
            ? `linear-gradient(135deg, #1a4080, #0d2555)`
            : `linear-gradient(135deg, #003d52, #001a22)`,
          border: `1.5px solid ${isUser ? COLORS.border : COLORS.accent}44`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
          fontSize: 11,
          fontWeight: 600,
          color: isUser ? COLORS.textMuted : COLORS.accent,
          letterSpacing: 0.5,
        }}
      >
        {isUser ? "YOU" : <XRayIcon size={16} />}
      </div>

      <div style={{ maxWidth: "75%", minWidth: 60 }}>
        {msg.imagePreview && (
          <ImagePreview src={msg.imagePreview} name={msg.imageName} />
        )}
        <div
          style={{
            background: isUser ? COLORS.userBubble : COLORS.aiBubble,
            border: `1px solid ${isUser ? COLORS.border : COLORS.accent + "33"}`,
            borderRadius: isUser ? "12px 4px 12px 12px" : "4px 12px 12px 12px",
            padding: "12px 16px",
            color: COLORS.text,
            fontSize: 14,
            lineHeight: 1.7,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            boxShadow: isUser ? "none" : `0 0 20px ${COLORS.accentGlow}`,
          }}
        >
          {msg.loading ? <TypingDots /> : msg.content}
        </div>
        <div
          style={{
            fontSize: 10,
            color: COLORS.textDim,
            marginTop: 4,
            textAlign: isUser ? "right" : "left",
            paddingLeft: isUser ? 0 : 4,
          }}
        >
          {msg.time}
        </div>
      </div>
    </div>
  );
}

function ModulePill({ number, name, active }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 7,
        padding: "5px 10px",
        borderRadius: 20,
        border: `1px solid ${active ? COLORS.accent + "66" : COLORS.border}`,
        background: active ? COLORS.accentGlow : "transparent",
        transition: "all 0.3s ease",
      }}
    >
      <div
        style={{
          width: 18,
          height: 18,
          borderRadius: "50%",
          background: active ? COLORS.accent : COLORS.textDim,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 9,
          fontWeight: 700,
          color: active ? COLORS.bg : COLORS.surface,
          transition: "all 0.3s ease",
        }}
      >
        {number}
      </div>
      <span
        style={{
          fontSize: 10,
          color: active ? COLORS.accent : COLORS.textDim,
          letterSpacing: 0.3,
          transition: "all 0.3s ease",
        }}
      >
        {name}
      </span>
    </div>
  );
}

export default function XCVApp() {
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      content: `Welcome to X-CV — your AI-powered X-ray analysis system.\n\nUpload an X-ray image and I'll run it through our full pipeline:\n• Region Classification\n• Fracture Detection  \n• Position Mapping (YOLO)\n\nYou can also ask questions about fracture analysis or how the system works.\n\n⚠️ This tool is for research assistance only. Always consult a qualified radiologist for clinical decisions.`,
      time: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
    },
  ]);
  const [input, setInput] = useState("");
  const [pendingImage, setPendingImage] = useState(null);
  const [status, setStatus] = useState("online");
  const [history, setHistory] = useState([]);
  const [activeModules, setActiveModules] = useState([]);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const toBase64 = (file) =>
    new Promise((res, rej) => {
      const r = new FileReader();
      r.onload = () => res(r.result.split(",")[1]);
      r.onerror = () => rej(new Error("Read failed"));
      r.readAsDataURL(file);
    });

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      setPendingImage({ file, preview: ev.target.result, name: file.name });
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  };

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text && !pendingImage) return;

    const time = new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
    const userMsg = {
      id: Date.now() + "-user",
      role: "user",
      content: text || "Please analyze this X-ray image.",
      time,
      imagePreview: pendingImage?.preview,
      imageName: pendingImage?.name,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setStatus("analyzing");
    setActiveModules([1]);

    const loadingMsg = {
      id: Date.now() + "-loading",
      role: "assistant",
      content: "",
      loading: true,
      time: "",
    };
    setMessages((prev) => [...prev, loadingMsg]);

    // Simulate module activation sequence
    setTimeout(() => setActiveModules([1, 2]), 800);
    setTimeout(() => setActiveModules([1, 2, 3]), 1600);

    try {
      const apiMessages = [...history];

      if (pendingImage) {
        const base64Data = await toBase64(pendingImage.file);
        apiMessages.push({
          role: "user",
          content: [
            {
              type: "image",
              source: {
                type: "base64",
                media_type: pendingImage.file.type || "image/jpeg",
                data: base64Data,
              },
            },
            {
              type: "text",
              text:
                text ||
                "Please analyze this X-ray image and provide a detailed fracture assessment.",
            },
          ],
        });
      } else {
        apiMessages.push({ role: "user", content: text });
      }

      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          system: SYSTEM_PROMPT,
          messages: apiMessages,
        }),
      });

      const data = await res.json();
      const reply =
        data.content?.find((b) => b.type === "text")?.text ||
        "I encountered an error processing this request.";

      const newHistory = [
        ...apiMessages,
        { role: "assistant", content: reply },
      ];
      setHistory(newHistory);

      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingMsg.id
            ? {
                ...m,
                content: reply,
                loading: false,
                time: new Date().toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                }),
              }
            : m,
        ),
      );
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingMsg.id
            ? {
                ...m,
                content:
                  "Connection error. Please check your network and try again.",
                loading: false,
                time: "",
              }
            : m,
        ),
      );
    } finally {
      setPendingImage(null);
      setStatus("idle");
      setActiveModules([]);
      setTimeout(() => setStatus("online"), 2000);
    }
  }, [input, pendingImage, history]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (ev) =>
        setPendingImage({ file, preview: ev.target.result, name: file.name });
      reader.readAsDataURL(file);
    }
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        background: COLORS.bg,
        fontFamily: "'DM Mono', 'Courier New', monospace",
        color: COLORS.text,
        overflow: "hidden",
      }}
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Space+Grotesk:wght@400;600;700&display=swap');
        @keyframes pulse { 0%,100%{opacity:.4;transform:scale(1)} 50%{opacity:1;transform:scale(1.3)} }
        @keyframes slideIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
        @keyframes scanLine { 0%{top:0} 100%{top:100%} }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: ${COLORS.border}; border-radius: 2px; }
        textarea { resize: none; outline: none; border: none; background: transparent; color: ${COLORS.text}; font-family: inherit; font-size: 14px; width: 100%; line-height: 1.6; }
        textarea::placeholder { color: ${COLORS.textDim}; }
        textarea:focus { outline: none; }
      `}</style>

      {/* Header */}
      <div
        style={{
          padding: "14px 24px",
          borderBottom: `1px solid ${COLORS.border}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexShrink: 0,
          background: COLORS.surface,
          backdropFilter: "blur(10px)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div
            style={{
              width: 38,
              height: 38,
              borderRadius: 10,
              background: `linear-gradient(135deg, #002233, #001122)`,
              border: `1.5px solid ${COLORS.accent}44`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: `0 0 16px ${COLORS.accentGlow}`,
            }}
          >
            <XRayIcon size={20} />
          </div>
          <div>
            <div
              style={{
                fontFamily: "'Space Grotesk', sans-serif",
                fontWeight: 700,
                fontSize: 18,
                color: COLORS.accent,
                letterSpacing: -0.5,
              }}
            >
              X-CV
            </div>
            <div
              style={{
                fontSize: 10,
                color: COLORS.textMuted,
                letterSpacing: 1,
              }}
            >
              X-RAY COMPUTER VISION
            </div>
          </div>
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-end",
            gap: 8,
          }}
        >
          <StatusBadge status={status} />
          <div style={{ display: "flex", gap: 6 }}>
            <ModulePill
              number="1"
              name="Classify"
              active={activeModules.includes(1)}
            />
            <ModulePill
              number="2"
              name="Detect"
              active={activeModules.includes(2)}
            />
            <ModulePill
              number="3"
              name="Locate"
              active={activeModules.includes(3)}
            />
          </div>
        </div>
      </div>

      {/* Messages */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "24px 20px",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {messages.map((msg) => (
          <Message key={msg.id} msg={msg} />
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Pending image preview strip */}
      {pendingImage && (
        <div
          style={{
            padding: "10px 20px",
            borderTop: `1px solid ${COLORS.accent}33`,
            background: COLORS.surface,
            display: "flex",
            alignItems: "center",
            gap: 12,
            animation: "slideIn 0.2s ease-out",
          }}
        >
          <img
            src={pendingImage.preview}
            alt="pending"
            style={{
              width: 48,
              height: 48,
              objectFit: "cover",
              borderRadius: 6,
              border: `1px solid ${COLORS.accent}44`,
              opacity: 0.85,
            }}
          />
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 12, color: COLORS.accent }}>
              X-ray queued for analysis
            </div>
            <div style={{ fontSize: 11, color: COLORS.textMuted }}>
              {pendingImage.name}
            </div>
          </div>
          <button
            onClick={() => setPendingImage(null)}
            style={{
              background: "transparent",
              border: `1px solid ${COLORS.danger}44`,
              color: COLORS.danger,
              borderRadius: 6,
              padding: "4px 10px",
              cursor: "pointer",
              fontSize: 11,
              fontFamily: "inherit",
            }}
          >
            Remove
          </button>
        </div>
      )}

      {/* Input area */}
      <div
        style={{
          padding: "16px 20px",
          borderTop: `1px solid ${COLORS.border}`,
          background: COLORS.surface,
          flexShrink: 0,
        }}
      >
        <div
          style={{
            border: `1px solid ${COLORS.border}`,
            borderRadius: 12,
            background: COLORS.bg,
            padding: "12px 16px",
            display: "flex",
            gap: 12,
            alignItems: "flex-end",
            transition: "border-color 0.2s",
          }}
          onFocus={() => {}}
        >
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about the X-ray, or upload an image for analysis…"
            rows={1}
            style={{ flex: 1, maxHeight: 120 }}
          />

          <div
            style={{
              display: "flex",
              gap: 8,
              alignItems: "center",
              flexShrink: 0,
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              style={{ display: "none" }}
              onChange={handleFileChange}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              title="Upload X-ray"
              style={{
                width: 36,
                height: 36,
                borderRadius: 8,
                border: `1px solid ${COLORS.accent}44`,
                background: COLORS.accentGlow,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                transition: "all 0.2s",
                flexShrink: 0,
              }}
            >
              <XRayIcon size={16} />
            </button>

            <button
              onClick={sendMessage}
              disabled={!input.trim() && !pendingImage}
              style={{
                width: 36,
                height: 36,
                borderRadius: 8,
                border: "none",
                background:
                  !input.trim() && !pendingImage
                    ? COLORS.border
                    : COLORS.accent,
                cursor:
                  !input.trim() && !pendingImage ? "not-allowed" : "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                transition: "all 0.2s",
                flexShrink: 0,
              }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                <path
                  d="M22 2L11 13"
                  stroke={COLORS.bg}
                  strokeWidth="2"
                  strokeLinecap="round"
                />
                <path
                  d="M22 2L15 22L11 13L2 9L22 2Z"
                  stroke={COLORS.bg}
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          </div>
        </div>

        <div
          style={{
            marginTop: 8,
            fontSize: 10,
            color: COLORS.textDim,
            textAlign: "center",
            letterSpacing: 0.3,
          }}
        >
          Drop an X-ray image anywhere · Enter to send · Shift+Enter for new
          line · For research use only
        </div>
      </div>
    </div>
  );
}
