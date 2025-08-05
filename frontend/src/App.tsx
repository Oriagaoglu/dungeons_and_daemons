// App.tsx
import React, { useState, useEffect, useRef } from "react";
import { v4 as uuidv4 } from "uuid";
import "./App.css";

// Types
interface Message {
  id: string;
  type: "user" | "assistant" | "system" | "dice_result" | "world_update";
  content: string;
  timestamp: string;
  metadata?: any;
}

interface CharacterSheet {
  name: string;
  race: string;
  class_name: string;
  level: number;
  stats: Record<string, number>;
  equipment: string[];
  backstory: string;
}

interface SystemStats {
  gpu?: {
    memory_used: number;
    memory_total: number;
    utilization: number;
  };
  system: {
    cpu_percent: number;
    memory_percent: number;
  };
  active_sessions: number;
}

// Custom Hooks
const useWebSocket = (sessionId: string) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);

  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8080/ws/${sessionId}`);

    ws.onopen = () => {
      setConnected(true);
      setSocket(ws);
      console.log("WebSocket connected");
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const message: Message = {
        id: uuidv4(),
        type: data.type || "assistant",
        content: data.content || JSON.stringify(data),
        timestamp: data.timestamp || new Date().toISOString(),
        metadata: data,
      };

      setMessages((prev) => [...prev, message]);
    };

    ws.onclose = () => {
      setConnected(false);
      setSocket(null);
      console.log("WebSocket disconnected");
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    return () => {
      ws.close();
    };
  }, [sessionId]);

  const sendMessage = (content: string, type: string = "chat") => {
    if (socket && connected) {
      const message = {
        type,
        content,
        timestamp: new Date().toISOString(),
      };

      socket.send(JSON.stringify(message));

      if (type === "chat") {
        // Add user message to local state immediately
        setMessages((prev) => [
          ...prev,
          {
            id: uuidv4(),
            type: "user",
            content,
            timestamp: new Date().toISOString(),
          },
        ]);
      }
    }
  };

  return { connected, messages, sendMessage };
};

// Components
const DiceRoller: React.FC<{ onRoll: (dice: string) => void }> = ({
  onRoll,
}) => {
  const [diceType, setDiceType] = useState("1d20");

  const rollDice = () => {
    onRoll(diceType);
  };

  return (
    <div className="dice-roller">
      <select
        value={diceType}
        onChange={(e) => setDiceType(e.target.value)}
        className="dice-select"
      >
        <option value="1d4">d4</option>
        <option value="1d6">d6</option>
        <option value="1d8">d8</option>
        <option value="1d10">d10</option>
        <option value="1d12">d12</option>
        <option value="1d20">d20</option>
        <option value="1d100">d100</option>
      </select>
      <button onClick={rollDice} className="dice-button">
        🎲 Roll
      </button>
    </div>
  );
};

const CharacterPanel: React.FC<{
  character: CharacterSheet | null;
  onUpdate: (character: CharacterSheet) => void;
}> = ({ character, onUpdate }) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editCharacter, setEditCharacter] = useState<CharacterSheet | null>(
    character
  );

  useEffect(() => {
    setEditCharacter(character);
  }, [character]);

  const handleSave = () => {
    if (editCharacter) {
      onUpdate(editCharacter);
      setIsEditing(false);
    }
  };

  if (!character && !isEditing) {
    return (
      <div className="character-panel">
        <button
          onClick={() => setIsEditing(true)}
          className="create-character-btn"
        >
          Create Character
        </button>
      </div>
    );
  }

  return (
    <div className="character-panel">
      <h3>Character Sheet</h3>
      {isEditing ? (
        <div className="character-editor">
          <input
            type="text"
            placeholder="Character Name"
            value={editCharacter?.name || ""}
            onChange={(e) =>
              setEditCharacter((prev) => ({ ...prev!, name: e.target.value }))
            }
          />
          <input
            type="text"
            placeholder="Race"
            value={editCharacter?.race || ""}
            onChange={(e) =>
              setEditCharacter((prev) => ({ ...prev!, race: e.target.value }))
            }
          />
          <input
            type="text"
            placeholder="Class"
            value={editCharacter?.class_name || ""}
            onChange={(e) =>
              setEditCharacter((prev) => ({
                ...prev!,
                class_name: e.target.value,
              }))
            }
          />
          <div className="character-actions">
            <button onClick={handleSave}>Save</button>
            <button onClick={() => setIsEditing(false)}>Cancel</button>
          </div>
        </div>
      ) : (
        <div className="character-display">
          <div>
            <strong>{character!.name}</strong>
          </div>
          <div>
            {character!.race} {character!.class_name}
          </div>
          <div>Level {character!.level}</div>
          <button onClick={() => setIsEditing(true)}>Edit</button>
        </div>
      )}
    </div>
  );
};

const SystemMonitor: React.FC<{ stats: SystemStats | null }> = ({ stats }) => {
  if (!stats) return null;

  return (
    <div className="system-monitor">
      <h4>System Status</h4>
      <div className="stats-grid">
        {stats.gpu && (
          <div className="stat-item">
            <span>GPU Memory:</span>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width: `${
                    (stats.gpu.memory_used / stats.gpu.memory_total) * 100
                  }%`,
                }}
              />
            </div>
            <span>
              {stats.gpu.memory_used}MB / {stats.gpu.memory_total}MB
            </span>
          </div>
        )}
        <div className="stat-item">
          <span>CPU:</span>
          <span>{stats.system.cpu_percent.toFixed(1)}%</span>
        </div>
        <div className="stat-item">
          <span>Memory:</span>
          <span>{stats.system.memory_percent.toFixed(1)}%</span>
        </div>
        <div className="stat-item">
          <span>Active Sessions:</span>
          <span>{stats.active_sessions}</span>
        </div>
      </div>
    </div>
  );
};

const MessageBubble: React.FC<{ message: Message }> = ({ message }) => {
  const renderContent = () => {
    switch (message.type) {
      case "dice_result":
        return (
          <div className="dice-result">
            🎲 Rolled {message.metadata.roll}:{" "}
            <strong>{message.metadata.result}</strong>
          </div>
        );
      case "world_update":
        return (
          <div className="world-update">
            🌍 <strong>World Update:</strong>
            <ul>
              {Object.entries(
                message.metadata?.updates as Record<string, string | string[]>
              ).map(([key, value]) => (
                <li key={key}>
                  {Array.isArray(value) ? value.join(", ") : value}
                </li>
              ))}
            </ul>
          </div>
        );
      default:
        return <div>{message.content}</div>;
    }
  };

  return (
    <div className={`message-bubble ${message.type}`}>
      {renderContent()}
      <div className="message-time">
        {new Date(message.timestamp).toLocaleTimeString()}
      </div>
    </div>
  );
};

// Main App Component
const App: React.FC = () => {
  const sessionId = useRef(uuidv4()).current;
  const { connected, messages, sendMessage } = useWebSocket(sessionId);

  const [inputMessage, setInputMessage] = useState("");
  const [character, setCharacter] = useState<CharacterSheet | null>(null);
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null);
  const [showSidebar, setShowSidebar] = useState(true);
  const [isTyping, setIsTyping] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Load character on startup
  useEffect(() => {
    loadCharacter();
    loadSystemStats();

    // Poll system stats every 30 seconds
    const statsInterval = setInterval(loadSystemStats, 30000);
    return () => clearInterval(statsInterval);
  }, []);

  // Focus input when connected
  useEffect(() => {
    if (connected && inputRef.current) {
      inputRef.current.focus();
    }
  }, [connected]);

  const loadCharacter = async () => {
    try {
      const response = await fetch(
        `http://localhost:8080/character/${sessionId}`
      );
      if (response.ok) {
        const characterData = await response.json();
        setCharacter(characterData);
      }
    } catch (error) {
      console.log("No existing character found");
    }
  };

  const loadSystemStats = async () => {
    try {
      const response = await fetch("http://localhost:8080/health");
      if (response.ok) {
        const stats = await response.json();
        setSystemStats(stats);
      }
    } catch (error) {
      console.error("Failed to load system stats:", error);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputMessage.trim() && connected) {
      setIsTyping(true);
      sendMessage(inputMessage);
      setInputMessage("");

      // Clear typing indicator after response
      setTimeout(() => setIsTyping(false), 2000);
    }
  };

  const handleDiceRoll = (dice: string) => {
    sendMessage(dice, "dice_roll");
  };

  const handleCharacterUpdate = async (updatedCharacter: CharacterSheet) => {
    try {
      const response = await fetch(
        `http://localhost:8080/character?session_id=${sessionId}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(updatedCharacter),
        }
      );

      if (response.ok) {
        setCharacter(updatedCharacter);
        // Notify the AI about character update
        sendMessage(
          `Character updated: ${updatedCharacter.name}, ${updatedCharacter.race} ${updatedCharacter.class_name}`,
          "character_update"
        );
      }
    } catch (error) {
      console.error("Failed to save character:", error);
    }
  };

  const triggerWorldUpdate = async () => {
    try {
      await fetch("http://localhost:8080/batch/world-update", {
        method: "POST",
      });
    } catch (error) {
      console.error("Failed to trigger world update:", error);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-left">
          <h1>🐲 Dungeons & Daemons AI</h1>
          <div className="connection-status">
            <div
              className={`status-indicator ${
                connected ? "connected" : "disconnected"
              }`}
            />
            {connected ? "Connected" : "Connecting..."}
          </div>
        </div>
        <div className="header-right">
          <button
            className="sidebar-toggle"
            onClick={() => setShowSidebar(!showSidebar)}
          >
            {showSidebar ? "❮" : "❯"}
          </button>
        </div>
      </header>

      <main className="app-main">
        <div className="chat-container">
          <div className="messages-container">
            {messages.length === 0 && (
              <div className="welcome-message">
                <h2>🏰 Welcome to your AI-powered D&D Adventure!</h2>
                <p>
                  Create a character and start your journey in the mystical
                  realm of Eldoria.
                </p>
                <div className="quick-actions">
                  <button onClick={() => sendMessage("Start a new adventure")}>
                    🗡️ Start Adventure
                  </button>
                  <button
                    onClick={() => sendMessage("Tell me about this world")}
                  >
                    🌍 Explore World
                  </button>
                  <button
                    onClick={() => sendMessage("Help me create a character")}
                  >
                    👤 Create Character
                  </button>
                </div>
              </div>
            )}

            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}

            {isTyping && (
              <div className="typing-indicator">
                <div className="typing-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                The Dungeon Master is thinking...
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSubmit} className="input-form">
            <div className="input-container">
              <input
                ref={inputRef}
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="What do you do, adventurer?"
                disabled={!connected}
                className="message-input"
              />
              <button
                type="submit"
                disabled={!connected || !inputMessage.trim()}
                className="send-button"
              >
                ⚡ Send
              </button>
            </div>
            <DiceRoller onRoll={handleDiceRoll} />
          </form>
        </div>

        {showSidebar && (
          <aside className="sidebar">
            <div className="sidebar-content">
              <CharacterPanel
                character={character}
                onUpdate={handleCharacterUpdate}
              />

              <div className="game-controls">
                <h3>🎮 Game Controls</h3>
                <button
                  onClick={triggerWorldUpdate}
                  className="world-update-btn"
                >
                  🔄 Update World
                </button>
                <button
                  onClick={() => sendMessage("Show my inventory")}
                  className="inventory-btn"
                >
                  🎒 Inventory
                </button>
                <button
                  onClick={() => sendMessage("Check my stats")}
                  className="stats-btn"
                >
                  📊 Stats
                </button>
              </div>

              <SystemMonitor stats={systemStats} />

              <div className="session-info">
                <h4>Session Info</h4>
                <div className="session-id">ID: {sessionId.slice(0, 8)}...</div>
                <div className="message-count">Messages: {messages.length}</div>
              </div>
            </div>
          </aside>
        )}
      </main>

      <footer className="app-footer">
        <div className="footer-content">
          <span>
            🧙‍♂️ Powered by AI • Built with React & FastAPI • RTX 2060 Optimized
          </span>
          <div className="tech-stack">
            <span className="tech-badge">React</span>
            <span className="tech-badge">TypeScript</span>
            <span className="tech-badge">WebSocket</span>
            <span className="tech-badge">vLLM</span>
            <span className="tech-badge">PostgreSQL</span>
            <span className="tech-badge">Redis</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
