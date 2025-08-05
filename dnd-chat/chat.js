const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");

let history = []; // Will only contain alternating user/assistant pairs

const systemPrompt = "You are a Dungeon Master. Narrate the adventure for the player based on their choices. Use vivid fantasy descriptions.";

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const prompt = userInput.value.trim();
  if (!prompt) return;

  addMessage("user", prompt);
  
  // For the first message, prepend the system prompt to the user message
  let userMessage;
  if (history.length === 0) {
    userMessage = systemPrompt + "\n\n" + prompt;
  } else {
    userMessage = prompt;
  }
  
  history.push({ role: "user", content: userMessage });
  userInput.value = "";

  // Send only the alternating user/assistant history
  const validMessages = [...history];

  try {
    console.log("Sending messages:", validMessages);

    const res = await fetch("http://localhost:8000/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        messages: validMessages,
        temperature: 0.8,
        max_tokens: 300
      })
    });

    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const reply = data.choices[0].message.content;

    addMessage("assistant", reply);
    history.push({ role: "assistant", content: reply });

  } catch (err) {
    console.error("⚠️ API error:", err);
    addMessage("assistant", "⚠️ The Dungeon Master encountered an error.");
  }
});

function addMessage(role, content) {
  const div = document.createElement("div");
  div.classList.add("chat-bubble", role);
  div.textContent = role === "user" ? `🧍 ${content}` : `🧙 ${content}`;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}