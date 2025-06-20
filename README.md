# ![PerceptusLabs Logo](https://github.com/Perceptus-Labs/Intentus/blob/main/public/logo.png?raw=true) Intentus - Omni Robotic Orchestrator

**Define the next generation of human-robot interaction.**
**Built by [PerceptusLabs](https://perceptuslabs.com)**

Intentus is a **scalable, self-learning orchestration server** that brings audio-visual intelligence to robots and autonomous agents. With memory, modular tools, and a structured planning system, it can perceive, reason, and act—adapting intelligently with each interaction.

> “The future isn’t just about robots that respond — it’s about robots that understand, plan, and grow.”

---

## ✨ Key Capabilities

* **Vision-Language-Action (VLA) Pipeline**
  Send structured text representations of audio/visual input; Intentus interprets, plans, executes, and adapts.

* **Web-Enabled Agent**
  Includes a web-browsing toolchain for open-ended question answering, research, or data retrieval.

* **Tool-Expandable**
  Add custom robot skills via natural language in under 100 lines. Tools live in `/tools` and can be hot-loaded.

* **Memory-Backed Planning**
  Multi-step agents with persistent memory and feedback-driven self-improvement.

* **Production-Ready**
  API-based architecture, secure with API key support, logging, and environment configuration.

* **Asynchronous & Scalable**
  Modern Python backend, fast async processing, with customizable agent parameters and execution lifecycles.

---

## 📦 Installation

```bash
git clone https://github.com/Perceptus-Labs/Intentus.git
cd intentus
python -m venv venv
source venv/bin/activate
pip install -e .
```

---

## 🚀 Quickstart: Local Demo

```bash
python intentus/examples/agent_demo.py
```

**Demo Task:** “What is the capital of France?”

You’ll see:

* Multi-step reasoning
* Query analysis
* Final output
* Agent memory dump
* Execution time and path

> 💡 See real-time agent logs in `example.log`, outputs saved to `example_outputs/`.

---

## 🧠 How It Works

1. **Your robot/system sends a structured intention**
   (transcribed speech, vision cues, context, etc.)
2. **Intentus parses & analyzes** the command with optional tools
3. **It plans actions, reasons over them**, and executes tools
4. **Feedback updates its memory**, and it continues iterating
5. **Returns a full report** on results, reasoning, and future suggestions

---

## 🛠️ Add Your Own Tools (VLA Plugins)

Developers can extend robot abilities via **natural language definitions** and lightweight Python handlers:

```bash
# Create a new tool file in intentus/tools/
# Describe your tool's purpose in plain English
# Add a function with input/output contracts
```

Your robot can learn to:

* Navigate new terrain
* Run factory checks
* Diagnose mechanical issues
* Interface with APIs or microcontrollers

> 🔧 Tool architecture is fully modular and scalable.

---

## 🧪 Run the API Server (Orchestrator)

The orchestrator receives intention payloads from your robot/Go backend:

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Set environment variables (either in .env or directly in the code)

```bash
export ORCHESTRATOR_API_KEY="your-api-key"
export ORCHESTRATOR_HOST="0.0.0.0"
export ORCHESTRATOR_PORT="8000"
```

### 3. Run the server

```bash
python main.py
```

> Now available at: `http://localhost:8000`

---

## 🔍 API Reference

### `POST /orchestrate`

Receives a structured intention payload:

```json
{
  "session_id": "session-123",
  "intention_type": "user_query",
  "description": "User asked about the weather",
  "confidence": 0.95,
  "transcript": "What's the weather like today?",
  "environment_context": "User is in San Francisco, CA",
  "timestamp": 1703123456
}
```

Returns:

```json
{
  "success": true,
  "query_analysis": "...",
  "base_response": "...",
  "final_output": "...",
  "execution_time": 2.5,
  "steps_taken": 3,
  "memory": ["Step 1: ...", "Step 2: ..."]
}
```

Authentication:
`Authorization: Bearer your-api-key`

---

## 🔍 Example Use Case

Imagine a robot assistant in a kitchen:

1. Sees an image of spilled flour (visual cue)
2. Hears “What do I do now?” (audio command)
3. Intentus processes the environment:

   * Identifies the spill
   * Suggests cleanup plan
   * Executes cleaning tool
   * Stores feedback if plan failed/succeeded

This is **closed-loop intention orchestration** — and it’s just the beginning.

---

## 🧰 Developer Configuration

Customize the agent in `main.py > get_agent()`:

```python
AgentConfig(
    llm_engine="gpt-4.1-mini",
    enabled_tools=["Wikipedia_Knowledge_Searcher_Tool"],
    verbose=True,
    max_steps=5,
    temperature=0.7,
)
```

---

## ✅ Testing

```bash
python test_orchestrator.py
```

Make sure `API_KEY` in the test script matches your environment variable.

---

## 🔐 Security

* `/orchestrate` is protected by an API key
* Set `ORCHESTRATOR_API_KEY`
* Authentication is optional but **highly recommended**

---

## 📈 For Investors & Partners

Intentus is a foundational system in **robot cognition**:

* It’s **tool-agnostic** and can support a fleet of heterogeneous robots
* It bridges **perception, language, and control**
* It allows **natural language tool creation** — no low-level firmware work required
* It logs full **memory and reasoning chains**, giving you total auditability
* It is **modular, API-first, and cloud-compatible**

Let’s redefine how robots understand and respond to the world.

---

## 👥 About PerceptusLabs

We are building the future of robotic cognition — creating agents that not only follow instructions, but **think, adapt, and evolve**.

Visit [perceptuslabs.ai](https://perceptuslabs.ai) for more.
