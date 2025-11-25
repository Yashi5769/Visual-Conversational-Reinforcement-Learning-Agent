# 🤖 VCRL-DKR: Visual-Conversational Reinforcement Learning Agent
### with Dynamic Knowledge Retrieval & Embodied AI

![Project Status](https://img.shields.io/badge/Status-Complete-success)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue)
![Framework](https://img.shields.io/badge/Framework-AI2--THOR-orange)
![AI Model](https://img.shields.io/badge/LLM-Llama--3-green)

## 📖 Executive Summary
**VCRL-DKR** is a state-of-the-art **Neuro-Symbolic AI system** designed to bridge the gap between high-level natural language understanding and low-level physical robotic interaction. 

Unlike traditional chatbots that live in text, this agent acts in a **photorealistic 3D world** (AI2-THOR). It utilizes a "Brain-Body" architecture where a Large Language Model (LLM) acts as the Planner, Computer Vision (YOLO) acts as the Eyes, and Reinforcement Learning (RL) provides autonomous navigation capabilities.

---

## 🌟 Key Capabilities

### 1. 🧠 Cognitive Reasoning (The Brain)
* **Contextual Awareness:** Uses **RAG (Retrieval-Augmented Generation)** to recall household facts (e.g., *"Where is the remote?"* -> *"It's usually on the sofa"*).
* **Dynamic Planning:** Converts abstract requests like *"I am hungry"* into actionable physical steps (`GOTO Kitchen` -> `OPEN Fridge` -> `GRAB Food`).

### 2. 👁️ Visual Perception (The Eyes)
* **Object Detection:** Integrated **YOLOv8** to identify objects in real-time from the robot's camera feed.
* **Semantic Mapping:** Maps natural language words (e.g., "Couch") to visual classes (e.g., "Sofa").

### 3. 🦾 Physical Interaction (The Body)
* **Manipulation:** Capable of **Opening** doors, **Closing** containers, **Grabbing** objects, and **Dropping** items.
* **Visual Servoing:** Implements intelligent **Approach Logic**—the robot spots an object from a distance, rotates to face it, and walks until it is within arm's reach (1.0m) before interacting.

### 4. 🧭 Hybrid Navigation (The Feet)
* **Teleportation:** For known semantic locations (Kitchen, Bedroom).
* **RL Wandering:** A trained **PPO (Proximal Policy Optimization)** agent that can autonomously explore unknown environments without colliding with walls.


## 🏗️ System Architecture

The project follows a modular pipeline:

```mermaid
graph LR
    A[User Input] --> B[RAG Memory];
    B --> C[LLM Planner];
    C --> D[JSON Execution Plan];
    D --> E[Robot Agent];
    E --> F[AI2-THOR Simulator];
    F -- Visual Feedback --> E;
    E -- Status Updates --> G[Web Dashboard];

exit:

## 📂 Directory Structure

```text
VCRL_DKR/
├── app.py                 # Main Web Application (Streamlit Interface)
├── config.py              # Configuration Settings & API Keys
├── requirements.txt       # Dependency List
├── .env                   # API Keys (Hidden)
├── models/
│   ├── yolov8n.pt         # Computer Vision Model
│   └── ppo_nav/           # Trained Reinforcement Learning Brain
└── modules/
    ├── agent.py           # Robot Control Logic (Movement, Physics, RL)
    ├── knowledge.py       # RAG System (ChromaDB Vector Store)
    ├── planner.py         # LLM Interface (Prompt Engineering)
    └── vision.py          # YOLO Wrapper for Object Detection
