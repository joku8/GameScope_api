# GameScope - Project Roadmap

## Phase 0+1: Data Collection & Exploratory Analysis

**Goal:** Build a clean, unified dataset and understand video game landcape through visualization.

### Core Work

- Ingest game metadata from public APIs (genres, mechanics, ratings, playtime)
- Normalize and deduplicate games across sources
- Handle missing and conflicting data explicitly
- Perform exploratory data analysis using interactive visualizations
- etc etc...

### MVP Deliverables

- Ingestion pipeline for data sources
- Unified game schema
- Visualizations:
  - Game landscape (clusters by mechanics/genres)
  - Critic vs player rating comparison
  - Time-to-complete vs user satisfaction
- Note teh observed patterns and biases

### Outcome

A reproducible dataset and visual understanding of how games relate to each other.
Simple machine learning groupings and clusterings

---

## Phase 2: Explainable Recommendation Engine

**Goal:** Recommend games based on known user preferences with fully transparent logic.

### Core Work

- **User onboarding:**
  - Rate known games
  - Answer structured preference questions (time, difficulty, novelty, etc.)
- Deterministic, weighted scoring algorithm (no black box ML)
- Recommendation explanations tied directly to scoring components

### MVP Deliverables

- User preference model derived from ratings + questions
- Recommendation engine producing ranked game lists
- Explanation view showing contribution of each factor
- Visualization overlaying user taste onto the game landscape

### Outcome

Users can see both what is recommended and why. Probobly going to be some api... Frontend later

---

## Phase 3: Reinforcement & Feedback Loop

**Goal:** Improve recommendations using explicit post-play feedback.

### Core Work

- Collect feedback after users try recommended games
- Adjust user preference weights based on success/failure signals
- Track system confidence vs actual fit

### MVP Deliverables

- Feedback form (fit rating + reason)
- Updated recommendations based on feedback
- Visualizations showing:
  - Preference changes over time
  - Where recommendations succeeded or failed

### Outcome

The system learns transparently from user experience without retraining opaque models.

---

## Phase 4: Frontend and User Interface

**Goal:** Deploy a user-facing interface for GameScope that makes recommendations understandable, actionable, and improvable through direct user feedback.

### Core Work

- Capture structured feedback after users try recommended games
- Incrementally adjust user preference weights based on:
  - Positive signals (enjoyed, finished, would recommend)
  - Negative signals (bounced early, disliked mechanics/theme)
- Track and compare:
  - System confidence in a recommendation
  - Actual user-perceived fit

### MVP Deliverables

- Authentication
  - Sign up, login, logout, persist session
- User game library
  - Collects info about played games (specific fields, tbd)
- Game details page
  - Pull info from igdb and also display user inputted fields (allow edit)
- Game recommendations
  - Pulled from backend
- Feedback on recommendations
  - Influences preference weights in backend

### Outcome

Create a frontend for the backend systems implemented in previous phases. Should enable game recommendations based on user tastes (ML) and allow indexing for played games.

---

## Phase 5: Cross-User Signals & Collaborative Influence

**Goal:** Enhance GameScope recommendations by incorporating anonymized patterns from other users with similar tastes, while preserving interpretability and user control.

### Core Work

- Identify taste similarity between users based on:
  - Library overlap
  - Feedback patterns
  - Preference weight vectors
- Blend collaborative signals with existing content-based recommendations
- Track when recommendations are influenced by:
  - Personal history
  - Similar users
  - Hybrid signals

### MVP Deliverables

- Similar-user scoring (backend-generated)
- Hybrid recommendation feed:
  - Backend + collaborative influence
- Lightweight explanation tag:
  - “Recommended because players with similar tastes liked this”
- Safeguards:
  - No direct user-to-user data exposure
- Fully anonymized aggregation

### Outcome

GameScope recommendations improve beyond a single-user model, leveraging collective taste patterns while remaining transparent, privacy-preserving, and explainable.
