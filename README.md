# GameScope

A game recommendation engine with explainable, transparent recommendations.

---

## Phase 1: Data Collection & Exploratory Analysis

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
