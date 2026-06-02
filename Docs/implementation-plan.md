# Zomato Restaurant Discovery Engine — Detailed Phase-Wise Implementation Plan

This implementation plan translates the architectural specifications in `Docs/architecture.md` into concrete, sequential development phases. Each phase defines clear deliverables, target files, structural tasks, and verification gates.

---

## Phase 1: Project Initialization & Dependency Management

**Goal:** Establish the directory layout, configure environment configurations, and install libraries including local embedding transformers.

### 1.1 Folder Structures Setup
Verify and build the canonical folder layout:
```text
zomato-milestone/
├── docs/
│   ├── context.md
│   ├── architecture.md
│   └── implementation-plan.md
├── data/
│   └── processed/
├── src/
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       ├── config.py
│       ├── models/
│       ├── ingestion/
│       ├── data/
│       ├── services/
│       └── api/
├── scripts/
├── requirements.txt
└── .env.example
```

### 1.2 Dependencies & Configurations
1.  **`requirements.txt`**: Add all necessary libraries:
    *   `pandas` (data structures)
    *   `datasets` (Hugging Face client loader)
    *   `transformers` & `torch` (or `sentence-transformers` for local embedding models)
    *   `numpy` (vector calculation and array storing)
    *   `pydantic` & `pydantic-settings` (model and configuration validation)
    *   `fastapi` & `uvicorn` (REST API layer)
    *   `streamlit` (if choosing Streamlit rapid interface)
    *   `groq` (strictly utilized external LLM API)
    *   `python-dotenv` (local environment loading)
2.  **`.env.example`**: Prepare environment configurations:
    ```env
    LLM_PROVIDER=groq
    GROQ_API_KEY=your_groq_api_key_here
    LLM_MODEL=llama-3.3-70b-versatile
    EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
    DATA_PATH=data/processed/restaurants.sqlite
    MAX_CANDIDATES=30
    ```
3.  **`src/app/config.py`**: Create Pydantic Settings class loading these environmental attributes gracefully.

### 1.3 Verification Gate
*   Run `pip install -r requirements.txt`.
*   Verify that `python -c "import transformers; import torch; print('PyTorch and Transformers loaded successfully')"` runs without errors.

---

## Phase 2: Ingestion & Local Embedding Generation

**Goal:** Load the Hugging Face dataset, clean the target fields, run a local transformers model to generate dense vectors for semantic text representations, and persist data locally.

### 2.1 Code Layout
1.  **`src/app/ingestion/loader.py`**: Load dataset `ManikaSaini/zomato-restaurant-recommendation` using the HF `datasets` library.
2.  **`src/app/ingestion/normalizer.py`**:
    *   Extract Restaurant Name, Location, Cuisine, rating, estimated_cost.
    *   Classify price into budget bands: `low` ($\le 500$), `medium` ($500 < \text{cost} \le 1500$), `high` ($> 1500$).
    *   Create a clean, standardized text summary string for each restaurant (combining cuisines, area, and tags) to vectorize.
3.  **`src/app/ingestion/pipeline.py`** or **`scripts/ingest.py`**:
    *   Initialize the local embedding pipeline (e.g., `all-MiniLM-L6-v2` via `sentence-transformers` or basic `transformers` pipeline).
    *   Vectorize the text summaries locally to produce dense embedding arrays (e.g., 384 dimensions).
    *   Store restaurant structured metadata inside a local SQLite table `data/processed/restaurants.sqlite`.
    *   Save generated vector arrays in `data/processed/embeddings.npy` or load directly into the DB.

### 2.2 Verification Gate
*   Run `python scripts/ingest.py`.
*   Assert that `data/processed/restaurants.sqlite` has been successfully created.
*   Confirm that `data/processed/embeddings.npy` exists and matches the row length of the SQLite table.

---

## Phase 3: Sequential Dependent Filters

**Goal:** Define canonical domain Pydantic models, write the repository access layer, and implement deterministic top-down hierarchy filters.

### 3.1 Code Layout
1.  **`src/app/models/domain.py`**: Establish core classes:
    *   `Restaurant` (metadata matching Zomato columns).
    *   `UserPreferences` (hierarchical properties, semantic text query, capacity $N$).
2.  **`src/app/data/repository.py`**:
    *   Implement `RestaurantRepository` connecting to the local SQLite database.
    *   Add query interfaces to return rows and map results back into lists of Pydantic domain items.
3.  **`src/app/services/filter_service.py`**:
    *   Implement the strict sequential hierarchy matching rules:
        *   **Step 1:** Filter strictly by `location` (case-insensitive locality string).
        *   **Step 2:** Filter matching `cuisine` parameters.
        *   **Step 3:** Filter rating values $\ge$ `min_rating`.
        *   **Step 4:** Filter records where `budget_band` matches.
    *   Expose a search entrance that returns isolated candidate lists. Implement short-circuiting: return an empty list early if zero records match.

### 3.2 Verification Gate
*   Run unit tests checking each hierarchical filtering dimension.
*   Verify that passing a non-existent location successfully triggers a zero-candidate short-circuit in the filter service.

---

## Phase 4: Local Vector Semantic Search Engine

**Goal:** Implement similarity logic to match qualitative, unstructured inputs against the deterministic candidate subset.

### 4.1 Code Layout
1.  **`src/app/services/vector_search.py`**:
    *   Load the local embedding model configured during app startup.
    *   Vectorize the user's qualitative preferences string (`additional_preferences`) into a dense search vector.
    *   Fetch the numpy embeddings array for the pre-filtered candidate subset.
    *   Compute cosine similarity scores between the user query embedding and candidate restaurant vectors.
    *   Rank the candidates by similarity score.
    *   Expose a search endpoint that returns the top $N$ candidates, where $1 \le N \le 10$.

### 4.2 Verification Gate
*   Write unit tests mapping similarity results.
*   Assert that the vector search successfully limits results strictly to the value of $N$ specified by the user.

---

## Phase 5: LLM Orchestration & Fallback Mitigation

**Goal:** Write prompt templates, configure connection clients, parse structured outputs, and implement error mitigation fallbacks.

### 5.1 Code Layout
1.  **`src/app/services/prompt_builder.py`**: Build instruction contexts including system constraints (grounding), structured candidate lists, user preferences, and a strict requirement to return outputs in valid JSON.
2.  **`src/app/services/llm_client.py`**: Design a wrapper around the Groq API (strictly utilizing the `groq` SDK) with low temperatures (0.3) utilizing the `llama-3.3-70b-versatile` model.
3.  **`src/app/services/response_parser.py`**: Parse structured lists of recommendation rankings and explanations. Incorporate regex capture for brackets `{}` or `[]` to isolate JSON content from markdown code wrappers.
4.  **`src/app/services/orchestrator.py`**:
    *   Coordinate components: `Input validation` $\rightarrow$ `Filter candidates` $\rightarrow$ `Semantic vector similarity search` $\rightarrow$ `Inject into Prompt Builder` $\rightarrow$ `Request LLM` $\rightarrow$ `Parse recommendations` $\rightarrow$ `Validate IDs and build final responses`.
    *   **Fallback Policies:**
        *   If the LLM connection fails or times out, fall back to sorting the vector candidates by rating, attaching a static, professional fallback explanation.
        *   Discard any recommendations containing restaurant IDs that were not in the vector search candidate list.

### 5.2 Verification Gate
*   Run integration tests using mocked LLM API connections.
*   Confirm that LLM timeouts and parsing failures recover gracefully without returning server/client failures.

---

## Phase 6: Application API & Zero-Scroll UI Viewport

**Goal:** Build endpoints using FastAPI and develop a premium web interface constrained to $100\text{vh}$ / $100\text{vw}$ that dynamically fits cards based on $N$.

### 6.1 Backend FastAPI Routers
*   **`src/app/api/routes.py`**: Expose metadata routes (`/locations`, `/cuisines`, `/health`) and the core route `POST /recommendations`.

### 6.2 Viewport UI Implementation
*   **Zero-Scroll Layout Shell:** Use custom CSS configuration properties to enforce absolute vertical and horizontal containment (`height: 100vh; overflow: hidden;`).
*   **Hierarchical Preference Panel:** Design a fixed control panel for top-down parameter inputs, plus a text input area for qualitative inputs.
*   **Self-Adjusting Card Grid:** Write programmatic styling rules (CSS grids or dynamic tailwind column maps) that adapt programmatically based on $N$ ($1 \le N \le 10$):
    *   **$N = 1$:** Single large dashboard card filling the dynamic layout.
    *   **$N \le 3$:** Grid renders cards in a single row with expanded columns.
    *   **$N \le 6$:** Grid switches to 2 rows of 3 columns, compressing paddings.
    *   **$N \le 10$:** Grid renders a highly compact 2x5 grid, adjusting fonts, margin dimensions, and text summaries to guarantee zero-scroll viewport compliance.
*   **Visual Elements:** Each card displays name, rating stars, location, cuisine badges, price category, and the custom generative explanation.

### 6.3 Verification Gate
*   Launch backend and frontend apps.
*   Validate end-to-end user flows for various values of $N$ (e.g., $N=1, 4, 10$) and confirm that no vertical scrollbars appear on standard screen dimensions.
