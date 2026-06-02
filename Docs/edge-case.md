# Zomato Restaurant Discovery Engine — Edge Cases & Fallback Policies

This document outlines critical edge cases, network failures, input anomalies, and viewport constraints along with their corresponding mitigation strategies and programmatic fallback policies.

---

## 1. Hierarchical Filtering Edge Cases

### 1.1 Over-Constrained Hierarchical Filters (Zero Candidates)
*   **Scenario:** The user inputs a combination of filters (e.g., specific location + rare cuisine + rating $\ge 4.9$ + low budget) that yields exactly zero matching restaurants in the local SQLite store.
*   **Mitigation (Early Exit):** The `FilterService` must immediately return an empty candidate response. The `RecommendationOrchestrator` catches this state and short-circuits the pipeline before calling the vector search engine or requesting the LLM API (saving cost and latency).
*   **Fallback UI Behavior:** The API returns a distinct 200 payload with empty results and a status indicator. The UI renders a dedicated empty state message: *"No restaurants match your exact filters. Try broadening your criteria (e.g., adjusting rating or budget)!"*

### 1.2 Location String Anomalies (Misspellings / Out-of-Bound areas)
*   **Scenario:** The user enters a location that does not exist in the ingested Hugging Face dataset.
*   **Mitigation:** 
    1.  **Datalists / Autocomplete:** The UI uses metadata endpoints (`/api/v1/metadata/locations`) to feed an interactive autocomplete search field, locking input options to valid values.
    2.  **Normalization & Matching:** The backend applies case-insensitive, trimmed, and substring matching (e.g., using SQL `LIKE %location%` or fuzzy ratio matching) during filtering. If still unmatched, it triggers the zero-candidate early exit.

---

## 2. Local Embedding & Vector Search Edge Cases

### 2.1 Blank or Empty Qualitative Input (`additional_preferences`)
*   **Scenario:** The user submits the search form without entering any free-text qualitative preference (e.g. they only selected Location, Cuisine, and Rating).
*   **Mitigation (Bypass Vector Search):** If the qualitative input is empty or contains only whitespace, the `VectorSearchEngine` skips semantic cosine similarity computation. It ranks candidates using the deterministic database sorting (sorted descending by rating and review votes) and limits the pool directly to $N$.
*   **Prompt Construction:** The prompt sent to the LLM adjusts instructions to rank and explain options based solely on their core parameters (cuisine matching, value, and rating).

### 2.2 Massive Matching Candidate Pools
*   **Scenario:** A broad query (e.g., "Bangalore" + "North Indian" + rating $\ge 3.0$) matches hundreds of database records, threatening LLM context windows and response latency.
*   **Mitigation (Hard Capping):** The `FilterService` applies a hard cap (e.g., top 30 sorted by rating/votes) before handing the candidates to the vector similarity engine. The similarity engine then matches qualitative embeddings and slices the subset strictly to the requested limit $N$ ($1 \le N \le 10$) before generating the LLM prompt.

---

## 3. LLM API Integration & Extraction Failures

### 3.1 LLM Provider Outage or Network Timeout
*   **Scenario:** The external Groq/Gemini/OpenAI API is unreachable, throttles requests (429 Rate Limited), or exceeds connection timeouts (504).
*   **Mitigation (Rating-Based Fallback Engine):** 
    1.  **API Client Wrapper:** The `LLMClient` implements a single connection retry with an exponential backoff of 1 second.
    2.  **Graceful Recovery:** If the retry fails, the `Orchestrator` catches the exception. Instead of returning a 500 error, it returns the top $N$ pre-filtered vector candidates. It maps each candidate into a card layout and attaches a pre-written static explanation: *"Recommended based on popularity, excellent cuisines, and high ratings in your locality."*
    3.  **UI Notification:** The interface displays a small, non-intrusive warning tag: *"AI Explanation Offline — serving verified database matches."*

### 3.2 Malformed LLM Response (Parsing / Schema Failure)
*   **Scenario:** The LLM responds with raw text, markdown-wrapped blocks, or truncated strings instead of conforming strictly to the requested JSON layout.
*   **Mitigation (Regex Isolate & Parse):** 
    1.  **Regex Capture:** The `ResponseParser` applies regex boundaries to locate and extract everything between the outer curly brackets `{ ... }` or square brackets `[ ... ]`.
    2.  **Schema Check:** If parsing the isolated block fails, or if critical attributes (e.g., `restaurant_id`) are missing, the system triggers the **Rating-Based Fallback Engine** to construct standard cards using database values.

### 3.3 Hallucinated Restaurant Recommendations
*   **Scenario:** The LLM suggests a highly rated restaurant that was NOT present in the candidate list passed to it.
*   **Mitigation (ID Verification):** During the data merger phase, the `Orchestrator` verifies each ID returned by the LLM against the original list of pre-filtered vector candidates. Any entry containing an unrecognized ID is stripped immediately.

---

## 4. Zero-Scroll Viewport Layout Viewport Edge Cases

### 4.1 UI Text Overflow (Long Restaurant Names & Descriptions)
*   **Scenario:** Long restaurant names (e.g., "The Grand Royal Spice Palace Bar and Bistro") or lengthy AI explanations cause grid cards to expand vertically, breaking the $100\text{vh}$ zero-scroll viewport height constraint.
*   **Mitigation:**
    1.  **Strict Height Styling:** Cards are styled with fixed height layouts (e.g., `max-h-[30vh]` or proportional grid allocations) and use `overflow-hidden` or `text-ellipsis` for long titles.
    2.  **Description Limits:** The Prompt Builder explicitly instructs the LLM to restrict each custom explanation to a single sentence (maximum 20 words).
    3.  **Scrollable Card Content:** Card bodies allow isolated scrollbars (`overflow-y-auto`) for the explanation text box only, keeping the parent grid layout static and contained.

### 4.2 Dynamic Card Sizing Calculations (Adaptive Layouts)
*   **Scenario:** High values of $N$ (e.g., $N=10$) crowd the layout on small monitors, while low values (e.g., $N=1$) leave excessive empty space.
*   **Mitigation (Adaptive CSS Grid):**
    *   **$N = 1$:** Grid allocates 1 prominent, widescreen dashboard card (`h-full w-full`).
    *   **$N \le 3$:** Grid aligns items in a single horizontal row (`grid-cols-3`), maximizing card width.
    *   **$N \le 6$:** Grid adjusts to a 2-row, 3-column matrix (`grid-rows-2 grid-cols-3`). Card margins and font sizes scale down automatically by $15\%$.
    *   **$N \le 10$:** Grid compresses into a highly compact 2x5 layout (`grid-rows-2 grid-cols-5`). Font sizes are reduced (`text-xs`), and card padding decreases (`p-2`) to guarantee all 10 cards stay within the viewport height.
*   **Resizing Event:** Uses CSS flex and viewport units (`vh`, `vw`, `dvh`) combined with Tailwind container queries (`@container`) instead of fixed pixel sizes to maintain proportions during browser window scaling.
