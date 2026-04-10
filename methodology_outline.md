## Methodology Outline

### 1. Schema and Ontology
- Define core entities and their roles in the recommendation pipeline.
  - Course: learning unit with metadata and skill outcomes.
  - Skill: canonical competency node used for retrieval and reasoning.
  - Candidate Profile: current skill state extracted from CV.
  - Job Requirement: target skill set extracted from JD.
- Define relationship types and semantics.
  - `TEACHES` (Course -> Skill): skill delivered after completing a course.
  - `REQUIRES` (Course -> Skill): prerequisite skill required before enrollment.
  - Optional analytical edges: similarity links between skills for soft matching.
- Specify property schema for each node/edge.
  - Identifiers, labels, category/domain, source, confidence, timestamps.
- State graph constraints and integrity rules.
  - Uniqueness constraints for `course_id`, `skill_id`.
  - Valid edge constraints (no dangling skill references).
  - Consistency checks for duplicated/ambiguous skill labels.
- Explain ontology design rationale.
  - Why this schema supports explainability.
  - Why prerequisite-aware ordering requires explicit `REQUIRES` semantics.

### 2. Skill Canonicalization and Graph Construction
- Input preparation.
  - Course descriptions and extracted raw skills from the data factory output.
  - Distinguish outcome skills vs. entry/prerequisite skills.
- Canonicalization pipeline (2-stage deduplication).
  - Stage 1: fuzzy matching to merge lexical variants/typos.
  - Stage 2: embedding-similarity merge for semantic variants.
- Canonical skill creation.
  - Select canonical label per cluster.
  - Preserve aliases for traceability and later retrieval.
  - Assign deterministic `skill_id` for stable graph references.
- Graph construction workflow.
  - Create skill nodes from canonical inventory.
  - Create course nodes from curriculum metadata.
  - Create `TEACHES` edges from skill outcomes.
  - Create `REQUIRES` edges from entry requirements.
- Storage and serving indexes.
  - Store graph in Neo4j for relational traversal.
  - Store canonical skill vectors in Elasticsearch for vector retrieval.
- Quality checks and build metrics.
  - Raw-to-canonical reduction ratio.
  - Number of orphan skills/courses.
  - Edge counts and connectivity statistics.

### 3. Automated Skill-Gap Identification
- Problem formulation.
  - Given target skills from JD and existing skills from CV, identify missing skills.
- Skill representation.
  - Encode JD skills and CV skills in a shared embedding space.
- Gap decision rule.
  - For each JD skill, compute maximum similarity to all CV skills.
  - Mark as gap if max similarity is below threshold.
- Threshold strategy.
  - Use default global threshold from configuration.
  - Support per-request threshold override for sensitivity tuning.
- Output artifacts.
  - Raw gap terms (human-readable).
  - Optional confidence scores for analysis and debugging.
- Error patterns to monitor.
  - False gaps due to lexical mismatch.
  - Missed gaps due to overly permissive threshold.

### 4. Hybrid Retrieval and Sequential Learning Path Reasoning
- Hybrid retrieval stage.
  - Map each gap term to canonical skills via vector search in Elasticsearch.
  - Aggregate unique canonical skill IDs as retrieval targets.
- Course candidate retrieval.
  - Query Neo4j for courses that `TEACHES` any target canonical skills.
  - Compute coverage-based ranking features (covered gaps, coverage count).
- Sequential learning path reasoning.
  - Use `REQUIRES` dependencies to order recommended courses.
  - Build prerequisite-aware path so foundational skills appear before advanced ones.
  - Resolve ties by coverage and dependency depth.
- Output structure.
  - Ranked course list with coverage explanation.
  - Ordered learning path with prerequisite logic.
- Explainability layer.
  - For each recommended course, expose which gaps are covered.
  - For each path transition, expose prerequisite justification.
- Evaluation perspective.
  - Compare baseline embedding-only retrieval vs. designed hybrid flow.
  - Report Precision@K, Recall@K, HitRate@K, nDCG@K.
  - Analyze trade-off between semantic matching strength and prerequisite coherence.
