"""
Evaluate skill mapping metrics on generated dataset

Directly loads generated dataset, performs embedding-based mapping,
and calculates metrics without converting to course format.

Metrics:
- Accuracy@1: Percentage of correct top-1 predictions
- MRR@K: Mean Reciprocal Rank at K
- Precision@K, Recall@K: At different K values
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

from src.embeddings.embedding_service import EmbeddingService
from src.io.esco_embedding_loader import load_esco_embeddings
from src.pipelines.course_skill_mapping_pipeline import CourseSkillMappingPipeline
from src.models.course_skill import CourseSkill
from src.settings import Settings
from src.utils.config_utils import load_config
from src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger("data_factory.evaluate_metrics")


def evaluate_generated_dataset(
    dataset_file: Path,
    pipeline: CourseSkillMappingPipeline,
    esco_metadata: List[Dict],
    esco_embeddings: np.ndarray,
    k_values: List[int] = [1, 3, 5, 10, 20],
    max_samples: int = None
) -> Dict:
    """
    Evaluate mapping on generated dataset using the actual pipeline
    
    Args:
        dataset_file: Path to skill_training_dataset.json
        pipeline: CourseSkillMappingPipeline instance
        esco_metadata: ESCO skill metadata
        esco_embeddings: ESCO skill embeddings
        k_values: K values for metrics
        max_samples: Maximum samples to evaluate (None = all)
    """
    # Load generated dataset
    LOGGER.info(f"Loading generated dataset from {dataset_file}")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    if max_samples:
        samples = samples[:max_samples]
    
    LOGGER.info(f"Evaluating on {len(samples)} samples")
    
    # Convert samples to CourseSkill objects for pipeline
    LOGGER.info("Converting samples to CourseSkill format...")
    course_skills = []
    for idx, sample in enumerate(samples):
        # Create minimal CourseSkill object with the variation text
        query_text = sample['query']
        course_skill = CourseSkill(
            course_id=f"GEN_{idx}",
            course_title=f"Generated Sample {idx}",
            skill_name=query_text[:100],
            skill_type="outcome",
            description=query_text,
            category=sample.get('metadata', {}).get('skill_type'),
            proficiency_level=None,
            bloom_taxonomy_level=None,
            source_file=Path("generated")
        )
        course_skills.append(course_skill)
    
    # Encode using pipeline's embedding service
    LOGGER.info("Encoding variations using pipeline...")
    payloads = [skill.to_embedding_payload() for skill in course_skills]
    course_embeddings = pipeline._embedding_service.encode(payloads)
    
    # Compute similarity matrix using same logic as pipeline
    LOGGER.info("Computing similarity matrix...")
    from src.pipelines.course_skill_mapping_pipeline import _normalize_vectors
    normalized_course = _normalize_vectors(course_embeddings)
    normalized_esco = _normalize_vectors(esco_embeddings)
    similarity_matrix = normalized_course @ normalized_esco.T
    LOGGER.info(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    # Extract top-K predictions for each sample
    LOGGER.info("Extracting top-K predictions...")
    all_predictions = []
    max_k = max(k_values)
    
    for idx, sample in enumerate(tqdm(samples, desc="Processing samples")):
        similarities = similarity_matrix[idx]
        
        # Get top-k indices
        top_k_indices = np.argsort(-similarities)[:max_k]
        
        # Get URIs and scores for top-k
        pred_uris = [esco_metadata[i]['skill_id'] for i in top_k_indices]
        pred_scores = [float(similarities[i]) for i in top_k_indices]
        
        all_predictions.append({
            'sample_idx': idx,
            'ground_truth_uri': sample['metadata']['skill_uri'],
            'predicted_uris': pred_uris,
            'scores': pred_scores
        })
    
    # Calculate metrics
    LOGGER.info("Calculating metrics...")
    results = {
        'total_samples': len(samples),
        'evaluated_samples': len(all_predictions),
    }
    
    for k in k_values:
        accuracy = calculate_accuracy_at_k(all_predictions, k)
        mrr = calculate_mrr_at_k(all_predictions, k)
        precision = calculate_precision_at_k(all_predictions, k)
        recall = calculate_recall_at_k(all_predictions, k)
        
        results[f'accuracy@{k}'] = accuracy
        results[f'mrr@{k}'] = mrr
        results[f'precision@{k}'] = precision
        results[f'recall@{k}'] = recall
    
    return results, all_predictions


def calculate_accuracy_at_k(predictions: List[Dict], k: int) -> float:
    """Calculate Accuracy@K"""
    correct = 0
    for pred in predictions:
        gt_uri = pred['ground_truth_uri']
        pred_uris = pred['predicted_uris'][:k]
        if gt_uri in pred_uris:
            correct += 1
    return correct / len(predictions) if predictions else 0.0


def calculate_mrr_at_k(predictions: List[Dict], k: int) -> float:
    """Calculate Mean Reciprocal Rank @K"""
    reciprocal_ranks = []
    for pred in predictions:
        gt_uri = pred['ground_truth_uri']
        pred_uris = pred['predicted_uris'][:k]
        
        try:
            rank = pred_uris.index(gt_uri) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def calculate_precision_at_k(predictions: List[Dict], k: int) -> float:
    """Calculate Precision@K (for single ground truth)"""
    precisions = []
    for pred in predictions:
        gt_uri = pred['ground_truth_uri']
        pred_uris = pred['predicted_uris'][:k]
        
        # For single GT: precision = 1/k if GT in top-k, else 0
        if gt_uri in pred_uris:
            precisions.append(1.0 / k)
        else:
            precisions.append(0.0)
    
    return np.mean(precisions) if precisions else 0.0


def calculate_recall_at_k(predictions: List[Dict], k: int) -> float:
    """Calculate Recall@K (for single ground truth)"""
    recalls = []
    for pred in predictions:
        gt_uri = pred['ground_truth_uri']
        pred_uris = pred['predicted_uris'][:k]
        
        # For single GT: recall = 1 if GT in top-k, else 0
        recalls.append(1.0 if gt_uri in pred_uris else 0.0)
    
    return np.mean(recalls) if recalls else 0.0


def print_results(results: Dict) -> None:
    """Pretty print evaluation results"""
    print("\n" + "=" * 70)
    print("📊 Skill Mapping Evaluation Results")
    print("=" * 70)
    
    print(f"\n📈 Dataset:")
    print(f"  Total samples: {results['total_samples']:,}")
    print(f"  Evaluated samples: {results['evaluated_samples']:,}")
    
    print(f"\n🎯 Accuracy@K:")
    for k in [1, 3, 5, 10, 20]:
        if f'accuracy@{k}' in results:
            acc = results[f'accuracy@{k}']
            print(f"  Accuracy@{k:2d}: {acc:.4f} ({acc*100:6.2f}%)")
    
    print(f"\n📊 MRR@K (Mean Reciprocal Rank):")
    for k in [1, 3, 5, 10, 20]:
        if f'mrr@{k}' in results:
            mrr = results[f'mrr@{k}']
            print(f"  MRR@{k:2d}:      {mrr:.4f}")
    
    print(f"\n📈 Precision@K:")
    for k in [1, 3, 5, 10, 20]:
        if f'precision@{k}' in results:
            prec = results[f'precision@{k}']
            print(f"  Precision@{k:2d}: {prec:.4f}")
    
    print(f"\n📉 Recall@K:")
    for k in [1, 3, 5, 10, 20]:
        if f'recall@{k}' in results:
            rec = results[f'recall@{k}']
            print(f"  Recall@{k:2d}:    {rec:.4f} ({rec*100:6.2f}%)")
    
    print("=" * 70 + "\n")


def main():
    # Setup logging
    default_logging = Path(__file__).resolve().parent.parent / "config" / "logging.yaml"
    if default_logging.exists():
        setup_logging(default_logging)
    
    settings = load_config()
    
    # Paths
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    dataset_file = base_dir / "data" / "processed" / "training_dataset" / "test_dataset.json"
    
    if not dataset_file.exists():
        LOGGER.error(f"Dataset file not found: {dataset_file}")
        return
    
    # Load ESCO embeddings
    LOGGER.info("Loading ESCO embeddings...")
    esco_metadata, esco_embeddings = load_esco_embeddings(
        settings.paths.processed_embeddings_dir
    )
    LOGGER.info(f"Loaded {len(esco_metadata)} ESCO skills")
    
    # Load embedding service
    LOGGER.info("Loading embedding service...")
    embedding_service = EmbeddingService(settings.embedding)
    
    # Create pipeline instance (without MySQL service since we don't need it)
    LOGGER.info("Creating pipeline instance...")
    pipeline = CourseSkillMappingPipeline(
        settings,
        embedding_service,
        mysql_service=None,
        reranker_service=None
    )
    
    # Evaluate
    results, predictions = evaluate_generated_dataset(
        dataset_file,
        pipeline,
        esco_metadata,
        esco_embeddings,
        k_values=[1, 3, 5, 10, 20],
        max_samples=None  # Set to e.g. 1000 for faster testing
    )
    
    # Print results
    print_results(results)
    
    # Save results to mapping_metrics folder
    output_dir = base_dir / "data" / "processed" / "mapping_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_file = output_dir / "evaluation_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Metrics saved to {metrics_file}")
    
    # Save detailed predictions
    predictions_file = output_dir / "detailed_predictions.jsonl"
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    LOGGER.info(f"Detailed predictions saved to {predictions_file}")


if __name__ == "__main__":
    main()
