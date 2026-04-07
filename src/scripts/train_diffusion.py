"""
Train the covariate diffusion model from ICPSR graph data.

Usage:
    python -m src.scripts.train_diffusion \
        --data_dir ICPSR_22140 \
        --std_name HIV \
        --epochs 4000 \
        --checkpoint_dir checkpoints/diffusion \
        --seed 42
"""

import argparse
from pathlib import Path

import numpy as np

from src.data.covariate_spec import validate_one_hot
from src.data.dataset import EdgePairDataset
from src.data.icpsr_loader import ICPSRGraphData
from src.models.covariate_model.ddpm_covariate_model import DDPMCovariateModel


def main():
    parser = argparse.ArgumentParser(description="Train covariate diffusion model")
    parser.add_argument("--data_dir", type=str, default="ICPSR_22140",
                        help="Path to ICPSR_22140 data directory")
    parser.add_argument("--std_name", type=str, default="HIV",
                        choices=["HIV", "Gonorrhea", "Chlamydia", "Syphilis", "Hepatitis"])
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/diffusion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--num_eval_samples", type=int, default=50,
                        help="Number of samples to generate for evaluation")
    args = parser.parse_args()

    # Load data
    print(f"Loading ICPSR data from {args.data_dir} ({args.std_name})...")
    graph_data = ICPSRGraphData(args.data_dir, args.std_name)
    print(f"  Total edge pairs: {len(graph_data.edge_pairs)}")
    print(f"  Total nodes: {graph_data.graph.number_of_nodes()}")

    # Train/test split
    train_pairs, test_pairs = graph_data.train_test_split(
        test_fraction=args.test_fraction, seed=args.seed
    )
    print(f"  Train pairs: {len(train_pairs)}")
    print(f"  Test pairs: {len(test_pairs)}")

    # Create dataset
    train_dataset = EdgePairDataset(train_pairs)

    # Create and train model
    model = DDPMCovariateModel(
        num_steps=args.diffusion_steps,
        device="auto",
    )
    print(f"  Device: {model.device}")

    metrics = model.train(
        train_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )

    # Save checkpoint
    checkpoint_path = str(Path(args.checkpoint_dir) / f"ddpm_{args.std_name}.pt")
    model.save(checkpoint_path)

    # Evaluation: sample from test parents and validate
    if test_pairs:
        print(f"\nEvaluating on {min(args.num_eval_samples, len(test_pairs))} test pairs...")
        eval_pairs = test_pairs[:args.num_eval_samples]
        parent_covs = np.array([p for p, _ in eval_pairs])
        real_children = np.array([c for _, c in eval_pairs])

        generated = model.sample(parent_covs, seed=args.seed)

        # Check validity
        all_valid = validate_one_hot(generated)
        print(f"  All generated covariates valid: {all_valid}")

        # Compare per-group marginals
        print("  Per-group marginal comparison (generated vs real):")
        from src.data.covariate_spec import COVARIATE_GROUPS
        for name, start, end in COVARIATE_GROUPS:
            gen_marginal = generated[:, start:end].mean(axis=0)
            real_marginal = real_children[:, start:end].mean(axis=0)
            diff = np.abs(gen_marginal - real_marginal).mean()
            print(f"    {name:10s}: mean abs diff = {diff:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
