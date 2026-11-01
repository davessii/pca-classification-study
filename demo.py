"""Runnable demonstration, separate from the original thesis notebook."""
import argparse
import csv
from pathlib import Path
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def compare(samples=900):
    X, y = make_classification(n_samples=samples, n_features=24, n_informative=8,
        n_redundant=12, n_clusters_per_class=2, class_sep=1.1, random_state=42)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2,
        stratify=y, random_state=42)
    results = []
    for family in ('Random Forest', 'MLP'):
        for use_pca in (False, True):
            estimator = (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
                if family == 'Random Forest' else MLPClassifier(hidden_layer_sizes=(32, 16),
                    max_iter=400, early_stopping=True, random_state=42))
            steps = [('scale', StandardScaler())]
            if use_pca:
                steps.append(('pca', PCA(n_components=0.95, svd_solver='full')))
            steps.append(('model', estimator))
            model = Pipeline(steps)
            start = time.perf_counter()
            model.fit(train_x, train_y)
            elapsed = time.perf_counter() - start
            predicted = model.predict(test_x)
            results.append(dict(model=family, pca=use_pca,
                features=int(model.named_steps['pca'].n_components_) if use_pca else X.shape[1],
                accuracy=float(accuracy_score(test_y, predicted)),
                f1=float(f1_score(test_y, predicted)), fit_seconds=elapsed))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results/demo'))
    parser.add_argument('--samples', type=int, default=900)
    args = parser.parse_args()
    if args.samples < 200:
        parser.error('--samples must be at least 200')
    rows = compare(args.samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output_dir/'metrics.json').write_text(json.dumps({
        'dataset': 'sklearn.make_classification', 'seed': 42, 'samples': args.samples,
        'scope': 'Synthetic software demonstration; not a reproduction of thesis results.',
        'results': rows}, indent=2), encoding='utf-8')
    labels = [r['model'] + (' + PCA' if r['pca'] else '') for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    bars = ax.bar(labels, [r['f1'] for r in rows], color=['#24577c', '#568cab', '#24577c', '#568cab'])
    ax.bar_label(bars, fmt='%.3f', padding=3)
    ax.set_ylim(0, 1.08); ax.set_ylabel('F1 on held-out synthetic data')
    ax.set_title('Runnable demo: Random Forest and MLP, with and without PCA')
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(args.output_dir/'comparison.svg')
    for row in rows:
        print(f"{row['model']:14} PCA={str(row['pca']):5} features={row['features']:2} F1={row['f1']:.3f}")


if __name__ == '__main__':
    main()
