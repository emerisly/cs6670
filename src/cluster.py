import torch
import numpy as np
from annoy import AnnoyIndex
from scipy.cluster.hierarchy import linkage, fcluster
import concurrent.futures


def load_embeddings():
    # load the embeddings from ./data/embeddings.npy
    embeddings = torch.load("./data/embeddings.npy")
    return embeddings


def compute_distances_for_range(start, end, all_embeddings, annoy_index):
    distances = []
    for i in range(start, end):
        for j in range(i + 1, len(all_embeddings)):
            distance = annoy_index.get_distance(i, j)
            distances.append(distance)
    return distances


def compute_distance_matrix(all_embeddings, annoy_index, num_workers=4):
    n = len(all_embeddings)
    distances = []

    # Calculate the range of indices each worker will handle
    ranges = [
        (i * n // num_workers, (i + 1) * n // num_workers) for i in range(num_workers)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_range = {
            executor.submit(
                compute_distances_for_range, r[0], r[1], all_embeddings, annoy_index
            ): r
            for r in ranges
        }

        for future in concurrent.futures.as_completed(future_to_range):
            # Collect results as they are completed
            distances.extend(future.result())

    return distances


def build_index(embeddings, n_trees=100):
    n_dimensions = embeddings.shape[1]

    annoy_index = AnnoyIndex(n_dimensions, "angular")
    for i, embedding in enumerate(embeddings):
        annoy_index.add_item(i, embedding)

    annoy_index.build(n_trees)
    return annoy_index


def cluster(distances, threshold=0.5):
    # Threshold for hierarchical clustering. Lower values will reduce false positives
    # but may miss more.
    condensed_distances = np.array(distances)
    Z = linkage(condensed_distances, method="average", optimal_ordering=True)
    return fcluster(Z, t=threshold, criterion="distance")


def get_clusters(embeddings, assignments):
    # Build a 2D numpy array of the cluster assignments, where arr[i] contains all the
    # embeddings in cluster i
    clusters = [np.array([]) for _ in range(max(assignments))]
    means = [0 for _ in range(max(assignments))]
    for i, assignment in enumerate(assignments):
        clusters[assignment - 1] = np.append(clusters[assignment - 1], embeddings[i])
        number_of_entries = len(clusters[assignment - 1])
        means[assignment - 1] += (
            means[assignment - 1] * (number_of_entries - 1) + embeddings[i]
        ) / number_of_entries
    return clusters, means


def test_cluster():
    embeddings = np.random.rand(10, 2)
    annoy_index = build_index(embeddings)
    distance_matrix = compute_distance_matrix(embeddings, annoy_index)
    assignments = cluster(distance_matrix)
    clusters, means = get_clusters(embeddings, assignments)
    print(clusters)
    print(means)


def main():
    test_cluster()


if __name__ == "__main__":
    main()
