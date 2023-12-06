import torch
import numpy as np
from annoy import AnnoyIndex
from scipy.cluster.hierarchy import linkage, fcluster
import concurrent.futures
import os
import shutil


def load_embeddings():
    # load the embeddings from ../data/images.npy
    save_dir = "../data/"
    images_load = np.load(save_dir+"images.npy")
    image_np = np.asarray(images_load)
    image_tensors_load = torch.from_numpy(image_np)
    image_tensors_load = torch.squeeze(image_tensors_load)
    image_tensors_load = image_tensors_load.type(torch.float16)
    print(type(image_tensors_load))
    print(image_tensors_load.size())

    return image_tensors_load

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
    index_assignments = [np.array([]) for _ in range(max(assignments))]

    for i, assignment in enumerate(assignments):
        # print(i, assignment)
        clusters[assignment - 1] = np.append(clusters[assignment - 1], embeddings[i])
        number_of_entries = len(clusters[assignment - 1])
        means[assignment - 1] += (
            means[assignment - 1] * (number_of_entries - 1) + embeddings[i]
        ) / number_of_entries
        index_assignments[assignment - 1] = np.append(index_assignments[assignment - 1], i)
    return clusters, means, index_assignments

def save_index_assignments(index_assignments):
    save_dir = "../data/cluster_indexes"
    
    # Remove the directory if it exists
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    # Create the directory
    os.makedirs(save_dir)

    for i in range(len(index_assignments)):
        var_path = os.path.join(save_dir+"/clusters_indexes_" + str(i) + ".npy")
        np_var = np.asarray(index_assignments[i],dtype=np.float32)
        np.save(var_path,np_var)

def save_clusters(clusters):
    save_dir = "../data/clusters"
    
    # Remove the directory if it exists
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    # Create the directory
    os.makedirs(save_dir)

    for i in range(len(clusters)):
        var_path = os.path.join(save_dir+"/clusters_" + str(i) + ".npy")
        np_var = np.asarray(clusters[i],dtype=np.float32)
        np.save(var_path,np_var)

def test_cluster():
    embeddings = np.random.rand(10, 2)
    save_dir = "../data/"
    test_path = os.path.join(save_dir+"/test_embedding.npy")

    # Check if the file exists
    if os.path.exists(test_path):
        # If it exists, delete the file
        os.remove(test_path)
    
    np.save(test_path, embeddings)

    annoy_index = build_index(embeddings)
    distance_matrix = compute_distance_matrix(embeddings, annoy_index)
    assignments = cluster(distance_matrix)
    clusters, means, index_assignments = get_clusters(embeddings, assignments)
    save_index_assignments(index_assignments)
    save_clusters(clusters)
    print("Embeddings")
    print(embeddings)
    print("\n")

    print("Clusters")
    print(clusters)
    print("\n")

    print("Means")
    print(means)
    print("\n")

    print("Index assignments")
    print(index_assignments)
    print("\n")

def embeddings_cluster():
    embeddings = load_embeddings()
    annoy_index = build_index(embeddings)
    distance_matrix = compute_distance_matrix(embeddings, annoy_index)
    assignments = cluster(distance_matrix)
    clusters, means, index_assignments = get_clusters(embeddings, assignments)
    # print(len(index_assignments))
    save_index_assignments(index_assignments)
    save_clusters(clusters)
    # print(means)


def main():
    # test_cluster()

    load_embeddings()
    embeddings_cluster()


if __name__ == "__main__":
    main()
