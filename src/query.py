import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_top_K_vectors(clusters_of_images, query_vector, K):
    """
    This function takes in three inputs: clusters_of_images, query_vector, and K. It finds the mean vector of each cluster and gets the top K clusters based on cosine similarity. Then, from each of the top K clusters, it finds the vector in the cluster that is closest to the query vector. Finally, it returns a numpy array of those top K vectors.

    Args:
        clusters_of_images (numpy.ndarray): A numpy array of dimensions (number_of_clusters, batch, embedding_size).
        query_vector (numpy.ndarray): A numpy array of size embedding_size.
        K (int): An integer representing the number of top clusters to return.

    Returns:
        numpy.ndarray: A numpy array of dimensions (K, 512) containing the top K vectors.
    """
    num_clusters, batch_size, embedding_size = np.shape(clusters_of_images)
    # Convert numpy arrays to PyTorch tensors
    clusters_of_images = torch.from_numpy(clusters_of_images)
    query_vector = torch.from_numpy(query_vector)

    # Calculate cosine similarity between each cluster mean and the query vector
    cluster_means = clusters_of_images.mean(
        dim=1
    )  # Calculate the mean for each cluster
    query_vector_flat = query_vector.view(1, -1)  # Flatten the query vector
    cos_sim = torch.nn.functional.cosine_similarity(
        cluster_means.view(-1, embedding_size), query_vector_flat, dim=1
    )

    # Get the indices of the top K clusters based on cosine similarity
    top_k_indices = cos_sim.topk(K).indices

    # Add assertions to verify the top_k_indices
    for i in range(K):
        assert top_k_indices[i] == torch.argsort(cos_sim, descending=True)[i].item()

    # Find the vector in each of the top K clusters that is closest to the query vector
    top_k_vectors = []
    for i in range(K):
        cluster_index = top_k_indices[i]

        # Calculate cosine similarity between the cluster vectors and the query vector
        cos_sim_cluster = torch.nn.functional.cosine_similarity(
            clusters_of_images[cluster_index].view(-1, embedding_size),
            query_vector_flat,
            dim=1,
        )

        # Find the closest vector within the cluster
        closest_vector_index = torch.argmax(cos_sim_cluster)
        top_k_vectors.append(
            clusters_of_images[cluster_index]
            .view(-1, embedding_size)[closest_vector_index]
            .numpy()
        )

    # Reshape the top K vectors to their original shape
    top_k_vectors = np.array(top_k_vectors).reshape(K, embedding_size)

    return top_k_vectors

# top K clusters and C vectors within those clusters
def find_top_K_C_vectors(clusters_of_images, query_vector, K, C=1):
    """
    This function finds the top K clusters and, for each of the top K clusters, returns the top C vectors based on cosine similarity.

    Args:
        clusters_of_images (numpy.ndarray): A numpy array of dimensions (number_of_clusters, batch, embedding_size).
        query_vector (numpy.ndarray): A numpy array of size (embedding_size).
        K (int): An integer representing the number of top clusters to return.
        C (int, optional): An integer representing the number of top vectors to return within each of the top K clusters. Default is 1.

    Returns:
        numpy.ndarray: A numpy array of dimensions (K, C, embedding_size) containing the top K clusters, each with the top C vectors.
    """
    num_clusters, batch_size, embedding_size = np.shape(clusters_of_images)
    # Convert numpy arrays to PyTorch tensors
    clusters_of_images = torch.from_numpy(clusters_of_images)
    query_vector = torch.from_numpy(query_vector)

    # Calculate cosine similarity between each cluster mean and the query vector
    cluster_means = clusters_of_images.mean(
        dim=1
    )  # Calculate the mean for each cluster
    query_vector_flat = query_vector.view(1, -1)  # Flatten the query vector
    cos_sim = torch.nn.functional.cosine_similarity(
        cluster_means.view(-1, embedding_size), query_vector_flat, dim=1
    )

    # Get the indices of the top K clusters based on cosine similarity
    top_k_indices = cos_sim.topk(K).indices

    top_K_C_vectors = []
    for i in range(K):
        cluster_index = top_k_indices[i]

        # Calculate cosine similarity between the cluster vectors and the query vector
        cos_sim_cluster = torch.nn.functional.cosine_similarity(
            clusters_of_images[cluster_index].view(-1, embedding_size),
            query_vector_flat,
            dim=1,
        )

        # Get the indices of the top C vectors within the cluster based on cosine similarity
        top_C_indices = torch.topk(cos_sim_cluster, C).indices

        # Extract and append the top C vectors from the cluster
        top_C_vectors = (
            clusters_of_images[cluster_index][top_C_indices]
            .numpy()
            .reshape(C, embedding_size)
        )
        top_K_C_vectors.append(top_C_vectors)

    # Reshape the top K C vectors to their original shape
    top_K_C_vectors = np.array(top_K_C_vectors).reshape(K * C, embedding_size)

    return top_K_C_vectors



## SAMPLE CODE TO TEST
# helper function for testing
def calculate_and_print_cosine_similarity(
    clusters_of_images, query_vector, K, top_k_vectors
):
    """
    Calculate and print cosine similarity between the top K vectors and the query vector, as well as all input vectors and the query vector.

    Args:
        clusters_of_images (numpy.ndarray): A numpy array of dimensions (number_of_clusters, batch, embedding_size).
        query_vector (numpy.ndarray): A numpy array of size embedding_size.
        K (int): An integer representing the number of top clusters to return.
        top_k_vectors (numpy.ndarray): A numpy array of dimensions (K, embedding_size) containing the top K vectors.

    Returns:
        None
    """
    # Calculate cosine similarity between the top K vectors and the query vector
    similarities_top_K = []
    for i in range(K):
        top_K_vector = top_k_vectors[i]
        top_K_vector_flat = top_K_vector.flatten()
        similarity = np.dot(query_vector.flatten(), top_K_vector_flat) / (
            np.linalg.norm(query_vector.flatten()) * np.linalg.norm(top_K_vector_flat)
        )
        similarities_top_K.append(similarity)

    # Calculate cosine similarity between all input vectors and the query vector
    similarities_all = []
    for cluster in clusters_of_images:
        for image in cluster:
            image_flat = image.flatten()
            similarity = np.dot(query_vector.flatten(), image_flat) / (
                np.linalg.norm(query_vector.flatten()) * np.linalg.norm(image_flat)
            )
            similarities_all.append(similarity)

    # Sort similarities in descending order
    similarities_top_K_sorted = sorted(similarities_top_K, reverse=True)
    similarities_all_sorted = sorted(similarities_all, reverse=True)

    print(
        "Similarities between the top K vectors and the query vector (in descending order):"
    )
    print(similarities_top_K_sorted)

    print(
        "\nSimilarities between all input vectors and the query vector (in descending order):"
    )
    print(similarities_all_sorted)


def testing():
    # Generate random data
    number_of_clusters = 10
    batch = 10
    # channels = 5
    # height = 48
    # width = 64
    embedding_size = 512
    clusters_of_images = np.random.rand(
        number_of_clusters, batch, embedding_size
    )
    query_vector = np.random.rand(embedding_size)
    K = 5

    print("TOP K RETRIEVAL")
    top_k_vectors = find_top_K_vectors(clusters_of_images, query_vector, K)
    calculate_and_print_cosine_similarity(
        clusters_of_images, query_vector, K, top_k_vectors
    )

    print("\nTOP K * C RETRIEVAL")
    top_K_C_vectors = find_top_K_C_vectors(clusters_of_images, query_vector, K, 2)
    calculate_and_print_cosine_similarity(
        clusters_of_images, query_vector, K * 2, top_K_C_vectors
    )
    # note here for calculate_and_print_cosine_similarity we pass in K*C. If we don't specify C for top_K_C_vectors, we should just pass in K for calculate_and_print_cosine_similarity


def main():
    testing()


if __name__ == "__main__":
    main()
