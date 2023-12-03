import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_query():
    save_dir = "../data/"
    query_tensors = np.load(save_dir+"/queries_test.npy")
    query_np = np.asarray(query_tensors)
    query_tensors_load = torch.from_numpy(query_np)
    query_tensors_load = torch.squeeze(query_tensors_load)
    # print("query tensor",query_tensors_load.shape)
    # print(query_tensors_load[0].size())

    return query_tensors_load

def load_test_embedding():
    save_dir = "../data/"
    embed_tensors = np.load(save_dir+"/test_embedding.npy")
    embed_np = np.asarray(embed_tensors)
    embed_tensors_load = torch.from_numpy(embed_np)
    embed_tensors_load = torch.squeeze(embed_tensors_load)
    # print("embeddings tensor",embed_tensors_load.shape)
    # print(embed_tensors_load[0].size())

    return embed_tensors_load

def load_clusters(embedding_size):
    save_dir = "../data/clusters/"

    directory = os.fsencode(save_dir)

    num_files = len(os.listdir(directory))
    clusters = [None] * num_files

    for i in range(num_files):
        filename = save_dir+os.fsdecode("clusters_"+str(i)+".npy")
        clusters_tensors = np.load(filename)
        clusters_np = np.asarray(clusters_tensors)
        clusters_np_reshaped = clusters_np.reshape(-1, embedding_size)
        clusters_tensors_load = torch.from_numpy(clusters_np_reshaped)
        clusters_tensors_load = torch.squeeze(clusters_tensors_load)
        clusters[i] = clusters_tensors_load
        # print(filename)
        # print("clusters tensor",clusters_tensors_load)
        
    return np.asarray(clusters)

def load_cluster_indexes():
    save_dir = "../data/cluster_indexes/"

    directory = os.fsencode(save_dir)

    num_files = len(os.listdir(directory))
    clusters_index = [None] * num_files

    for i in range(num_files):
        filename = save_dir+os.fsdecode("clusters_indexes_"+str(i)+".npy")
        clusters_index_load = np.load(filename)
        clusters_index_np = np.asarray(clusters_index_load)
        clusters_index[i] = clusters_index_np
        # print(filename)
        # print("clusters index tensor",clusters_index_np)
        
    return clusters_index

def find_top_K_vectors(clusters_of_images, index_assignments, query_vector, K, embedding_size):
    # Convert query_vector to a PyTorch tensor
    query_vector = torch.from_numpy(query_vector)

    # Calculate cosine similarity between each cluster mean and the query vector
    cluster_means = torch.cat([cluster.mean(dim=0, keepdim=True) if len(cluster.shape) > 1 else cluster.unsqueeze(0) for cluster in clusters_of_images])
    query_vector_flat = query_vector.view(1, -1)  # Flatten the query vector

    # Ensure both tensors have the same size at non-singleton dimension 1
    if cluster_means.dim() == 1:
        cluster_means = cluster_means.unsqueeze(0)
    if query_vector_flat.dim() == 1:
        query_vector_flat = query_vector_flat.unsqueeze(0)

    cos_sim = torch.nn.functional.cosine_similarity(
        cluster_means, query_vector_flat, dim=1
    )

    # Get the indices of the top K clusters based on cosine similarity
    top_k_indices = cos_sim.topk(K).indices

    # Find the vector in each of the top K clusters that is closest to the query vector
    top_k_vectors = []
    top_k_vector_indices = []
    for i in range(K):
        cluster_index = top_k_indices[i].item()  # Extract the cluster index from the tensor

        # Find the closest vector within the cluster
        cluster_vectors = clusters_of_images[cluster_index]
        cluster_indices = index_assignments[cluster_index]
        if len(cluster_vectors.shape) > 1:
            cos_sim_cluster = torch.nn.functional.cosine_similarity(
                cluster_vectors, query_vector_flat, dim=1
            )
            closest_vector_index = torch.argmax(cos_sim_cluster).item()
            top_k_vectors.append(cluster_vectors[closest_vector_index].numpy())
            top_k_vector_indices.append(cluster_indices[closest_vector_index])
        else:
            # For one-dimensional tensors, use the tensor itself
            top_k_vectors.append(cluster_vectors.numpy())
            top_k_vector_indices.append(index_assignments[cluster_index])

    # Reshape the top K vectors to their original shape
    top_k_vectors = np.array(top_k_vectors).reshape(K, -1)

    return top_k_vectors, top_k_vector_indices

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



# SAMPLE CODE TO TEST
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

    embeddings = load_test_embedding()
    embedding_size = 2
    clusters_of_images = load_clusters(embedding_size) #SPECIFY EMBEDDING SIZE
    cluster_index_assignments = load_cluster_indexes()
    query_vector = np.random.rand(embedding_size)
    K = len(cluster_index_assignments)

    print("TOP K RETRIEVAL")
    print("CLUSTERS")
    print(clusters_of_images)
    print("\n")

    print("CLUSTER INDEX ASSIGNMENTS")
    print(cluster_index_assignments)
    print("\n")

    print("QUERY")
    print(query_vector)
    print("\n")
    top_k_vectors, top_k_indices = find_top_K_vectors(clusters_of_images, cluster_index_assignments, query_vector, K, embedding_size)
    calculate_and_print_cosine_similarity(
        clusters_of_images, query_vector, K, top_k_vectors
    )

    print("Top K Vectors")
    print(top_k_vectors)
    print("\n")

    print("Top K Indices")
    print(top_k_indices)
    print("\n")

    # print("\nTOP K * C RETRIEVAL")
    # top_K_C_vectors = find_top_K_C_vectors(clusters_of_images, query_vector, K, 2)
    # calculate_and_print_cosine_similarity(
    #     clusters_of_images, query_vector, K * 2, top_K_C_vectors
    # )
    # note here for calculate_and_print_cosine_similarity we pass in K*C. If we don't specify C for top_K_C_vectors, we should just pass in K for calculate_and_print_cosine_similarity


def main():
    testing()


if __name__ == "__main__":
    main()
