import os
from collections import defaultdict
from pathlib import Path
import numpy as np

def cosine_similarity(vector1, vector2):
    # Ensure the vectors are numpy arrays
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    
    # Compute the dot product and magnitudes
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # Handle the case where the magnitude is zero to avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # Compute cosine similarity
    cosine_sim = dot_product / (magnitude1 * magnitude2)
    
    return cosine_sim

class FaceDatabase:
    cosine_threshold = 0.48

    def __init__(self):
        self.database = defaultdict(dict)

    def load_database_embeddings(self, database_path):
        print(f'loading database "{database_path}"...', end='', flush=True)
        self.database = defaultdict(dict)

        embedding_paths = []
        # Walk through directory recursively
        for root, dirs, files in os.walk(database_path):
            for file in files:
                if file.lower().endswith('embed'):
                    # Full path to the image
                    embed = np.loadtxt(os.path.join(root, file))
                    name = Path(root).name
                    self.database[name][file] = embed
        print(f'Done.')

    def delete_profile(self, profile_name):
        if profile_name in self.database:
            self.database.pop(profile_name)

    def delete_embedding(self, profile_name, embedding_file_name):
        if embedding_file_name in self.database[profile_name]:
            self.database[profile_name].pop(embedding_file_name)

    def add_to_database(self, embedding, profile_image_path):
        # Save embedding 
        embed_path = profile_image_path.replace('.jpg', '.embed')
        np.savetxt(f'{embed_path}', embedding)

        # Update database
        file_name = Path(embed_path).name
        profile = embed_path.split('/')[-2]
        self.database[profile][file_name] = embedding

    def find(self, target_embedding):
        profile_name, max_distance = 'Unknown', float('-inf')

        all_distances = []
        all_hits = []
        distance_dict = defaultdict(list)
        for name, db_embeddings in self.database.items():
            distances = []
            if not db_embeddings:
                continue

            for (file_name, db_embedding) in db_embeddings.items():
                distance_dict[name].append(cosine_similarity(db_embedding, target_embedding))

        all_distances = [(name, np.max(dist)) for name, dist in distance_dict.items()]
        all_distances = sorted(all_distances, key=lambda x: x[1], reverse=True)

        if not all_distances:
            return 'Unknown', all_distances

        if all_distances[0][1] > self.cosine_threshold:
            profile_name = all_distances[0][0]

        return profile_name, all_distances
