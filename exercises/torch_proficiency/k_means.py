import torch
import pdb


n_points = 2
n_centroids = 5
dim = 10
coord_points = torch.randn([n_points,dim]).unsqueeze(0)
coord_centroids = torch.randn([n_centroids,dim]).unsqueeze(0)

def closest_centroid(coord_points: torch.Tensor, coord_centroids):
    #[batch_size, n_points, n_centroids]
    distances_points_centroids = torch.cdist(coord_points, coord_centroids)

    closest_centroids_idx = torch.argmin(distances_points_centroids,dim = -1)

    #closest_centroids_coords = coord_centroids[closest_centroids_idx]
    closest_centroids_idx_expanded = closest_centroids_idx.unsqueeze(-1).expand(-1,-1,coord_centroids.size(-1)) #[1,n_points,1]
    closest_centroids_coords = torch.gather(coord_centroids,1, closest_centroids_idx_expanded)
    #under the hood:
    #coord_centroids[i, closest_centroids_idx_expanded[i,j,k], k]

    return closest_centroids_idx, closest_centroids_coords

idxs, coords = closest_centroid(coord_points,coord_centroids)
pdb.set_trace()





# add k as parameter 
# different ways to do it, explore broadcasting and torch.argmin