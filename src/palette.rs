use smallvec::SmallVec;

use crate::math::Vector;

struct Cluster<T> {
    centroid: T,
    vectors: SmallVec<[T; 16]>,
}

impl<T> Cluster<T>
where
    T: Vector,
{
    fn total_squared_error(&self) -> f32 {
        self.centroid.total_squared_error(&self.vectors)
    }

    fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

pub fn find_cluster<T, V>(sample: T, centroids: &[V], tv: impl Fn(T) -> V + Copy) -> usize
where
    V: Vector,
{
    let vector = tv(sample);

    let mut min_error = f32::INFINITY;
    let mut best_cluster = 0;

    for (i, &centroid) in centroids.iter().enumerate() {
        let error = centroid.distance_squared(vector);
        if error < min_error {
            min_error = error;
            best_cluster = i;
        }
    }

    best_cluster
}

/// Builds a palette from samples,
/// clustering values into a predefined number of voronoi cells.
/// Finds optimal palette to minimize total squared error across all samples.
pub fn build_palette<T, V>(
    samples: impl Iterator<Item = T>,
    len: usize,
    tv: impl Fn(T) -> V + Copy,
) -> Vec<V>
where
    T: Copy,
    V: Vector,
{
    let all_samples = samples.collect::<SmallVec<[T; 16]>>();
    let all_vectors = all_samples
        .iter()
        .copied()
        .map(tv)
        .collect::<SmallVec<[V; 16]>>();

    let centroid = V::centroid(&all_vectors);

    let mut clusters = vec![Cluster {
        centroid,
        vectors: all_vectors,
    }];

    while clusters.len() < len {
        let mut best_gain = 0.0f32;
        let mut best_split = None;

        for i in 0..clusters.len() {
            let tse = clusters[i].total_squared_error();
            let (left, right) = cluster_split(&clusters[i]);

            if left.is_empty() || right.is_empty() {
                continue;
            }

            let l_tse = left.total_squared_error();
            let r_tse = right.total_squared_error();

            let gain = tse - (l_tse + r_tse);
            if gain > best_gain {
                best_gain = gain;
                best_split = Some((i, left, right));
            }
        }

        match best_split {
            None => break,
            Some((i, left, right)) => {
                clusters[i] = left;
                clusters.push(right);
            }
        }

        let mut last_error = rebuild_clusters(&all_samples, &mut clusters, tv);

        for _ in 0..2 {
            let error = rebuild_clusters(&all_samples, &mut clusters, tv);
            if error + 0.0001 >= last_error {
                break;
            }
            last_error = error;
        }
    }

    let mut last_error = rebuild_clusters(&all_samples, &mut clusters, tv);

    for _ in 0..10 {
        let error = rebuild_clusters(&all_samples, &mut clusters, tv);
        if error + 0.0001 >= last_error {
            break;
        }
        last_error = error;
    }

    clusters.into_iter().map(|c| c.centroid).collect()
}

fn rebuild_clusters<T, V>(
    samples: &[T],
    clusters: &mut Vec<Cluster<V>>,
    tv: impl Fn(T) -> V + Copy,
) -> f32
where
    T: Copy,
    V: Vector,
{
    for c in &mut *clusters {
        c.vectors.clear();
    }

    let mut total_error = 0.0f32;

    for &sample in samples {
        let vector = tv(sample);

        let mut min_error = f32::INFINITY;
        let mut best_cluster = 0;

        for (i, cluster) in clusters.iter().enumerate() {
            let error = cluster.centroid.distance_squared(vector);
            if error < min_error {
                min_error = error;
                best_cluster = i;
            }
        }

        total_error += min_error;

        clusters[best_cluster].vectors.push(vector);
    }

    clusters.retain(|c| !c.is_empty());

    for c in &mut *clusters {
        c.centroid = V::centroid(&c.vectors);
    }

    total_error
}

/// Calculate the gain from splitting a cluster of samples along the principal axis.
fn cluster_split<V>(cluster: &Cluster<V>) -> (Cluster<V>, Cluster<V>)
where
    V: Vector,
{
    let axis = V::principal_axis(&cluster.vectors);
    let centroid_projection = cluster.centroid.project(axis);

    let (left, right) = cluster
        .vectors
        .iter()
        .copied()
        .partition::<SmallVec<[V; 16]>, _>(|s| s.project(axis) < centroid_projection);

    let left_centroid = V::centroid(&left);
    let right_centroid = V::centroid(&right);

    let left = Cluster {
        centroid: left_centroid,
        vectors: left,
    };
    let right = Cluster {
        centroid: right_centroid,
        vectors: right,
    };

    (left, right)
}
