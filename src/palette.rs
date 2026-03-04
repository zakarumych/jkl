use smallvec::SmallVec;

use crate::math::Vector;

struct Cluster<T> {
    centroid: T,
    samples: SmallVec<[T; 16]>,
}

impl<T> Cluster<T>
where
    T: Vector,
{
    fn total_squared_error(&self) -> f32 {
        self.centroid.total_squared_error(&self.samples)
    }

    fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Builds a palette from samples,
/// clustering values into a predefined number of voronoi cells.
/// Finds optimal palette to minimize total squared error across all samples.
pub fn build_palette<T>(samples: impl Iterator<Item = T>, len: usize) -> Vec<T>
where
    T: Vector,
{
    let all_samples = samples.collect::<SmallVec<[T; 16]>>();

    let centroid = T::centroid(&all_samples);

    let mut clusters = vec![Cluster {
        centroid,
        samples: all_samples.clone(),
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

        let mut last_error = rebuild_clusters(&all_samples, &mut clusters);

        for _ in 0..2 {
            let error = rebuild_clusters(&all_samples, &mut clusters);
            if error + 0.0001 >= last_error {
                break;
            }
            last_error = error;
        }
    }

    let mut last_error = rebuild_clusters(&all_samples, &mut clusters);

    for _ in 0..10 {
        let error = rebuild_clusters(&all_samples, &mut clusters);
        if error + 0.0001 >= last_error {
            break;
        }
        last_error = error;
    }

    clusters.into_iter().map(|c| c.centroid).collect()
}

fn rebuild_clusters<T>(samples: &[T], clusters: &mut Vec<Cluster<T>>) -> f32
where
    T: Vector,
{
    for c in &mut *clusters {
        c.samples.clear();
    }

    let mut total_error = 0.0f32;

    for &sample in samples {
        let mut min_error = f32::INFINITY;
        let mut best_cluster = 0;

        for (i, cluster) in clusters.iter().enumerate() {
            let error = cluster.centroid.distance_squared(sample);
            if error < min_error {
                min_error = error;
                best_cluster = i;
            }
        }

        total_error += min_error;

        clusters[best_cluster].samples.push(sample);
    }

    clusters.retain(|c| !c.is_empty());

    for c in &mut *clusters {
        c.centroid = T::centroid(&c.samples);
    }

    total_error
}

/// Calculate the gain from splitting a cluster of samples along the principal axis.
fn cluster_split<T>(cluster: &Cluster<T>) -> (Cluster<T>, Cluster<T>)
where
    T: Vector,
{
    let axis = T::principal_axis(&cluster.samples);
    let centroid_projection = cluster.centroid.project(axis);

    let (left, right) = cluster
        .samples
        .iter()
        .copied()
        .partition::<SmallVec<[T; 16]>, _>(|s| s.project(axis) < centroid_projection);

    let left_centroid = T::centroid(&left);
    let right_centroid = T::centroid(&right);

    let left = Cluster {
        centroid: left_centroid,
        samples: left,
    };
    let right = Cluster {
        centroid: right_centroid,
        samples: right,
    };

    (left, right)
}
