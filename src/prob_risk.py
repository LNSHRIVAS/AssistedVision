import numpy as np
def predict_constant_velocity(state, dt, steps):
    cx, cy, vx, vy = state
    ts = np.arange(1, steps+1) * dt
    xs = cx + np.outer(ts, vx)
    ys = cy + np.outer(ts, vy)
    return np.stack([xs, ys], axis=-1)
def sample_trajectories(state, cov, dt, steps, n_samples=200):
    mean = np.array(state)
    samples = np.random.multivariate_normal(mean, cov, size=n_samples)
    trajs = []
    for s in samples:
        trajs.append(predict_constant_velocity(s, dt, steps))
    return np.stack(trajs, axis=0)
def collision_probability(samples, ego_point, radius=30.0):
    dists = np.linalg.norm(samples - np.array(ego_point)[None,None,:], axis=-1)
    collisions = (dists <= radius).any(axis=1)
    return float(collisions.mean())
def compute_risk_prob(ego_point, obj_state, obj_cov=None, dt=0.2, horizon=3.0, n_samples=300, radius=30.0):
    steps = max(1, int(horizon / dt))
    if obj_cov is None:
        px_var = 25.0
        speed = np.linalg.norm(obj_state[2:4])
        sp_var = max(1.0, (speed*2.0))
        obj_cov = np.diag([px_var, px_var, sp_var, sp_var])
    samples = sample_trajectories(obj_state, obj_cov, dt, steps, n_samples=n_samples)
    prob = collision_probability(samples, ego_point, radius=radius)
    hit_times = []
    for s in samples:
        dists = np.linalg.norm(s - np.array(ego_point)[None,:], axis=-1)
        hits = np.where(dists <= radius)[0]
        if hits.size>0:
            hit_times.append((hits[0]+1)*dt)
    ttc_est = float(np.median(hit_times)) if hit_times else None
    return prob, ttc_est
