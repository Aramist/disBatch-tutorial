import time
from itertools import product
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
from joblib import Parallel, delayed
from tqdm import tqdm


def pad_rir(rir):
    # assuming only one sound source
    flat_rirs = [r[0] for r in rir]
    max_length = max([len(r) for r in flat_rirs])
    padded_rirs = [np.pad(r, (0, max_length - len(r))) for r in flat_rirs]
    return np.stack(padded_rirs, axis=0)


def arena_random_point(arena_dims, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    x = rng.uniform(1e-2, arena_dims[0] - 1e-2)
    y = rng.uniform(1e-2, arena_dims[1] - 1e-2)
    # z = rng.uniform(1e-2, 5e-2)
    z = 5e-2
    return np.array([x, y, z])


sample_rate = 125000

# room dimensions
r_height = 14.5 * 2.54 / 100  # z dim
r_width = 22.5 * 2.54 / 100  # x dim
r_length = 14 * 2.54 / 100  # y dim
r_offset = 1 * 2.54 / 100  # The ceiling is 1 inch wider on each side than the floor

# Corner points of the room
floor_corners = np.array(
    [
        [0, 0, 0],
        [r_width, 0, 0],
        [r_width, r_length, 0],
        [0, r_length, 0],
    ]
)

ceiling_corners = np.array(
    [
        [0 - r_offset, 0 - r_offset, r_height],
        [r_width + r_offset, 0 - r_offset, r_height],
        [r_width + r_offset, r_length + r_offset, r_height],
        [0 - r_offset, r_length + r_offset, r_height],
    ]
)

all_vertices = np.concatenate((floor_corners, ceiling_corners), axis=0)

# Construct walls
# left, bottom, right, top, floor, ceiling
wall_vertex_indices = np.array(
    [
        [0, 4, 7, 3],  # left wall
        [1, 2, 6, 5],  # right wall
        [0, 1, 5, 4],  # bottom wall
        [3, 7, 6, 2],  # top wall
        [0, 1, 2, 3],  # floor
        [4, 5, 6, 7],  # ceiling
    ]
)


def get_rir(
    absorption: float, scattering: float, max_order: int, sound_source_pos: np.ndarray
):
    absorption_arr = [absorption] * 4 + [0.95] * 2
    materials = [pra.Material(a, scattering) for a in absorption_arr]

    walls = [
        pra.room.wall_factory(
            corners=all_vertices[v_idx, :].T,
            absorption=m.absorption_coeffs,
            scattering=m.scattering_coeffs,
            name=f"wall_{n}",
        )
        for n, (v_idx, m) in enumerate(zip(wall_vertex_indices, materials))
    ]

    # Construct room
    room = pra.room.Room(
        walls=walls,
        fs=sample_rate,
        max_order=max_order,
        use_rand_ism=False,
        ray_tracing=False,
    )
    room.add_source(sound_source_pos.tolist())

    # Microphone positions
    # Microphones are located on the ceiling, above the corner points of the floor
    microphone_pos = np.array(
        [
            [r_width, 0, r_height - 0.01],
            [0, 0, r_height - 0.01],
            [0, r_length, r_height - 0.01],
            [r_width, r_length, r_height - 0.01],
        ]
    )

    room_center_3d = np.array(
        [
            r_width / 2,
            r_length / 2,
            0,
        ]
    )
    # Point microphones at room_center_3d

    mic_direction_vectors = room_center_3d - microphone_pos
    mic_direction_vectors /= np.linalg.norm(
        mic_direction_vectors, axis=1, keepdims=True
    )
    all_directivities = []
    for i in range(mic_direction_vectors.shape[0]):
        orientation = pra.directivities.DirectionVector(
            azimuth=np.arctan2(
                mic_direction_vectors[i, 1], mic_direction_vectors[i, 0]
            ),
            colatitude=np.arccos(mic_direction_vectors[i, 2]),
            degrees=False,
        )
        pattern = pra.directivities.DirectivityPattern.SUBCARDIOID
        directivity = pra.directivities.CardioidFamily(orientation, pattern)
        all_directivities.append(directivity)

    mic_array = pra.MicrophoneArray(
        microphone_pos.T, fs=sample_rate, directivity=all_directivities
    )
    room.add_microphone_array(mic_array)
    room.compute_rir()

    return pad_rir(room.rir)


# Make a huge dataset of rirs


def make_dataset(save_path: Path, absorption: float, scattering: float, max_order: int):
    def job():
        # want to make it unlikely two workers have the same seed
        seed = int(time.time() * 1e6) % 2**16
        rng = np.random.default_rng(seed)
        sound_source_pos = arena_random_point([r_width, r_length, r_height], rng=rng)
        rir = get_rir(absorption, scattering, max_order, sound_source_pos)
        scaled_sound_source_pos = (
            sound_source_pos - np.array([r_width / 2, r_length / 2, 0])
        ) * 1000
        return rir, scaled_sound_source_pos

    NUM_INSTANCES = 10000
    NUM_WORKERS = -2
    print(f"Using {NUM_WORKERS} workers to generate {NUM_INSTANCES} RIRs")

    rir_database = Parallel(n_jobs=NUM_WORKERS)(
        delayed(job)() for _ in range(NUM_INSTANCES)
    )

    # Write data to disk
    num_mics = rir_database[0][0].shape[0]
    full_rirs = np.ascontiguousarray(
        np.concatenate([r[0] for r in rir_database], axis=1).T
    )
    rir_lengths = np.array([r[0].shape[1] for r in rir_database])
    rir_full_length = rir_lengths.sum()
    with h5py.File(save_path, "w") as f:
        f.create_dataset("rir", shape=(rir_full_length, num_mics), data=full_rirs)
        f.create_dataset(
            "locations",
            shape=(NUM_INSTANCES, 3),
            data=np.stack([r[1] for r in rir_database], axis=0),
        )
        f.create_dataset("rir_length_idx", data=np.cumsum(np.insert(rir_lengths, 0, 0)))
        # for n, (start, end) in enumerate(
        #     zip(f["rir_length_idx"][:-1], f["rir_length_idx"][1:])
        # ):
        #     f["rir"][start:end, :] = rir_database[n][0].T


if __name__ == "__main__":
    candidate_absorption = np.arange(0.05, 0.301, 0.025)
    candidate_scattering = (1.00,)
    candidate_max_order = (9,)

    sweep_size = (
        len(candidate_absorption) * len(candidate_scattering) * len(candidate_max_order)
    )

    save_dir = Path("/mnt/home/atanelus/ceph/february/pretraining_datasets/")
    save_dir.mkdir(exist_ok=True)
    for a, s, m_o in tqdm(
        product(candidate_absorption, candidate_scattering, candidate_max_order),
        total=sweep_size,
    ):
        save_path = save_dir / f"rir_dataset_{a:0.2f}_{s:0.2f}_{m_o:0>2d}.h5"
        if not save_path.exists():
            make_dataset(save_path, a, s, m_o)
