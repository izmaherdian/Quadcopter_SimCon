import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from obstacles import make_obstacles
from waypoints import makeWaypoints

class APF:
    """
    Algoritma Artificial Potential Field (APF) untuk perencanaan jalur UAV 3D.
    """

    def __init__(self, obstacles, obstacle_radii):
        # Koordinat pusat hambatan (obstacles)
        self.obstacles = obstacles
        # Radius masing-masing hambatan
        self.obstacle_radii = obstacle_radii

        t_wps, wps, y_wps, v_wp, wp_ini = makeWaypoints()
        self.t_wps = t_wps
        self.wps  = wps
        self.y_wps = y_wps
        self.v_wp  = v_wp
        self.start = wp_ini
        self.goal = np.array([9, 10, 6])


        # Parameter APF
        self.step_size = 0.1
        self.max_iterations = 1000
        self.epsilon = 0.8  # faktor tarik
        self.eta = 0.2      # faktor tolak
        self.d_goal = 5     # jarak tarik maksimal
        self.r0 = 0.3       # jarak pengaruh tolak dari permukaan obstacle
        self.threshold = 0.3
        # Parameter momentum smoothing
        self.alpha = 0.7    # koefisien momentum
        self.velocity = np.zeros(3)
        # Simpan jalur
        self.path = np.zeros((0, 3))

    def distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Hitung jarak Euclidean antara dua titik 3D.
        """
        return np.linalg.norm(p1 - p2)

    def attraction(self, q: np.ndarray) -> np.ndarray:
        """
        Hitung gaya tarik menuju goal.
        """
        r = self.distance(q, self.goal)
        dir_vec = self.goal - q
        if r <= self.d_goal:
            return self.epsilon * dir_vec
        return self.d_goal * self.epsilon * dir_vec / (r + 1e-6)

    def repulsion(self, q: np.ndarray) -> np.ndarray:
        """
        Hitung gaya tolak berdasarkan jarak ke permukaan obstacle.
        Gaya hanya bekerja jika 0 < d_surface <= r0.
        """
        total = np.zeros(3)
        for center, radius in zip(self.obstacles, self.obstacle_radii):
            # Jarak pusat UAV ke pusat obstacle
            d_center = self.distance(q, center)
            # Jarak dari UAV ke permukaan obstacle
            d_surface = d_center - radius
            # Terapkan repulsi hanya saat dalam jangkauan
            if 0 < d_surface <= self.r0:
                # Arah unit dari obstacle ke UAV
                dir_obs = (q - center) / (d_center + 1e-6)
                # Magnitudo gaya repulsi (hukum kuadrat terbalik)
                mag = self.eta * (1.0 / d_surface - 1.0 / self.r0) / (d_surface**2)
                total += mag * dir_obs
        return total

    def plan_path(self):
        """
        Simulasi perencanaan jalur menggunakan APF dengan momentum smoothing.
        """
        q = self.start.copy()
        self.path = [q.copy()]
        print("Start point:", q)  # cetak titik awal

        for _ in range(self.max_iterations):
            f_attr = self.attraction(q)
            f_rep = self.repulsion(q)
            f_total = f_attr + f_rep
            # Jika tidak ada gaya, berhenti
            if np.allclose(f_total, 0):
                break

            # Normalisasi arah gaya total
            dir_unit = f_total / (np.linalg.norm(f_total) + 1e-6)
            # Momentum smoothing
            self.velocity = self.alpha * self.velocity + (1 - self.alpha) * dir_unit
            vel_unit = self.velocity / (np.linalg.norm(self.velocity) + 1e-6)

            # Update posisi
            q = q + self.step_size * vel_unit
            self.path.append(q.copy())
            # CETAK TITIK TERBARU SAJA
            self.desPos = q
            print("Desired position:", self.desPos)
            # Ubah tipe data desPos
            self.desPos = np.array(self.desPos, dtype=float)
            # Tampilkan Tipe data despos
            print("Tipe data desPos:", type(self.desPos))
            # Tampilkan dimensi desPos
            

            # Cek pencapaian goal
            if self.distance(self.desPos, self.goal) < self.threshold:
                self.path.append(self.goal.copy())
                print("Goal reached at:", self.goal)
                break

        # jika masih butuh, bisa cetak panjang total path:
        # print(f"Total points: {len(self.path)}")

    def set_axes_equal(self, ax):
        """
        Seting sumbu X, Y, Z agar skala sama.
        """
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max(x_range, y_range, z_range) / 2
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        ax.set_xlim3d([x_middle - max_range, x_middle + max_range])
        ax.set_ylim3d([y_middle - max_range, y_middle + max_range])
        ax.set_zlim3d([z_middle - max_range, z_middle + max_range])

    def animate(self):
        """
        Animasi pergerakan UAV dan hambatan sederhana untuk kinerja optimal.
        Obstacles dan UAV digambar sebagai titik scatter, bukan sphere yang berat.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot obstacles sebagai scatter point dengan ukuran proporsional radius
        sizes = (self.obstacle_radii / np.max(self.obstacle_radii)) * 300
        ax.scatter(self.obstacles[:,0], self.obstacles[:,1], self.obstacles[:,2],
                   c='green', s=sizes, alpha=0.5, label='Obstacle')

        # Plot start dan goal
        ax.scatter(*self.start, c='blue', s=100, label='Start')
        ax.scatter(*self.goal, c='red', s=100, label='Goal')

        # Inisialisasi UAV sebagai scatter point
        uav_size = 100
        uav_scatter = ax.scatter([self.start[0]], [self.start[1]], [self.start[2]],
                                 c='blue', s=uav_size, label='UAV')

        # Plot jalur kosong (line)
        path_line, = ax.plot([], [], [], c='magenta', linewidth=2, label='Jalur UAV')

        # Atur batas sumbu dan aspek sama
        all_pts = np.vstack((self.path, self.obstacles, self.goal[np.newaxis, :]))
        ax.set_xlim(np.min(all_pts[:,0]) - 1, np.max(all_pts[:,0]) + 1)
        ax.set_ylim(np.min(all_pts[:,1]) - 1, np.max(all_pts[:,1]) + 1)
        ax.set_zlim(np.min(all_pts[:,2]) - 1, np.max(all_pts[:,2]) + 1)
        self.set_axes_equal(ax)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True)
        ax.legend(loc='best')

        def update(frame):
            # Update UAV posisi
            x, y, z = self.path[frame]
            uav_scatter._offsets3d = ([x], [y], [z])
            # Update jalur
            coords = np.array(self.path[:frame + 1])
            path_line.set_data(coords[:, 0], coords[:, 1])
            path_line.set_3d_properties(coords[:, 2])
            return path_line, uav_scatter

        anim = animation.FuncAnimation(fig, update, frames=len(self.path),
                                       interval=100, blit=False)
        plt.show()

def check_collision(apf: APF) -> np.ndarray:
    """
    Periksa indeks hambatan jika jalur menabrak.
    """
    collisions = []
    for point in apf.path:
        for idx, center in enumerate(apf.obstacles):
            if apf.distance(point, center) <= apf.obstacle_radii[idx]:
                collisions.append(idx)
    return np.array(sorted(set(collisions)))

if __name__ == "__main__":
    obstacles, obstacle_radii = make_obstacles()
    apf = APF(obstacles, obstacle_radii)
    apf.plan_path()
    # Deteksi tabrakan dan cetak keterangan
    collisions = check_collision(apf)
    if collisions.size > 0:
        print("Jalur menabrak hambatan pada indeks:", collisions)
        for idx in collisions:
            print(f" - Hambatan di koordinat: {apf.obstacles[idx]}")
    else:
        print(f"Jalur aman. Total jarak: {sum(np.linalg.norm(apf.path[i+1]-apf.path[i]) for i in range(len(apf.path)-1)):.2f}")
    apf.animate()
