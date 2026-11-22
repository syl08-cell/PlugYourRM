
# custom/schemes/your_scheme/your_tracker.py
"""你的跟踪器：一帧观测进，一帧预测出；火控在内部自己发 ShotSolution。

契约（core/interfaces/tracker.py）：
    输入     Observation + SelfState
    输出     List[Prediction]，第一条是"该打谁"
    没目标   返回空列表

云台只认 ShotSolution —— 要让云台动，必须在 step() 里
public_bus().publish(PublicMessages.CONTROL, solution)。


一、整体结构（对应 docs/writing_your_tracker.md 的升级路径）

    数据关联  每块板一条 track；最近邻 + 门限，门限小于相邻板间距
    状态估计  每条 track 一个匀速 KalmanFilter（位置 + 速度，6 维）
    旋转模型  在 KF 之上再估"绕车中心的转动"：圆心 + 半径 + 角速度
    选板      按"离当前瞄点角度 + 距离 + 目标类型"打分
    火控      弹道解 yaw / pitch / 飞行时间 -> 前推瞄点 -> 判该不该开火


二、两个必须解释的设计决定

**1）内部一律用世界系。**
cfg.sim.frame 默认是 G（相对云台中心）。自车一动，G 系里的"静止目标"
也在动，速度估计会被自车运动污染。所以观测进来先加 gimbal.pos 换成世界系，
出去时再减回去。跟踪器内部只认世界系。

**2）为什么要"圆心 + 角速度"模型，而不只用匀速模型。**
装甲板不是直线运动的。一辆小陀螺以 omega 旋转时，板的轨迹是圆，
而匀速模型（CV）只会沿切线直着推——角速度越大，前推 0.2 s 的偏差越大：

    omega = 2 rad/s -> 0.2 s 转 23 度，误差几厘米
    omega = 6 rad/s -> 0.2 s 转 69 度，误差半米以上（实测命中率掉到 0）

关键洞察：**板的法向永远背离车心**。所以对同一条 track 的历史样本
    pos_i = center + r * n_i
是关于 (center, r) 的**线性**方程（n_i 是板的世界法向）。四个以上样本
就能最小二乘解出圆心和半径，再从 psi 的历史差分得到 omega。前推就变成
"绕圆心转 omega*delay"，而不是"沿切线直推"。

这个模型对前哨站的旋转塔同样成立（它也是绕自身轴匀速转），
所以一套代码同时修好小陀螺和前哨。


三、同一个时间戳只更新一次

sensor_fps=50，而 keeper 按 300 Hz 叫 tick()，同一份 Observation 会被反复
送进来。时间戳没变就跳过量测更新，否则滤波器会被同一帧更新六次。
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from config import cfg
from core.algorithms.filters.kalman import KalmanFilter
from core.algorithms.math.angle import safe_angle_sub
from core.algorithms.math.transform import rpy_to_R
from core.bus import public_bus
from core.bus.dataflow import Prediction, ShotSolution
from core.bus.messages import PublicMessages
from core.bus.public_bus import debug_plot, debug_text
from core.entities.property.entity_types import EntityType, TargetType
from core.interfaces.tracker import Tracker, register_tracker
from custom.components.traj_solvers.tjurm_traj_solver import TjurmTrajSolver

# ------------------------------------------------------------------ 可调参数
_STATE_DIM = 6            # [x, y, z, vx, vy, vz]

_ACCEL_NOISE = 20.0       # 过程噪声谱密度 (m^2/s^3)
_MEAS_SIGMA = 0.06        # 单帧位置噪声 (m)。实测约束 PnP 后约 2~3 cm，留余量
_INIT_POS_VAR = 0.25
_INIT_VEL_VAR = 25.0

# 关联门限必须**小于相邻装甲板的间距**，否则观测会被配到隔壁那块板上，
# 两条 track 互相拽、位姿来回跳（现象是"观测准、跟踪飘"）。
# 参考间距：步兵板在 length/2 半径上，间距约 0.5 m；
# 前哨三块板在半径 0.275 m 的圆上，间距 0.275*sqrt(3) ≈ 0.476 m。
_GATE_BASE = 0.18
_GATE_GROWTH = 1.5

_COAST_LIMIT = 0.12       # 最后一次观测后还能"续瞄"多久 (s)
_TRACK_LIFE = 0.35        # 彻底丢失多久后删掉这条 track (s)
_MIN_HITS = 3             # 至少命中几帧才允许被选中

# 旋转模型：历史窗口与有效性判据
_RING_WINDOW = 1.0        # 保留多久的历史 (s)
_RING_MIN_SAMPLES = 4     # 少于这个数不拟合
_RING_MAX_RESIDUAL = 0.06 # 拟合残差 (m) 超过它认为"不是绕轴转"
_RING_RANGE = (0.10, 1.20)  # 半径的合理区间 (m)

_FIRE_YAW_TOL = math.radians(0.3)
_FIRE_PITCH_TOL = math.radians(0.3)
_FIRE_FACING_MIN = 0.20

_DEFAULT_FLY = 0.20
_FIRE_DELAY = 0.02

_TYPE_RANK = {TargetType.ROBOT: 0.0, TargetType.OUTPOST: 0.15, TargetType.RUNE: 0.5}


# ------------------------------------------------------------------ 旋转模型
@dataclass
class _Ring:
    """一条 track 的"绕圆心旋转"模型。

    静止时 center 无意义；只要半径和残差合理就当成有效模型。
    """

    center: np.ndarray
    radius: float
    omega: float
    residual: float
    omega_t: float          # omega 对应的时刻


def _fit_ring(samples) -> Optional[_Ring]:
    """从 (t, pos, normal, psi) 样本拟合 center / radius / omega。

    模型（**只在水平面内**）：pos_xy = center_xy + r * n_xy
    板绕 z 轴转，法向的水平分量永远从车心指向板心，所以对 (center_xy, r)
    是线性的，直接最小二乘。

    **为什么不用三维。** 装甲板的 z 高度由配置的 high/low_height 决定，
    和安装倾角 theta_0 无关（见 core/entities/rigids/chassis.py：
    rel_pos 的 z 来自 height 字段，而 mount_R 的 x 轴带 sin(theta_0) 的 z 分量）。
    两者不是同一个几何来源，硬套三维模型会在 z 上自相矛盾，
    最小二乘只好折中——实测会把圆心和半径带偏，进而把 omega 低估 15%。
    旋转本来就只发生在水平面内，降到二维既正确又更稳。
    """
    if len(samples) < _RING_MIN_SAMPLES:
        return None

    A = np.zeros((2 * len(samples), 3), float)
    b = np.zeros(2 * len(samples), float)
    for i, (_t, pos, normal, _psi) in enumerate(samples):
        n_xy = np.asarray(normal, float)[:2]
        norm_xy = float(np.linalg.norm(n_xy))
        if norm_xy < 1e-6:
            return None                    # 板面朝上下，没有水平法向
        n_xy = n_xy / norm_xy
        A[2 * i:2 * i + 2, 0:2] = np.eye(2)
        A[2 * i:2 * i + 2, 2] = n_xy
        b[2 * i:2 * i + 2] = np.asarray(pos, float)[:2]

    try:
        # Ridge（很小的正则）：小陀螺在一个窗口里只扫过一小段圆弧，
        # center 的切向分量几乎不可观；完全无正则时最小二乘会把圆心
        # 甩到很远、半径顶到上界。加 1e-3 的岭项把它钉在样本中心附近。
        ATA = A.T @ A
        ATb = A.T @ b
        lam = 1e-3 * float(np.trace(ATA)) / 3.0
        sol = np.linalg.solve(ATA + lam * np.eye(3), ATb)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(sol)):
        return None

    center_xy = sol[0:2]
    radius = float(sol[2])
    if not (_RING_RANGE[0] <= abs(radius) <= _RING_RANGE[1]):
        return None
    radius = abs(radius)

    residual = 0.0
    for _t, pos, normal, _psi in samples:
        n_xy = np.asarray(normal, float)[:2]
        n_xy = n_xy / max(float(np.linalg.norm(n_xy)), 1e-9)
        residual += float(np.linalg.norm(np.asarray(pos, float)[:2]
                                         - center_xy - radius * n_xy))
    residual /= len(samples)
    if residual > _RING_MAX_RESIDUAL:
        return None

    # center 保留样本的平均高度：旋转只动 xy，z 用不着，但保持可读
    center = np.array([center_xy[0], center_xy[1],
                       float(np.mean([np.asarray(s[1], float)[2] for s in samples]))], float)

    # omega：位置相对圆心的**相位**对时间的最小二乘斜率。
    #
    # 不用 psi 的差分：psi 是"板在绝对系下的偏航"，每转一圈回绕一次，
    # 板在背面时又没有观测（漏一段），跨缺口解包会累错。
    # 位置相位 θ = atan2(p_y - c_y, p_x - c_x) 正是预测时"绕圆心转"的角，
    # 用它估 omega 与预测模型天然一致。
    ts = np.array([s[0] for s in samples], float)
    thetas = np.array([
        math.atan2(float(pos[1] - center_xy[1]), float(pos[0] - center_xy[0]))
        for _t, pos, _normal, _psi in samples
    ], float)
    for i in range(1, len(thetas)):
        thetas[i] = thetas[i - 1] + float(safe_angle_sub(thetas[i], thetas[i - 1]))
    if float(ts[-1] - ts[0]) < 0.05:
        return None
    omega = float(np.polyfit(ts, thetas, 1)[0])

    return _Ring(center=center, radius=radius, omega=omega,
                 residual=residual, omega_t=float(ts[-1]))


def _rotate_z(vec, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    return np.array([c * x - s * y, s * x + c * y, z], float)


# ------------------------------------------------------------------ 单条轨迹
@dataclass
class _Track:
    track_id: int
    entity_type: EntityType
    armor_size: str
    kf: KalmanFilter
    last_t: float
    last_update: float
    hits: int
    normal_world: np.ndarray
    lit: bool
    psi: float = 0.0
    history: Deque = field(default_factory=lambda: deque(maxlen=64))
    ring: Optional[_Ring] = None

    @property
    def pos(self) -> np.ndarray:
        return self.kf.x[:3].copy()

    @property
    def vel(self) -> np.ndarray:
        return self.kf.x[3:].copy()

    def predict_at(self, t: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """前推到绝对时刻 t，返回 (位置, 板法向, 角速度)。

        有旋转模型就绕圆心转（这对关联同样重要：小陀螺转过背面时观测会断，
        匀速外推会把门限甩到圆的切线方向，从而丢掉这条 track）；
        否则退回匀速外推。
        """
        ring = self.ring
        if ring is not None:
            angle = ring.omega * (t - ring.omega_t)
            offset = self.pos - ring.center
            pos = ring.center + _rotate_z(offset, angle)
            return pos, _rotate_z(self.normal_world, angle), ring.omega
        dt = t - self.last_t
        return self.pos + self.vel * dt, self.normal_world.copy(), 0.0

    def predict_aim(self, delay: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """前推 delay 秒（相对当前时刻）。"""
        return self.predict_at(self.last_t + delay)


def _transition(dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """匀速模型的状态转移 F 与过程噪声 Q（连续白噪声加速度模型）。"""
    F = np.eye(_STATE_DIM)
    F[0, 3] = F[1, 4] = F[2, 5] = dt

    q = _ACCEL_NOISE
    dt2, dt3, dt4 = dt * dt, dt * dt * dt, dt * dt * dt * dt
    Q = np.zeros((_STATE_DIM, _STATE_DIM))
    for i in range(3):
        Q[i, i] = q * dt4 / 4.0
        Q[i, i + 3] = Q[i + 3, i] = q * dt3 / 2.0
        Q[i + 3, i + 3] = q * dt2
    return F, Q


def _target_type(entity_type: EntityType) -> TargetType:
    if entity_type is EntityType.Rune:
        return TargetType.RUNE
    if entity_type is EntityType.Outpost:
        return TargetType.OUTPOST
    return TargetType.ROBOT


# ------------------------------------------------------------------ 跟踪器
@register_tracker("yours")
class YourTracker(Tracker):
    description = "我的方案：KF 多目标 + 旋转模型前推 + 弹道火控"

    def __init__(self):
        self.frame = str(cfg.sim.frame).upper()
        if self.frame not in ("G", "W"):
            raise ValueError(f"sim.frame 必须是 'G' 或 'W'，实得 {self.frame!r}")

        self.traj = TjurmTrajSolver()

        a = cfg.sim.aim
        self.rune_cooldown = float(a.rune_cooldown)
        self.max_range = float(cfg.sim.camera.max_range)
        self.fire_delay = float(cfg.sim.bullet.fire_delay)

        self._tracks: Dict[int, _Track] = {}
        self._next_id = 0
        self._last_t: Optional[float] = None

        self._cache_t: Optional[float] = None
        self._cache_solution: Optional[ShotSolution] = None

        self._prev_cmd_t: Optional[float] = None
        self._prev_yaw = 0.0
        self._prev_pitch = 0.0

    # ------------------------------------------------------------ 契约入口
    def step(self, obs, self_state) -> List[Prediction]:
        if obs is None or self_state is None:
            return []

        t = float(obs.timestamp)

        # 同一帧被反复送进来：不重复更新滤波器。
        # 控制指令重发一次——电控每帧取最新一份，重发不改变结果。
        if self._cache_t is not None and t == self._cache_t:
            if self._cache_solution is not None:
                public_bus().publish(PublicMessages.CONTROL, self._cache_solution)
            return []

        gimbal_pos = np.asarray(self_state.gimbal.pos, float).reshape(3)

        self._predict_all(t)
        self._associate(obs, t, gimbal_pos)
        self._prune(t)
        self._last_t = t

        track = self._select(self_state, t)
        if track is None:
            self._cache_t = t
            self._cache_solution = None
            return []

        solution, aim_armor, fly_time = self._build_solution(track, t, self_state)
        pred = self._build_prediction(track, t, solution, aim_armor, gimbal_pos)

        self._cache_t = t
        self._cache_solution = solution

        public_bus().publish(PublicMessages.CONTROL, solution)
        self._debug(track, solution, self_state)

        return [pred]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0
        self._last_t = None
        self._cache_t = None
        self._cache_solution = None
        self._prev_cmd_t = None
        self.traj.reset()

    # ------------------------------------------------------------ 坐标系
    def _to_world(self, rel_pos, gimbal_pos) -> np.ndarray:
        pos = np.asarray(rel_pos, float).reshape(3)
        return pos if self.frame == "W" else pos + gimbal_pos

    def _from_world(self, pos_world, gimbal_pos) -> np.ndarray:
        pos = np.asarray(pos_world, float).reshape(3)
        return pos if self.frame == "W" else pos - gimbal_pos

    # ------------------------------------------------------------ 滤波
    def _predict_all(self, t: float) -> None:
        for track in self._tracks.values():
            dt = t - track.last_t
            if dt <= 0.0:
                continue
            F, Q = _transition(dt)
            track.kf.predict(F, Q)
            track.last_t = t

    def _associate(self, obs, t: float, gimbal_pos) -> None:
        measurements = []
        for armor in obs.obs_armors:
            pos_world = self._to_world(armor.rel_pos, gimbal_pos)
            if not np.all(np.isfinite(pos_world)):
                continue
            rpy = np.asarray(armor.rel_rpy, float).reshape(3)
            normal = rpy_to_R(rpy)[:, 0]
            measurements.append((armor, pos_world, normal, float(rpy[2])))

        if not measurements:
            return

        candidates = []
        for track in self._tracks.values():
            gate = _GATE_BASE + _GATE_GROWTH * max(t - track.last_update, 0.0)
            # 用旋转模型外推到本帧时刻再比距离：小陀螺转过背面时观测会断，
            # 匀速外推会把预测甩到切线方向，白丢一条 track。
            predicted, _n_pred, _w_pred = track.predict_at(t)
            for mi, (_armor, pos_world, _normal, _psi) in enumerate(measurements):
                dist = float(np.linalg.norm(pos_world - predicted))
                if dist <= gate:
                    candidates.append((dist, track.track_id, mi))
        candidates.sort()

        used_tracks, used_meas = set(), set()
        for _dist, track_id, mi in candidates:
            if track_id in used_tracks or mi in used_meas:
                continue
            used_tracks.add(track_id)
            used_meas.add(mi)
            self._update_track(self._tracks[track_id], measurements[mi], t)

        for mi, (armor, pos_world, normal, psi) in enumerate(measurements):
            if mi not in used_meas:
                self._spawn_track(armor, pos_world, normal, psi, t)

    def _spawn_track(self, armor, pos_world, normal, psi, t: float) -> None:
        P0 = np.diag([_INIT_POS_VAR] * 3 + [_INIT_VEL_VAR] * 3)
        x0 = np.concatenate([pos_world, np.zeros(3)])
        track = _Track(
            track_id=self._next_id,
            entity_type=armor.entity_type,
            armor_size=armor.armor_size,
            kf=KalmanFilter(x0=x0, P0=P0),
            last_t=t,
            last_update=t,
            hits=1,
            normal_world=normal.copy(),
            lit=bool(armor.lit),
            psi=float(psi),
        )
        track.history.append((t, pos_world.copy(), normal.copy(), float(psi)))
        self._tracks[self._next_id] = track
        self._next_id += 1

    def _update_track(self, track: _Track, measurement, t: float) -> None:
        armor, pos_world, normal, psi = measurement
        H = np.zeros((3, _STATE_DIM))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        R = np.eye(3) * (_MEAS_SIGMA ** 2)
        track.kf.update(pos_world, H, R)
        track.last_update = t
        track.hits += 1
        track.normal_world = normal.copy()
        track.lit = bool(armor.lit)
        track.psi = float(psi)

        track.history.append((t, pos_world.copy(), normal.copy(), float(psi)))
        while track.history and t - track.history[0][0] > _RING_WINDOW:
            track.history.popleft()
        # 圆靶是圆盘，没有"绕轴转的板"语义，不做旋转模型
        if track.entity_type is not EntityType.Rune:
            track.ring = _fit_ring(list(track.history))

    def _prune(self, t: float) -> None:
        dead = [tid for tid, tr in self._tracks.items()
                if t - tr.last_update > _TRACK_LIFE]
        for tid in dead:
            del self._tracks[tid]

    # ------------------------------------------------------------ 选板
    def _selection_score(self, track: _Track, self_state, t: float) -> Optional[float]:
        """一条 track 的分数（越小越该打）；不可用时返回 None。"""
        if track.hits < _MIN_HITS:
            return None
        if self._last_t - track.last_update > _COAST_LIMIT:
            return None
        if track.entity_type is EntityType.Rune and not track.lit:
            return None

        gimbal = self_state.gimbal
        origin = np.asarray(gimbal.pos, float).reshape(3)
        # 用"飞行后的位置"打分：转得快的目标，当前角度和该打的角度不是一回事
        aim, _normal, _omega = track.predict_aim(_DEFAULT_FLY)

        offset = aim - origin
        dist = float(np.linalg.norm(offset))
        if dist < 0.5 or dist > self.max_range:
            return None

        yaw_t = math.atan2(float(offset[1]), float(offset[0]))
        pitch_t = math.atan2(float(offset[2]),
                             math.hypot(float(offset[0]), float(offset[1])))
        angle_err = math.hypot(
            float(safe_angle_sub(yaw_t, float(gimbal.rpy[2]))),
            pitch_t - (-float(gimbal.rpy[1])))

        return angle_err + 0.02 * dist + _TYPE_RANK[_target_type(track.entity_type)]

    def _select(self, self_state, t: float) -> Optional[_Track]:
        best, best_score = None, float("inf")
        for track in self._tracks.values():
            score = self._selection_score(track, self_state, t)
            if score is not None and score < best_score:
                best, best_score = track, score
        return best

    # ------------------------------------------------------------ 弹道与火控
    def _build_solution(self, track: _Track, t: float, self_state):
        """解 (yaw, pitch, 飞行时间)；瞄点按延迟迭代前推。"""
        # 第一轮用固定延迟拿一个飞行时间，第二轮用真实延迟再解一次
        aim, _normal, _omega = track.predict_aim(_DEFAULT_FLY)
        result = self.traj.solve_aim(self._express_for_traj(aim, self_state), self_state)
        fly = _DEFAULT_FLY
        if result is not None:
            fly = result[2]
            aim, _normal, _omega = track.predict_aim(fly + self.fire_delay)
            result = self.traj.solve_aim(
                self._express_for_traj(aim, self_state), self_state)

        if result is None:
            yaw, pitch_up, fly = self._fallback_angles(aim, self_state)
        else:
            yaw, pitch_up, fly = result

        yaw_rate, pitch_rate = self._rates(t, yaw, pitch_up)
        fire, fire_mode, fire_key, cooldown = self._fire_decision(
            track, self_state, yaw, pitch_up, fly)

        solution = ShotSolution(
            yaw=float(yaw),
            pitch=float(pitch_up),
            yaw_rate=float(yaw_rate),
            pitch_rate=float(pitch_rate),
            fire=bool(fire),
            fire_mode=fire_mode,
            fire_key=fire_key,
            cooldown=cooldown,
            timestamp=float(t),
        )
        return solution, aim, float(fly)

    def _express_for_traj(self, aim_world, self_state) -> np.ndarray:
        gimbal_pos = np.asarray(self_state.gimbal.pos, float).reshape(3)
        return self._from_world(aim_world, gimbal_pos)

    def _fallback_angles(self, aim_world, self_state):
        """弹道无解时的直线指向。"""
        muzzle = np.asarray(self_state.muzzle.pos, float).reshape(3)
        d = np.asarray(aim_world, float) - muzzle
        yaw = math.atan2(float(d[1]), float(d[0]))
        pitch_up = math.atan2(float(d[2]),
                              max(math.hypot(float(d[0]), float(d[1])), 1e-6))
        return yaw, pitch_up, _DEFAULT_FLY

    def _rates(self, t: float, yaw: float, pitch_up: float):
        """瞄点视线的角速度（前馈）：相邻两帧角度差 / dt。"""
        if self._prev_cmd_t is None:
            self._prev_cmd_t = t
            self._prev_yaw = yaw
            self._prev_pitch = pitch_up
            return 0.0, 0.0

        dt = t - self._prev_cmd_t
        if dt <= 1e-6:
            return 0.0, 0.0

        yaw_rate = float(safe_angle_sub(yaw, self._prev_yaw)) / dt
        pitch_rate = (pitch_up - self._prev_pitch) / dt
        self._prev_cmd_t = t
        self._prev_yaw = yaw
        self._prev_pitch = pitch_up
        return yaw_rate, pitch_rate

    def _fire_decision(self, track: _Track, self_state, yaw: float,
                       pitch_up: float, fly: float):
        """该不该扣扳机。只判"自瞄视角"的合理性，物理许可归电控。"""
        gimbal = self_state.gimbal
        yaw_err = abs(float(safe_angle_sub(yaw, float(gimbal.rpy[2]))))
        pitch_err = abs(pitch_up - (-float(gimbal.rpy[1])))
        on_target = yaw_err < _FIRE_YAW_TOL and pitch_err < _FIRE_PITCH_TOL

        # 命中那一刻板法向是否还朝着自车（转了背面当然打不中）
        _aim, normal, _omega = track.predict_aim(fly + self.fire_delay)
        to_me = np.asarray(gimbal.pos, float).reshape(3) - track.pos
        norm = float(np.linalg.norm(to_me))
        facing = 0.0 if norm < 1e-6 else float(np.dot(normal, to_me / norm))

        if track.entity_type is EntityType.Rune:
            # 符用连发 + 0.8 s 建议冷却，不是"单发 + fire_key"。
            # single 模式下 fire_key 不变会被电控判 same_target 永久锁死，
            # 同一块灯臂漏了就补不了第二发。灯臂窗口 2.5 s，0.8 s 能打 3 发。
            fire = bool(track.lit and on_target)
            return fire, "auto", int(track.track_id), self.rune_cooldown

        fire = bool(on_target and facing > _FIRE_FACING_MIN)
        return fire, "auto", None, 0.0

    # ------------------------------------------------------------ 输出
    def _build_prediction(self, track: _Track, t: float, solution: ShotSolution,
                          aim_world: np.ndarray, gimbal_pos: np.ndarray) -> Prediction:
        pred_pos = self._from_world(aim_world, gimbal_pos)
        return Prediction(
            target_type=_target_type(track.entity_type),
            center=pred_pos.copy(),
            armors=[pred_pos.copy()],
            timestamp=float(t),
            is_tracking=True,
            aim_rate=np.array([solution.yaw_rate, solution.pitch_rate], float),
        )

    # ------------------------------------------------------------ 调试
    def _debug(self, track: _Track, solution: ShotSolution, self_state) -> None:
        """曲线面板 + 信息面板的数据源（不发就一直是 no data）。"""
        gimbal_pos = np.asarray(self_state.gimbal.pos, float).reshape(3)
        offset = track.pos - gimbal_pos
        obs_yaw = math.atan2(float(offset[1]), float(offset[0]))
        obs_pitch = math.atan2(float(offset[2]),
                               math.hypot(float(offset[0]), float(offset[1])))

        debug_plot(obs_yaw=obs_yaw, obs_pitch=obs_pitch,
                   pred_yaw=solution.yaw, pred_pitch=solution.pitch)

        dist = float(np.linalg.norm(offset))
        spin = "-" if track.ring is None else f"{track.ring.omega:+.1f}"
        status = "fire" if solution.fire else "hold"
        debug_text(
            f"id={track.track_id} {track.entity_type.name} dist={dist:.2f} "
            f"spin={spin} {status}",
            color=(255, 255, 0) if solution.fire else (200, 200, 200))
