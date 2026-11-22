# custom/schemes/your_scheme/your_pose_solver.py
"""你的位姿解算：一帧检测进，一帧观测出。

契约（core/interfaces/pose_solver.py）：
    输入     DetectionFrame + SelfState
    输出     Observation，每块看到的板一条 ArmorObservation

流程（对应 docs/coordinate_systems.md §五 的四步）：

    1) 遍历 frame.detections
    2) 每块板：先用 components 里的单板 PnP（ippe）拿初值，
       再用"板的安装角已知"这个先验做一次约束精化（见 _refine）
    3) 出口在 C'（绝对相机系），换到 G（frame="G"）或 W（frame="W"）
    4) 打包成 Observation

为什么要做第 2 步的精化：

    ippe 是全 6 自由度解，不用任何先验。一块 13.5 cm 宽的小板在 4 m 外，
    高度方向只占 10 个像素，深度方向的约束非常弱——实测单帧位置误差
    1 cm 量级是靠"正好无噪声"才拿到的，稍有噪声就掉到十几厘米。
    但装甲板在物理上**只绕 yaw 转**：滚转为 0、俯仰等于安装角
    （cfg.robot.armor_tilt / cfg.outpost.armor_tilt）。把这四个自由度锁死，
    未知量从 6 降到 4（位置 + psi），重投影才能把深度真正钉住。

    这就是 docs/coordinate_systems.md §5.4 说的"用关节角增强的 PnP"。

两处例外：
    - 圆靶（EntityType.Rune）没有板面方向可解，检测端已给出靶心 3D（世界系），
      这里只换表达系，不进 PnP。
    - 某块板解失败就地丢掉：宁可这一帧少一条观测，也不给跟踪器一个错位姿。
"""
from __future__ import annotations

import math

import numpy as np

from config import cfg
from core.bus.dataflow import ArmorObservation, Observation
from core.entities.property.entity_types import EntityType
from core.interfaces.pose_solver import PoseSolver, register_pose_solver
from custom.components.armor_solvers import get_solver
from custom.components.armor_solvers.ippe import object_points

_REFINE_ITERS = 10
_REFINE_TOL = 1e-10
#: 重投影误差上限 (px) 的基准。超过它认为这块板本帧解不可信，整条观测丢掉。
_MAX_REPROJ_PX = 3.0


@register_pose_solver("yours")
class YourPoseSolver(PoseSolver):
    description = "我的方案：单板 PnP + 安装角先验精化 + C' -> G/W 换系"

    def __init__(self):
        # 单板求解器由 cfg.sim.pnp_method 选（当前是 "ippe"），只用它拿初值
        self.solver = get_solver(cfg.sim.pnp_method)
        self.frame = str(cfg.sim.frame).upper()
        if self.frame not in ("G", "W"):
            raise ValueError(f"sim.frame 必须是 'G' 或 'W'，实得 {self.frame!r}")

        # 安装俯仰 theta_0：配置里是**度**，这里换成弧度。
        # 它是目标板的安装角，不是自车 pitch（见坐标系文档 §5.4）。
        self.robot_pitch = math.radians(float(cfg.robot.armor_tilt))
        self.outpost_pitch = math.radians(float(cfg.outpost.armor_tilt))

        # 像素噪声越大，允许的重投影残差越宽；否则正常帧会被无谓丢掉。
        noise = float(getattr(cfg.sim, "pixel_noise_sigma", 0.0) or 0.0)
        self.reproj_limit = _MAX_REPROJ_PX + 2.0 * noise

    # ---------------------------------------------------------------- 主入口
    def solve(self, frame, self_state) -> Observation:
        timestamp = float(frame.timestamp)

        K = np.asarray(self_state.camera.K, float)
        # R^{C'}_{C1}：相机系 -> 绝对方向系 = 相机在 world 下的姿态
        R_camera_to_abs = self._rpy_to_R(self_state.camera.rpy)
        camera_pos = np.asarray(self_state.camera.pos, float).reshape(3)
        gimbal_pos = np.asarray(self_state.gimbal.pos, float).reshape(3)

        armors = []
        for det in frame.detections:
            if det.entity_type is EntityType.Rune:
                item = self._rune(det, gimbal_pos)
            else:
                item = self._armor(det, K, R_camera_to_abs, camera_pos, gimbal_pos)
            if item is not None:
                armors.append(item)

        return Observation(armors, timestamp)

    # ---------------------------------------------------------------- 工具
    @staticmethod
    def _rpy_to_R(rpy):
        """欧拉角 [roll, pitch, yaw] -> R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。

        与 core.algorithms.math.transform.rpy_to_R 同一个公式；
        就地展开只是为了避免每次 solve() 都 import 一次热路径函数。
        """
        roll, pitch, yaw = (float(v) for v in np.asarray(rpy, float).reshape(3))
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], float)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], float)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], float)
        return Rz @ Ry @ Rx

    # ---------------------------------------------------------------- 装甲板
    def _armor(self, det, K, R_camera_to_abs, camera_pos, gimbal_pos):
        """一块装甲板：像素四角点 -> C' 位姿 -> 世界 -> 表达系。"""
        theta_0 = (self.outpost_pitch
                   if det.entity_type is EntityType.Outpost
                   else self.robot_pitch)
        img = np.asarray(det.image_points_2d, float).reshape(-1, 2)
        if img.shape[0] != 4:
            return None

        # --- 初值：ippe 的全 6 自由度解 ---
        pos_c, rpy_c = self.solver.estimate(
            camera_k=K,
            armor_size=det.armor_size,
            image_points_2d=img,
            rot_camera_to_abs_camera=R_camera_to_abs,
            mount_pitch=theta_0,
        )
        if pos_c is None or rpy_c is None:
            return None
        pos_c = np.asarray(pos_c, float).reshape(3)
        if not np.all(np.isfinite(pos_c)):
            return None

        # --- 精化：锁死滚转与俯仰，只解位置 + psi ---
        t_flu, psi = self._refine(img, K, R_camera_to_abs, det.armor_size,
                                  theta_0, pos_c, float(rpy_c[2]))
        if t_flu is None:
            return None

        # 相机系（FLU）-> 世界：M 把相机系方向转到绝对方向，平移是光心位置
        pos_world = R_camera_to_abs @ t_flu + camera_pos

        return ArmorObservation(
            entity_type=det.entity_type,
            rel_pos=self._express(pos_world, gimbal_pos),
            rel_rpy=np.array([0.0, float(theta_0), float(psi)], float),
            armor_size=det.armor_size,
            image_points_2d=img,
            lit=bool(det.lit),
        )

    # ---------------------------------------------------------------- 精化
    def _refine(self, img, K, R_camera_to_abs, armor_size, theta_0,
                pos_c_init, psi_init):
        """约束 PnP：未知量 (t_flu, psi)，板姿态 = Rz(psi) @ Ry(theta_0)。

        投影链就地展开成最简形式（与 camera.world_to_pixel 同一模型）：
            P_flu = R(psi) @ obj + t
            rdf   = FLU2RDF @ P_flu = (-P_y, -P_z, P_x)
            u     = f * rdf_x / rdf_z = -f * P_y / P_x
            v     = f * rdf_y / rdf_z = -f * P_z / P_x

        三角化需要 P_x > 0（板在相机前方）。
        返回 (t_flu, psi)；失败或解不可信返回 (None, None)。
        """
        obj = object_points(armor_size)          # (4,3)，板面在 x = 0
        f = float(K[0, 0])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        M = np.asarray(R_camera_to_abs, float).reshape(3, 3)

        # C' 位置 -> 相机 FLU 位置
        t = M.T @ np.asarray(pos_c_init, float).reshape(3)
        if t[0] <= 1e-3:
            return None, None        # ippe 给到了相机背后：初值不能用

        psi = float(psi_init)
        ct, st = math.cos(theta_0), math.sin(theta_0)
        Ry = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]], float)

        for _ in range(_REFINE_ITERS):
            cpsi, spsi = math.cos(psi), math.sin(psi)
            Rz = np.array([[cpsi, -spsi, 0.0],
                           [spsi, cpsi, 0.0],
                           [0.0, 0.0, 1.0]], float)
            dRz = np.array([[-spsi, -cpsi, 0.0],
                            [cpsi, -spsi, 0.0],
                            [0.0, 0.0, 0.0]], float)

            R = M.T @ Rz @ Ry                     # armor -> camera FLU
            dR = M.T @ dRz @ Ry                   # dR/dpsi
            P = (R @ obj.T).T + t                 # (4,3) camera FLU
            dP = (dR @ obj.T).T                   # (4,3) dP/dpsi

            px = P[:, 0]
            if np.any(px <= 1e-3):
                return None, None

            u = -f * P[:, 1] / px + cx
            v = -f * P[:, 2] / px + cy
            residual = np.empty(8, float)
            residual[0::2] = u - img[:, 0]
            residual[1::2] = v - img[:, 1]
            if not np.all(np.isfinite(residual)):
                return None, None

            # 投影对 P 的偏导
            du = np.empty((4, 3), float)
            dv = np.empty((4, 3), float)
            du[:, 0] = f * P[:, 1] / (px * px)
            du[:, 1] = -f / px
            du[:, 2] = 0.0
            dv[:, 0] = f * P[:, 2] / (px * px)
            dv[:, 1] = 0.0
            dv[:, 2] = -f / px

            J = np.empty((8, 4), float)
            # 对 t 的偏导就是 du/dP、dv/dP
            J[0::2, 0:3] = du
            J[1::2, 0:3] = dv
            # 对 psi 的偏导再乘 dP/dpsi
            J[0::2, 3] = np.einsum("ij,ij->i", du, dP)
            J[1::2, 3] = np.einsum("ij,ij->i", dv, dP)

            try:
                delta = np.linalg.solve(J.T @ J + 1e-9 * np.eye(4), -J.T @ residual)
            except np.linalg.LinAlgError:
                return None, None
            if not np.all(np.isfinite(delta)):
                return None, None

            # 阻尼：单步别超过 30 cm / 0.2 rad，避免从坏初值冲出去
            step = float(np.linalg.norm(delta[:3]))
            if step > 0.3:
                delta[:3] *= 0.3 / step
            if abs(delta[3]) > 0.2:
                delta[3] *= 0.2 / abs(delta[3])

            t = t + delta[:3]
            psi = float((psi + float(delta[3]) + math.pi) % (2.0 * math.pi) - math.pi)

            if float(np.linalg.norm(delta)) < _REFINE_TOL:
                break

        if t[0] <= 1e-3 or not np.all(np.isfinite(t)):
            return None, None

        # 质量门限：重投影误差过大说明这块板本帧的几何自相矛盾，
        # 与其把错位姿喂给跟踪器，不如这一帧不给观测（跟踪器会续瞄）。
        reproj = self._reproj_error(t, psi, obj, img, K, M, theta_0, f, cx, cy)
        if reproj is None or reproj > self.reproj_limit:
            return None, None
        return t, psi

    # ---------------------------------------------------------------- 校验
    @staticmethod
    def _reproj_error(t, psi, obj, img, K, M, theta_0, f, cx, cy):
        """(t_flu, psi) 下四个角点的重投影 RMS 误差（像素）；失败返回 None。"""
        cpsi, spsi = math.cos(psi), math.sin(psi)
        Rz = np.array([[cpsi, -spsi, 0.0], [spsi, cpsi, 0.0], [0.0, 0.0, 1.0]], float)
        ct, st = math.cos(theta_0), math.sin(theta_0)
        Ry = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]], float)
        R = M.T @ Rz @ Ry
        P = (R @ obj.T).T + t
        px = P[:, 0]
        if np.any(px <= 1e-3):
            return None
        u = -f * P[:, 1] / px + cx
        v = -f * P[:, 2] / px + cy
        err = np.hypot(u - img[:, 0], v - img[:, 1])
        if not np.all(np.isfinite(err)):
            return None
        return float(np.sqrt(np.mean(err * err)))

    # ---------------------------------------------------------------- 圆靶
    def _rune(self, det, gimbal_pos):
        """圆靶：靶心世界坐标由检测端直接给出，原样透传（只换表达系）。

        灯臂的法线恒为世界 -x（见 core/entities/rigids/rune.py），
        所以 rel_rpy 写死 [0, 0, pi]——渲染层拿它画朝向箭头。
        """
        if det.center is None:
            return None
        pos_world = np.asarray(det.center, float).reshape(3)
        return ArmorObservation(
            entity_type=EntityType.Rune,
            rel_pos=self._express(pos_world, gimbal_pos),
            rel_rpy=np.array([0.0, 0.0, math.pi], float),
            armor_size=det.armor_size,
            image_points_2d=(None if det.image_points_2d is None
                             else np.asarray(det.image_points_2d, float)),
            lit=bool(det.lit),
        )

    # ---------------------------------------------------------------- 表达系
    def _express(self, pos_world, gimbal_pos):
        """世界坐标 -> cfg.sim.frame 指定的表达系（G 相对云台中心，W 绝对）。"""
        if self.frame == "W":
            return np.asarray(pos_world, float).copy()
        return np.asarray(pos_world, float) - gimbal_pos
